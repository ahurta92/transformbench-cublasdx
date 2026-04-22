#pragma once
#include <cassert>
#include "util.h"

// transform_blocked.h — block-distributed 3D transform with AMD MFMA.
//
// One wavefront owns one K×K block of the tensor throughout the computation:
//   wave s initially holds A[:, :, k'=s];
//   after the corner turn, wave s holds result[a=s, :, :].
//
// Flow:
//   distribute      wave s loads A[:, :, k'=s]          (strided read from HBM)
//   Pass 1 (local)  blk_s <- B^T · blk_s                (a, j')
//   Pass 2 (local)  blk_s <- blk_s · B                  (a, b)
//   corner turn     LDS all-to-all: wave t ends up holding
//                   temp2[a=t, b, k']  stored as (b, k')
//   Pass 3 (local)  blk_t <- blk_t · B                  (b, c) — canonical
//   store           wave s writes result[s, :, :] to HBM
//
// Local GEMMs use v_mfma_f64_16x16x4f64 (GFX90A).  One MFMA does a 16×16
// output with K=4 contraction; for K=16 we chain 4 MFMAs per pass.
//
// LDS layout (34 KB at K=16):
//   buf[K³]     single K³ scratch, in-place across passes via register stash
//   B_lds[K²]   cached B matrix, shared across all waves
//
// Thread-block size: 64 × K = 1024 threads at K=16.

#if defined(__HIP__)
typedef double v4f64 __attribute__((ext_vector_type(4)));

__device__ inline v4f64 mfma_16x16x4_f64(double a, double b, v4f64 c) {
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}
#endif

template<typename T, int K>
__global__
__launch_bounds__(K * 64, 1)
void transform_kernel_blocked(int nfuncs,
                              const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C,
                              T* __restrict__ /*workspace unused*/)
{
    static_assert(std::is_same<T, double>::value, "MFMA path is FP64 only");
    static_assert(K == 16, "MFMA path: K=16 only for now");
    static_assert(K * K % 64 == 0, "K^2 must be a multiple of the wavefront size (64)");

    constexpr int K2 = K * K;
    constexpr int K3 = K * K * K;
    constexpr int ELEMS_PER_LANE = K2 / 64;  // 4 at K=16
    constexpr int NMFMA = K / 4;              // 4 MFMAs per K=16 pass

    // LDS bank-conflict pad: give each "row" inside a wave's K×K region an
    // extra column of dead storage so within-wave stride-K accesses (pass 2/3
    // A_frag loads, corner-turn cross-wave writes) don't collide on the same
    // 128-byte bank row.  Per-wave region becomes K × (K+1) instead of K × K.
    constexpr int BK = K + 1;          // padded inner (row) stride
    constexpr int WB = K * BK;         // per-wave region size (padded)

    extern __shared__ unsigned char _smem_blk[];
    T* buf   = reinterpret_cast<T*>(_smem_blk);
    T* B_lds = buf + K * WB;

    const int s = threadIdx.y;    // wave index (0..K-1)
    const int t = threadIdx.x;    // lane within wave (0..63)
    const int tid  = s * 64 + t;
    const int nthr = blockDim.x * blockDim.y;

    // Cache B into LDS once.
    for (int i = tid; i < K2; i += nthr) {
        B_lds[i] = B[i];
    }
    __syncthreads();

    for (int cube = blockIdx.x; cube < nfuncs; cube += gridDim.x) {
        const T* a_ptr = A + (size_t)cube * K3;
        T*       c_ptr = C + (size_t)cube * K3;

        // --- Distribute: wave s reads A[:, :, k'=s] into buf[s, :, :] ---
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = t + e * 64;
            int i = idx / K;
            int j = idx % K;
            buf[s*WB + i*BK + j] = a_ptr[i*K2 + j*K + s];
        }
        // No __syncthreads: pass 1 reads wave s's own region (intra-wave).

        // --- Pass 1 MFMA: out = B^T · blk  (rows: i' -> a) ---
        // GFX90A v_mfma_f64_16x16x4f64 confirmed layouts:
        //   A_frag (M=16, K=4):  lane t -> A[t%16][t/16]         (col-major)
        //   B_frag (K=4, N=16):  lane t -> B[t/16][t%16]         (row-major)
        //   D      (M=16, N=16): lane t acc[e] -> D[(t/16)+4*e][t%16]  (stride-4 rows)
        //
        // Pass 1: A_frag = B^T, B_frag = blk.
        //   A_frag[m][k] = B^T[m][p*4+k] = B[p*4+k][m]
        //   thread t contributes B[p*4 + t/16][t%16]
        //   B_frag[k][n] = blk[p*4+k][n]
        //   thread t contributes blk[p*4 + t/16][t%16]
        v4f64 acc = v4f64{0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for (int p = 0; p < NMFMA; ++p) {
            double a_val = B_lds[(p*4 + (t >> 4)) * K + (t & 15)];
            double b_val = buf[s*WB + (p*4 + (t >> 4)) * BK + (t & 15)];
            acc = mfma_16x16x4_f64(a_val, b_val, acc);
        }
        // No __syncthreads: wave in lockstep; reads done before writes start.

        // Store: thread t acc[e] -> D[(t/16) + 4*e][t%16]
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int row = (t >> 4) + 4 * e;      // a
            int col = t & 15;                // j
            buf[s*WB + row*BK + col] = acc[e];
        }
        // No __syncthreads: pass 2 reads wave s's own region (intra-wave).

        // --- Pass 2 MFMA: out = blk · B  (cols: j' -> b) ---
        //   A_frag[m][k] = blk[m][p*4+k]
        //   thread t contributes blk[t%16][p*4 + t/16]
        //   B_frag[k][n] = B[p*4+k][n]
        //   thread t contributes B[p*4 + t/16][t%16]
        acc = v4f64{0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for (int p = 0; p < NMFMA; ++p) {
            double a_val = buf[s*WB + (t & 15) * BK + (p*4 + (t >> 4))];
            double b_val = B_lds[(p*4 + (t >> 4)) * K + (t & 15)];
            acc = mfma_16x16x4_f64(a_val, b_val, acc);
        }
        // No __syncthreads: wave in lockstep; reads done before writes start.

        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int row = (t >> 4) + 4 * e;      // a
            int col = t & 15;                // b
            buf[s*WB + row*BK + col] = acc[e];
        }
        // No __syncthreads: corner turn stash reads wave s's own region (intra-wave).

        // --- Corner turn (in-place with stash; cross-wave writes) ---
        //   buf[a, b, s]  <-  buf[s, a, b]
        T stash[ELEMS_PER_LANE];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = t + e * 64;
            int a = idx / K;
            int b = idx % K;
            stash[e] = buf[s*WB + a*BK + b];
        }
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = t + e * 64;
            int a = idx / K;
            int b = idx % K;
            buf[a*WB + b*BK + s] = stash[e];
        }
        __syncthreads();

        // --- Pass 3 MFMA + store: out = blk · B, write directly to HBM ---
        // Same orientation as pass 2 (blk has (b, k') layout).
        acc = v4f64{0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for (int p = 0; p < NMFMA; ++p) {
            double a_val = buf[s*WB + (t & 15) * BK + (p*4 + (t >> 4))];
            double b_val = B_lds[(p*4 + (t >> 4)) * K + (t & 15)];
            acc = mfma_16x16x4_f64(a_val, b_val, acc);
        }
        // No sync: acc is in registers, about to write to HBM.

        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            int row = (t >> 4) + 4 * e;      // b
            int col = t & 15;                // c
            c_ptr[s*K2 + row*K + col] = acc[e];
        }
        // No __syncthreads: next iter's distribute writes only wave s's own
        // buf region; other waves may still be in pass 3 on the current
        // tensor reading their own region — no cross-wave conflict.
    }
}

template<typename T>
inline size_type blocked_shmem_size(int K) {
    // Padded buf (K × K × (K+1)) + B cache (K × K)
    return (size_type)((K * K * (K + 1) + K * K) * sizeof(T));
}

template<typename T>
inline Dim3 blocked_blockdim(int K) {
    return Dim3(64, K, 1);
}

template<typename T>
inline void submit_transform_bench_blocked(int nfuncs, int nblocks, int K,
                                           const T* A, const T* B, T* C, T* workspace,
                                           Stream stream)
{
    if (K == 16) {
        constexpr int Kv = 16;
        Dim3      td   = blocked_blockdim<T>(Kv);
        size_type smem = blocked_shmem_size<T>(Kv);
        CONFIGURE_KERNEL((transform_kernel_blocked<T, Kv>), smem);
        CALL_KERNEL((transform_kernel_blocked<T, Kv>), std::min(nfuncs, nblocks), td, smem, stream,
                    (nfuncs, A, B, C, workspace));
    } else {
        fprintf(stderr, "blocked transform: K=%d not supported (MFMA path: K=16 only)\n", K);
        assert(false);
    }
}
