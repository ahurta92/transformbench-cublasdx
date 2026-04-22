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

        // --- Distribute: cooperative coalesced load of the K^3 tensor ---
        // All blockDim threads pull K^3 doubles from HBM with stride-1 reads
        // (one cache line per 16 consecutive lanes), placing them in buf in
        // canonical (i, j, k) layout: buf[i*WB + j*BK + k] = A[i, j, k].
        //
        // This changes buf's meaning:
        //   OLD  (per-wave):  buf[s*WB + i*BK + j] = A[i, j, k'=s]   (s slowest)
        //   NEW  (canonical): buf[i*WB + j*BK + k] = A[i, j, k]      (k fastest)
        // The strides WB and BK coincide numerically, so post-corner-turn
        // accesses (which treat buf as (a, b, k')) continue to work with the
        // same indexing.  Only the distribute and pass-1 reads need to change.
        {
            constexpr int NTHR = K * 64;    // block size
            constexpr int NITER = K3 / NTHR; // 4 for K=16
            #pragma unroll
            for (int e = 0; e < NITER; ++e) {
                int idx = tid + e * NTHR;
                int i = idx / K2;
                int j = (idx / K) % K;
                int k = idx % K;
                buf[i*WB + j*BK + k] = a_ptr[idx];
            }
        }
        __syncthreads();  // cross-wave writes visible before pass-1 reads

        // --- Pass 1 MFMA (SWAPPED OPERANDS): compute blk^T · B instead of B^T · blk ---
        // GFX90A v_mfma_f64_16x16x4f64 confirmed layouts:
        //   A_frag (M=16, K=4):  lane t -> A[t%16][t/16]         (col-major)
        //   B_frag (K=4, N=16):  lane t -> B[t/16][t%16]         (row-major)
        //   D      (M=16, N=16): lane t acc[e] -> D[(t/16)+4*e][t%16]
        //
        // By feeding A_frag = blk^T (from buf, treating (i,j) as if transposed)
        // and B_frag = B (from B_lds), MFMA produces D where
        //   D[j][a] = sum_i blk[i][j] * B[i][a] = temp1[a][j]
        // i.e. temp1 stored with its axes swapped in the register file:
        //   lane t, acc[e]  ==  temp1[t%16][(t/16) + 4*e]
        //
        // This is exactly the layout pass 2's A_frag wants at iter p = e,
        // so pass 2 can consume pass 1's acc directly -- no LDS round trip.
        // With canonical buf layout, thread t's A_frag contribution at iter p
        // lives at buf[i*WB + j*BK + k] with i=p*4+t/16, j=t%16, k=s.
        v4f64 acc1 = v4f64{0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for (int p = 0; p < NMFMA; ++p) {
            double a_val = buf[(p*4 + (t >> 4)) * WB + (t & 15) * BK + s];
            double b_val = B_lds[(p*4 + (t >> 4)) * K + (t & 15)];
            acc1 = mfma_16x16x4_f64(a_val, b_val, acc1);
        }
        // No LDS writeback: acc1[p] is pass 2's a_val at iter p directly.

        // --- Pass 2 MFMA: out = temp1 · B  (cols: j' -> b) ---
        //   A_frag = temp1 (from acc1), B_frag = B.
        //   thread t at iter p uses acc1[p] = temp1[t%16][p*4 + t/16]
        v4f64 acc = v4f64{0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for (int p = 0; p < NMFMA; ++p) {
            double a_val = acc1[p];
            double b_val = B_lds[(p*4 + (t >> 4)) * K + (t & 15)];
            acc = mfma_16x16x4_f64(a_val, b_val, acc);
        }
        // Pass 2 output is in acc.  The corner-turn stash step used to read
        // pass 2's buf at positions (a = t/16 + 4e, b = t%16) -- exactly the
        // positions MFMA already deposited in acc.  So acc[e] IS the stash;
        // we can skip both the pass-2 store and the stash read, and write
        // acc directly into the cross-wave destination below.

        __syncthreads();  // all waves done with pass-2 reads from own region

        // --- Corner turn (cross-wave write, fused with pass-2 store) ---
        //   buf[a*WB + b*BK + s]  <-  acc[e]   where (a, b) = (t/16 + 4e, t%16)
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = t + e * 64;
            int a = idx / K;
            int b = idx % K;
            buf[a*WB + b*BK + s] = acc[e];
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
