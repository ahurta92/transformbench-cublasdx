#pragma once
#include <cassert>
#include "util.h"

// transform_blocked.h — block-distributed 3D transform.
//
// One wavefront owns one K×K block of the tensor throughout the computation:
//   wave s initially holds A[:, :, k'=s];
//   after the corner turn, wave s holds result[a=s, :, :].
//
// Flow:
//   distribute      wave s loads A[:, :, k'=s]          (strided read from HBM)
//   Pass 1 (local)  blk_s <- B^T · blk_s                (a, j')
//   Pass 2 (local)  blk_s <- blk_s · B                  (a, b)
//   corner turn     exchange through LDS: wave t ends up holding
//                   temp2[a=t, b, k']  stored as (b, k')
//   Pass 3 (local)  blk_t <- blk_t · B                  (b, c) — canonical
//   store           wave s writes result[s, :, :] to HBM
//
// Pass 3 right-multiplies by B so the output lands in canonical (b, c) order
// with no post-store transpose.
//
// This file: scalar FP64 ops, single-buffer + B-in-LDS.  K=16 only.
// Thread-block size: 64 × K = 1024 threads at K=16 (wavefront size × K waves).
//
// LDS layout (34 KB at K=16, double):
//   buf[K³]     single K³ scratch, reused across passes via register staging
//   B_lds[K²]   cached B matrix, shared across all waves
//
// Each pass computes its K²/64 output elements per lane into a register stash
// (acc[]), __syncthreads, then writes back to buf -- allowing the same buffer
// to serve as both input and output of the pass.  The corner turn uses the
// same stash pattern to exchange across wave boundaries in place.

template<typename T, int K>
__global__
__launch_bounds__(K * 64, 1)
void transform_kernel_blocked(int nfuncs,
                              const T* __restrict__ A,
                              const T* __restrict__ B,
                              T* __restrict__ C,
                              T* __restrict__ /*workspace unused*/)
{
    static_assert(K * K % 64 == 0, "K^2 must be a multiple of the wavefront size (64)");

    constexpr int K2 = K * K;
    constexpr int K3 = K * K * K;
    constexpr int ELEMS_PER_LANE = K2 / 64;  // 4 at K=16

    extern __shared__ unsigned char _smem_blk[];
    T* buf   = reinterpret_cast<T*>(_smem_blk);
    T* B_lds = buf + K3;

    const int s    = threadIdx.y;    // wave index (0..K-1); k' initially, a after corner turn
    const int lane = threadIdx.x;    // 0..63
    const int tid  = s * 64 + lane;
    const int nthr = blockDim.x * blockDim.y;

    // Cache B into LDS once per kernel invocation.
    #pragma unroll
    for (int i = tid; i < K2; i += nthr) {
        B_lds[i] = B[i];
    }
    __syncthreads();

    // Per-lane register stash for in-place passes.
    T acc[ELEMS_PER_LANE];

    for (int cube = blockIdx.x; cube < nfuncs; cube += gridDim.x) {
        const T* a_ptr = A + (size_t)cube * K3;
        T*       c_ptr = C + (size_t)cube * K3;

        // --- Distribute: wave s reads slab A[:, :, k'=s] into buf[s*K² ..] ---
        // buf[s, i, j] = A[i, j, s]   (strided read from A's flat layout)
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int i = idx / K;
            int j = idx % K;
            buf[s*K2 + i*K + j] = a_ptr[i*K2 + j*K + s];
        }
        __syncthreads();

        // --- Pass 1 (in-place): blk_s <- B^T · blk_s   (rows: i' -> a) ---
        // acc[e] = sum_i B[i, a] * buf[s, i, j]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int j = idx % K;
            T sum = T(0);
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                sum += B_lds[i*K + a] * buf[s*K2 + i*K + j];
            }
            acc[e] = sum;
        }
        __syncthreads();  // all reads done across wave before any lane writes
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int j = idx % K;
            buf[s*K2 + a*K + j] = acc[e];
        }
        __syncthreads();

        // --- Pass 2 (in-place): blk_s <- blk_s · B   (cols: j' -> b) ---
        // acc[e] = sum_j buf[s, a, j] * B[j, b]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int b = idx % K;
            T sum = T(0);
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                sum += buf[s*K2 + a*K + j] * B_lds[j*K + b];
            }
            acc[e] = sum;
        }
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int b = idx % K;
            buf[s*K2 + a*K + b] = acc[e];
        }
        __syncthreads();

        // --- Corner turn (in-place with cross-wave writes): stash, sync, write ---
        // Wave s reads its own region (wave s owns temp2[s, :, :]) into acc,
        // then writes to destination wave = a at position (b, k'=s).
        //   buf[a*K² + b*K + s] <- buf[s*K² + a*K + b]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int b = idx % K;
            acc[e] = buf[s*K2 + a*K + b];
        }
        __syncthreads();  // every wave has read its entire region
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int a = idx / K;
            int b = idx % K;
            buf[a*K2 + b*K + s] = acc[e];
        }
        __syncthreads();

        // --- Pass 3 + store (fused): blk_t <- blk_t · B, write directly to HBM ---
        // Wave s now represents a=s.  Data lives at buf[s, b, k'].
        // C[s, b, c] = sum_k buf[s, b, k] * B[k, c]
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int idx = lane + e * 64;
            int b = idx / K;
            int c = idx % K;
            T sum = T(0);
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                sum += buf[s*K2 + b*K + k] * B_lds[k*K + c];
            }
            c_ptr[s*K2 + b*K + c] = sum;
        }
        // Barrier before next `cube` iteration overwrites buf in the distribute step.
        __syncthreads();
    }
}

template<typename T>
inline size_type blocked_shmem_size(int K) {
    // One K³ scratch + one K² B cache
    return (size_type)((K * K * K + K * K) * sizeof(T));
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
        fprintf(stderr, "blocked transform: K=%d not supported (first cut: K=16 only)\n", K);
        assert(false);
    }
}
