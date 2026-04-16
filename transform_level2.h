#pragma once
#include <cassert>
#include "util.h"
#include "mxm.h"
#include "mxm_level2.h"

// transform_level2.h — L2: B staged into LDS once per block, then three mTxmq passes.

// Use a byte-typed shared buffer to avoid extern __shared__ type conflicts across TUs.
template <typename T>
__global__ void transform_kernel_L2(int nfuncs, int K,
                                     const T* __restrict__ A,
                                     const T* __restrict__ B,
                                     T* __restrict__ C,
                                     T* __restrict__ workspace)
{
    extern __shared__ unsigned char _shmem_l2[];
    T* b_shm = reinterpret_cast<T*>(_shmem_l2);

    // Stage B into LDS once per block (shared across all cubes in grid-stride loop).
    int nthr = blockDim.x * blockDim.y;
    int tid  = blockDim.x * threadIdx.y + threadIdx.x;
    for (int i = tid; i < K * K; i += nthr) b_shm[i] = B[i];
    __syncthreads();  // ensure B visible to all threads before first mTxmq

    const int K3 = K * K * K;
    T* w = workspace + (size_t)blockIdx.x * K3;  // one workspace slab per block

    for (int cube = blockIdx.x; cube < nfuncs; cube += gridDim.x) {
        const T* a = A + (size_t)cube * K3;
        T*       c = C + (size_t)cube * K3;

        // Three passes: ping-pong between c (output) and w (workspace).
        // Pass 1 → c, Pass 2 → w, Pass 3 → c  (result in c).
        T* t0 = c;
        T* t1 = w;
        const int dimi = K * K, dimj = K;

        mra::mTxmq(dimi, dimj, dimj, t0, a,  b_shm);  // pass 1 → c
        mra::mTxmq(dimi, dimj, dimj, t1, t0, b_shm);  // pass 2 → w
        mra::mTxmq(dimi, dimj, dimj, t0, t1, b_shm);  // pass 3 → c
        // mTxmq ends with SYNCTHREADS, so c is visible to all before next iteration.
    }
}

template<typename T>
inline size_type transform_L2_shmem_size(int K) {
    return mra::mTxmq_L2_shmem_size<T>(K);
}

template<typename T>
inline void submit_transform_bench_L2(int nfuncs, int nblocks, int K,
                                       const T* A, const T* B, T* C, T* workspace,
                                       Stream stream)
{
    Dim3 thread_dims = mra::mTxmq_L2_blockdim<T>(K);
    assert(block_size(thread_dims) <= MAX_THREADS_PER_BLOCK);
    size_type smem_size = mra::mTxmq_L2_shmem_size<T>(K);
    CONFIGURE_KERNEL(transform_kernel_L2<T>, smem_size);
    CALL_KERNEL(transform_kernel_L2<T>, std::min(nfuncs, nblocks), thread_dims, smem_size, stream,
                (nfuncs, K, A, B, C, workspace));
}
