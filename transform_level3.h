#pragma once
#include <cassert>
#include "util.h"
#include "mxm_level3.h"
#include "transform_level2.h"  // fallback for K not in {16, 32}

// transform_level3.h — L3: K-templated register-blocking kernel.
// Falls back to L2 for K values other than 16 and 32.

template<typename T, int K>
__global__ void transform_kernel_L3(int nfuncs,
                                     const T* __restrict__ A,
                                     const T* __restrict__ B,
                                     T* __restrict__ C,
                                     T* __restrict__ workspace)
{
    extern __shared__ unsigned char _shmem_l3[];
    T* b_shm = reinterpret_cast<T*>(_shmem_l3);

    // Stage B into LDS once per block.
    constexpr int K2 = K * K;
    int nthr = blockDim.x * blockDim.y;
    int tid  = blockDim.x * threadIdx.y + threadIdx.x;
    for (int i = tid; i < K2; i += nthr) b_shm[i] = B[i];
    __syncthreads();

    constexpr int K3 = K * K * K;
    T* w = workspace + (size_t)blockIdx.x * K3;

    for (int cube = blockIdx.x; cube < nfuncs; cube += gridDim.x) {
        const T* a = A + (size_t)cube * K3;
        T*       c = C + (size_t)cube * K3;

        // Pass 1 → c, Pass 2 → w, Pass 3 → c.
        mra::mTxmq_L3<T, K>(c, a,  b_shm);  // pass 1 → c
        mra::mTxmq_L3<T, K>(w, c,  b_shm);  // pass 2 → w
        mra::mTxmq_L3<T, K>(c, w,  b_shm);  // pass 3 → c
    }
}

template<typename T>
inline void submit_transform_bench_L3(int nfuncs, int nblocks, int K,
                                       const T* A, const T* B, T* C, T* workspace,
                                       Stream stream)
{
    if (K == 16) {
        constexpr int Kv = 16;
        Dim3      td   = mra::mTxmq_L3_blockdim<T>(Kv);
        size_type smem = mra::L3_shmem_size<T>(Kv);
        CONFIGURE_KERNEL((transform_kernel_L3<T, Kv>), smem);
        CALL_KERNEL((transform_kernel_L3<T, Kv>), std::min(nfuncs, nblocks), td, smem, stream,
                    (nfuncs, A, B, C, workspace));
    } else if (K == 32) {
        constexpr int Kv = 32;
        Dim3      td   = mra::mTxmq_L3_blockdim<T>(Kv);
        size_type smem = mra::L3_shmem_size<T>(Kv);
        CONFIGURE_KERNEL((transform_kernel_L3<T, Kv>), smem);
        CALL_KERNEL((transform_kernel_L3<T, Kv>), std::min(nfuncs, nblocks), td, smem, stream,
                    (nfuncs, A, B, C, workspace));
    } else {
        // Fall back to L2 for other K values.
        //submit_transform_bench_L2<T>(nfuncs, nblocks, K, A, B, C, workspace, stream);
        fprintf(stderr, "Unsupported K=%d for L3 kernel\n", K);
        assert(false);

    }
}
