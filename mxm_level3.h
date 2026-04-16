#pragma once
#include "util.h"

// mxm_level3.h — L3: register-blocked mTxmq, K compile-time template parameter.
//
// Algorithm (per thread, linear tid over K² rows):
//   acc[K] = 0                            // K doubles in VGPRs
//   for k in 0..K-1:
//       aki = a[k * K² + i]               // one global load per k
//       for j in 0..K-1:  [unrolled]
//           acc[j] += aki * b_shm[k*K+j]  // LDS reads
//   for j in 0..K-1:  [unrolled]
//       c[i*K + j] = acc[j]               // one store per j
//
// c(i,j) = sum_k a(k,i)*b(k,j)  [Q-mode: c zeroed via acc initialisation]

namespace mra {

template<typename T, int K>
__device__ void mTxmq_L3(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b_shm)
{
    constexpr int dimi = K * K;  // number of output rows
    constexpr int dimj = K;       // number of output cols

    int nthr = blockDim.x * blockDim.y;
    int tid  = blockDim.x * threadIdx.y + threadIdx.x;

    for (int i = tid; i < dimi; i += nthr) {
        T acc[K];
        #pragma unroll
        for (int j = 0; j < K; ++j) acc[j] = T(0);

        for (int k = 0; k < K; ++k) {
            T aki = a[k * dimi + i];   // global load: a(k, i)
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                acc[j] += aki * b_shm[k * dimj + j];  // LDS: b(k, j)
            }
        }

        #pragma unroll
        for (int j = 0; j < K; ++j) {
            c[i * dimj + j] = acc[j];
        }
    }
    __syncthreads();  // output fully written before caller reads it
}

template<typename T>
constexpr size_type L3_shmem_size(int K) {
    return K * K * sizeof(T);  // same as L2 — one K×K matrix in LDS
}

template<typename T>
constexpr Dim3 mTxmq_L3_blockdim(int K) {
    return max_thread_dims(K);
}

} // namespace mra
