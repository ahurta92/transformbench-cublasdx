#pragma once
#include "util.h"

// mxm_level2.h — L2 metadata: B staged in LDS by the transform kernel.
// The actual mTxmq computation reuses mra::mTxmq from mxm.h (B pointer is LDS).

namespace mra {

template<typename T>
constexpr size_type mTxmq_L2_shmem_size(size_type K) {
    return (size_type)(K * K * sizeof(T));
}

template<typename T>
constexpr Dim3 mTxmq_L2_blockdim(int K) {
    return max_thread_dims(K);
}

} // namespace mra
