#pragma once

/**
 * Level 6: Single GEMM using the Kronecker product.
 *
 * The 3-pass transform  C = B^T x (B^T x (B^T x A))  can be written as a
 * single matrix-vector multiply per tensor:
 *
 *   vec(C) = (B^T ⊗ B^T ⊗ B^T) · vec(A)
 *
 * where ⊗ is the Kronecker product, giving a K³×K³ matrix.
 * All nfuncs tensors are batched into one GEMM:
 *
 *   C[K³ × N] = KronMat[K³ × K³] × A[K³ × N]
 *
 * Pros: one API call, large GEMM → high GPU utilization for small K.
 * Cons: K⁶ FLOPs vs 3K⁴ for the sequential approach (12-33× more work
 *       for K=6..10), and K⁶ memory for KronMat.
 */

#include "util.h"
#ifdef MRA_HAVE_HIP
#  include <hipblas/hipblas.h>
   using blasHandle_t = hipblasHandle_t;
#  define BLAS_OP_N    HIPBLAS_OP_N
#  define blasCreate   hipblasCreate
#  define blasDestroy  hipblasDestroy
#  define blasSetStream hipblasSetStream
#  define blasDgemm    hipblasDgemm
#elif defined(MRA_HAVE_CUDA)
#  include <cublas_v2.h>
   using blasHandle_t = cublasHandle_t;
#  define BLAS_OP_N    CUBLAS_OP_N
#  define blasCreate   cublasCreate
#  define blasDestroy  cublasDestroy
#  define blasSetStream cublasSetStream
#  define blasDgemm    cublasDgemm
#endif

// ---------------------------------------------------------------------------
// Kernel: build the K³×K³ Kronecker product matrix (column-major).
//
//   KronMat[I, J] = B^T[i₀,j₀] · B^T[i₁,j₁] · B^T[i₂,j₂]
//
// Index decomposition (first index fastest = column-major vector):
//   I = i₀ + K·i₁ + K²·i₂
//   J = j₀ + K·j₁ + K²·j₂
//
// B is row-major K×K, so B^T[i,j] = B[j·K + i].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void build_kron_kernel(int K, const T* __restrict__ B,
                                   T* __restrict__ KronMat)
{
    const int K3 = K * K * K;
    const int I  = blockIdx.x * blockDim.x + threadIdx.x;
    const int J  = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= K3 || J >= K3) return;

    const int i0 = I % K,        j0 = J % K;
    const int i1 = (I / K) % K,  j1 = (J / K) % K;
    const int i2 = I / (K * K),  j2 = J / (K * K);

    // B^T[i,j] = B[j*K + i]  (B is row-major)
    KronMat[I + J * K3] = B[j0*K + i0] * B[j1*K + i1] * B[j2*K + i2];
}

// ---------------------------------------------------------------------------
// Build the Kronecker matrix on the device (call once before timing).
// KronMat must already be allocated with K³×K³ elements.
// ---------------------------------------------------------------------------
template <typename T>
inline void build_kron_matrix(int K, const T* B_dev, T* KronMat_dev,
                               Stream stream)
{
    const int K3 = K * K * K;
    dim3 block(16, 16);
    dim3 grid((K3 + 15) / 16, (K3 + 15) / 16);
    CALL_KERNEL(build_kron_kernel<T>, grid, block, 0, stream,
                (K, B_dev, KronMat_dev));
}

// ---------------------------------------------------------------------------
// Submit one round of the Kronecker GEMM (called inside the timing loop).
//
//   C[K³ × nfuncs] = KronMat[K³ × K³] × A[K³ × nfuncs]
//
// A and C are treated as column-major (each contiguous K³-block = one tensor).
// ---------------------------------------------------------------------------
template <typename T>
inline void submit_transform_kron_bench(int nfuncs, int K,
                                         const T* A, const T* KronMat, T* C,
                                         blasHandle_t blas_handle,
                                         Stream stream)
{
    const int K3 = K * K * K;
    const double alpha = 1.0, beta = 0.0;
    blasSetStream(blas_handle, stream);
    blasDgemm(blas_handle,
              BLAS_OP_N, BLAS_OP_N,
              K3, nfuncs, K3,
              &alpha,
              KronMat, K3,
              A,       K3,
              &beta,
              C,       K3);
}

// Required by the benchmark dispatch (values are unused for level 6).
template <typename T>
inline int kron_shmem_size(int /*K*/) { return 0; }

inline Dim3 kron_blockdim(int /*K*/) { return {1, 1, 1}; }
