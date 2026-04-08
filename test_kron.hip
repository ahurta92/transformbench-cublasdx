/**
 * Correctness test: compare level 3 (reference) vs level 6 (Kronecker GEMM).
 *
 * Runs both on the same A and B, then checks max absolute difference.
 */

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "util.h"
#include "transform_level3.h"
#include "transform_kron.h"

template <typename T>
void test_kron(int K, int nfuncs) {
    const int K3 = K * K * K;

    // Allocate and fill host arrays with random data
    std::vector<T> h_A(nfuncs * K3), h_B(K * K), h_Cref(nfuncs * K3), h_Ckron(nfuncs * K3);
    std::srand(42);
    for (auto& v : h_A) v = (T)std::rand() / RAND_MAX;
    for (auto& v : h_B) v = (T)std::rand() / RAND_MAX;

    // Device allocations
    T *d_A, *d_B, *d_Cref, *d_Ckron, *d_workspace, *d_KronMat;
    MALLOC(&d_A,         nfuncs * K3 * sizeof(T));
    MALLOC(&d_B,         K * K * sizeof(T));
    MALLOC(&d_Cref,      nfuncs * K3 * sizeof(T));
    MALLOC(&d_Ckron,     nfuncs * K3 * sizeof(T));
    MALLOC(&d_workspace, nfuncs * K3 * sizeof(T));  // enough for level 3
    MALLOC(&d_KronMat,   (size_t)K3 * K3 * sizeof(T));

    // Copy inputs to device
#ifdef MRA_HAVE_HIP
    hipMemcpy(d_A, h_A.data(), nfuncs * K3 * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), K * K * sizeof(T),       hipMemcpyHostToDevice);
#else
    cudaMemcpy(d_A, h_A.data(), nfuncs * K3 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * K * sizeof(T),       cudaMemcpyHostToDevice);
#endif

    Stream stream;
    CREATE_STREAM(&stream);

    // --- Reference: level 3 ---
    submit_transform_level3_bench<T>(nfuncs, nfuncs, K, d_A, d_B, d_Cref, d_workspace, stream);
    SYNC_STREAM(stream);

    // --- Level 6: Kronecker GEMM ---
    build_kron_matrix<T>(K, d_B, d_KronMat, stream);
    SYNC_STREAM(stream);

    blasHandle_t blas_handle;
    blasCreate(&blas_handle);
    submit_transform_kron_bench<T>(nfuncs, K, d_A, d_KronMat, d_Ckron, blas_handle, stream);
    SYNC_STREAM(stream);
    blasDestroy(blas_handle);

    // Copy results to host
#ifdef MRA_HAVE_HIP
    hipMemcpy(h_Cref.data(),  d_Cref,  nfuncs * K3 * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(h_Ckron.data(), d_Ckron, nfuncs * K3 * sizeof(T), hipMemcpyDeviceToHost);
#else
    cudaMemcpy(h_Cref.data(),  d_Cref,  nfuncs * K3 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ckron.data(), d_Ckron, nfuncs * K3 * sizeof(T), cudaMemcpyDeviceToHost);
#endif

    // Compare
    T max_abs_err = 0, max_rel_err = 0;
    for (int i = 0; i < nfuncs * K3; ++i) {
        T abs_err = std::abs(h_Cref[i] - h_Ckron[i]);
        T rel_err = abs_err / (std::abs(h_Cref[i]) + 1e-14);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    std::cout << "K=" << K << " nfuncs=" << nfuncs
              << "  max_abs_err=" << max_abs_err
              << "  max_rel_err=" << max_rel_err
              << (max_rel_err < 1e-10 ? "  PASS" : "  FAIL")
              << "\n";

    FREE(d_A); FREE(d_B); FREE(d_Cref); FREE(d_Ckron);
    FREE(d_workspace); FREE(d_KronMat);
    hipStreamDestroy(stream);
}

int main() {
    for (int K : {4, 6, 8, 10}) {
        test_kron<double>(K, 16);
    }
}
