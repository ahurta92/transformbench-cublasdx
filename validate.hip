/**
 * validate.hip — compare GPU L1 output against the CPU reference transform.
 *
 * CPU reference mirrors madness/src/madness/tensor/transform3d.cc :: transform3d<T>():
 *
 *   result(a,b,c) = sum(i',j',k') A(i',j',k') C(i',a) C(j',b) C(k',c)
 *
 * implemented as three sequential mTxm passes:
 *   mTxm(K², K, K, r,   A, B)    pass 1: contract dim-0
 *   mTxm(K², K, K, tmp, r, B)    pass 2: contract dim-1
 *   mTxm(K², K, K, r, tmp, B)    pass 3: contract dim-2
 *
 * where  mTxm(dimi, dimj, dimk, c, a, b):
 *   c(i,j) += sum(k)  a(k,i) * b(k,j)
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "transform.h"
#include "util.h"

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

// c(i,j) += sum(k) a(k,i) * b(k,j)   (from MADNESS mTxm)
static void cpu_mTxm(int dimi, int dimj, int dimk,
                     double* c, const double* a, const double* b) {
    for (int i = 0; i < dimi; ++i) {
        double* ci         = c + i * dimj;
        const double* aik  = a + i;          // a(k,i) strides by dimi
        for (int k = 0; k < dimk; ++k, aik += dimi) {
            double aki = *aik;
            for (int j = 0; j < dimj; ++j)
                ci[j] += aki * b[k * dimj + j];
        }
    }
}

// Three-pass transform matching transform3d.cc
static void cpu_transform3d(int K, const double* A, const double* B, double* result) {
    int K3 = K * K * K;
    std::vector<double> tmp(K3, 0.0);
    std::fill(result, result + K3, 0.0);

    cpu_mTxm(K * K, K, K, result,    A,           B);  // pass 1
    cpu_mTxm(K * K, K, K, tmp.data(), result,     B);  // pass 2
    std::fill(result, result + K3, 0.0);
    cpu_mTxm(K * K, K, K, result,    tmp.data(),  B);  // pass 3
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto opt = OptionParser(argc, argv);
    int K = opt.parse("-K", 16);
    int N = opt.parse("-N", 8);

    const int K3 = K * K * K;

    // Deterministic input
    std::vector<double> h_A(N * K3);
    std::vector<double> h_B(K * K);
    for (int i = 0; i < N * K3; ++i) h_A[i] = (double)(i % 13) * 0.1 - 0.6;
    for (int i = 0; i < K * K;  ++i) h_B[i] = (double)(i % 7)  * 0.1 - 0.3;

    // CPU reference
    std::vector<double> h_ref(N * K3);
    for (int n = 0; n < N; ++n)
        cpu_transform3d(K, h_A.data() + n * K3, h_B.data(), h_ref.data() + n * K3);

    // GPU L1
    double *d_A, *d_B, *d_C, *d_workspace;
    MALLOC(&d_A,         N * K3    * sizeof(double));
    MALLOC(&d_B,             K * K * sizeof(double));
    MALLOC(&d_C,         N * K3    * sizeof(double));
    MALLOC(&d_workspace,     K3    * sizeof(double));  // 1 block

    MEMCPY_H2D(d_A, h_A.data(), N * K3    * sizeof(double));
    MEMCPY_H2D(d_B, h_B.data(),     K * K * sizeof(double));

    Stream stream;
    CREATE_STREAM(&stream);
    submit_transform_bench<double>(N, /*nblocks=*/1, K, d_A, d_B, d_C, d_workspace, stream);
    SYNC_STREAM(stream);

    std::vector<double> h_gpu(N * K3);
    MEMCPY_D2H(h_gpu.data(), d_C, N * K3 * sizeof(double));

    FREE(d_A); FREE(d_B); FREE(d_C); FREE(d_workspace);

    // Compare
    double max_abs_err = 0.0, max_ref_val = 0.0;
    for (int i = 0; i < N * K3; ++i) {
        double err = std::abs(h_gpu[i] - h_ref[i]);
        double ref = std::abs(h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (ref > max_ref_val) max_ref_val = ref;
    }
    double rel_err = (max_ref_val > 0.0) ? max_abs_err / max_ref_val : max_abs_err;

    bool pass = rel_err < 1e-10;
    std::cout << "K=" << K
              << " N=" << N
              << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << rel_err
              << (pass ? "  PASS" : "  FAIL")
              << std::endl;

    return pass ? 0 : 1;
}
