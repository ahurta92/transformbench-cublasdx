/**
 * validate_levels.hip — correctness test for all optimization levels.
 *
 * Compares GPU output from the selected level against the CPU reference transform:
 *   result(a,b,c) = sum(i,j,k) A(i,j,k) * B(i,a) * B(j,b) * B(k,c)
 * implemented via three sequential mTxm passes.
 *
 * Usage:
 *   ./validate_levels -l <level> -K <K> -N <N>
 *   level: 1=L1-global, 2=L2-lds, 3=L3-regblk, 7=blocked (corner-turn)  (default: 3)
 *   K: transform order (default: 16)
 *   N: number of cubes (default: 8)
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "transform.h"
#include "transform_level2.h"
#include "transform_level3.h"
#include "transform_blocked.h"
#include "transform_blocked_rocwmma.h"
#include "util.h"

// ---------------------------------------------------------------------------
// CPU reference  (matches madness/src/madness/tensor/transform3d.cc)
// c(i,j) += sum(k) a(k,i) * b(k,j)
// ---------------------------------------------------------------------------
static void cpu_mTxm(int dimi, int dimj, int dimk,
                     double* c, const double* a, const double* b)
{
    for (int i = 0; i < dimi; ++i) {
        double*       ci  = c + i * dimj;
        const double* aik = a + i;          // a(k,i) strides by dimi
        for (int k = 0; k < dimk; ++k, aik += dimi) {
            double aki = *aik;
            for (int j = 0; j < dimj; ++j)
                ci[j] += aki * b[k * dimj + j];
        }
    }
}

static void cpu_transform3d(int K, const double* A, const double* B, double* result)
{
    int K3 = K * K * K;
    std::vector<double> tmp(K3, 0.0);
    std::fill(result, result + K3, 0.0);

    cpu_mTxm(K * K, K, K, result,      A,           B);  // pass 1
    cpu_mTxm(K * K, K, K, tmp.data(),  result,      B);  // pass 2
    std::fill(result, result + K3, 0.0);
    cpu_mTxm(K * K, K, K, result,      tmp.data(),  B);  // pass 3
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    auto opt   = OptionParser(argc, argv);
    int  K     = opt.parse("-K", 16);
    int  N     = opt.parse("-N", 8);
    int  level = opt.parse("-l", 3);

    const int K3 = K * K * K;

    // Deterministic inputs (same as validate.hip)
    std::vector<double> h_A(N * K3);
    std::vector<double> h_B(K * K);
    for (int i = 0; i < N * K3; ++i) h_A[i] = (double)(i % 13) * 0.1 - 0.6;
    for (int i = 0; i < K * K;  ++i) h_B[i] = (double)(i % 7)  * 0.1 - 0.3;

    // CPU reference
    std::vector<double> h_ref(N * K3);
    for (int n = 0; n < N; ++n)
        cpu_transform3d(K, h_A.data() + n * K3, h_B.data(), h_ref.data() + n * K3);

    // GPU buffers
    double *d_A, *d_B, *d_C, *d_workspace;
    MALLOC(&d_A,         (size_t)N * K3    * sizeof(double));
    MALLOC(&d_B,         (size_t)K * K     * sizeof(double));
    MALLOC(&d_C,         (size_t)N * K3    * sizeof(double));
    MALLOC(&d_workspace, (size_t)K3        * sizeof(double));  // 1 block

    MEMCPY_H2D(d_A, h_A.data(), (size_t)N * K3 * sizeof(double));
    MEMCPY_H2D(d_B, h_B.data(), (size_t)K * K  * sizeof(double));

    Stream stream;
    CREATE_STREAM(&stream);

    // Dispatch to selected level
    const int nblocks = 1;
    switch (level) {
        case 1:
            submit_transform_bench<double>(N, nblocks, K, d_A, d_B, d_C, d_workspace, stream);
            break;
        case 2:
            submit_transform_bench_L2<double>(N, nblocks, K, d_A, d_B, d_C, d_workspace, stream);
            break;
        case 3:
            submit_transform_bench_L3<double>(N, nblocks, K, d_A, d_B, d_C, d_workspace, stream);
            break;
        case 7:
            submit_transform_bench_blocked<double>(N, nblocks, K, d_A, d_B, d_C, d_workspace, stream);
            break;
        case 8:
            submit_transform_bench_blocked_rocwmma<double>(N, nblocks, K, d_A, d_B, d_C, d_workspace, stream);
            break;
        default:
            std::cerr << "Unknown level " << level << " (supported: 1, 2, 3, 7, 8)\n";
            return 1;
    }
    SYNC_STREAM(stream);

    std::vector<double> h_gpu(N * K3);
    MEMCPY_D2H(h_gpu.data(), d_C, (size_t)N * K3 * sizeof(double));

    FREE(d_A);  FREE(d_B);  FREE(d_C);  FREE(d_workspace);

    // Compare
    double max_abs_err = 0.0, max_ref_val = 0.0;
    for (int i = 0; i < N * K3; ++i) {
        double err = std::abs(h_gpu[i] - h_ref[i]);
        double ref = std::abs(h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (ref > max_ref_val) max_ref_val = ref;
    }
    double rel_err = (max_ref_val > 0.0) ? max_abs_err / max_ref_val : max_abs_err;
    bool   pass    = (rel_err < 1e-10);

    std::cout << "level=L" << level
              << " K=" << K
              << " N=" << N
              << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << rel_err
              << (pass ? "  PASS" : "  FAIL")
              << std::endl;

    return pass ? 0 : 1;
}
