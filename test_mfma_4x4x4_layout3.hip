// Third probe pass for v_mfma_f64_4x4x4f64.
// Goal: nail down the B fragment layout.
//
// From the single-A-lane probes we learned:
//   A at lane S (a=1, rest 0)  -->  output 1 at 4 lanes = 4*m + {0,1,2,3}
//   where m = 4*(S%4) + (S/4)%4, k = S/16.
//   (i.e. instruction does D_{16x4} = A_{16x4} * B_{4x4},
//    A[m][k] at lane (16k + (m%4)*4 + m/4).)
//
// Now: isolate B. Feed a=1 everywhere, b=1 only at one specific lane X.
// D[m][n] = sum_k A[m][k] * B[k][n] * 1_{lane has that value}
//         = A[m][k_X] * 1 * [n == n_X]   (since A=1, only k_X contributes)
// Output=1 wherever t%4 == n_X.  Reveals n_X per lane.

#include <cstdio>
#include <hip/hip_runtime.h>

__device__ inline double mfma(double a, double b, double c) {
    return __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}

template<int X>
__global__ void probe_b(double* out) {
    int t = threadIdx.x;
    double b = (t == X) ? 1.0 : 0.0;
    out[t] = mfma(1.0, b, 0.0);
}

static void run(const char* name, void (*kernel)(double*)) {
    double *d_out, h[64];
    hipMalloc(&d_out, 64 * sizeof(double));
    kernel<<<1, 64>>>(d_out);
    hipDeviceSynchronize();
    hipMemcpy(h, d_out, 64 * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_out);

    printf("\n=== %s ===\n", name);
    // Only show where output != 0 and mark block/row structure.
    int nnz = 0;
    for (int t = 0; t < 64; ++t) if (h[t] != 0.0) ++nnz;
    printf("  nonzero lanes: %d\n", nnz);
    for (int t = 0; t < 64; ++t) {
        if (h[t] != 0.0) {
            int m = t / 4;
            int n = t % 4;
            printf("    lane %2d -> D[m=%2d][n=%d] = %.1f\n", t, m, n, h[t]);
        }
    }
}

int main() {
    run("B at lane 0 only",  probe_b<0>);
    run("B at lane 1 only",  probe_b<1>);
    run("B at lane 2 only",  probe_b<2>);
    run("B at lane 3 only",  probe_b<3>);
    run("B at lane 4 only",  probe_b<4>);
    run("B at lane 15 only", probe_b<15>);
    run("B at lane 16 only", probe_b<16>);
    run("B at lane 17 only", probe_b<17>);
    return 0;
}
