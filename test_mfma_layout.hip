// Diagnostic: probe the MFMA output layout for v_mfma_f64_16x16x4f64.
// A = identity-like (A[m][k] = 1 if m==k else 0), B = all ones.
// Expected D[m][n] = sum_k A[m][k] * B[k][n] = (m < 4 ? 1.0 : 0.0)
//
// Write acc[0..3] per-thread to host-readable memory, inspect layout.

#include <cstdio>
#include <hip/hip_runtime.h>

typedef double v4f64 __attribute__((ext_vector_type(4)));

__global__ void probe_mfma(double* out) {
    int t = threadIdx.x;

    // A: col-major (lane t -> A[t%16][t/16]);  value A[m][k] = 1
    // B: hypothesis lane t -> B[t/16][t%16];   value B[k][n] = n
    // -> D[m][n] = sum_k 1 * n = 4n  (for K=4)
    // Distinct D per column -> reveals col mapping per lane.
    double a_val = 1.0;
    double b_val = (double)(t & 15);

    v4f64 acc = v4f64{0.0, 0.0, 0.0, 0.0};
    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(a_val, b_val, acc, 0, 0, 0);

    for (int e = 0; e < 4; ++e) {
        out[t * 4 + e] = acc[e];
    }
}

int main() {
    double *d_out, h_out[256];
    hipMalloc(&d_out, 256 * sizeof(double));
    probe_mfma<<<1, 64>>>(d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, 256 * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_out);

    // A has diagonal 1s (m==k for m,k in 0..3), zero elsewhere.  B = all ones (1 everywhere).
    // Expected D[m][n] = 1 if m<4 else 0 (for all n in 0..15).
    //
    // Print acc[0..3] per lane so we can work out which (m, n) each acc[e] corresponds to.
    printf("lane | acc[0] acc[1] acc[2] acc[3]\n");
    for (int t = 0; t < 64; ++t) {
        printf("%3d  | %4.1f %4.1f %4.1f %4.1f\n",
               t, h_out[t*4], h_out[t*4+1], h_out[t*4+2], h_out[t*4+3]);
    }
    return 0;
}
