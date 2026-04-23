// Diagnostic: probe the v_mfma_f64_4x4x4f64 layout on GFX90A.
//
// Output type is a single double per lane (64 lanes * 1 = 64 outputs total =
// four 4x4 tiles, or four replicas of one 4x4 tile).  Question: is it
// 4-broadcast (single A, single B, four D replicas) or 4-independent
// (four separate 4x4 GEMMs)?
//
// By analogy with v_mfma_f64_16x16x4f64, the natural layout hypothesis is:
//   A (M=4, K=4):   lane t -> A[t%4][(t/4)%4]   (col-major within 16 lanes)
//   B (K=4, N=4):   lane t -> B[(t/4)%4][t%4]   (row-major within 16 lanes)
//   D (M=4, N=4):   lane t -> D[(t/4)%4][t%4]   (row-major within 16 lanes)
// with the block/replica index = t/16.
//
// Tests below feed distinguishing patterns to confirm (or refute) each piece.

#include <cstdio>
#include <hip/hip_runtime.h>

__device__ inline double mfma_4x4x4_f64(double a, double b, double c) {
    return __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}

// Test 1: all-ones. Expected D[m][n] = 4 everywhere.  Any deviation
// (including zeros) exposes subtle lane participation rules.
__global__ void probe_ones(double* out) {
    int t = threadIdx.x;
    double acc = mfma_4x4x4_f64(1.0, 1.0, 0.0);
    out[t] = acc;
}

// Test 2: broadcast vs independent.  Zero out A from lanes 16..63.
// If broadcast: D = 4 in ALL four blocks (t/16 = 0,1,2,3) because the
// hardware reads A only from lanes 0..15 and replicates.
// If independent: D = 4 in block 0 only; D = 0 in blocks 1..3 because
// those blocks' A was zeroed.
__global__ void probe_broadcast_A(double* out) {
    int t = threadIdx.x;
    double a = (t < 16) ? 1.0 : 0.0;
    double b = 1.0;
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

// Test 3: same but zero B for lanes 16..63.
__global__ void probe_broadcast_B(double* out) {
    int t = threadIdx.x;
    double a = 1.0;
    double b = (t < 16) ? 1.0 : 0.0;
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

// Test 4: reveal n-dim of D.
// Hypothesis: B[k][n] under lane t<16 is B[(t/4)][t%4].
// Feed b = t%4 (so B[k][n] = n, independent of k).
// Then D[m][n] = sum_k 1 * n = 4n, independent of m.
// The output tells us which lanes correspond to which n.
__global__ void probe_reveal_n(double* out) {
    int t = threadIdx.x;
    double a = 1.0;
    double b = (double)(t & 3);
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

// Test 5: reveal m-dim of D.
// Hypothesis: A[m][k] under lane t<16 is A[t%4][(t/4)].
// Feed a = t%4 (so A[m][k] = m, independent of k).
// Then D[m][n] = sum_k m * 1 = 4m.
// (Note under broadcast, lanes 16..63 also need to provide sensible values;
// feeding a = t%4 keeps A[m][k] = m consistent across replicas.)
__global__ void probe_reveal_m(double* out) {
    int t = threadIdx.x;
    double a = (double)(t & 3);
    double b = 1.0;
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

// Test 6: full (m, n) probe.
// A[m][k] = m, B[k][n] = n  =>  D[m][n] = sum_k m*n = 4mn.
// Encodes 4m*n = 0..36 into each lane.  Combined with test 4/5 this
// uniquely identifies (m, n) per lane.
__global__ void probe_mn(double* out) {
    int t = threadIdx.x;
    double a = (double)(t & 3);
    double b = (double)(t & 3);
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

// Test 7: reveal A's k-dim.
// A[m][k] = 10k (so rows don't matter).
// Under hypothesis lane t<16 -> A[t%4][t/4], feed a = 10*(t/4).
// Then D[m][n] = sum_k 10k * 1 = 10*(0+1+2+3) = 60, uniform.
// If we feed a = 10*((t/4)%4), lanes 0..63 still feed A[m][k]=10k consistently.
// Uniform 60 across all lanes means A's k-dim is loaded correctly from 4 lanes.
__global__ void probe_reveal_Ak(double* out) {
    int t = threadIdx.x;
    double a = 10.0 * (double)((t >> 2) & 3);
    double b = 1.0;
    out[t] = mfma_4x4x4_f64(a, b, 0.0);
}

static void run(const char* name, void (*kernel)(double*)) {
    double *d_out, h[64];
    hipMalloc(&d_out, 64 * sizeof(double));
    kernel<<<1, 64>>>(d_out);
    hipDeviceSynchronize();
    hipMemcpy(h, d_out, 64 * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_out);

    printf("\n=== %s ===\n", name);
    printf("lane (t/16,t%%16) : value\n");
    for (int t = 0; t < 64; ++t) {
        int block = t >> 4;
        int sub   = t & 15;
        printf("%2d (%d,%2d): %6.1f", t, block, sub, h[t]);
        if ((t & 3) == 3) printf("\n"); else printf("   ");
    }
}

int main() {
    run("Test 1: A=1, B=1  (expect D=4 if broadcast, all lanes)", probe_ones);
    run("Test 2: A nonzero only for t<16, B=1  (broadcast? all D=4 vs tile-0 only)", probe_broadcast_A);
    run("Test 3: A=1, B nonzero only for t<16  (mirror of Test 2)", probe_broadcast_B);
    run("Test 4: reveal n-dim  (D[m][n] = 4n; expect 0,4,8,12 pattern vs n)", probe_reveal_n);
    run("Test 5: reveal m-dim  (D[m][n] = 4m; expect 0,4,8,12 pattern vs m)", probe_reveal_m);
    run("Test 6: D[m][n] = 4mn  (uniquely identifies (m,n) per lane)", probe_mn);
    run("Test 7: A_k probe (expect 60 everywhere if A's k-dim loaded from 4 lanes)", probe_reveal_Ak);
    return 0;
}
