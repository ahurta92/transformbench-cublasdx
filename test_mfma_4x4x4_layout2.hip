// Additional probes to pin down v_mfma_f64_4x4x4f64 fragment layout.
// Strategy: feed values that uniquely encode (row, col) per lane to reveal
// which subset of lanes contributes to each output cell.

#include <cstdio>
#include <hip/hip_runtime.h>

__device__ inline double mfma(double a, double b, double c) {
    return __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, c, 0, 0, 0);
}

// P1: a = lane_id, b = 1.  Output tells us sum of A[m][0..3] where
// A[m][k] is loaded from "some" lanes; the sum reveals the identity
// of contributing lanes at row m.
__global__ void p1(double* out) {
    int t = threadIdx.x;
    out[t] = mfma((double)t, 1.0, 0.0);
}

// P2: a = 1, b = lane_id.
__global__ void p2(double* out) {
    int t = threadIdx.x;
    out[t] = mfma(1.0, (double)t, 0.0);
}

// P3: a = t&3, b = 1 (variant of prior test 5 with accumulator verification).
__global__ void p3(double* out) {
    int t = threadIdx.x;
    out[t] = mfma((double)(t & 3), 1.0, 0.0);
}

// P4: a = 1, b = (t>>4) (reveal B's batch/block dim if any).
__global__ void p4(double* out) {
    int t = threadIdx.x;
    out[t] = mfma(1.0, (double)((t >> 4) & 3), 0.0);
}

// P5: a = (t>>4), b = 1 (reveal A's batch/block dim if any).
__global__ void p5(double* out) {
    int t = threadIdx.x;
    out[t] = mfma((double)((t >> 4) & 3), 1.0, 0.0);
}

// P6: a = lane_id only for a selected lane (t == TARGET); zero elsewhere.
// Shows how much that single a-value contributes to each output lane.
template<int TARGET>
__global__ void p_single(double* out) {
    int t = threadIdx.x;
    double a = (t == TARGET) ? 1.0 : 0.0;
    double b = 1.0;
    out[t] = mfma(a, b, 0.0);
}

static void run(const char* name, void (*kernel)(double*)) {
    double *d_out, h[64];
    hipMalloc(&d_out, 64 * sizeof(double));
    kernel<<<1, 64>>>(d_out);
    hipDeviceSynchronize();
    hipMemcpy(h, d_out, 64 * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_out);

    printf("\n=== %s ===\n", name);
    for (int t = 0; t < 64; ++t) {
        printf("%2d:%5.1f ", t, h[t]);
        if ((t & 7) == 7) printf("\n");
    }
}

int main() {
    run("P1: a=lane_id, b=1", p1);
    run("P2: a=1, b=lane_id", p2);
    run("P3: a=t&3, b=1 (sanity, repeats prior test 5)", p3);
    run("P4: a=1, b=(t>>4)&3  (does B depend on t/16?)", p4);
    run("P5: a=(t>>4)&3, b=1  (does A depend on t/16?)", p5);
    run("P-lane0 only A",   p_single<0>);
    run("P-lane1 only A",   p_single<1>);
    run("P-lane4 only A",   p_single<4>);
    run("P-lane15 only A",  p_single<15>);
    run("P-lane16 only A",  p_single<16>);
    run("P-lane32 only A",  p_single<32>);
    run("P-lane48 only A",  p_single<48>);
    return 0;
}
