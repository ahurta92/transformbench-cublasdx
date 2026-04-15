#include <iostream>
#include <chrono>

#include "transform.h"
#include "util.h"

template<typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K, int num_streams) {

  std::vector<Stream> streams(num_streams);
  T* A, *B, *C, *workspace;
  MALLOC(&A, nfuncs * K * K * K * sizeof(T));
  MALLOC(&B, K * K * sizeof(T));
  MALLOC(&C, nfuncs * K * K * K * sizeof(T));
  MALLOC(&workspace, nblocks * K * K * K * sizeof(T));

  for (int i = 0; i < num_streams; ++i) {
    CREATE_STREAM(&streams[i]);
  }

  Dim3 thread_dims = mra::mTxmq_blockdim<T>(K);
  int smem_size    = mra::mTxmq_shmem_size<T>(K);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  for (int i = 0; i < nreps+1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntasks; ++t) {
      submit_transform_bench(nfuncs, nblocks, K, A, B, C, workspace, streams[t % num_streams]);
    }
    for (int t = 0; t < num_streams; ++t) {
      SYNC_STREAM(streams[t]);
    }
    end = std::chrono::high_resolution_clock::now();

    if (i > 0) {
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 * nfuncs;
      std::cout << "Transform"
                << ";level=L1-global"
                << ";nfuncs=" << nfuncs
                << ";nblocks=" << nblocks
                << ";K=" << K
                << ";tasks=" << ntasks
                << ";threads={" << thread_dims.x << "," << thread_dims.y << "," << thread_dims.z << "}"
                << ";smem=" << smem_size
                << ";Time(us)=" << us
                << ";GFlop=" << flops * 1e-9
                << ";Gflop/s=" << (1e-3 * flops) / us
                << std::endl;
    }
  }

  FREE(A);
  FREE(B);
  FREE(C);
  FREE(workspace);
}

int main(int argc, char **argv) {

  auto opt = OptionParser(argc, argv);

  int nreps       = opt.parse("-r", 5);
  int ntasks      = opt.parse("-n", 500);
  int N           = opt.parse("-N", 2048);
  int K           = opt.parse("-K", 16);
  int M           = opt.parse("-M", 512);
  int num_streams = opt.parse("-s", 4);

  std::cout << "Running benchmark"
            << " nreps=" << nreps
            << " ntasks=" << ntasks
            << " N=" << N
            << " K=" << K
            << " M=" << M
            << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, num_streams);
}
