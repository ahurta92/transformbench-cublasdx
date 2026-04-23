#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "transform.h"
#include "transform_level2.h"
#include "transform_level3.h"
#include "transform_blocked.h"
#include "transform_blocked_rocwmma.h"
#include "util.h"

template <typename T>
void transform_bench(int nreps, int ntasks, int nfuncs, int nblocks, int K,
                     int num_streams, int level) {
  // Auto-select level if not specified.
  if (level < 1) {
    // crash out

    fprintf(stderr, "Invalid level %d. Must be 1, 2, or 3.\n", level);
    assert(false);
    return;
  }

  // Level metadata for output line.
  Dim3 thread_dims;
  int smem_size;
  std::string level_name;

  switch (level) {
  case 2:
    thread_dims = mra::mTxmq_L2_blockdim<T>(K);
    smem_size = (int)mra::mTxmq_L2_shmem_size<T>(K);
    level_name = "L2-lds";
    break;
  case 3:
    thread_dims = mra::mTxmq_L3_blockdim<T>(K);
    smem_size = (int)mra::L3_shmem_size<T>(K);
    level_name = "L3-regblk";
    break;
  case 7:
    thread_dims = blocked_blockdim<T>(K);
    smem_size = (int)blocked_shmem_size<T>(K);
    level_name = "L7-blocked";
    break;
  case 8:
    thread_dims = blocked_rocwmma_blockdim<T>(K);
    smem_size = (int)blocked_rocwmma_shmem_size<T>(K);
    level_name = "L8-blocked-rocwmma";
    break;
  default: // level == 1
    thread_dims = mra::mTxmq_blockdim<T>(K);
    smem_size = 0;
    level_name = "L1-global";
    break;
  }

  std::vector<Stream> streams(num_streams);
  T *A;
  T *B;
  T *C;
  T *workspace;
  MALLOC(&A, (size_t)nfuncs * K * K * K * sizeof(T));
  MALLOC(&B, (size_t)K * K * sizeof(T));
  MALLOC(&C, (size_t)nfuncs * K * K * K * sizeof(T));
  MALLOC(&workspace, (size_t)nblocks * K * K * K * sizeof(T));

  for (int i = 0; i < num_streams; ++i)
    CREATE_STREAM(&streams[i]);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  for (int i = 0; i < nreps + 1; ++i) {
    beg = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < ntasks; ++t) {
      Stream s = streams[t % num_streams];
      switch (level) {
      case 2:
        submit_transform_bench_L2<T>(nfuncs, nblocks, K, A, B, C, workspace, s);
        break;
      case 3:
        submit_transform_bench_L3<T>(nfuncs, nblocks, K, A, B, C, workspace, s);
        break;
      case 7:
        submit_transform_bench_blocked<T>(nfuncs, nblocks, K, A, B, C, workspace, s);
        break;
      case 8:
        submit_transform_bench_blocked_rocwmma<T>(nfuncs, nblocks, K, A, B, C, workspace, s);
        break;
      default:
        submit_transform_bench<T>(nfuncs, nblocks, K, A, B, C, workspace, s);
        break;
      }
    }
    for (int t = 0; t < num_streams; ++t)
      SYNC_STREAM(streams[t]);
    end = std::chrono::high_resolution_clock::now();

    if (i > 0) {
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - beg)
                    .count();
      uint64_t flops = (uint64_t)ntasks * K * K * K * K * 3 * 2 * nfuncs;
      std::cout << "Transform"
                << ";level=" << level_name << ";nfuncs=" << nfuncs
                << ";nblocks=" << nblocks << ";K=" << K << ";tasks=" << ntasks
                << ";threads={" << thread_dims.x << "," << thread_dims.y << ","
                << thread_dims.z << "}"
                << ";smem=" << smem_size << ";Time(us)=" << us
                << ";GFlop=" << flops * 1e-9
                << ";Gflop/s=" << (1e-3 * flops) / us << std::endl;
    }
  }

  FREE(A);
  FREE(B);
  FREE(C);
  FREE(workspace);
}

int main(int argc, char **argv) {
  auto opt = OptionParser(argc, argv);

  int nreps = opt.parse("-r", 5);
  int ntasks = opt.parse("-n", 500); // Number
  int N = opt.parse("-N", 2048);     // Number of functions
  int K = opt.parse("-K", 16);
  int M = opt.parse("-M", 512);
  int num_streams = opt.parse("-s", 4);
  int level = opt.parse("-l", -1);

  std::cout << "Running benchmark"
            << " nreps=" << nreps << " ntasks=" << ntasks << " N=" << N
            << " K=" << K << " M=" << M << " level=" << level << std::endl;

  transform_bench<double>(nreps, ntasks, N, M, K, num_streams, level);
}
