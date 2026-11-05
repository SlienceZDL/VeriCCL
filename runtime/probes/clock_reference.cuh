#ifndef VERICCL_CLOCK_REFERENCE_CUH_
#define VERICCL_CLOCK_REFERENCE_CUH_
#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <time.h>

inline uint64_t vericclHostTimeNs() {
  timespec value;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value)) throw std::runtime_error("host clock read failed");
  return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}
__global__ static void vericclReadGlobaltimer(unsigned long long* output) {
  if (blockIdx.x || threadIdx.x) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  *output = value;
#else
  *output = 0;  // This contract requires the nanosecond globaltimer domain.
#endif
}
inline void vericclMpiCheck(int status) {
  if (status != MPI_SUCCESS) throw std::runtime_error("clock-reference MPI exchange failed");
}
inline std::string vericclClockSamples(int rank, int ranks, cudaStream_t stream) {
  const int count = 16;
  int64_t offset = 0;
  uint64_t uncertainty = 0;
  if (rank == 0) {
    for (int peer = 1; peer < ranks; peer++) {
      for (int i = 0; i < count; i++) {
        uint64_t request;
        vericclMpiCheck(MPI_Recv(&request, 1, MPI_UINT64_T, peer, 419, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        uint64_t response = vericclHostTimeNs();
        vericclMpiCheck(MPI_Send(&response, 1, MPI_UINT64_T, peer, 420, MPI_COMM_WORLD));
      }
    }
  } else {
    uncertainty = UINT64_MAX;
    for (int i = 0; i < count; i++) {
      uint64_t before = vericclHostTimeNs(), reference;
      vericclMpiCheck(MPI_Send(&before, 1, MPI_UINT64_T, 0, 419, MPI_COMM_WORLD));
      vericclMpiCheck(MPI_Recv(&reference, 1, MPI_UINT64_T, 0, 420, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      uint64_t after = vericclHostTimeNs();
      uint64_t bound = (after - before + 1) / 2;
      if (bound < uncertainty) {
        uncertainty = bound;
        offset = int64_t(reference) - int64_t(before + (after - before) / 2);
      }
    }
  }
  vericclMpiCheck(MPI_Barrier(MPI_COMM_WORLD));
  unsigned long long* device_timer = nullptr;
  VC_CUDA(cudaMalloc(&device_timer, sizeof(*device_timer)));
  std::ostringstream result;
  try {
    for (int i = 0; i < count; i++) {
      uint64_t before = vericclHostTimeNs();
      vericclReadGlobaltimer<<<1, 1, 0, stream>>>(device_timer);
      VC_CUDA(cudaGetLastError());
      unsigned long long ticks = 0;
      VC_CUDA(cudaMemcpyAsync(&ticks, device_timer, sizeof(ticks), cudaMemcpyDeviceToHost, stream));
      VC_CUDA(cudaStreamSynchronize(stream));
      uint64_t after = vericclHostTimeNs();
      if (!ticks) throw std::runtime_error("nanosecond GPU globaltimer is unavailable");
      result << "VERICCL_CLOCK_SYNC " << rank << ' ' << ticks << ' ' << before << ' ' << after << ' '
             << offset << ' ' << uncertainty << '\n';
    }
  } catch (...) { cudaFree(device_timer); throw; }
  VC_CUDA(cudaFree(device_timer));
  return result.str();
}
#endif
