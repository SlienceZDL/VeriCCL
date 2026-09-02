#include <cuda_runtime.h>
#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t result = (call);                                             \
    if (result != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(result));    \
      MPI_Abort(MPI_COMM_WORLD, 2);                                          \
    }                                                                        \
  } while (0)


static uint64_t monotonicNs() {
  struct timespec value;
#if defined(CLOCK_MONOTONIC_RAW)
  clock_gettime(CLOCK_MONOTONIC_RAW, &value);
#else
  clock_gettime(CLOCK_MONOTONIC, &value);
#endif
  return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}


static void selectLocalDevice(int worldRank) {
  MPI_Comm localComm = MPI_COMM_NULL;
  MPI_Comm_split_type(
      MPI_COMM_WORLD,
      MPI_COMM_TYPE_SHARED,
      worldRank,
      MPI_INFO_NULL,
      &localComm);
  int localRank = 0;
  MPI_Comm_rank(localComm, &localRank);
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 1) {
    fprintf(stderr, "no CUDA device is visible to MPI rank %d\n", worldRank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
  MPI_Comm_free(&localComm);
}


__global__ static void readGpuTimer(unsigned long long* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  unsigned long long value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  *output = value;
#else
  *output = clock64();
#endif
}


static void estimateReferenceOffset(
    int rank,
    int worldSize,
    int sampleCount,
    int64_t* offsetNs,
    uint64_t* uncertaintyNs) {
  *offsetNs = 0;
  *uncertaintyNs = 0;
  if (worldSize == 1) return;

  if (rank == 0) {
    for (int peer = 1; peer < worldSize; ++peer) {
      for (int sample = 0; sample < sampleCount; ++sample) {
        uint64_t request = 0;
        MPI_Recv(
            &request,
            1,
            MPI_UNSIGNED_LONG_LONG,
            peer,
            100,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        uint64_t response = monotonicNs();
        MPI_Send(
            &response,
            1,
            MPI_UNSIGNED_LONG_LONG,
            peer,
            101,
            MPI_COMM_WORLD);
      }
    }
    return;
  }

  uint64_t bestUncertainty = UINT64_MAX;
  int64_t bestOffset = 0;
  for (int sample = 0; sample < sampleCount; ++sample) {
    uint64_t before = monotonicNs();
    MPI_Send(
        &before,
        1,
        MPI_UNSIGNED_LONG_LONG,
        0,
        100,
        MPI_COMM_WORLD);
    uint64_t reference = 0;
    MPI_Recv(
        &reference,
        1,
        MPI_UNSIGNED_LONG_LONG,
        0,
        101,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    uint64_t after = monotonicNs();
    uint64_t uncertainty = (after - before) / 2;
    int64_t midpoint = (int64_t)(before + (after - before) / 2);
    int64_t offset = (int64_t)reference - midpoint;
    if (uncertainty < bestUncertainty) {
      bestUncertainty = uncertainty;
      bestOffset = offset;
    }
  }
  *offsetNs = bestOffset;
  *uncertaintyNs = bestUncertainty;
}


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int worldSize = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  selectLocalDevice(rank);

  int sampleCount = 16;
  if (argc == 2) sampleCount = atoi(argv[1]);
  if (sampleCount < 2 || sampleCount > 1024) {
    if (rank == 0) {
      fprintf(stderr, "sample count must be between 2 and 1024\n");
    }
    MPI_Finalize();
    return 1;
  }

  int64_t referenceOffsetNs = 0;
  uint64_t referenceUncertaintyNs = 0;
  estimateReferenceOffset(
      rank,
      worldSize,
      sampleCount,
      &referenceOffsetNs,
      &referenceUncertaintyNs);
  MPI_Barrier(MPI_COMM_WORLD);

  unsigned long long* deviceTimer = NULL;
  CUDA_CHECK(cudaMalloc((void**)&deviceTimer, sizeof(*deviceTimer)));
  readGpuTimer<<<1, 1>>>(deviceTimer);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long warmupGpuTicks = 0;
  CUDA_CHECK(cudaMemcpy(
      &warmupGpuTicks,
      deviceTimer,
      sizeof(warmupGpuTicks),
      cudaMemcpyDeviceToHost));

  for (int sample = 0; sample < sampleCount; ++sample) {
    uint64_t hostBeforeNs = monotonicNs();
    readGpuTimer<<<1, 1>>>(deviceTimer);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long gpuTicks = 0;
    CUDA_CHECK(cudaMemcpy(
        &gpuTicks,
        deviceTimer,
        sizeof(gpuTicks),
        cudaMemcpyDeviceToHost));
    uint64_t hostAfterNs = monotonicNs();
    printf(
        "VERICCL_CLOCK_SYNC %d %llu %llu %llu %lld %llu\n",
        rank,
        gpuTicks,
        (unsigned long long)hostBeforeNs,
        (unsigned long long)hostAfterNs,
        (long long)referenceOffsetNs,
        (unsigned long long)referenceUncertaintyNs);
  }
  fflush(stdout);

  CUDA_CHECK(cudaFree(deviceTimer));
  MPI_Finalize();
  return 0;
}
