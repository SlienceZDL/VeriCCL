#include "timing.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cmath>

template<typename T> __global__ void fill_kernel(T* out, size_t n, double value) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
    out[i] = T(value);
}
template<typename T> __device__ double numeric(T value) { return double(value); }
template<> __device__ double numeric(__half value) { return __half2float(value); }
template<> __device__ double numeric(__nv_bfloat16 value) { return __bfloat162float(value); }
template<typename T> __global__ void check_kernel(const T* data, size_t n, double expected,
                                                  double atol, double rtol, unsigned long long* bad) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x) {
    double actual = numeric(data[i]);
    if (!isfinite(actual) || fabs(actual - expected) > atol + rtol * fabs(expected)) atomicAdd(bad, 1ULL);
  }
}

#define VC_TYPES(F) \
  F(ncclInt8, int8_t) F(ncclUint8, uint8_t) F(ncclInt32, int32_t) F(ncclUint32, uint32_t) \
  F(ncclInt64, int64_t) F(ncclUint64, uint64_t) F(ncclFloat16, __half) F(ncclFloat32, float) \
  F(ncclFloat64, double) F(ncclBfloat16, __nv_bfloat16)

size_t datatype_bytes(ncclDataType_t type) {
  switch (type) {
#define VC_SIZE(tag, T) case tag: return sizeof(T);
    VC_TYPES(VC_SIZE)
#undef VC_SIZE
    default: throw std::runtime_error("unsupported datatype");
  }
}
void fill_values(void* data, size_t n, ncclDataType_t type, double value, cudaStream_t stream) {
  const int blocks = int(std::min(size_t(4096), (n + 255) / 256));
  if (!n) return;
  switch (type) {
#define VC_FILL(tag, T) case tag: fill_kernel<T><<<blocks, 256, 0, stream>>>((T*)data, n, value); break;
    VC_TYPES(VC_FILL)
#undef VC_FILL
    default: throw std::runtime_error("unsupported datatype");
  }
  VC_CUDA(cudaGetLastError());
}
unsigned long long check_values(const void* data, size_t n, ncclDataType_t type,
                                double expected, double atol, double rtol, cudaStream_t stream) {
  unsigned long long* bad = nullptr;
  unsigned long long result = 0;
  VC_CUDA(cudaMalloc(&bad, sizeof(*bad)));
  VC_CUDA(cudaMemsetAsync(bad, 0, sizeof(*bad), stream));
  const int blocks = int(std::min(size_t(4096), (n + 255) / 256));
  switch (type) {
#define VC_CHECK(tag, T) case tag: check_kernel<T><<<blocks, 256, 0, stream>>>((const T*)data, n, expected, atol, rtol, bad); break;
    VC_TYPES(VC_CHECK)
#undef VC_CHECK
    default: cudaFree(bad); throw std::runtime_error("unsupported datatype");
  }
  VC_CUDA(cudaGetLastError());
  VC_CUDA(cudaMemcpyAsync(&result, bad, sizeof(result), cudaMemcpyDeviceToHost, stream));
  VC_CUDA(cudaStreamSynchronize(stream));
  VC_CUDA(cudaFree(bad));
  return result;
}
