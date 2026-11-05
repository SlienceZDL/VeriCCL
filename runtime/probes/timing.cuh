#ifndef VERICCL_PROBE_TIMING_CUH
#define VERICCL_PROBE_TIMING_CUH
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include "vericcl_probe.h"

inline void cuda_check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
inline void nccl_check(ncclResult_t status, const char* expression) {
  if (status != ncclSuccess) throw std::runtime_error(std::string(expression) + ": " + ncclGetErrorString(status));
}
#define VC_CUDA(expr) cuda_check((expr), #expr)
#define VC_NCCL(expr) nccl_check((expr), #expr)

size_t datatype_bytes(ncclDataType_t type);
void fill_values(void* data, size_t elements, ncclDataType_t type, double value, cudaStream_t stream);
unsigned long long check_values(const void* data, size_t elements, ncclDataType_t type,
                                double expected, double atol, double rtol, cudaStream_t stream);
#endif
