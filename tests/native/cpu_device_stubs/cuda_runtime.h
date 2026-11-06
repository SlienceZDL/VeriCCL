#ifndef VERICCL_TEST_CUDA_RUNTIME_H
#define VERICCL_TEST_CUDA_RUNTIME_H
/* CPU boundary doubles for the real API/worker/cleanup implementation. */
#include <atomic>
#include <cstddef>
struct CpuStream { std::atomic<bool> ready{true}; };
struct CpuEvent {};
using cudaStream_t=CpuStream*;
using cudaEvent_t=CpuEvent*;
enum cudaError_t { cudaSuccess, cudaErrorNotReady };
enum { cudaMemoryTypeDevice=2, cudaMemcpyDeviceToHost=1 };
struct cudaPointerAttributes { int type,device; };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes*,const void*);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);
cudaError_t cudaMemGetInfo(size_t*,size_t*);
cudaError_t cudaMalloc(void**,size_t);
template<class T> cudaError_t cudaMalloc(T** p,size_t bytes){return cudaMalloc(reinterpret_cast<void**>(p),bytes);}
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpyAsync(void*,const void*,size_t,int,cudaStream_t);
#endif
