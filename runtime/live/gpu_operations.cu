#include "gpu_operations.h"
#include "../probes/timing.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <stdint.h>
template<typename T> __device__ double number(T x){return double(x);}
template<> __device__ double number(__half x){return __half2float(x);}
template<> __device__ double number(__nv_bfloat16 x){return __bfloat162float(x);}
__device__ double original(int rank,size_t offset){return 1+((size_t(rank)*3+offset)%5);}
template<typename T> __global__ void fill_input(T* data,size_t n,int rank) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)data[i]=T(original(rank,i));
}
template<typename T> __global__ void scale_output(T* data,size_t n,int divisor) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)data[i]=T(number(data[i])/divisor);
}
__global__ void sample_timer(unsigned long long* data) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  unsigned long long v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));*data=v;
#else
  *data=0;
#endif
}
/* Reference semantics are indexed by rank and logical element, independently
 * of the XML routes, channels, TB order or buffer-allocation implementation.
 * collective: broadcast/reduce/allgather/allreduce/alltoall/reduce_scatter=0..5.
 * reduction: sum/prod/min/max/avg=0..4. */
template<typename T> __global__ void check_output(const T* data,size_t n,int coll,int op,int rank,int ranks,int root,size_t input_n,double atol,double rtol,unsigned long long* bad) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
    double expected=0;
    if(coll==0)expected=original(root,i);
    else if(coll==2)expected=original(int(i/input_n),i%input_n);
    else if(coll==4){size_t block=input_n/ranks;expected=original(int(i/block),size_t(rank)*block+i%block);}
    else {
      size_t index=coll==5?size_t(rank)*(input_n/ranks)+i:i;
      expected=original(0,index);
      for(int peer=1;peer<ranks;peer++) {
        double value=original(peer,index);
        if(op==1)expected*=value;else if(op==2)expected=fmin(expected,value);else if(op==3)expected=fmax(expected,value);else expected+=value;
      }
      if(op==4)expected/=ranks;
      /* Integral reference follows the selected storage type's modular result.
       * Tests use small ranks to avoid floating conversion losing int64 bits. */
      expected=number(T(expected));
    }
    double actual=number(data[i]);
    if(!isfinite(actual) || !isfinite(expected) || fabs(actual-expected)>atol+rtol*fabs(expected))atomicAdd(bad,1ULL);
  }
}
#define VC_ALL_TYPES(F) F(ncclInt8,int8_t) F(ncclUint8,uint8_t) F(ncclInt32,int32_t) F(ncclUint32,uint32_t) F(ncclInt64,int64_t) F(ncclUint64,uint64_t) F(ncclFloat16,__half) F(ncclFloat32,float) F(ncclFloat64,double) F(ncclBfloat16,__nv_bfloat16)
static int blocks(size_t n){return int(std::min(size_t(4096),(n+255)/256));}
void vericcl_fill_input(void* p,size_t n,ncclDataType_t type,int rank,cudaStream_t stream) {
  if(!n)return;
  switch(type){
#define F(tag,T) case tag:fill_input<T><<<blocks(n),256,0,stream>>>((T*)p,n,rank);break;
    VC_ALL_TYPES(F)
#undef F
    default:throw std::invalid_argument("unsupported datatype");
  }VC_CUDA(cudaGetLastError());
}
void vericcl_scale_output(void* p,size_t n,ncclDataType_t type,int divisor,cudaStream_t stream) {
  if(!n)return;
  switch(type){
#define F(tag,T) case tag:scale_output<T><<<blocks(n),256,0,stream>>>((T*)p,n,divisor);break;
    F(ncclFloat16,__half) F(ncclFloat32,float) F(ncclFloat64,double) F(ncclBfloat16,__nv_bfloat16)
#undef F
    default:throw std::invalid_argument("integer avg is unavailable");
  }VC_CUDA(cudaGetLastError());
}
void vericcl_sample_timer(unsigned long long* p,cudaStream_t stream){sample_timer<<<1,1,0,stream>>>(p);VC_CUDA(cudaGetLastError());}
unsigned long long vericcl_check_output(const void* p,size_t n,ncclDataType_t type,int coll,int op,int rank,int ranks,int root,size_t input_n,double atol,double rtol,cudaStream_t stream) {
  if(!n)return 0;
  unsigned long long* bad=nullptr,result=0;VC_CUDA(cudaMalloc(&bad,sizeof(*bad)));VC_CUDA(cudaMemsetAsync(bad,0,sizeof(*bad),stream));
  switch(type){
#define F(tag,T) case tag:check_output<T><<<blocks(n),256,0,stream>>>((const T*)p,n,coll,op,rank,ranks,root,input_n,atol,rtol,bad);break;
    VC_ALL_TYPES(F)
#undef F
    default:throw std::invalid_argument("unsupported datatype");
  }
  VC_CUDA(cudaGetLastError());VC_CUDA(cudaMemcpyAsync(&result,bad,sizeof(result),cudaMemcpyDeviceToHost,stream));VC_CUDA(cudaStreamSynchronize(stream));VC_CUDA(cudaFree(bad));return result;
}
