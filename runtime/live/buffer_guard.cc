#include "buffer_guard.h"
#include "../probes/timing.cuh"
#include <thread>
namespace vericcl {
void wait_stream_stopped(void* raw,Deadline deadline){
  for(;;){auto status=cudaStreamQuery(static_cast<cudaStream_t>(raw));if(status==cudaSuccess)return;
    if(status!=cudaErrorNotReady)cuda_check(status,"stream termination");
    if(Clock::now()>=deadline)throw std::runtime_error("GPU stream termination deadline expired");
    std::this_thread::sleep_for(std::chrono::microseconds(100));}
}
void BufferGuard::protect(Deadline deadline){
  if(needs_backup_ && input_bytes_){VC_CUDA(cudaMalloc(&backup_,input_bytes_));VC_CUDA(cudaMemcpyAsync(backup_,input_,input_bytes_,cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream_)));}
  wait_stream_stopped(stream_,deadline);original_=true;
}
void BufferGuard::restore(Deadline deadline,bool stopped){
  if(!stopped || !original_)throw std::runtime_error("input restore requires original snapshot and acknowledged old-work termination");
  if(output_bytes_)VC_CUDA(cudaMemsetAsync(output_,0,output_bytes_,static_cast<cudaStream_t>(stream_)));
  if(needs_backup_ && input_bytes_)VC_CUDA(cudaMemcpyAsync(input_,backup_,input_bytes_,cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream_)));
  wait_stream_stopped(stream_,deadline);
}
void BufferGuard::release(Deadline deadline){
  wait_stream_stopped(stream_,deadline);if(backup_){VC_CUDA(cudaFree(backup_));backup_=nullptr;}original_=false;
}
}
