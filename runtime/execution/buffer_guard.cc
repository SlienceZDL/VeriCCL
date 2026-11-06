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
void BufferGuard::release(Deadline deadline){
  wait_stream_stopped(stream_,deadline);
}
}
