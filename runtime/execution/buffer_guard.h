#ifndef VERICCL_BUFFER_GUARD_H
#define VERICCL_BUFFER_GUARD_H
#include "contract.h"
namespace vericcl {
void wait_stream_stopped(void* stream,Deadline deadline);
class BufferGuard {
  void* stream_;
public:
  explicit BufferGuard(void* stream):stream_(stream){}
  // No implicit CUDA free on exception or a public timeout. The State keeps
  // this guard alive until release() follows acknowledged stream termination.
  void release(Deadline deadline);
  void* stream()const{return stream_;}
};
}
#endif
