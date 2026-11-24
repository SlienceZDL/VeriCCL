#ifndef VERICCL_BUFFER_GUARD_H
#define VERICCL_BUFFER_GUARD_H
#include "contract.h"
namespace vericcl {
void wait_stream_stopped(void* stream,Deadline deadline);
class BufferGuard {
  void* input_;void* output_;void* stream_;void* backup_=nullptr;
  uint64_t input_bytes_,output_bytes_;bool needs_backup_,original_=false;
public:
  BufferGuard(void* input,uint64_t input_bytes,void* output,uint64_t output_bytes,void* stream,bool needs_backup):
    input_(input),output_(output),stream_(stream),input_bytes_(input_bytes),output_bytes_(output_bytes),needs_backup_(needs_backup){}
  // No implicit CUDA free on exception or a public timeout. The State keeps
  // this guard alive until release() follows acknowledged stream termination.
  void protect(Deadline deadline);
  void restore(Deadline deadline,bool old_work_terminated);
  void release(Deadline deadline);
  bool original_available()const{return original_;}
  uint64_t backup_bytes()const{return needs_backup_?input_bytes_:0;}
  void* stream()const{return stream_;}
};
}
#endif
