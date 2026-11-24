#ifndef VERICCL_RECOVERY_H
#define VERICCL_RECOVERY_H
#include "contract.h"
namespace vericcl {
class RecoveryLedger {
  uint64_t maximum_, attempt_=0;
  Deadline deadline_;
  bool gpu_=false,communication_=false,group_=false;
public:
  RecoveryLedger(uint64_t maximum,Deadline deadline):maximum_(maximum),deadline_(deadline){}
  Deadline deadline() const{return deadline_;}
  void confirm_stopped(bool gpu,bool communication,bool group){gpu_=gpu;communication_=communication;group_=group;}
  uint64_t next_attempt(bool original_available,bool legal_path) {
    if(!gpu_ || !communication_ || !group_)throw std::runtime_error("old GPU, communication or fixed membership termination is unresolved");
    if(!original_available || !legal_path)throw std::runtime_error("replay requires original input and a legal path");
    if(Clock::now()>=deadline_)throw std::runtime_error("total recovery deadline expired");
    if(attempt_>=maximum_)throw std::runtime_error("maximum additional retries reached");
    gpu_=communication_=group_=false;return ++attempt_;
  }
};
}
#endif
