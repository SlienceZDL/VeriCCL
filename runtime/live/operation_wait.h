#ifndef VERICCL_OPERATION_WAIT_H
#define VERICCL_OPERATION_WAIT_H
#include "contract.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace vericcl {
struct OperationWait {
  std::mutex mutex;
  std::condition_variable condition;
  bool done=false;
  Deadline public_deadline=Deadline::max();
  void set_phase_deadline(Deadline value){
    {std::lock_guard<std::mutex> lock(mutex);public_deadline=value;}
    condition.notify_all();
  }
  void complete(){
    {std::lock_guard<std::mutex> lock(mutex);done=true;}
    condition.notify_all();
  }
  bool wait_public(Deadline parent){
    std::unique_lock<std::mutex> lock(mutex);
    while(!done){
      auto deadline=std::min(parent,public_deadline);
      if(Clock::now()>=deadline)return false;
      condition.wait_until(lock,deadline);
    }
    return true;
  }
};
}
#endif
