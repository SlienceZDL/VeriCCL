#include "operation_wait.h"
#include <cassert>
#include <future>
#include <iostream>
using namespace vericcl;
int main(){
  OperationWait operation;operation.public_deadline=after(2);
  std::atomic<bool> owned{true};
  std::promise<void> release;auto release_wait=release.get_future();
  auto blocked=std::async(std::launch::async,[&]{
    operation.set_phase_deadline(after(.03));
    release_wait.wait();
    owned=false;operation.complete();
  });
  auto begin=Clock::now();
  assert(!operation.wait_public(after(2)));
  assert(std::chrono::duration<double>(Clock::now()-begin).count()<.12);
  assert(owned && !operation.done); // Timed-out abort/cleanup is still owned.
  release.set_value();blocked.get();assert(!owned && operation.wait_public(after(.1)));
  OperationWait phases;phases.public_deadline=after(.04);
  auto progresses=std::async(std::launch::async,[&]{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));phases.set_phase_deadline(after(.08));
    std::this_thread::sleep_for(std::chrono::milliseconds(45));phases.complete();
  });
  assert(phases.wait_public(after(.2)));progresses.get();
  std::cout<<"Actual native wait path: shorter recovery deadline returns with blocked ownership retained; phase changes wake the waiter\n";
}
