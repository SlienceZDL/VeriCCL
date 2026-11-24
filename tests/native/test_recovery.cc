#include "recovery.h"
#include <cassert>
#include <iostream>
#include <vector>
using namespace vericcl;
template<class F> void rejects(F call){bool failed=false;try{call();}catch(...){failed=true;}assert(failed);}
int main(){
  auto deadline=after(2);RecoveryLedger recovery(2,deadline);
  std::vector<int> input{3},backup=input;int contribution=7;
  input[0]+=contribution; // An injected partially completed in-place request.
  recovery.confirm_stopped(false,true,true);
  rejects([&]{recovery.next_attempt(true,true);});assert(input[0]==10);
  recovery.confirm_stopped(true,true,true);
  assert(recovery.next_attempt(true,true)==1);
  input=backup;input[0]+=contribution;assert(input[0]==10 && input[0]!=17);
  assert(recovery.deadline()==deadline);
  recovery.confirm_stopped(true,true,true);assert(recovery.next_attempt(true,true)==2);
  recovery.confirm_stopped(true,true,true);rejects([&]{recovery.next_attempt(true,true);});
  RecoveryLedger missing(2,deadline);missing.confirm_stopped(true,true,true);
  rejects([&]{missing.next_attempt(false,true);});rejects([&]{missing.next_attempt(true,false);});
  RecoveryLedger expired(2,Clock::now());expired.confirm_stopped(true,true,true);
  rejects([&]{expired.next_attempt(true,true);});
  std::cout<<"CPU recovery: original 3+7 restored to 10; termination, membership, path and fixed deadline required\n";
}
