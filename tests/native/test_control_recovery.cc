#include "control_transport.h"
#include "recovery.h"
#include <array>
#include <cassert>
#include <future>
#include <iostream>
using namespace vericcl;

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  int port=std::stoi(argv[1]);
  std::array<int,2> final{}, launches{}, commits{};
  auto member=[&](int rank) {
    ControlTransport transport(rank,2,"combined-group","combined-session",
        "127.0.0.1","127.0.0.1",port,after(5));
    const int original=rank?7:3;
    int buffer=original;
    RecoveryLedger recovery(2,after(3));
    const auto deadline=recovery.deadline();
    for(unsigned attempt=0;attempt<2;++attempt) {
      Json identity={{"request",9},{"attempt",attempt},{"version","immutable-A"}};
      if(rank==0)transport.inject_lost_replies_for_test("COMMIT",1);
      auto committed=transport.gather("COMMIT",identity,deadline);
      assert(committed[0]==identity && committed[1]==identity);
      ++commits[rank];
      if(rank==0)transport.inject_lost_replies_for_test("START",1);
      auto started=transport.gather("START",identity,deadline);
      assert(started[0]==identity && started[1]==identity);
      ++launches[rank];
      // A CPU buffer models a partially completed in-place reduction. The
      // actual production transport and recovery admission code are used.
      buffer+=rank?3:7;
      assert(buffer==10);
      if(attempt==0) {
        recovery.confirm_stopped(true,true,false);
        bool blocked=false;
        try{recovery.next_attempt(true,true);}catch(...){blocked=true;}
        assert(blocked && buffer==10); // No restoration before group agreement.
        auto stopped=transport.gather("RECOVERY_STOPPED",{{"stopped",true}},deadline);
        recovery.confirm_stopped(true,true,stopped[0]["stopped"]==true && stopped[1]["stopped"]==true);
        assert(recovery.next_attempt(true,true)==1);
        assert(recovery.deadline()==deadline);
        buffer=original;
        transport.gather("RECOVERY_READY",identity,deadline);
      }
    }
    final[rank]=buffer;
    // Retried TCP replies do not create extra logical launches or commits.
    assert(launches[rank]==2 && commits[rank]==2 && !transport.broken());
    transport.gather("COMPLETE",{{"result",buffer}},deadline);
  };
  auto first=std::async(std::launch::async,member,0);
  auto second=std::async(std::launch::async,member,1);
  first.get();second.get();
  assert(final[0]==10 && final[1]==10);
  std::cout<<"CPU combined COMMIT/START reply loss and partial in-place replay passed; no CUDA execution\n";
}
