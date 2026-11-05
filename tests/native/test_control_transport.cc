#include "control_transport.h"
#include <cassert>
#include <future>
#include <iostream>
#include <unistd.h>
using namespace vericcl;
int main(int argc,char**argv) {
  if(argc!=2)return 2;
  int port=std::stoi(argv[1]);
  auto member=[&](int rank) {
    ControlTransport transport(rank,2,"test-group","test-session","127.0.0.1","127.0.0.1",port,after(3));
    auto members=transport.gather("READY",Json{{"rank",rank}},after(1));
    assert(members[0]["rank"]==0 && members[1]["rank"]==1);
    if(rank==0)transport.inject_lost_replies_for_test("COMMIT",1);
    auto commit=transport.gather("COMMIT",{{"request",7},{"attempt",0},{"version","A"}},after(1));
    assert(commit[0]==commit[1]);
    if(rank==0)transport.inject_lost_replies_for_test("START",1);
    auto started=transport.gather("START",{{"request",7},{"attempt",0},{"version","A"}},after(1));
    assert(started[0]==started[1]);
    bool rejected=false;
    try{transport.gather(rank?"wrong-phase":"COMMIT",{},after(.15));}catch(...){rejected=true;}
    assert(rejected && transport.broken());
  };
  auto first=std::async(std::launch::async,member,0);
  usleep(2000);
  auto second=std::async(std::launch::async,member,1);
  first.get();second.get();
  std::promise<void> peer_ready,coordinator_gone;
  auto peer_ready_wait=peer_ready.get_future();auto gone_wait=coordinator_gone.get_future();
  auto coordinator=std::async(std::launch::async,[&]{
    {ControlTransport transport(0,2,"loss-group","loss-session","127.0.0.1","127.0.0.1",port,after(2));
      transport.gather("READY",{},after(1));peer_ready_wait.wait();}
    coordinator_gone.set_value();
  });
  auto survivor=std::async(std::launch::async,[&]{
    ControlTransport transport(1,2,"loss-group","loss-session","127.0.0.1","127.0.0.1",port,after(2));
    transport.gather("READY",{},after(1));peer_ready.set_value();gone_wait.wait();
    auto begin=Clock::now();bool failed=false;try{transport.gather("COMMIT",{},after(.15));}catch(...){failed=true;}
    assert(failed && transport.broken() && std::chrono::duration<double>(Clock::now()-begin).count()<.5);
  });
  coordinator.get();survivor.get();
  std::promise<void> member_gone;auto member_wait=member_gone.get_future();
  auto remaining=std::async(std::launch::async,[&]{
    ControlTransport transport(0,2,"member-loss","member-session","127.0.0.1","127.0.0.1",port,after(2));
    transport.gather("READY",{},after(1));member_wait.wait();bool failed=false;
    try{transport.gather("START",{},after(.15));}catch(...){failed=true;}assert(failed && transport.broken());
  });
  auto departing=std::async(std::launch::async,[&]{
    {ControlTransport transport(1,2,"member-loss","member-session","127.0.0.1","127.0.0.1",port,after(2));transport.gather("READY",{},after(1));}
    member_gone.set_value();
  });
  remaining.get();departing.get();
  std::cout<<"TCP membership, immutable COMMIT/START retry, coordinator/member loss and bounded mismatch checks passed; no GPU execution\n";
}
