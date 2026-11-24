#include "control_transport.h"
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <unistd.h>

namespace vericcl {
Socket::~Socket() { if (fd>=0) ::close(fd); }
Socket& Socket::operator=(Socket&& other) noexcept {
  if (this != &other) { if(fd>=0)::close(fd);fd=other.fd;other.fd=-1; } return *this;
}
static void wait_fd(int fd, short events, Deadline deadline) {
  for (;;) {
    auto left = std::chrono::duration<double,std::milli>(deadline-Clock::now()).count();
    if (left<=0) throw std::runtime_error("control deadline expired");
    pollfd item{fd,events,0};
    int rc=poll(&item,1,int(std::min(2147483647.,std::ceil(left))));
    if (rc<0 && errno==EINTR) continue;
    if (rc<=0) throw std::runtime_error(rc==0 ? "control deadline expired" : "control poll failed");
    if((item.revents&events) || (events==POLLOUT && (item.revents&(POLLERR|POLLHUP))))return;
    throw std::runtime_error("control peer disconnected");
  }
}
static void transfer(int fd, void* buffer, size_t bytes, bool sending, Deadline deadline) {
  auto p=static_cast<char*>(buffer);
  while(bytes) {
    wait_fd(fd,sending?POLLOUT:POLLIN,deadline);
    ssize_t n=sending ? send(fd,p,bytes,MSG_NOSIGNAL|MSG_DONTWAIT) : recv(fd,p,bytes,MSG_DONTWAIT);
    if(n<0 && (errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK))continue;
    if(n<=0)throw std::runtime_error("control frame interrupted");
    p+=n;bytes-=size_t(n);
  }
}
void send_json(int fd,const Json& value,Deadline deadline) {
  auto text=value.dump();
  if(text.empty() || text.size()>4*1024*1024)throw std::invalid_argument("control frame exceeds 4 MiB");
  uint32_t length=htonl(uint32_t(text.size()));
  transfer(fd,&length,4,true,deadline);transfer(fd,text.data(),text.size(),true,deadline);
}
Json receive_json(int fd,Deadline deadline) {
  uint32_t length;transfer(fd,&length,4,false,deadline);length=ntohl(length);
  if(!length || length>4*1024*1024)throw std::invalid_argument("invalid control frame length");
  std::string text(length,'\0');transfer(fd,text.data(),text.size(),false,deadline);return strict_json(text);
}
static sockaddr_in endpoint(const std::string& address,uint16_t port) {
  sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(port);
  if(!port || inet_pton(AF_INET,address.c_str(),&addr.sin_addr)!=1)
    throw std::invalid_argument("control endpoint needs a numeric IPv4 address and nonzero port");
  return addr;
}
// The coordinator retains each immutable all-rank decision until every rank
// has advanced. A disconnected caller re-sends the same phase/sequence; it
// obtains the cached decision instead of repeating a logical state transition.
struct ControlTransport::Coordinator {
  struct Round {std::string phase;std::vector<Json> members;Json reply;size_t bytes=0;};
  struct Client {
    Socket socket;std::string incoming,outgoing;size_t expected=0,sent=0;
    uint64_t sequence=UINT64_MAX;bool parsed=false;Deadline expiry=after(2);
    explicit Client(int fd):socket(fd){}
  };
  Socket listener;std::string group,session;int ranks;
  std::atomic<bool> stop{false};std::thread worker;
  std::map<uint64_t,Round> rounds;uint64_t next=0;std::string fatal;
  std::vector<std::unique_ptr<Client>> clients;
  std::mutex injection_mutex;std::string injection_phase;unsigned dropped=0;
  Coordinator(int count,std::string g,std::string s,const std::string& bind,uint16_t port):group(std::move(g)),session(std::move(s)),ranks(count) {
    listener=Socket(socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0));
    if(listener.fd<0)throw std::runtime_error("control socket creation failed");
    int one=1;setsockopt(listener.fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));auto addr=endpoint(bind,port);
    if(::bind(listener.fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)) || listen(listener.fd,std::min(ranks+8,4096)))
      throw std::runtime_error("control bind/listen failed: "+std::string(strerror(errno)));
    worker=std::thread([this]{loop();});
  }
  ~Coordinator(){stop=true;if(worker.joinable())worker.join();}
  void reply(Client& client,const Json& value){
    auto text=value.dump();if(text.size()>4*1024*1024)throw std::runtime_error("control group result exceeds 4 MiB");
    uint32_t length=htonl(uint32_t(text.size()));client.outgoing.assign(reinterpret_cast<const char*>(&length),4);client.outgoing+=text;
  }
  void request(Client& client,const Json& message){
    exact_keys(message,{"group","session","sequence","phase","rank","payload"},{"group","session","sequence","phase","rank","payload"});
    if(message.at("group")!=group || message.at("session")!=session)throw std::invalid_argument("control session identity mismatch");
    auto sequence=integer(message.at("sequence"),"sequence");int rank=int(integer(message.at("rank"),"rank",ranks-1));
    auto phase=message.at("phase").get<std::string>();if(phase.empty() || phase.size()>128)throw std::invalid_argument("invalid control phase");
    if(sequence>next || (sequence<next && !rounds.count(sequence)))throw std::invalid_argument("control sequence is outside the retained decision boundary");
    auto found=rounds.find(sequence);
    if(found==rounds.end())found=rounds.emplace(sequence,Round{phase,std::vector<Json>(ranks),Json(),0}).first;
    auto& round=found->second;
    if(round.phase!=phase)throw std::invalid_argument("control phase/sequence identity mismatch");
    Json envelope={{"payload",message.at("payload")}};
    if(!round.members[rank].is_null() && round.members[rank]!=envelope)throw std::invalid_argument("duplicate control identity has different contents");
    if(round.members[rank].is_null()){
      round.bytes+=envelope.dump().size();if(round.bytes>3*1024*1024)throw std::invalid_argument("control group payload exceeds 3 MiB");
      round.members[rank]=envelope;
    }
    if(round.reply.is_null() && std::all_of(round.members.begin(),round.members.end(),[](const Json& v){return !v.is_null();})){
      Json members=Json::array();for(const auto& item:round.members)members.push_back(item.at("payload"));
      round.reply={{"group",group},{"session",session},{"sequence",sequence},{"phase",phase},{"members",members}};
      next++;while(rounds.size()>2)rounds.erase(rounds.begin());
    }
    client.sequence=sequence;client.parsed=true;
  }
  void loop(){
    while(!stop){
      pollfd descriptor{listener.fd,POLLIN,0};poll(&descriptor,1,1);
      if(descriptor.revents&POLLIN)for(;;){
        int fd=accept4(listener.fd,nullptr,nullptr,SOCK_CLOEXEC|SOCK_NONBLOCK);if(fd<0)break;
        if(clients.size()>=size_t(ranks+8)){::close(fd);continue;}
        int one=1;setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));clients.push_back(std::make_unique<Client>(fd));
      }
      for(auto it=clients.begin();it!=clients.end();){
        auto& c=**it;bool remove=false;
        try{
          if(!c.parsed && c.outgoing.empty()){
            char block[65536];ssize_t n=recv(c.socket.fd,block,sizeof(block),MSG_DONTWAIT);
            if(n==0)remove=true;else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)remove=true;
            else if(n>0){
              c.incoming.append(block,size_t(n));
              if(!c.expected && c.incoming.size()>=4){uint32_t length;memcpy(&length,c.incoming.data(),4);c.expected=ntohl(length);if(!c.expected || c.expected>4*1024*1024)throw std::invalid_argument("invalid control frame length");}
              if(c.expected && c.incoming.size()>=c.expected+4){if(c.incoming.size()!=c.expected+4)throw std::invalid_argument("one phase per control connection is required");request(c,strict_json(c.incoming.substr(4)));c.incoming.clear();}
            }
          }
          if(c.outgoing.empty() && !fatal.empty())reply(c,{{"error",fatal}});
          if(c.parsed && c.outgoing.empty()){
            auto round=rounds.find(c.sequence);
            // All ranks have already advanced past an evicted round. A stale
            // disconnected duplicate must not poison the current group.
            if(round==rounds.end())remove=true;
            else if(!round->second.reply.is_null()){
              std::lock_guard<std::mutex> lock(injection_mutex);
              if(dropped && injection_phase==round->second.phase){dropped--;remove=true;}
              else reply(c,round->second.reply);
            }
          }
          if(!remove && !c.outgoing.empty()){
            ssize_t n=send(c.socket.fd,c.outgoing.data()+c.sent,c.outgoing.size()-c.sent,MSG_NOSIGNAL|MSG_DONTWAIT);
            if(n>0)c.sent+=size_t(n);else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)remove=true;
            if(c.sent==c.outgoing.size())remove=true;
          }
        }catch(const std::exception& error){fatal=error.what();try{reply(c,{{"error",fatal}});}catch(...){remove=true;}}
        if(Clock::now()>=c.expiry)remove=true;
        if(remove)it=clients.erase(it);else++it;
      }
    }
  }
};
static Socket connect_control(const std::string& address,uint16_t port,Deadline deadline){
  Socket peer(socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0));if(peer.fd<0)throw std::runtime_error("control socket creation failed");
  auto addr=endpoint(address,port);int rc=connect(peer.fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr));
  if(rc<0 && errno==EINPROGRESS){wait_fd(peer.fd,POLLOUT,deadline);int error=0;socklen_t size=sizeof(error);if(getsockopt(peer.fd,SOL_SOCKET,SO_ERROR,&error,&size) || error)throw std::runtime_error("control connection not ready");}
  else if(rc<0)throw std::runtime_error("control connection unavailable");
  int one=1;setsockopt(peer.fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));return peer;
}
ControlTransport::ControlTransport(int rank,int ranks,const std::string& group,const std::string& session,
    const std::string& address,const std::string& bind,uint16_t port,Deadline deadline):
    rank_(rank),ranks_(ranks),group_(group),session_(session),address_(address),port_(port){
  if(ranks<1 || ranks>4096 || rank<0 || rank>=ranks || group.empty() || session.empty())throw std::invalid_argument("invalid control identity");
  endpoint(address,port);endpoint(bind,port);
  if(rank==0)coordinator_=std::make_unique<Coordinator>(ranks,group,session,bind,port);
  auto members=gather("MEMBERSHIP",{{"rank",rank},{"ranks",ranks}},deadline);
  for(int member=0;member<ranks;member++)if(members.at(member)!=Json{{"rank",member},{"ranks",ranks}})throw std::runtime_error("control membership differs");
}
ControlTransport::~ControlTransport()=default;
void ControlTransport::inject_lost_replies_for_test(const std::string& phase,unsigned count){
  if(!coordinator_)throw std::invalid_argument("only test coordinator may inject reply loss");
  std::lock_guard<std::mutex> lock(coordinator_->injection_mutex);coordinator_->injection_phase=phase;coordinator_->dropped=count;
}
Json ControlTransport::gather(const std::string& phase,const Json& payload,Deadline deadline){
  if(broken_)throw std::runtime_error("control session is unresolved");
  Json message={{"group",group_},{"session",session_},{"sequence",sequence_},{"phase",phase},{"rank",rank_},{"payload",payload}};
  std::string failure="control deadline expired";
  while(Clock::now()<deadline){
    // Every reconnect uses this same immutable message and original deadline.
    auto trial=std::min(deadline,after(.25));
    try{
      auto peer=connect_control(address_,port_,trial);send_json(peer.fd,message,trial);auto reply=receive_json(peer.fd,trial);
      if(reply.contains("error")){broken_=true;throw std::invalid_argument(reply.at("error").get<std::string>());}
      exact_keys(reply,{"group","session","sequence","phase","members"},{"group","session","sequence","phase","members"});
      for(const auto& key:{"group","session","sequence","phase"})if(reply.at(key)!=message.at(key))throw std::invalid_argument("control reply identity mismatch");
      auto members=reply.at("members");if(!members.is_array() || members.size()!=size_t(ranks_))throw std::invalid_argument("control membership incomplete");
      sequence_++;return members;
    }catch(const std::invalid_argument&){broken_=true;throw;}
    catch(const std::exception& error){failure=error.what();}
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  broken_=true;throw std::runtime_error("control outcome unresolved after bounded query/retransmission: "+failure);
}
} // namespace vericcl
