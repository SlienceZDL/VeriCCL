#include "local_control.h"
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace vericcl {
struct LocalControlServer::Impl {
  struct Client {
    Socket socket;
    std::string incoming,outgoing;
    size_t sent=0,expected=0;
    uint64_t waiting=0;
    bool parsed=false;
    Deadline expires=after(2);
    explicit Client(int fd):socket(fd){}
  };
  std::string path,session;
  Socket listener;
  dev_t device{};ino_t inode{};
  std::atomic<bool> stop{false};std::atomic<size_t> client_count{0};std::thread worker;
  std::mutex mutex;
  Json active,last_command,last_reply;
  bool dispatched=false;uint64_t last_id=0;
  std::function<Json()> status;
  std::vector<std::unique_ptr<Client>> clients;
  Impl(std::string p,std::string s,std::function<Json()> query):path(std::move(p)),session(std::move(s)),status(std::move(query)) {
    sockaddr_un address{};address.sun_family=AF_UNIX;
    if(path.empty() || path[0]!='/' || path.size()>=sizeof(address.sun_path))throw std::invalid_argument("Unix socket needs an absolute path shorter than 108 bytes");
    std::memcpy(address.sun_path,path.c_str(),path.size()+1);
    listener=Socket(socket(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0));
    if(listener.fd<0 || bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)) || chmod(path.c_str(),0600) || listen(listener.fd,16))
      throw std::runtime_error("cannot create owned Unix socket; existing paths are not replaced");
    struct stat st{};if(lstat(path.c_str(),&st))throw std::runtime_error("cannot identify Unix socket");device=st.st_dev;inode=st.st_ino;
    worker=std::thread([this]{loop();});
  }
  ~Impl(){stop=true;if(worker.joinable())worker.join();struct stat st{};if(!lstat(path.c_str(),&st) && st.st_dev==device && st.st_ino==inode && S_ISSOCK(st.st_mode))unlink(path.c_str());}
  void reply(Client& client,const Json& value){
    auto text=value.dump();if(text.size()>4*1024*1024)text=Json{{"error","control response exceeds 4 MiB"}}.dump();uint32_t n=htonl(uint32_t(text.size()));
    client.outgoing.assign(reinterpret_cast<const char*>(&n),4);client.outgoing+=text;client.expires=after(2);
  }
  void request(Client& client,const Json& message){
    exact_keys(message,{"session","command_id","action","manifest_path","manifest_sha256","warmup","timed","capture_trace","remaining_s"},{"session","command_id","action"});
    if(message.at("session")!=session)throw std::invalid_argument("Unix session identity differs");
    auto id=integer(message.at("command_id"),"command_id");auto action=message.at("action").get<std::string>();
    if(action=="status") {
      if(id!=0)throw std::invalid_argument("status uses command_id 0");auto state=status();
      std::lock_guard<std::mutex> lock(mutex);
      reply(client,{{"status",state},{"last_command_id",last_id},{"last_command_reply",last_reply},
        {"pending_command_id",active.is_null()?Json(nullptr):active.at("command_id")}});return;
    }
    if(!id || !std::set<std::string>{"execute","prepare","sample","close"}.count(action))throw std::invalid_argument("invalid live command");
    if(action=="prepare" && (!message.contains("manifest_path") || !message.contains("manifest_sha256")))throw std::invalid_argument("prepare requires immutable manifest path and SHA-256");
    std::lock_guard<std::mutex> lock(mutex);
    if(!active.is_null()) {
      if(active!=message)throw std::invalid_argument("another live command is pending or duplicate contents differ");
    } else if(id==last_id) {
      if(message!=last_command)throw std::invalid_argument("completed command identity reused with different contents");reply(client,last_reply);return;
    } else {
      if(id!=last_id+1)throw std::invalid_argument("live command sequence is not the next unresolved boundary");active=message;dispatched=false;
    }
    client.waiting=id;client.expires=Deadline::max();
  }
  void loop() {
    while(!stop) {
      pollfd poller{listener.fd,POLLIN,0};poll(&poller,1,5);
      if(poller.revents&POLLIN)for(;;){int fd=accept4(listener.fd,nullptr,nullptr,SOCK_NONBLOCK|SOCK_CLOEXEC);if(fd<0)break;if(clients.size()>=16){close(fd);continue;}clients.push_back(std::make_unique<Client>(fd));client_count=clients.size();}
      for(auto it=clients.begin();it!=clients.end();) {
        auto& c=**it;bool remove=false;
        try {
          if(!c.parsed) {
            char block[65536];ssize_t n=recv(c.socket.fd,block,sizeof(block),MSG_DONTWAIT);
            if(n==0)remove=true;else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)remove=true;
            else if(n>0) {
              c.incoming.append(block,size_t(n));
              if(!c.expected && c.incoming.size()>=4){uint32_t len;std::memcpy(&len,c.incoming.data(),4);c.expected=ntohl(len);if(!c.expected || c.expected>4*1024*1024)throw std::invalid_argument("Unix frame length invalid");}
              if(c.expected && c.incoming.size()>=c.expected+4){if(c.incoming.size()!=c.expected+4)throw std::invalid_argument("one command per Unix connection is required");c.parsed=true;request(c,strict_json(c.incoming.substr(4)));}
            }
          }
          if(c.waiting && c.outgoing.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            if(c.waiting==last_id)reply(c,last_reply);
            else if(c.waiting<last_id)reply(c,{{"error","result superseded; inspect durable command evidence"}});
            else {char byte;ssize_t n=recv(c.socket.fd,&byte,1,MSG_PEEK|MSG_DONTWAIT);if(n==0)remove=true;}
          }
          if(!c.outgoing.empty()) {
            ssize_t n=send(c.socket.fd,c.outgoing.data()+c.sent,c.outgoing.size()-c.sent,MSG_DONTWAIT|MSG_NOSIGNAL);
            if(n>0)c.sent+=size_t(n);else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)remove=true;
            if(c.sent==c.outgoing.size())remove=true;
          }
        }catch(const std::exception& error){c.parsed=true;c.waiting=0;reply(c,{{"error",error.what()}});}
        if(Clock::now()>=c.expires)remove=true;
        if(remove){it=clients.erase(it);client_count=clients.size();}else++it;
      }
    }
  }
};
LocalControlServer::LocalControlServer(const std::string& path,const std::string& session,std::function<Json()> status):impl_(std::make_unique<Impl>(path,session,std::move(status))){}
LocalControlServer::~LocalControlServer()=default;
bool LocalControlServer::take(Json& command){std::lock_guard<std::mutex> lock(impl_->mutex);if(impl_->active.is_null() || impl_->dispatched)return false;impl_->dispatched=true;command=impl_->active;return true;}
void LocalControlServer::complete(uint64_t id,const Json& result){std::lock_guard<std::mutex> lock(impl_->mutex);if(impl_->active.is_null() || impl_->active.at("command_id")!=id)throw std::invalid_argument("completion has no matching live command");impl_->last_id=id;impl_->last_command=impl_->active;impl_->last_reply={{"command_id",id},{"result",result}};impl_->active=nullptr;impl_->dispatched=false;}
void LocalControlServer::wait_idle(double timeout){auto deadline=after(timeout);while(impl_->client_count && Clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(2));}
}
