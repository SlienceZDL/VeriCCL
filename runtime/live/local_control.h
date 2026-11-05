#ifndef VERICCL_LOCAL_CONTROL_H
#define VERICCL_LOCAL_CONTROL_H
#include "control_transport.h"
#include <functional>
#include <memory>
namespace vericcl {
class LocalControlServer {
  struct Impl;
  std::unique_ptr<Impl> impl_;
 public:
  LocalControlServer(const std::string& path,const std::string& session,std::function<Json()> status);
  ~LocalControlServer();
  bool take(Json& command);
  void complete(uint64_t command_id,const Json& result);
  void wait_idle(double timeout_s);
  LocalControlServer(const LocalControlServer&)=delete;
  LocalControlServer& operator=(const LocalControlServer&)=delete;
};
}
#endif
