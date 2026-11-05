#include "local_control.h"
#include <cassert>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
using namespace vericcl;
Json call(const std::string& path,Json request){
  Socket socket_fd(socket(AF_UNIX,SOCK_STREAM,0));sockaddr_un address{};address.sun_family=AF_UNIX;std::snprintf(address.sun_path,sizeof(address.sun_path),"%s",path.c_str());
  if(connect(socket_fd.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)))throw std::runtime_error("connect failed");
  send_json(socket_fd.fd,request,after(2));return receive_json(socket_fd.fd,after(2));
}
int main(int argc,char**argv){
  if(argc!=2)return 2;std::string path=argv[1];
  LocalControlServer server(path,"session",[]{return Json{{"native_busy",true}};});
  auto request=Json{{"session","session"},{"command_id",1},{"action","execute"}};
  auto client=std::async(std::launch::async,[&]{return call(path,request);});
  Json command;
  for(int i=0;i<100 && !server.take(command);i++)usleep(1000);
  assert(command==request);
  auto status=call(path,{{"session","session"},{"command_id",0},{"action","status"}});
  assert(status["status"]["native_busy"]==true);
  auto duplicate=std::async(std::launch::async,[&]{return call(path,request);});
  auto conflict=call(path,{{"session","session"},{"command_id",1},{"action","close"}});
  assert(conflict.contains("error"));
  server.complete(1,{{"success",true},{"request_id",1}});
  assert(client.get()==duplicate.get());
  assert(call(path,request)["result"]["request_id"]==1);
  assert(!server.take(command));
  std::cout<<"Unix status remains responsive; duplicate commands execute once\n";
}
