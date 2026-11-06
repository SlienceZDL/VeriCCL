/* Executes the production C API with CPU doubles only at CUDA/NCCL calls. */
#include "../../runtime/execution/api.cc"
#include <filesystem>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>

static std::atomic<bool> pause_attributes{false};
static std::atomic<unsigned> launches{0},default_queries{0};
const char* cudaGetErrorString(cudaError_t){return "CPU device stub";}
cudaError_t cudaSetDevice(int){return cudaSuccess;}
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes,const void*){
  while(pause_attributes)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  *attributes={cudaMemoryTypeDevice,0};return cudaSuccess;
}
cudaError_t cudaStreamQuery(cudaStream_t stream){
  if(!stream){++default_queries;return cudaSuccess;}
  return stream->ready?cudaSuccess:cudaErrorNotReady;
}
cudaError_t cudaEventCreate(cudaEvent_t* event){*event=new CpuEvent;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t event){delete event;return cudaSuccess;}
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t){return cudaSuccess;}
cudaError_t cudaEventQuery(cudaEvent_t){return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* time,cudaEvent_t,cudaEvent_t){*time=.01;return cudaSuccess;}
cudaError_t cudaMemGetInfo(size_t* free,size_t* total){*free=*total=uint64_t(1)<<40;return cudaSuccess;}
cudaError_t cudaMalloc(void** pointer,size_t bytes){*pointer=std::malloc(bytes);return cudaSuccess;}
cudaError_t cudaFree(void* pointer){std::free(pointer);return cudaSuccess;}
cudaError_t cudaMemcpyAsync(void* to,const void* from,size_t bytes,int,cudaStream_t){std::memcpy(to,from,bytes);return cudaSuccess;}
const char* ncclGetErrorString(ncclResult_t){return "CPU communication stub";}
ncclResult_t ncclGetUniqueId(ncclUniqueId* id){*id={};return ncclSuccess;}
ncclResult_t ncclCommGetAsyncError(ncclComm_t,ncclResult_t* result){*result=ncclSuccess;return ncclSuccess;}
ncclResult_t ncclCommDestroy(ncclComm_t comm){delete comm;return ncclSuccess;}
ncclResult_t ncclCommAbort(ncclComm_t comm){delete comm;return ncclSuccess;}
ncclResult_t ncclCustomCollective(const void*,void*,size_t,ncclDataType_t,ncclRedOp_t,int,ncclComm_t,cudaStream_t){++launches;return ncclSuccess;}
extern "C" ncclResult_t vericclNcclCommInitRank(ncclComm_t* comm,int,ncclUniqueId,int,const VericclNcclPlanOptions*){*comm=new CpuComm;return ncclSuccess;}
extern "C" ncclResult_t vericclNcclPlanAdmission(ncclComm_t,size_t,ncclDataType_t,ncclRedOp_t,int* resident){*resident=1;return ncclSuccess;}
extern "C" ncclResult_t vericclNcclAbortSignal(ncclComm_t){return ncclSuccess;}
extern "C" ncclResult_t vericclTracePause(ncclComm_t){return ncclSuccess;}
extern "C" ncclResult_t vericclTraceReset(ncclComm_t,size_t){return ncclSuccess;}
extern "C" ncclResult_t vericclTraceFinish(ncclComm_t){return ncclSuccess;}
void vericcl_scale_output(void*,size_t,ncclDataType_t,int,cudaStream_t){}
void vericcl_sample_timer(unsigned long long* value,cudaStream_t){*value=1;}

static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static VericclResult result(){VericclResult r{};r.struct_size=sizeof(r);r.abi_version=VERICCL_ABI_VERSION;return r;}
static uint16_t port(){
  Socket descriptor(socket(AF_INET,SOCK_STREAM,0));sockaddr_in address{};address.sin_family=AF_INET;
  address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);socklen_t size=sizeof(address);
  require(bind(descriptor.fd,reinterpret_cast<sockaddr*>(&address),size)==0,"bind test port");
  require(getsockname(descriptor.fd,reinterpret_cast<sockaddr*>(&address),&size)==0,"read test port");
  return ntohs(address.sin_port);
}
struct Fixture {
  std::shared_ptr<State> state=std::make_shared<State>();
  VericclContext* context=new VericclContext{state};
  VericclWorkload workload;
  CpuStream stream;
  int input[4]{},output[4]{};
  VericclBuffer in{sizeof(VericclBuffer),VERICCL_ABI_VERSION,input,sizeof(input)};
  VericclBuffer out{sizeof(VericclBuffer),VERICCL_ABI_VERSION,output,sizeof(output)};
  explicit Fixture(const std::string& directory){
    std::filesystem::create_directories(directory);state->directory=directory;
    state->config.rank_count=1;state->config.request_prepare_timeout_s=.03;
    state->config.execution_attempt_timeout_s=.5;state->config.control_phase_timeout_s=.5;
    state->config.close_cleanup_timeout_s=.03;
    state->group="ownership";state->session=directory;
    state->transport=std::make_unique<ControlTransport>(0,1,state->group,state->session,"127.0.0.1","127.0.0.1",port(),after(1));
    auto v=std::make_shared<Version>();v->ready=v->comm_initialized=true;v->comm=new CpuComm;
    cudaEventCreate(&v->start);cudaEventCreate(&v->done);
    v->spec.buffers={16,16,0,0,false};v->spec.datatype="float32";v->spec.reduction="sum";
    v->spec.runtime_bytes=16;v->spec.element_bytes=4;v->spec.version="fixed";v->spec.contract="contract";
    state->version=v;workload.state=state;workload.version=v;
  }
  void wait_worker(){
    auto job=state->job;require(bool(job),"worker exists");std::unique_lock<std::mutex> lock(job->mutex);
    require(job->condition.wait_for(lock,std::chrono::seconds(2),[&]{return job->done;}),"worker must finish");
  }
  void finish(){
    stream.ready=true;pause_attributes=false;wait_worker();auto closed=result();
    require(vericcl_close(context,&closed)==VERICCL_OK && closed.buffers_reusable && closed.resources_released,"completed user stream can be released");
    require(vericcl_destroy(&context)==VERICCL_OK,"closed context can be destroyed");
  }
};
static void stalled_producer(const std::string& root,bool during_attributes){
  Fixture fixture(root);fixture.stream.ready=false;pause_attributes=during_attributes;
  launches=default_queries=0;auto receipt=result();
  require(vericcl_execute(&fixture.workload,&fixture.in,&fixture.out,&fixture.stream,&receipt)==VERICCL_UNRESOLVED,"producer preparation must time out");
  require(!receipt.buffers_reusable && !receipt.output_valid && !launches,"timeout must not release buffers or enqueue collective");
  if(during_attributes){
    auto closed=result();require(vericcl_close(fixture.context,&closed)==VERICCL_UNRESOLVED && !closed.buffers_reusable,"blocked worker cannot release buffers");
    pause_attributes=false;
  }
  fixture.wait_worker();auto closed=result();
  auto close_code=vericcl_close(fixture.context,&closed);
  require(close_code==VERICCL_UNRESOLVED && !closed.buffers_reusable && !closed.resources_released,
          "close released caller buffers while the original nondefault stream was still busy");
  fixture.wait_worker();
  VericclStatus status{};status.struct_size=sizeof(status);status.abi_version=VERICCL_ABI_VERSION;
  require(vericcl_status(fixture.context,&status)==VERICCL_OK && status.caller_buffers_retained,"status must retain the caller buffers");
  require(!default_queries,"cleanup must not substitute the default stream");
  fixture.finish();
}
static void successful_execution(const std::string& root){
  Fixture fixture(root);launches=0;auto receipt=result();
  require(vericcl_execute(&fixture.workload,&fixture.in,&fixture.out,&fixture.stream,&receipt)==VERICCL_OK,"ready stream executes");
  require(receipt.output_valid && receipt.buffers_reusable && receipt.gpu_completed && launches==1,"execution returns a complete release receipt");
  fixture.finish();
}
static void rejected_buffer(const std::string& root){
  Fixture fixture(root);auto bad=fixture.in;bad.bytes=0;auto receipt=result();launches=0;
  require(vericcl_execute(&fixture.workload,&bad,&fixture.out,&fixture.stream,&receipt)==VERICCL_INVALID,"short buffer must be rejected");
  VericclStatus status{};status.struct_size=sizeof(status);status.abi_version=VERICCL_ABI_VERSION;
  require(vericcl_status(fixture.context,&status)==VERICCL_OK && !status.caller_buffers_retained && !status.request_id && !launches,"pure input rejection must not start a request");
  fixture.finish();
}
int main(int argc,char** argv){
  try{
    require(argc==2,"provide a private fixture directory");std::string root=argv[1];
    stalled_producer(root+"/producer",false);stalled_producer(root+"/attributes",true);
    successful_execution(root+"/success");rejected_buffer(root+"/rejected");
    std::cout<<"production execute/status/close retain the original stream across preparation timeouts; CPU device doubles only\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
