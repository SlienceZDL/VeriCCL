#include "vericcl.h"
#include "vericcl_plan.h"
#include "vericcl_trace_control.h"
#include "control_transport.h"
#include "schedule_store.h"
#include "gpu_operations.h"
#include "internal_api.h"
#include "buffer_guard.h"
#include "operation_wait.h"
#include "../probes/timing.cuh"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace vericcl {
struct Version {
  WorkSpec spec;
  std::string owned_xml;
  VericclNcclPlanOptions options{};
  ncclComm_t comm=nullptr;
  cudaEvent_t start=nullptr,done=nullptr;
  bool ready=false, comm_initialized=false, event_recorded=false, gpu_complete=true, release_failed=false;
  uint64_t reservation=0;
};
struct Job : OperationWait {
  VericclCode code=VERICCL_FAILED;VericclResult result{};
  std::shared_ptr<Version> version;Json json;std::string text;
};
struct State {
  VericclConfig config{};
  std::string group,session,address,bind,directory;
  std::mutex mutex;
  bool busy=false,uncertain=false,closed=false,buffers_retained=false,request_active=false;
  uint64_t next_request=1,request_id=0,attempt=0,successes=0,reserved=0,version_count=0;
  std::string committed,message;
  SingleScheduleIdentity schedule_identity;
  std::shared_ptr<Job> job;
  std::unique_ptr<ControlTransport> transport;
  std::shared_ptr<Version> version;
  std::unique_ptr<VericclWorkload> handle;
  std::unique_ptr<BufferGuard> guard;
  unsigned long long* clock_timer=nullptr;
  unsigned long long clock_value=0;
  void* clock_stream=nullptr;
};
static void string_copy(char* out,size_t size,const std::string& text){if(size){std::snprintf(out,size,"%s",text.c_str());}}
template<class T> static bool valid_struct(const T* p){return p && p->struct_size==sizeof(T) && p->abi_version==VERICCL_ABI_VERSION;}
static void result_identity(State& s,VericclResult& r){r.struct_size=sizeof(r);r.abi_version=VERICCL_ABI_VERSION;r.request_id=s.request_id;r.attempt=s.attempt;r.logical_successes=s.successes;string_copy(r.version_id,sizeof(r.version_id),s.committed);}
static ncclDataType_t type_of(const std::string& name){
  static const std::map<std::string,ncclDataType_t> names={{"int8",ncclInt8},{"uint8",ncclUint8},{"int32",ncclInt32},{"uint32",ncclUint32},{"int64",ncclInt64},{"uint64",ncclUint64},{"float16",ncclFloat16},{"float32",ncclFloat32},{"float64",ncclFloat64},{"bfloat16",ncclBfloat16}};return names.at(name);
}
static ncclRedOp_t op_of(const std::string& name){
  static const std::map<std::string,ncclRedOp_t> names={{"sum",ncclSum},{"avg",ncclSum},{"prod",ncclProd},{"min",ncclMin},{"max",ncclMax}};return names.at(name);
}
static Json exchange(State& s,const std::string& phase,const Json& payload,Deadline parent){
  if(!s.transport)throw std::runtime_error("control transport unavailable");
  return s.transport->gather(phase,payload,child_deadline(parent,s.config.control_phase_timeout_s));
}
static void unanimous(const Json& values,const std::vector<std::string>& keys){
  for(const auto& value:values) {
    if(value.contains("error") && !value.at("error").is_null())throw std::invalid_argument("rank rejected phase: "+value.at("error").get<std::string>());
    for(const auto& key:keys)if(value.at(key)!=values[0].at(key))throw std::invalid_argument("rank contract mismatch: "+key);
  }
}
static VericclCode submit(const std::shared_ptr<State>& s,Deadline deadline,
    std::function<void(State&,Job&,Deadline)> fn,std::shared_ptr<Job>& returned,bool allow_uncertain=false) {
  auto job=std::make_shared<Job>();
  job->public_deadline=deadline;
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    if(s->closed)return VERICCL_CLOSED;
    if(s->busy)return VERICCL_BUSY;
    if(s->uncertain && !allow_uncertain)return VERICCL_UNRESOLVED;
    s->busy=true;s->job=job;
  }
  returned=job;
  try {
    std::thread([s,job,fn,deadline] {
      auto start=Clock::now();
      try {fn(*s,*job,deadline);job->code=VERICCL_OK;}
      catch(const std::invalid_argument& error){job->code=VERICCL_INVALID;string_copy(job->result.message,sizeof(job->result.message),error.what());}
      catch(const std::exception& error){job->code=VERICCL_FAILED;string_copy(job->result.message,sizeof(job->result.message),error.what());}
      catch(...){job->code=VERICCL_FAILED;string_copy(job->result.message,sizeof(job->result.message),"unknown native failure");}
      {
        std::lock_guard<std::mutex> lock(s->mutex);
        if(job->code!=VERICCL_OK && (s->buffers_retained || s->request_active || (s->transport && s->transport->broken())))s->uncertain=true;
        if(s->uncertain && job->code!=VERICCL_OK)job->code=VERICCL_UNRESOLVED;
        s->message=job->result.message;s->busy=false;
        result_identity(*s,job->result);job->result.code=job->code;job->result.wall_time_us=elapsed_us(start);
      }
      job->complete();
    }).detach(); // State and Job own the complete operation until its receipt.
  }catch(...){std::lock_guard<std::mutex> lock(s->mutex);s->busy=false;return VERICCL_FAILED;}
  if(!job->wait_public(deadline)) {
    std::lock_guard<std::mutex> lock(s->mutex);s->uncertain=true;s->message="native worker deadline expired; ownership retained";return VERICCL_UNRESOLVED;
  }
  return job->code;
}
static void wait_event(Version& version,Deadline deadline) {
  for(;;) {
    auto status=cudaEventQuery(version.done);
    if(status==cudaSuccess){version.gpu_complete=true;return;}
    if(status!=cudaErrorNotReady)cuda_check(status,"completion event");
    ncclResult_t async_error=ncclSuccess;VC_NCCL(ncclCommGetAsyncError(version.comm,&async_error));VC_NCCL(async_error);
    if(Clock::now()>=deadline)throw std::runtime_error("execution attempt deadline expired before GPU completion");
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}
static uint64_t reservation(const WorkSpec& p){
  const char* text=getenv("NCCL_BUFFSIZE");if(!text)throw std::invalid_argument("NCCL_BUFFSIZE must be explicitly set before process start");
  size_t end=0;uint64_t bytes=std::stoull(text,&end);if(end!=strlen(text) || bytes<8 || bytes>INT32_MAX)throw std::invalid_argument("invalid NCCL_BUFFSIZE");
  // Reserve all base ring channels plus plan peer connections, both directions,
  // and LL/LL128 overhead. The actual free-memory delta is recorded as well.
  uint64_t communication=multiply(2*(32+multiply(p.channels,std::max(1,p.ranks-1))),bytes+5*1024*1024);
  return communication+128*1024*1024+p.scratch_bytes+multiply(p.trace_records,80)+
      p.message_bytes+p.buffers.output_bytes;
}
static std::string hex_id(const ncclUniqueId& id){std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto c:id.internal)out<<std::setw(2)<<int(uint8_t(c));return out.str();}
static ncclUniqueId parse_id(const std::string& text){ncclUniqueId id{};if(text.size()!=2*sizeof(id))throw std::invalid_argument("invalid communicator identity");for(size_t i=0;i<sizeof(id);i++){auto part=text.substr(2*i,2);size_t end=0;int v=std::stoi(part,&end,16);if(end!=2)throw std::invalid_argument("invalid communicator identity");id.internal[i]=char(v);}return id;}
} // namespace vericcl

struct VericclContext {std::shared_ptr<vericcl::State> state;};
struct VericclWorkload {std::weak_ptr<vericcl::State> state;std::shared_ptr<vericcl::Version> version;};

using namespace vericcl;
extern "C" VericclCode vericcl_init(const VericclConfig* config,VericclContext** out) {
  if(!valid_struct(config) || !out)return VERICCL_INVALID;*out=nullptr;
  try {
    if(config->rank_count<1 || config->rank_count>4096 || config->rank<0 || config->rank>=config->rank_count || config->device<0 || config->reserved)
      throw std::invalid_argument("invalid rank/device configuration");
    validate_execution_config(*config);
    auto copy=[](const char* v){if(!v || !*v || strlen(v)>4096)throw std::invalid_argument("invalid configuration string");return std::string(v);};
    auto s=std::make_shared<State>();s->config=*config;s->group=copy(config->group_id);s->session=copy(config->session_id);s->address=copy(config->coordinator_address);s->bind=copy(config->bind_address);s->directory=copy(config->evidence_directory);
    struct stat st{};if(s->directory[0]!='/' || lstat(s->directory.c_str(),&st) || !S_ISDIR(st.st_mode) || st.st_uid!=getuid())throw std::invalid_argument("evidence directory must be an existing owned absolute directory");
    auto handle=new VericclContext{s};*out=handle;std::shared_ptr<Job> job;
    return submit(s,after(config->control_phase_timeout_s),[](State& s,Job&,Deadline deadline){
      s.transport=std::make_unique<ControlTransport>(s.config.rank,s.config.rank_count,s.group,s.session,s.address,s.bind,s.config.coordinator_port,deadline);
      Json config={{"group",s.group},{"ranks",s.config.rank_count},{"request_prepare",s.config.request_prepare_timeout_s},{"execution",s.config.execution_attempt_timeout_s},{"version_prepare",s.config.version_prepare_timeout_s},{"control",s.config.control_phase_timeout_s},{"close",s.config.close_cleanup_timeout_s}};
      auto members=exchange(s,"INIT",{{"config",config}},deadline);unanimous(members,{"config"});
    },job);
  }catch(...){return VERICCL_INVALID;}
}
static VericclCode prepare_until(VericclContext* context,const char* text,VericclWorkload** out,Deadline deadline) {
  if(!context || !text || !out)return VERICCL_INVALID;*out=nullptr;
  auto s=context->state;std::string manifest(text);std::shared_ptr<Job> job;
  auto code=submit(s,deadline,[manifest](State& s,Job& job,Deadline deadline){
    WorkSpec spec;std::string error;
    try{spec=WorkSpec::parse(manifest,s.config.rank,s.config.rank_count);s.schedule_identity.admit(spec);}catch(const std::exception& e){error=e.what();}
    Json payload={{"error",error.empty()?Json(nullptr):Json(error)},{"version",spec.version},{"contract",spec.contract}};
    auto members=exchange(s,"PREPARE_VALIDATE",payload,deadline);unanimous(members,{"version","contract"});
    if(s.version) {
      if(!s.version->ready)throw std::invalid_argument("the context's schedule is no longer ready; create a new context");
      job.version=s.version;return;
    }
    auto version=std::make_shared<Version>();version->spec=spec;
    size_t free_before=0,total=0;error.clear();
    try {
      VC_CUDA(cudaSetDevice(s.config.device));VC_CUDA(cudaMemGetInfo(&free_before,&total));
      version->reservation=reservation(spec);
      if(version->reservation>free_before || (s.config.device_memory_budget_bytes && s.reserved+version->reservation>s.config.device_memory_budget_bytes))throw std::invalid_argument("schedule exceeds device memory reservation");
      version->owned_xml=immutable_xml_copy(spec,s.directory);
    }catch(const std::exception& e){error=e.what();}
    unanimous(exchange(s,"PREPARE_ADMISSION",{{"error",error.empty()?Json(nullptr):Json(error)}},deadline),{});
    s.version=version;s.reserved=version->reservation;
    {std::lock_guard<std::mutex> lock(s.mutex);s.version_count=1;}
    ncclUniqueId id{};if(s.config.rank==0)VC_NCCL(ncclGetUniqueId(&id));
    auto identities=exchange(s,"PREPARE_ID",{{"uid",s.config.rank==0?hex_id(id):""}},deadline);id=parse_id(identities[0].at("uid"));
    version->options={sizeof(VericclNcclPlanOptions),VERICCL_NCCL_PLAN_ABI,version->owned_xml.c_str(),spec.trace_records,spec.trace_prefix.c_str()};
    // options.trace_prefix must point to storage retained by this version.
    version->options.trace_prefix=version->spec.trace_prefix.c_str();
    error.clear();int resident=0;
    try {
      VC_NCCL(vericclNcclCommInitRank(&version->comm,s.config.rank_count,id,s.config.rank,&version->options));
      version->comm_initialized=true;
      VC_NCCL(vericclNcclPlanAdmission(version->comm,spec.runtime_bytes,type_of(spec.datatype),op_of(spec.reduction),&resident));
      VC_CUDA(cudaEventCreate(&version->start));VC_CUDA(cudaEventCreate(&version->done));
      if(spec.trace_records)VC_NCCL(vericclTracePause(version->comm));
    }catch(const std::exception& e){error=e.what();}
    auto ready=exchange(s,"READY",{{"error",error.empty()?Json(nullptr):Json(error)},{"version",spec.version},{"contract",spec.contract}},deadline);
    try{unanimous(ready,{"version","contract"});}catch(...){std::lock_guard<std::mutex> lock(s.mutex);s.uncertain=true;throw;}
    version->ready=true;job.version=version;size_t free_after=0;VC_CUDA(cudaMemGetInfo(&free_after,&total));
    atomic_evidence(s.directory+"/prepared-"+spec.version+"-seq-"+std::to_string(s.transport->sequence())+".json",{{"version",spec.version},{"contract",spec.contract},{"rank",s.config.rank},{"resident_tb_limit",resident},{"reserved_bytes",version->reservation},{"free_bytes_before",free_before},{"free_bytes_after",free_after},{"ready",true}});
  },job);
  if(code==VERICCL_OK && job && job->version) {
    std::lock_guard<std::mutex> lock(s->mutex);
    if(!s->handle){s->handle=std::make_unique<VericclWorkload>();s->handle->state=s;s->handle->version=job->version;}
    *out=s->handle.get();
  }
  return code;
}
extern "C" VericclCode vericcl_prepare(VericclContext* context,const char* text,VericclWorkload** out) {
  if(!context)return VERICCL_INVALID;
  return prepare_until(context,text,out,after(context->state->config.version_prepare_timeout_s));
}
#include "execute_impl.h"
static VericclCode release_until(VericclWorkload* workload,VericclResult* result,Deadline deadline) {
  if(!workload || !valid_struct(result))return VERICCL_INVALID;auto s=workload->state.lock();if(!s)return VERICCL_CLOSED;auto v=workload->version;std::shared_ptr<Job> job;
  *result={};result->struct_size=sizeof(*result);result->abi_version=VERICCL_ABI_VERSION;
  auto code=submit(s,deadline,[v](State& s,Job& job,Deadline deadline){
    Json check={{"version",v->spec.version},{"error",nullptr}};
    if(s.committed==v->spec.version || !v->ready || !v->gpu_complete || v->release_failed)check["error"]="version is committed, unready or still referenced by unresolved GPU work";
    unanimous(exchange(s,"RELEASE_VERSION",check,deadline),{"version"});VC_CUDA(cudaSetDevice(s.config.device));
    try{if(v->comm){VC_NCCL(ncclCommDestroy(v->comm));v->comm=nullptr;}if(v->start){VC_CUDA(cudaEventDestroy(v->start));v->start=nullptr;}if(v->done){VC_CUDA(cudaEventDestroy(v->done));v->done=nullptr;}}
    catch(...){v->release_failed=true;std::lock_guard<std::mutex> lock(s.mutex);s.uncertain=true;throw;}
    v->ready=false;s.reserved=0;
    {std::lock_guard<std::mutex> lock(s.mutex);s.version_count=0;}
    job.result.resources_released=1;job.result.buffers_reusable=1;
  },job);
  if(job){std::lock_guard<std::mutex> lock(job->mutex);if(job->done)*result=job->result;}result->code=code;return code;
}
extern "C" VericclCode vericcl_release(VericclWorkload* workload,VericclResult* result) {
  if(!workload)return VERICCL_INVALID;auto s=workload->state.lock();if(!s)return VERICCL_CLOSED;
  return release_until(workload,result,after(s->config.close_cleanup_timeout_s));
}
extern "C" VericclCode vericcl_status(VericclContext* context,VericclStatus* status) {
  if(!context || !valid_struct(status))return VERICCL_INVALID;auto s=context->state;std::lock_guard<std::mutex> lock(s->mutex);
  *status={};status->struct_size=sizeof(*status);status->abi_version=VERICCL_ABI_VERSION;status->busy=s->busy;status->uncertain=s->uncertain;status->closed=s->closed;status->caller_buffers_retained=s->buffers_retained;
  status->request_id=s->request_id;status->attempt=s->attempt;status->logical_successes=s->successes;status->retained_versions=s->version_count;status->code=s->closed?VERICCL_CLOSED:s->uncertain?VERICCL_UNRESOLVED:VERICCL_OK;
  string_copy(status->version_id,sizeof(status->version_id),s->committed);string_copy(status->message,sizeof(status->message),s->message);return VERICCL_OK;
}
static VericclCode close_until(VericclContext* context,VericclResult* result,Deadline deadline) {
  if(!context || !valid_struct(result))return VERICCL_INVALID;auto s=context->state;
  *result={};result->struct_size=sizeof(*result);result->abi_version=VERICCL_ABI_VERSION;
  std::shared_ptr<Job> previous;{std::lock_guard<std::mutex> lock(s->mutex);previous=s->job;if(s->closed){*result={};result->struct_size=sizeof(*result);result->abi_version=VERICCL_ABI_VERSION;result->resources_released=1;result->buffers_reusable=1;return VERICCL_OK;}}
  if(previous){std::unique_lock<std::mutex> wait(previous->mutex);if(!previous->condition.wait_until(wait,deadline,[&]{return previous->done;})){result->code=VERICCL_UNRESOLVED;result->resources_released=0;result->buffers_reusable=0;return VERICCL_UNRESOLVED;}}
  std::shared_ptr<Job> job;auto code=submit(s,deadline,[](State& s,Job& job,Deadline deadline){
    VC_CUDA(cudaSetDevice(s.config.device));
    bool uncertain;{std::lock_guard<std::mutex> lock(s.mutex);uncertain=s.uncertain;}
    if(s.clock_timer){wait_stream_stopped(s.clock_stream,deadline);VC_CUDA(cudaFree(s.clock_timer));s.clock_timer=nullptr;}
    if(s.version){auto& v=*s.version;
      if(v.release_failed)throw std::runtime_error("previous native release failed; ownership remains unresolved");
      if(v.comm && !v.comm_initialized)throw std::runtime_error("partial communicator initialization has unresolved ownership");
      if(!v.gpu_complete || (uncertain && v.comm)){
        if(!v.gpu_complete && !s.guard)throw std::runtime_error("GPU completion unresolved without an owned stream");
        stop_schedule(v,s.guard?s.guard->stream():nullptr,deadline);
      }
      try{if(v.comm){VC_NCCL(ncclCommDestroy(v.comm));v.comm=nullptr;}if(v.start){VC_CUDA(cudaEventDestroy(v.start));v.start=nullptr;}if(v.done){VC_CUDA(cudaEventDestroy(v.done));v.done=nullptr;}}
      catch(...){v.release_failed=true;throw;}
      if(Clock::now()>=deadline)throw std::runtime_error("close deadline exhausted");
    }
    if(s.guard){s.guard->release(deadline);s.guard.reset();}
    {std::lock_guard<std::mutex> lock(s.mutex);s.closed=true;s.buffers_retained=false;s.request_active=false;s.reserved=0;s.version_count=0;}
    job.result.resources_released=1;job.result.buffers_reusable=1;
  },job,true);
  if(job){std::lock_guard<std::mutex> lock(job->mutex);if(job->done){*result=job->result;return code;}}
  result->code=code;result->resources_released=0;result->buffers_reusable=0;return code;
}
extern "C" VericclCode vericcl_close(VericclContext* context,VericclResult* result) {
  if(!context)return VERICCL_INVALID;
  return close_until(context,result,after(context->state->config.close_cleanup_timeout_s));
}
static VericclCode trace_control_until(VericclWorkload* workload,int action,Deadline deadline) {
  if(!workload || action<0 || action>2)return VERICCL_INVALID;auto s=workload->state.lock();if(!s)return VERICCL_CLOSED;auto v=workload->version;std::shared_ptr<Job> job;
  return submit(s,deadline,[v,action](State& s,Job&,Deadline deadline){
    if(!v->spec.trace_records || !v->gpu_complete)throw std::invalid_argument("trace control requires an idle trace-enabled version");VC_CUDA(cudaSetDevice(s.config.device));
    unanimous(exchange(s,"TRACE_CONTROL",{{"action",action},{"version",v->spec.version}},deadline),{"action","version"});
    if(action==0)VC_NCCL(vericclTracePause(v->comm));else if(action==1)VC_NCCL(vericclTraceReset(v->comm,v->spec.expected_steps));else VC_NCCL(vericclTraceFinish(v->comm));
  },job);
}
extern "C" VericclCode vericcl_trace_control(VericclWorkload* workload,int action) {
  if(!workload)return VERICCL_INVALID;auto s=workload->state.lock();if(!s)return VERICCL_CLOSED;
  return trace_control_until(workload,action,after(s->config.control_phase_timeout_s));
}
extern "C" VericclCode vericcl_destroy(VericclContext** context) {
  if(!context || !*context)return VERICCL_INVALID;auto s=(*context)->state;
  {std::lock_guard<std::mutex> lock(s->mutex);if(!s->closed || s->busy)return VERICCL_UNRESOLVED;}
  delete *context;*context=nullptr;return VERICCL_OK;
}
extern "C" const char* vericcl_error_string(VericclCode code){switch(code){case VERICCL_OK:return "completed";case VERICCL_INVALID:return "invalid contract";case VERICCL_BUSY:return "another operation is active";case VERICCL_TIMEOUT:return "deadline expired";case VERICCL_UNRESOLVED:return "ownership or group outcome is unresolved";case VERICCL_CLOSED:return "context closed";default:return "native operation failed";}}
namespace vericcl {
VericclCode driver_close(VericclContext* context,VericclResult* result,double seconds) {
  if(!context || !std::isfinite(seconds) || seconds<=0)return VERICCL_INVALID;
  return close_until(context,result,after(std::min(seconds,context->state->config.close_cleanup_timeout_s)));
}
VericclCode driver_exchange(VericclContext* context,const std::string& phase,const Json& local,Json* all,double timeout) {
  if(!context || !all || !std::isfinite(timeout) || timeout<=0)return VERICCL_INVALID;std::shared_ptr<Job> job;auto code=submit(context->state,after(std::min(timeout,context->state->config.control_phase_timeout_s)),[phase,local](State& s,Job& job,Deadline deadline){job.json=exchange(s,phase,local,deadline);},job);
  if(code==VERICCL_OK)*all=job->json;return code;
}
static uint64_t host_time(){timespec t;if(clock_gettime(CLOCK_MONOTONIC_RAW,&t))throw std::runtime_error("host clock failed");return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
VericclCode clock_samples(VericclWorkload* workload,void* raw_stream,std::string* output,double seconds) {
  if(!workload || !output || !std::isfinite(seconds) || (seconds<=0 && seconds!=-1))return VERICCL_INVALID;auto s=workload->state.lock();if(!s)return VERICCL_CLOSED;auto v=workload->version;std::shared_ptr<Job> job;
  auto code=submit(s,after(seconds<0?s->config.control_phase_timeout_s:std::min(seconds,s->config.control_phase_timeout_s)),[v,raw_stream](State& s,Job& job,Deadline deadline){
    VC_CUDA(cudaSetDevice(s.config.device));auto stream=static_cast<cudaStream_t>(raw_stream);uint64_t uncertainty=UINT64_MAX;int64_t offset=0;
    for(int i=0;i<16;i++) {
      auto before=host_time();exchange(s,"CLOCK_BARRIER",{},deadline);auto reference=s.config.rank==0?host_time():0;
      auto values=exchange(s,"CLOCK_REFERENCE",{{"reference",reference}},deadline);auto end=host_time();uint64_t bound=(end-before+1)/2;
      if(bound<uncertainty){uncertainty=bound;offset=int64_t(values[0].at("reference").get<uint64_t>())-int64_t(before+(end-before)/2);}
    }
    if(s.clock_timer)throw std::runtime_error("previous clock sampling still owns its device allocation");
    s.clock_stream=raw_stream;VC_CUDA(cudaMalloc(&s.clock_timer,sizeof(*s.clock_timer)));std::ostringstream text;
    for(int i=0;i<16;i++){
      auto before=host_time();vericcl_sample_timer(s.clock_timer,stream);s.clock_value=0;VC_CUDA(cudaMemcpyAsync(&s.clock_value,s.clock_timer,sizeof(s.clock_value),cudaMemcpyDeviceToHost,stream));wait_stream_stopped(stream,deadline);auto end=host_time();auto ticks=s.clock_value;
      if(!ticks)throw std::runtime_error("nanosecond globaltimer unavailable");text<<"VERICCL_CLOCK_SYNC "<<s.config.rank<<' '<<ticks<<' '<<before<<' '<<end<<' '<<offset<<' '<<uncertainty<<'\n';
      if(Clock::now()>=deadline)throw std::runtime_error("clock sampling deadline expired");
    }
    VC_CUDA(cudaFree(s.clock_timer));s.clock_timer=nullptr;job.text=text.str();
  },job);if(code==VERICCL_OK)*output=job->text;return code;
}
}
