#include "vericcl.h"
#include "schedule_store.h"
#include "internal_api.h"
#include "gpu_operations.h"
#include "../probes/timing.cuh"
#include <mpi.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <thread>
#include <unistd.h>

using namespace vericcl;
static void api(VericclCode code,const char* name,VericclContext* context=nullptr) {
  if(code==VERICCL_OK)return;std::string message=std::string(name)+": "+vericcl_error_string(code);
  if(context){VericclStatus s{};s.struct_size=sizeof(s);s.abi_version=VERICCL_ABI_VERSION;if(vericcl_status(context,&s)==VERICCL_OK)message+="; "+std::string(s.message);}
  throw std::runtime_error(message);
}
static ncclDataType_t datatype(const std::string& name){static const std::map<std::string,ncclDataType_t> values={{"int8",ncclInt8},{"uint8",ncclUint8},{"int32",ncclInt32},{"uint32",ncclUint32},{"int64",ncclInt64},{"uint64",ncclUint64},{"float16",ncclFloat16},{"float32",ncclFloat32},{"float64",ncclFloat64},{"bfloat16",ncclBfloat16}};return values.at(name);}
static VericclResult result(){VericclResult r{};r.struct_size=sizeof(r);r.abi_version=VERICCL_ABI_VERSION;return r;}
static VericclBuffer buffer(void* data,uint64_t bytes){return {sizeof(VericclBuffer),VERICCL_ABI_VERSION,data,bytes};}
static int named(const std::string& name,const std::vector<std::string>& values){auto p=std::find(values.begin(),values.end(),name);if(p==values.end())throw std::invalid_argument("invalid operation name");return int(p-values.begin());}
static void mpi(int status){if(status!=MPI_SUCCESS)throw std::runtime_error("MPI startup identity exchange failed");}

int main(int argc,char** argv) {
  int rank=-1,ranks=0;bool initialized=false,gpu_released=false,correct=false;
  std::string directory,rank_directory,clock_text;Json report;
  VericclContext* context=nullptr;VericclWorkload* workload=nullptr;
  void* allocation_in=nullptr;void* allocation_out=nullptr;cudaStream_t stream=nullptr;
  Deadline close_deadline=Deadline::max();double cleanup_timeout_s=0;
  try {
    std::string config_path;
    for(int i=1;i<argc;i++) {
      std::string key(argv[i]);if(i==1 && key=="schedule")continue;
      if(key=="--config" && i+1<argc)config_path=argv[++i];else throw std::invalid_argument("usage: vericcl-native schedule --config FILE");
    }
    if(config_path.empty())throw std::invalid_argument("native execution requires explicit --config");
    auto settings=strict_json(read_file(config_path));
    const std::set<std::string> keys={"schema_version","group_id","session_id","rank_map","coordinator_address","bind_address","port","output_directory","manifest_path","timeouts","device_memory_budget_bytes","warmup","timed","capture_trace","self_test_api"};
    exact_keys(settings,keys,keys);if(settings.at("schema_version")!=1)throw std::invalid_argument("native launch schema mismatch");
    directory=settings.at("output_directory").get<std::string>();
    mpi(MPI_Init(&argc,&argv));initialized=true;mpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN));mpi(MPI_Comm_rank(MPI_COMM_WORLD,&rank));mpi(MPI_Comm_size(MPI_COMM_WORLD,&ranks));
    const auto& mapping=settings.at("rank_map");if(!mapping.is_array() || mapping.size()!=size_t(ranks))throw std::invalid_argument("MPI rank count differs from explicit mapping");
    // MPI is restricted to startup identity agreement. All subsequent request,
    // schedule, completion and clock-reference phases use the CPU TCP session.
    auto digest=sha256(settings.dump());std::vector<char> hashes(size_t(ranks)*64);
    mpi(MPI_Allgather(digest.data(),64,MPI_CHAR,hashes.data(),64,MPI_CHAR,MPI_COMM_WORLD));
    for(int peer=0;peer<ranks;peer++)if(std::string(hashes.data()+peer*64,64)!=digest)throw std::invalid_argument("rank launch configuration differs");
    for(int i=0;i<ranks;i++){exact_keys(mapping[i],{"rank","host","device"},{"rank","host","device"});if(mapping[i].at("rank")!=i)throw std::invalid_argument("rank mapping must be dense and ordered");}
    int device=int(integer(mapping[rank].at("device"),"device",INT_MAX));
    rank_directory=directory+"/rank-"+std::to_string(rank);std::filesystem::create_directory(rank_directory);
    auto manifest=read_file(settings.at("manifest_path"),128*1024*1024);auto spec=WorkSpec::parse(manifest,rank,ranks);auto type=datatype(spec.datatype);
    auto group=settings.at("group_id").get<std::string>(),session=settings.at("session_id").get<std::string>();
    auto address=settings.at("coordinator_address").get<std::string>(),bind=settings.at("bind_address").get<std::string>();
    auto timeouts=settings.at("timeouts");
    const std::set<std::string> timeout_keys={"request_prepare_timeout_s","execution_attempt_timeout_s","version_prepare_timeout_s","control_phase_timeout_s","close_cleanup_timeout_s"};exact_keys(timeouts,timeout_keys,timeout_keys);
    VericclConfig config{};config.struct_size=sizeof(config);config.abi_version=VERICCL_ABI_VERSION;config.rank=rank;config.rank_count=ranks;config.device=device;
    config.coordinator_port=uint16_t(integer(settings.at("port"),"port",65535));config.group_id=group.c_str();config.session_id=session.c_str();config.coordinator_address=address.c_str();config.bind_address=bind.c_str();config.evidence_directory=rank_directory.c_str();
#define TIME(name) config.name=timeouts.at(#name).get<double>();
    TIME(request_prepare_timeout_s) TIME(execution_attempt_timeout_s) TIME(version_prepare_timeout_s) TIME(control_phase_timeout_s) TIME(close_cleanup_timeout_s)
#undef TIME
    finite_positive(config.close_cleanup_timeout_s,"close timeout");cleanup_timeout_s=config.close_cleanup_timeout_s;
    config.device_memory_budget_bytes=integer(settings.at("device_memory_budget_bytes"),"device_memory_budget_bytes");
    int warmup=int(integer(settings.at("warmup"),"warmup",1000000)),timed=int(integer(settings.at("timed"),"timed",1000000));
    if(!timed || !settings.at("capture_trace").is_boolean() || !settings.at("self_test_api").is_boolean())throw std::invalid_argument("invalid sampling configuration");
    bool trace=settings.at("capture_trace");if(trace!=(spec.trace_records>0) || (trace && timed!=1))throw std::invalid_argument("trace requires exactly one timed invocation and allocated records");
    // Evaluate initialization before reading its returned context. Function
    // argument order could otherwise pass the old null context to api(),
    // losing the native failure details needed to diagnose startup failures.
    auto init_code=vericcl_init(&config,&context);
    api(init_code,"init",context);api(vericcl_prepare(context,manifest.c_str(),&workload),"prepare",context);
    VC_CUDA(cudaSetDevice(device));VC_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    void* input;void* output;
    if(spec.inplace) {
      uint64_t bytes=std::max(spec.buffers.input_offset+spec.buffers.input_bytes,spec.buffers.output_offset+spec.buffers.output_bytes);
      VC_CUDA(cudaMalloc(&allocation_in,std::max(uint64_t(1),bytes)));input=static_cast<char*>(allocation_in)+spec.buffers.input_offset;output=static_cast<char*>(allocation_in)+spec.buffers.output_offset;
    }else {VC_CUDA(cudaMalloc(&allocation_in,std::max(uint64_t(1),spec.buffers.input_bytes)));VC_CUDA(cudaMalloc(&allocation_out,std::max(uint64_t(1),spec.buffers.output_bytes)));input=allocation_in;output=allocation_out;}
    auto in=buffer(input,spec.buffers.input_bytes),out=buffer(output,spec.buffers.output_bytes);
    bool api_checks=false;
    if(settings.at("self_test_api").get<bool>()) {
      auto short_input=in;short_input.bytes=0;auto receipt=result();
      if(vericcl_execute(workload,&short_input,&out,stream,&receipt)!=VERICCL_INVALID)throw std::runtime_error("short buffer was accepted");
      VericclStatus status{};status.struct_size=sizeof(status);status.abi_version=VERICCL_ABI_VERSION;api(vericcl_status(context,&status),"status");
      if(status.request_id || status.logical_successes || status.caller_buffers_retained || status.uncertain)throw std::runtime_error("rejected buffer changed execution state");api_checks=true;
    }
    std::vector<double> times;std::vector<uint64_t> requests;unsigned long long bad=0;std::vector<double> all_times;
    int collective=named(spec.collective,{"broadcast","reduce","allgather","allreduce","alltoall","reduce_scatter"});
    int reduction=named(spec.reduction,{"sum","prod","min","max","avg"});
    double atol=spec.manifest.at("tolerances").value("absolute",0.),rtol=spec.manifest.at("tolerances").value("relative",0.);
    auto run=[&](bool measured){
      if(spec.buffers.output_bytes)VC_CUDA(cudaMemsetAsync(output,0,spec.buffers.output_bytes,stream));
      vericcl_fill_input(input,spec.buffers.input_bytes/spec.element_bytes,type,rank,stream);VC_CUDA(cudaStreamSynchronize(stream));
      auto receipt=result();api(vericcl_execute(workload,&in,&out,stream,&receipt),"execute",context);
      if(!receipt.gpu_completed || !receipt.output_valid || !receipt.buffers_reusable)throw std::runtime_error("enqueue was returned without complete GPU/group receipt");
      auto incorrect=vericcl_check_output(output,spec.buffers.output_bytes/spec.element_bytes,type,collective,reduction,rank,ranks,spec.root,spec.message_bytes/spec.element_bytes,atol,rtol,stream);
      if(spec.input_read_only && !spec.inplace)incorrect+=vericcl_check_output(input,spec.buffers.input_bytes/spec.element_bytes,type,0,0,rank,ranks,rank,spec.message_bytes/spec.element_bytes,0,0,stream);
      Json peers;api(driver_exchange(context,"NUMERICAL_CHECK",{{"incorrect",incorrect}},&peers,config.control_phase_timeout_s),"numerical agreement",context);
      atomic_evidence(rank_directory+"/numerical-request-"+std::to_string(receipt.request_id)+"-attempt-"+std::to_string(receipt.attempt)+".json",
        {{"request_id",receipt.request_id},{"attempt",receipt.attempt},{"version_id",spec.version},{"local_incorrect_elements",incorrect},{"ranks",peers}});
      for(const auto& peer:peers)bad+=peer.at("incorrect").get<unsigned long long>();
      if(bad)throw std::runtime_error("independent numerical reference check failed");
      all_times.push_back(receipt.device_time_us);if(measured){times.push_back(receipt.device_time_us);requests.push_back(receipt.request_id);}return receipt;
    };
    for(int i=0;i<warmup;i++)run(false);
    if(trace){std::string sample;api(clock_samples(workload,stream,&sample),"before clock samples",context);clock_text+=sample;api(vericcl_trace_control(workload,1),"trace reset",context);}
    for(int i=0;i<timed;i++)run(true);
    if(trace){api(vericcl_trace_control(workload,2),"trace finish",context);std::string sample;api(clock_samples(workload,stream,&sample),"after clock samples",context);clock_text+=sample;}
    cudaDeviceProp props{};VC_CUDA(cudaGetDeviceProperties(&props,device));int runtime_version=0,driver_version=0,nccl_version=0;
    VC_CUDA(cudaRuntimeGetVersion(&runtime_version));VC_CUDA(cudaDriverGetVersion(&driver_version));VC_NCCL(ncclGetVersion(&nccl_version));
    std::ostringstream uuid;uuid<<std::hex<<std::setfill('0');for(char c:props.uuid.bytes)uuid<<std::setw(2)<<int(uint8_t(c));
    correct=true;auto closed=result();
    if(close_deadline==Deadline::max())close_deadline=after(cleanup_timeout_s);
    double close_seconds=std::chrono::duration<double>(close_deadline-Clock::now()).count();
    api(driver_close(context,&closed,close_seconds),"close",context);
    if(!closed.resources_released || !closed.buffers_reusable)throw std::runtime_error("close did not acknowledge ownership release");
    VC_CUDA(cudaFree(allocation_in));allocation_in=nullptr;if(allocation_out){VC_CUDA(cudaFree(allocation_out));allocation_out=nullptr;}VC_CUDA(cudaStreamDestroy(stream));stream=nullptr;
    gpu_released=true;
    report={{"rank",rank},{"device",device},{"correct",true},{"gpu_completed",true},{"gpu_released",true},{"driver_kind","native_msccl"},
      {"warmup_iterations",warmup},{"timed_iterations",timed},{"device_times_us",times},{"request_ids",requests},{"mean_device_time_us",times.empty()?0.:std::accumulate(times.begin(),times.end(),0.)/times.size()},
      {"trace_enabled",trace},{"reset_after_warmup",trace},{"executed_after_reset",trace?timed:0},{"target_event_time_us",trace?times[0]:0.},{"clock_domain",trace?"globaltimer_ns":"none"},
      {"version_id",spec.version},{"contract_digest",spec.contract},{"api_preflight_checks",api_checks},
      {"runtime_identity",{{"gpu_uuid",uuid.str()},{"gpu_model",props.name},{"compute_capability",std::to_string(props.major)+"."+std::to_string(props.minor)},{"cuda_runtime",runtime_version},{"cuda_driver",driver_version},{"nccl_version",nccl_version},{"protocol","Simple"}}}};
    atomic_evidence(directory+"/rank-"+std::to_string(rank)+".json",report);
    if(trace){std::ofstream log(directory+"/clock.rank-"+std::to_string(rank)+".log",std::ios::binary);log<<clock_text;log.flush();if(!log)throw std::runtime_error("clock log write failed");}
    api(vericcl_destroy(&context),"destroy");
    mpi(MPI_Finalize());return 0;
  }catch(const std::exception& error) {
    std::cerr<<"VeriCCL native rank "<<rank<<": "<<error.what()<<'\n';
    // Closing may itself remain unresolved. Never free caller buffers merely
    // because the public wait ended or a process is about to terminate.
    bool safe=gpu_released;
    if(context && !safe){
      if(close_deadline==Deadline::max() && cleanup_timeout_s>0)close_deadline=after(cleanup_timeout_s);
      double remaining=std::chrono::duration<double>(close_deadline-Clock::now()).count();
      if(close_deadline!=Deadline::max() && remaining>0){auto closed=result();auto code=driver_close(context,&closed,remaining);safe=code==VERICCL_OK && closed.resources_released && closed.buffers_reusable;}
    }
    if(safe && !gpu_released) {
      if(allocation_in && cudaFree(allocation_in)!=cudaSuccess)safe=false;
      if(allocation_out && cudaFree(allocation_out)!=cudaSuccess)safe=false;
      if(stream && cudaStreamDestroy(stream)!=cudaSuccess)safe=false;
      gpu_released=safe;
    }
    try{if(rank>=0 && !directory.empty())atomic_evidence(directory+"/rank-"+std::to_string(rank)+".json",{{"rank",rank},{"correct",correct},{"gpu_released",gpu_released},{"error",error.what()},{"driver_kind","native_msccl"}});}catch(...){}
    // MPI_Finalize is a collective and cannot close a failed fixed membership.
    // The owned launcher handles peer termination; missing GPU receipts remain
    // quarantined by the Python supervisor.
    if(initialized)std::_Exit(1);return 1;
  }
}
