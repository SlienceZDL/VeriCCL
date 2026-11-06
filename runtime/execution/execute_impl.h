/* Included once by api.cc after the private State and schedule definitions. */
static void stop_schedule(Version& version,void* stream,Deadline deadline) {
  if(version.release_failed)throw std::runtime_error("previous communicator release remains unresolved");
  if(version.comm && !version.comm_initialized)throw std::runtime_error("partial communicator initialization remains unresolved");
  if(version.comm)VC_NCCL(vericclNcclAbortSignal(version.comm));
  wait_stream_stopped(stream,deadline);version.gpu_complete=true;
  if(version.comm) {
    try{VC_NCCL(ncclCommAbort(version.comm));version.comm=nullptr;version.comm_initialized=false;}
    catch(...){version.release_failed=true;throw;}
  }
  version.ready=false;version.event_recorded=false;
}
static VericclCode execute_until(VericclWorkload* workload,const VericclBuffer* input,const VericclBuffer* output,void* raw_stream,VericclResult* result,Deadline deadline) {
  if(!workload || !valid_struct(input) || !valid_struct(output) || !valid_struct(result))return VERICCL_INVALID;
  auto state=workload->state.lock();if(!state)return VERICCL_CLOSED;auto version=workload->version;auto in=*input,out=*output;
  std::shared_ptr<Job> job;
  auto code=submit(state,deadline,[version,in,out,raw_stream](State& s,Job& job,Deadline parent){
    auto& v=*version;auto begin=Clock::now();auto prep=child_deadline(parent,s.config.request_prepare_timeout_s);
    job.set_phase_deadline(prep);
    auto stream=static_cast<cudaStream_t>(raw_stream);std::string error;
    try {
      if(!v.ready || !v.comm || !v.gpu_complete || s.request_active || s.guard)throw std::invalid_argument("workload is not ready or an unfinished request is retained");
      validate_buffers(v.spec.buffers,in.data,in.bytes,out.data,out.bytes);
      // Retain the caller's exact stream before CUDA validation or producer
      // synchronization can outlive the public wait. close() must inspect
      // this stream even if no collective has been enqueued yet.
      s.guard=std::make_unique<BufferGuard>(raw_stream);
      {std::lock_guard<std::mutex> lock(s.mutex);s.buffers_retained=true;}
      VC_CUDA(cudaSetDevice(s.config.device));
      for(auto pointer:{std::make_pair(in.data,v.spec.buffers.input_bytes),std::make_pair(out.data,v.spec.buffers.output_bytes)})if(pointer.second){
        cudaPointerAttributes attributes{};VC_CUDA(cudaPointerGetAttributes(&attributes,pointer.first));
        if(attributes.type!=cudaMemoryTypeDevice || attributes.device!=s.config.device)throw std::invalid_argument("buffer is not device memory on this rank's GPU");
      }
      wait_stream_stopped(raw_stream,prep);
    }catch(const std::exception& failure){error=failure.what();}
    uint64_t request;{std::lock_guard<std::mutex> lock(s.mutex);request=s.next_request;}
    Json identity={{"request",request},{"attempt",0},{"version",v.spec.version},{"contract",v.spec.contract}};
    auto ready=identity;ready["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"REQUEST_PREPARE",ready,prep),{"request","attempt","version","contract"});
    {std::lock_guard<std::mutex> lock(s.mutex);s.request_id=request;s.attempt=0;s.request_active=true;s.buffers_retained=true;}
    job.result.preparation_time_us=elapsed_us(begin);
    auto control_start=Clock::now();
    unanimous(exchange(s,"EXECUTE_READY",identity,prep),{"request","attempt","version","contract"});
    unanimous(exchange(s,"EXECUTE_ACK",identity,prep),{"request","attempt","version","contract"});
    {std::lock_guard<std::mutex> lock(s.mutex);s.committed=v.spec.version;}
    auto execution=child_deadline(parent,s.config.execution_attempt_timeout_s);error.clear();
    job.set_phase_deadline(execution);
    try {
      unanimous(exchange(s,"START",identity,execution),{"request","attempt","version","contract"});
      unanimous(exchange(s,"START_ACK",identity,execution),{"request","attempt","version","contract"});
      job.result.control_time_us=elapsed_us(control_start);
      v.gpu_complete=false;v.event_recorded=false;
      VC_CUDA(cudaEventRecord(v.start,stream));
      VC_NCCL(ncclCustomCollective(in.data,out.data,v.spec.runtime_bytes/v.spec.element_bytes,type_of(v.spec.datatype),op_of(v.spec.reduction),0,v.comm,stream));
      if(v.spec.reduction=="avg")vericcl_scale_output(out.data,v.spec.buffers.output_bytes/v.spec.element_bytes,type_of(v.spec.datatype),s.config.rank_count,stream);
      VC_CUDA(cudaEventRecord(v.done,stream));v.event_recorded=true;wait_event(v,execution);
      float ms=0;VC_CUDA(cudaEventElapsedTime(&ms,v.start,v.done));job.result.device_time_us=ms*1000.;job.result.gpu_completed=1;
    }catch(const std::exception& failure){error=failure.what();}
    job.set_phase_deadline(parent);
    auto outcome=identity;outcome["error"]=error;outcome["gpu_completed"]=v.gpu_complete;
    auto outcomes=exchange(s,"EXECUTION_OUTCOME",outcome,parent);
    for(const auto& peer:outcomes){
      for(const auto& key:{"request","attempt","version","contract"})if(peer.at(key)!=identity.at(key))throw std::runtime_error("execution outcome identity mismatch");
      if(!peer.at("error").get<std::string>().empty() || !peer.at("gpu_completed").get<bool>())
        throw std::runtime_error("schedule execution failed; cleanup is required before a new context can be used");
    }
    error.clear();
    try{atomic_evidence(s.directory+"/request-"+std::to_string(request)+"-attempt-0.json",
      {{"group",s.group},{"request",request},{"attempt",0},{"version",v.spec.version},{"rank",s.config.rank},
       {"gpu_completed",true},{"completion_receipt",true},{"device_time_us",job.result.device_time_us}});
    }catch(const std::exception& failure){error=failure.what();}
    auto receipt=identity;receipt["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"COMPLETION_EVIDENCE",receipt,parent),{"request","attempt","version","contract"});
    error.clear();try{s.guard->release(parent);s.guard.reset();}catch(const std::exception& failure){error=failure.what();}
    receipt["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"COMPLETION_RELEASE",receipt,parent),{"request","attempt","version","contract"});
    {std::lock_guard<std::mutex> lock(s.mutex);s.successes++;s.next_request++;s.request_active=false;s.buffers_retained=false;job.result.output_valid=1;job.result.buffers_reusable=1;}
  },job);
  if(job){std::lock_guard<std::mutex> lock(job->mutex);if(job->done){*result=job->result;return code;}}
  *result={};result->struct_size=sizeof(*result);result->abi_version=VERICCL_ABI_VERSION;result->code=code;
  {std::lock_guard<std::mutex> lock(state->mutex);result_identity(*state,*result);string_copy(result->message,sizeof(result->message),state->message);}
  return code;
}
extern "C" VericclCode vericcl_execute(VericclWorkload* workload,const VericclBuffer* input,const VericclBuffer* output,void* raw_stream,VericclResult* result) {
  if(!workload)return VERICCL_INVALID;auto state=workload->state.lock();if(!state)return VERICCL_CLOSED;
  return execute_until(workload,input,output,raw_stream,result,after(state->config.request_prepare_timeout_s+state->config.execution_attempt_timeout_s));
}
