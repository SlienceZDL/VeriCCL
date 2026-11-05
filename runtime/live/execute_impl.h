/* Included once by api.cc after the private State and version definitions. */
static void stop_version_for_replay(State& state,Version& version,void* stream,Deadline deadline) {
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
static void prepare_replay_version(State& state,Version& version,Deadline deadline) {
  ncclUniqueId id{};if(state.config.rank==0)VC_NCCL(ncclGetUniqueId(&id));
  auto identities=exchange(state,"RECOVERY_COMM_ID",{{"uid",state.config.rank==0?hex_id(id):""}},deadline);
  id=parse_id(identities[0].at("uid"));std::string error;int resident=0;
  try {
    VC_NCCL(vericclNcclCommInitRank(&version.comm,state.config.rank_count,id,state.config.rank,&version.options));version.comm_initialized=true;
    VC_NCCL(vericclNcclPlanAdmission(version.comm,version.spec.runtime_bytes,type_of(version.spec.datatype),op_of(version.spec.reduction),&resident));
    if(version.spec.trace_records)VC_NCCL(vericclTracePause(version.comm));
    if(version.counters && version.light_active)VC_NCCL(vericclProbeAttach(version.comm,version.counters,version.spec.threadblocks*256));
  }catch(const std::exception& failure){error=failure.what();}
  unanimous(exchange(state,"RECOVERY_READY",{{"version",version.spec.version},{"error",error.empty()?Json(nullptr):Json(error)}},deadline),{"version"});
  version.ready=true;
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
      validate_buffers(v.spec.buffers,in.data,in.bytes,out.data,out.bytes);VC_CUDA(cudaSetDevice(s.config.device));
      for(auto pointer:{std::make_pair(in.data,v.spec.buffers.input_bytes),std::make_pair(out.data,v.spec.buffers.output_bytes)})if(pointer.second){
        cudaPointerAttributes attributes{};VC_CUDA(cudaPointerGetAttributes(&attributes,pointer.first));
        if(attributes.type!=cudaMemoryTypeDevice || attributes.device!=s.config.device)throw std::invalid_argument("buffer is not device memory on this rank's GPU");
      }
    }catch(const std::exception& failure){error=failure.what();}
    uint64_t request;{std::lock_guard<std::mutex> lock(s.mutex);request=s.next_request;}
    Json identity={{"request",request},{"attempt",0},{"version",v.spec.version},{"contract",v.spec.contract}};
    auto ready=identity;ready["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"REQUEST_PREPARE",ready,prep),{"request","attempt","version","contract"});
    {std::lock_guard<std::mutex> lock(s.mutex);s.request_id=request;s.attempt=0;s.request_active=true;s.buffers_retained=true;}
    s.guard=std::make_unique<BufferGuard>(in.data,v.spec.buffers.input_bytes,out.data,v.spec.buffers.output_bytes,raw_stream,v.spec.inplace || !v.spec.input_read_only);
    auto backup_start=Clock::now();error.clear();
    try{s.guard->protect(prep);}catch(const std::exception& failure){error=failure.what();}
    auto protected_inputs=exchange(s,"ORIGINAL_INPUT_READY",{{"error",error.empty()?Json(nullptr):Json(error)}},prep);
    bool cancel_prepare=std::any_of(protected_inputs.begin(),protected_inputs.end(),[](const Json& item){return !item.at("error").is_null();});
    if(cancel_prepare){
      error.clear();try{s.guard->release(prep);s.guard.reset();}catch(const std::exception& failure){error=failure.what();}
      unanimous(exchange(s,"REQUEST_CANCELLED",{{"request",request},{"committed",false},{"error",error.empty()?Json(nullptr):Json(error)}},prep),{"request","committed"});
      {std::lock_guard<std::mutex> lock(s.mutex);s.request_active=false;s.buffers_retained=false;}
      throw std::invalid_argument("request cancelled unanimously before COMMIT because original input protection failed");
    }
    job.result.backup_time_us=elapsed_us(backup_start);job.result.preparation_time_us=elapsed_us(begin);
    std::unique_ptr<RecoveryLedger> recovery;Deadline recovery_start{};uint64_t attempt=0;
    for(;;) {
      auto active_parent=recovery?recovery->deadline():parent;
      auto control_start=Clock::now();identity["attempt"]=attempt;
      unanimous(exchange(s,"COMMIT",identity,recovery?active_parent:prep),{"request","attempt","version","contract"});
      unanimous(exchange(s,"COMMIT_ACK",identity,recovery?active_parent:prep),{"request","attempt","version","contract"});
      {std::lock_guard<std::mutex> lock(s.mutex);s.committed=v.spec.version;s.attempt=attempt;}
      auto execution=child_deadline(active_parent,s.config.execution_attempt_timeout_s);error.clear();
      job.set_phase_deadline(execution);
      try {
        unanimous(exchange(s,"START",identity,execution),{"request","attempt","version","contract"});
        unanimous(exchange(s,"START_ACK",identity,execution),{"request","attempt","version","contract"});
        job.result.control_time_us+=elapsed_us(control_start);
        if(v.counters && v.light_active)VC_CUDA(cudaMemsetAsync(v.counters,0,v.spec.threadblocks*256*sizeof(VericclProbeCounter),stream));
        v.gpu_complete=false;v.event_recorded=false;
        VC_CUDA(cudaEventRecord(v.start,stream));
        VC_NCCL(ncclCustomCollective(in.data,out.data,v.spec.runtime_bytes/v.spec.element_bytes,type_of(v.spec.datatype),op_of(v.spec.reduction),0,v.comm,stream));
        if(v.spec.reduction=="avg")vericcl_scale_output(out.data,v.spec.buffers.output_bytes/v.spec.element_bytes,type_of(v.spec.datatype),s.config.rank_count,stream);
        VC_CUDA(cudaEventRecord(v.done,stream));v.event_recorded=true;wait_event(v,execution);
        float ms=0;VC_CUDA(cudaEventElapsedTime(&ms,v.start,v.done));job.result.device_time_us=ms*1000.;job.result.gpu_completed=1;
        if(s.config.fault_inject_request==request && attempt<s.config.fault_inject_attempts)
          throw std::runtime_error("injected failure after real GPU completion; output intentionally withheld");
      }catch(const std::exception& failure){error=failure.what();}
      if(!error.empty() && !recovery){recovery_start=Clock::now();recovery=std::make_unique<RecoveryLedger>(s.config.max_retries,child_deadline(parent,s.config.recovery_timeout_s));}
      if(recovery)job.set_phase_deadline(recovery->deadline());
      auto outcome=identity;outcome["error"]=error;outcome["gpu_completed"]=v.gpu_complete;
      auto outcomes=exchange(s,"ATTEMPT_OUTCOME",outcome,recovery?recovery->deadline():parent);
      bool failed=false;
      for(const auto& peer:outcomes){for(const auto& key:{"request","attempt","version","contract"})if(peer.at(key)!=identity.at(key))throw std::runtime_error("attempt outcome identity mismatch");failed|=!peer.at("error").get<std::string>().empty() || !peer.at("gpu_completed").get<bool>();}
      if(!failed)break;
      if(!recovery){recovery_start=Clock::now();recovery=std::make_unique<RecoveryLedger>(s.config.max_retries,child_deadline(parent,s.config.recovery_timeout_s));}
      job.set_phase_deadline(recovery->deadline());
      atomic_evidence(s.directory+"/recovery-request-"+std::to_string(request)+"-attempt-"+std::to_string(attempt)+".json",
        {{"request",request},{"attempt",attempt},{"version",v.spec.version},{"outcomes",outcomes},
         {"source",s.config.fault_inject_request==request && attempt<s.config.fault_inject_attempts?"fault_injection":"runtime"},
         {"original_available",s.guard->original_available()},{"backup_bytes",s.guard->backup_bytes()},
         {"recovery_deadline_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(recovery->deadline().time_since_epoch()).count()}});
      error.clear();try{stop_version_for_replay(s,v,raw_stream,recovery->deadline());}catch(const std::exception& failure){error=failure.what();}
      unanimous(exchange(s,"RECOVERY_STOPPED",{{"request",request},{"attempt",attempt},{"gpu_stopped",v.gpu_complete},
        {"communication_stopped",v.comm==nullptr},{"error",error.empty()?Json(nullptr):Json(error)}},recovery->deadline()),{"request","attempt","gpu_stopped","communication_stopped"});
      recovery->confirm_stopped(v.gpu_complete,v.comm==nullptr,true);
      // No device/link identity is inferred from a timeout. Explicit failed
      // links are excluded by the controller before a version is prepared.
      attempt=recovery->next_attempt(s.guard->original_available(),true);
      s.guard->restore(recovery->deadline(),true);
      prepare_replay_version(s,v,recovery->deadline());
    }
    if(recovery)job.result.recovery_time_us=elapsed_us(recovery_start);
    auto success_parent=recovery?recovery->deadline():parent;
    job.set_phase_deadline(success_parent);
    v.telemetry=Json::array();
    if(v.counters && v.light_active){
      std::vector<VericclProbeCounter> counters(v.spec.threadblocks*256);VC_CUDA(cudaMemcpy(counters.data(),v.counters,counters.size()*sizeof(VericclProbeCounter),cudaMemcpyDeviceToHost));
      for(auto step:v.spec.local.at("steps")){
        auto position=step.at("tb_id").get<size_t>()*256+step.at("step_index").get<size_t>();const auto& counter=counters.at(position);
        step["service_ns"]=counter.service_ns;step["dependency_wait_ns"]=counter.dependency_wait_ns;step["samples"]=counter.samples;step["invalid_samples"]=counter.invalid_samples;step["source"]="software_device_counters";v.telemetry.push_back(step);
      }
    }
    error.clear();
    try{atomic_evidence(s.directory+"/request-"+std::to_string(request)+"-attempt-"+std::to_string(attempt)+".json",
      {{"group",s.group},{"request",request},{"attempt",attempt},{"version",v.spec.version},{"rank",s.config.rank},
       {"gpu_completed",true},{"completion_receipt",true},{"device_time_us",job.result.device_time_us},
       {"backup_time_us",job.result.backup_time_us},{"recovery_time_us",job.result.recovery_time_us},{"telemetry",v.telemetry}});
    }catch(const std::exception& failure){error=failure.what();}
    auto receipt=identity;receipt["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"COMPLETION_EVIDENCE",receipt,success_parent),{"request","attempt","version","contract"});
    error.clear();try{s.guard->release(success_parent);s.guard.reset();}catch(const std::exception& failure){error=failure.what();}
    receipt["error"]=error.empty()?Json(nullptr):Json(error);
    unanimous(exchange(s,"COMPLETION_RELEASE",receipt,success_parent),{"request","attempt","version","contract"});
    {std::lock_guard<std::mutex> lock(s.mutex);s.successes++;s.next_request++;s.request_active=false;s.buffers_retained=false;job.result.output_valid=1;job.result.buffers_reusable=1;}
  },job);
  if(job){std::lock_guard<std::mutex> lock(job->mutex);if(job->done){*result=job->result;return code;}}
  *result={};result->struct_size=sizeof(*result);result->abi_version=VERICCL_ABI_VERSION;result->code=code;
  {std::lock_guard<std::mutex> lock(state->mutex);result_identity(*state,*result);string_copy(result->message,sizeof(result->message),state->message);}
  return code;
}
extern "C" VericclCode vericcl_execute(VericclWorkload* workload,const VericclBuffer* input,const VericclBuffer* output,void* raw_stream,VericclResult* result) {
  if(!workload)return VERICCL_INVALID;auto state=workload->state.lock();if(!state)return VERICCL_CLOSED;
  return execute_until(workload,input,output,raw_stream,result,after(state->config.request_prepare_timeout_s+state->config.execution_attempt_timeout_s+state->config.recovery_timeout_s));
}
