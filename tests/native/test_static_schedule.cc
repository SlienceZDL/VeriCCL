#include "schedule_store.h"
#include <cassert>
#include <fstream>
#include <iostream>
using namespace vericcl;

template<class F> void rejects(F fn) {
  bool failed=false;
  try{fn();}catch(const std::invalid_argument&){failed=true;}
  assert(failed);
}
int main(int argc,char** argv) {
  assert(argc==2);
  std::string xml="<algo/>",path=argv[1];
  std::ofstream(path)<<xml;
  Json contract={{"collective","broadcast"},{"datatype","float32"},{"reduction_op",nullptr},
    {"root",0},{"inplace",false},{"rank_count",1},{"input_size_bytes",16}};
  Json manifest=contract;
  manifest.update({{"schema_version",1},{"version_id",sha256(xml)},{"xml_sha256",sha256(xml)},
    {"xml_path",path},{"contract_digest",sha256(contract.dump())},{"slice_bytes",16},
    {"runtime_bytes",16},{"channels",1},{"trace_prefix",""},{"tolerances",Json::object()},
    {"ranks",Json::array({{{"rank",0},{"input_bytes",16},{"output_bytes",16},{"scratch_bytes",0},
      {"input_offset_bytes",0},{"output_offset_bytes",0},{"input_read_only",true},{"threadblocks",0},
      {"expected_steps",0},{"trace_records",0},{"steps",Json::array()}}})}});
  auto obsolete=manifest;obsolete["light_telemetry"]=true;
  rejects([&]{WorkSpec::parse(obsolete.dump(),0,1);});
  auto spec=WorkSpec::parse(manifest.dump(),0,1);
  assert(spec.version==sha256(xml) && spec.trace_records==0);
  SingleScheduleIdentity identity;
  identity.admit(spec);
  identity.admit(spec); // Repeated preparation of the same manifest is idempotent.
  std::string second_xml="<algo channels=\"2\"/>",second_path=path+".second";
  std::ofstream(second_path)<<second_xml;
  auto second=manifest;
  second["version_id"]=second["xml_sha256"]=sha256(second_xml);second["xml_path"]=second_path;
  auto replacement=WorkSpec::parse(second.dump(),0,1);
  rejects([&]{identity.admit(replacement);});
  identity.admit(spec); // A rejected replacement does not change the binding.
  auto trace=manifest;trace["ranks"][0]["trace_records"]=1;trace["trace_prefix"]=path+".trace";
  auto traced=WorkSpec::parse(trace.dump(),0,1);
  rejects([&]{identity.admit(traced);});
  VericclConfig config{};
  config.request_prepare_timeout_s=config.execution_attempt_timeout_s=config.version_prepare_timeout_s=
      config.control_phase_timeout_s=config.close_cleanup_timeout_s=1;
  validate_execution_config(config);
  for(double VericclConfig::*field:{&VericclConfig::background_tuning_task_timeout_s,&VericclConfig::recovery_timeout_s}) {
    config.*field=1;rejects([&]{validate_execution_config(config);});config.*field=0;
  }
  for(uint32_t VericclConfig::*field:{&VericclConfig::max_retries,&VericclConfig::fault_inject_attempts}) {
    config.*field=1;rejects([&]{validate_execution_config(config);});config.*field=0;
  }
  config.fault_inject_request=1;rejects([&]{validate_execution_config(config);});config.fault_inject_request=0;
  config.reserved=1;rejects([&]{validate_execution_config(config);});config.reserved=0;
  config.execution_attempt_timeout_s=0;rejects([&]{validate_execution_config(config);});
  std::cout<<"static schedule manifest contracts passed; no GPU execution\n";
}
