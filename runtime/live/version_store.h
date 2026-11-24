#ifndef VERICCL_VERSION_STORE_H
#define VERICCL_VERSION_STORE_H
#include "contract.h"
namespace vericcl {
struct WorkSpec {
  Json manifest, local;
  std::string version, contract, xml, collective, datatype, reduction, trace_prefix;
  int ranks=0, rank=0, root=-1, channels=0;
  uint64_t message_bytes=0, slice_bytes=0, runtime_bytes=0, element_bytes=0;
  uint64_t scratch_bytes=0, trace_records=0, expected_steps=0, threadblocks=0;
  bool inplace=false, input_read_only=true, light_telemetry=true;
  BufferGeometry buffers{};
  static WorkSpec parse(const std::string&, int rank, int ranks);
};
std::string immutable_xml_copy(const WorkSpec&, const std::string& directory);
void atomic_evidence(const std::string& path, const Json& value);
}
#endif
