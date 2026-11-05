#ifndef VERICCL_INTERNAL_API_H
#define VERICCL_INTERNAL_API_H
#include "vericcl.h"
#include "contract.h"
/* Driver-only helpers. They retain the same serialized worker and parent
 * deadlines as the public API; none exposes the underlying communicator. */
namespace vericcl {
VericclCode driver_close(VericclContext*, VericclResult*, double remaining_s);
VericclCode driver_prepare(VericclContext*, const char*, VericclWorkload**, double remaining_s);
VericclCode driver_execute(VericclWorkload*, const VericclBuffer*, const VericclBuffer*, void*, VericclResult*, double remaining_s);
VericclCode driver_release(VericclWorkload*, VericclResult*, double remaining_s);
VericclCode driver_trace_control(VericclWorkload*, int action, double remaining_s);
VericclCode driver_light_control(VericclWorkload*, int enabled, double remaining_s);
VericclCode driver_exchange(VericclContext*, const std::string& phase, const Json& local, Json* all, double timeout_s);
VericclCode clock_samples(VericclWorkload*, void* stream, std::string* output, double remaining_s=-1);
}
#endif
