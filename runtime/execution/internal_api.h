#ifndef VERICCL_INTERNAL_API_H
#define VERICCL_INTERNAL_API_H
#include "vericcl.h"
#include "contract.h"
/* Driver-only helpers. They retain the same serialized worker and parent
 * deadlines as the public API; none exposes the underlying communicator. */
namespace vericcl {
VericclCode driver_close(VericclContext*, VericclResult*, double remaining_s);
VericclCode driver_exchange(VericclContext*, const std::string& phase, const Json& local, Json* all, double timeout_s);
VericclCode clock_samples(VericclWorkload*, void* stream, std::string* output, double remaining_s=-1);
}
#endif
