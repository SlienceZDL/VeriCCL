#ifndef VERICCL_PROBE_H
#define VERICCL_PROBE_H
#include <stdint.h>
#include <stddef.h>
#include <nccl.h>

/* One writer per interpreter TB/step, used only by the owned probe process.
 * All fields are reset after warmup and read only after CUDA completion. */
typedef struct VericclProbeCounter {
  uint64_t service_ns;
  uint64_t dependency_wait_ns;
  uint64_t samples;
  uint64_t first_start_ns;
  uint64_t last_end_ns;
  uint64_t invalid_samples;
  uint32_t kind;
  uint32_t channel;
} VericclProbeCounter;

#ifdef __cplusplus
extern "C" {
#endif
/* Synchronizes the calling process's current device before changing ownership.
 * The caller allocates/frees the device table. Passing NULL/0 detaches it. */
ncclResult_t vericclProbeAttach(ncclComm_t comm, VericclProbeCounter* device_table, size_t table_entries);
#ifdef __cplusplus
}
#endif
#endif
