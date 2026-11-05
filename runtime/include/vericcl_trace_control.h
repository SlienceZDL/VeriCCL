#ifndef VERICCL_TRACE_CONTROL_H_
#define VERICCL_TRACE_CONTROL_H_
#include <stddef.h>
#include <nccl.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Call with the communicator's device current and no concurrent enqueue.
 * Allocate trace storage using VERICCL_TRACE_* before communicator creation.
 * Pause before warmup; Reset after warmup; enqueue one collective; Finish.
 * All calls synchronize that device. Finish disables further recording. */
ncclResult_t vericclTracePause(ncclComm_t comm);
ncclResult_t vericclTraceReset(ncclComm_t comm, size_t expected_records);
ncclResult_t vericclTraceFinish(ncclComm_t comm);
#ifdef __cplusplus
}
#endif
#endif
