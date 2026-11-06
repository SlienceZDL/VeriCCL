#ifndef VERICCL_PUBLIC_H
#define VERICCL_PUBLIC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define VERICCL_ABI_VERSION 2U
typedef struct VericclContext VericclContext;
typedef struct VericclWorkload VericclWorkload;
typedef enum VericclCode {
  VERICCL_OK = 0, VERICCL_INVALID = 1, VERICCL_BUSY = 2,
  VERICCL_TIMEOUT = 3, VERICCL_UNRESOLVED = 4, VERICCL_FAILED = 5,
  VERICCL_CLOSED = 6
} VericclCode;
typedef struct VericclConfig {
  uint32_t struct_size, abi_version;
  int32_t rank, rank_count, device;
  uint16_t coordinator_port;
  uint16_t reserved;
  const char *group_id, *session_id, *coordinator_address, *bind_address;
  const char *evidence_directory;
  double request_prepare_timeout_s, execution_attempt_timeout_s;
  double background_tuning_task_timeout_s; /* Reserved: must be zero. */
  double version_prepare_timeout_s, control_phase_timeout_s;
  double recovery_timeout_s; /* Reserved: must be zero. */
  double close_cleanup_timeout_s;
  /* Includes communication, scratch and detailed trace storage. Zero
   * selects the currently free-device-memory bound; caller buffers also count. */
  uint64_t device_memory_budget_bytes;
  uint32_t max_retries; /* Reserved: must be zero. */
  uint32_t fault_inject_attempts; /* Reserved: must be zero. */
  uint64_t fault_inject_request; /* Reserved: must be zero. */
} VericclConfig;
typedef struct VericclBuffer {
  uint32_t struct_size, abi_version;
  void *data;
  uint64_t bytes;
} VericclBuffer;
typedef struct VericclResult {
  uint32_t struct_size, abi_version;
  int32_t code;
  uint32_t gpu_completed, output_valid, buffers_reusable, resources_released;
  uint64_t request_id, attempt, logical_successes;
  double device_time_us, preparation_time_us, control_time_us, wall_time_us;
  double backup_time_us, recovery_time_us; /* Reserved: always zero. */
  char version_id[65], message[512];
} VericclResult;
typedef struct VericclStatus {
  uint32_t struct_size, abi_version;
  int32_t code;
  uint32_t busy, uncertain, closed, caller_buffers_retained;
  uint64_t request_id, attempt, logical_successes, retained_versions;
  char version_id[65], message[512];
} VericclStatus;

/* Calls are collective over the fixed rank membership, except status. A
 * context permits one active API operation. Strings are copied on entry.
 * Each finite deadline includes the operation's child phases. */
VericclCode vericcl_init(const VericclConfig*, VericclContext**);
/* Accepts a manifest exported by VeriCCL after complete static checking.
 * It verifies XML identity and prepares one communicator. A context binds to
 * its first manifest for its lifetime; a different manifest requires a new
 * context. Repeating the identical preparation is idempotent while ready. */
VericclCode vericcl_prepare(VericclContext*, const char* workload_json, VericclWorkload**);
/* Release an unused prepared workload. The context keeps its fixed identity. */
VericclCode vericcl_release(VericclWorkload*, VericclResult*);
/* The stream is a cudaStream_t cast to void*. Input/output remain owned by
 * the caller. On UNRESOLVED, keep both buffers and stream alive until status
 * or a later close explicitly acknowledges their release. Enqueue is never
 * reported as logical success. Callers synchronize their producer work on
 * this stream and do not mutate any retained input during the request. */
VericclCode vericcl_execute(VericclWorkload*, const VericclBuffer* input,
    const VericclBuffer* output, void* stream, VericclResult*);
VericclCode vericcl_status(VericclContext*, VericclStatus*);
/* A successful close invalidates workload handles. An unresolved close
 * keeps the context valid for status/close; it never frees active buffers. */
VericclCode vericcl_close(VericclContext*, VericclResult*);
/* Frees only an already-closed identity handle and nulls the caller pointer. */
VericclCode vericcl_destroy(VericclContext**);
/* Detailed trace control uses the same managed worker. */
VericclCode vericcl_trace_control(VericclWorkload*, int action);
const char* vericcl_error_string(VericclCode);
#ifdef __cplusplus
}
#endif
#endif
