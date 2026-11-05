#ifndef VERICCL_PLAN_H
#define VERICCL_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include <nccl.h>
#ifdef __cplusplus
extern "C" {
#endif
#define VERICCL_NCCL_PLAN_ABI 1U

typedef struct VericclNcclPlanOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* xml_path;
  uint64_t trace_records;
  const char* trace_prefix;
} VericclNcclPlanOptions;

/* Synchronous explicit-plan initialization. Invoke from a managed worker and
 * outside ncclGroupStart/End. The options are borrowed only until this returns.
 * With trace_records=0 this communicator ignores global trace environment. */
ncclResult_t vericclNcclCommInitRank(ncclComm_t* comm, int ranks,
    ncclUniqueId unique_id, int rank, const VericclNcclPlanOptions* options);
/* Check the actual CustomCollective kernel and prepared scratch allocation,
 * without enqueueing a collective. Returns the selected resident-TB limit. */
ncclResult_t vericclNcclPlanAdmission(ncclComm_t comm, size_t runtime_bytes,
    ncclDataType_t datatype, ncclRedOp_t op, int* resident_tb_limit);
/* Signal device/proxy termination without freeing communicator resources.
 * The caller must confirm its GPU stream stopped before ncclCommAbort. */
ncclResult_t vericclNcclAbortSignal(ncclComm_t comm);
#ifdef __cplusplus
}
#endif
#endif
