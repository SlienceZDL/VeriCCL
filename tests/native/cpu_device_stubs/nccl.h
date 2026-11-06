#ifndef VERICCL_TEST_NCCL_H
#define VERICCL_TEST_NCCL_H
#include "cuda_runtime.h"
struct CpuComm {};
using ncclComm_t=CpuComm*;
struct ncclUniqueId { char internal[128]; };
enum ncclResult_t { ncclSuccess };
enum ncclDataType_t { ncclInt8,ncclUint8,ncclInt32,ncclUint32,ncclInt64,ncclUint64,
                     ncclFloat16,ncclFloat32,ncclFloat64,ncclBfloat16 };
enum ncclRedOp_t { ncclSum,ncclProd,ncclMin,ncclMax };
const char* ncclGetErrorString(ncclResult_t);
ncclResult_t ncclGetUniqueId(ncclUniqueId*);
ncclResult_t ncclCommGetAsyncError(ncclComm_t,ncclResult_t*);
ncclResult_t ncclCommDestroy(ncclComm_t);
ncclResult_t ncclCommAbort(ncclComm_t);
ncclResult_t ncclCustomCollective(const void*,void*,size_t,ncclDataType_t,ncclRedOp_t,int,ncclComm_t,cudaStream_t);
#endif
