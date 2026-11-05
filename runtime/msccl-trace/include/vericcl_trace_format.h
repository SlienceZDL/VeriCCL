#ifndef VERICCL_TRACE_FORMAT_H_
#define VERICCL_TRACE_FORMAT_H_

#include <stddef.h>
#include <stdint.h>

#define VERICCL_TRACE_MAGIC 0x5643434cU
#define VERICCL_TRACE_VERSION 1U
#define VERICCL_TRACE_FLAG_OVERFLOW 0x1U

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint32_t record_size;
  uint32_t rank;
  uint64_t count;
  uint64_t capacity;
  uint32_t overflow;
  uint32_t reserved;
} VericclRawTraceHeader;

typedef struct {
  uint32_t rank;
  uint16_t tb_id;
  uint16_t step_index;
  uint16_t endpoint_type;
  int16_t peer;
  uint16_t channel;
  uint32_t iteration;
  uint64_t tb_reach;
  uint64_t dependency_done;
  uint64_t transfer_start;
  uint64_t transfer_end;
  uint32_t flags;
  uint32_t reserved;
} VericclRawStepTraceRecord;

#if defined(__cplusplus)
static_assert(sizeof(VericclRawTraceHeader) == 40,
              "VericclRawTraceHeader layout changed");
static_assert(sizeof(VericclRawStepTraceRecord) == 64,
              "VericclRawStepTraceRecord layout changed");
static_assert(offsetof(VericclRawStepTraceRecord, tb_reach) == 24,
              "VericclRawStepTraceRecord alignment changed");
#else
_Static_assert(sizeof(VericclRawTraceHeader) == 40,
               "VericclRawTraceHeader layout changed");
_Static_assert(sizeof(VericclRawStepTraceRecord) == 64,
               "VericclRawStepTraceRecord layout changed");
_Static_assert(offsetof(VericclRawStepTraceRecord, tb_reach) == 24,
               "VericclRawStepTraceRecord alignment changed");
#endif

#endif
