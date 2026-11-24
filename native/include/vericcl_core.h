/* C ABI of the VeriCCL native core (loaded by vericcl/native via ctypes).
 *
 * Every entry point takes one flat input buffer and returns one malloc'ed
 * output buffer (release it with vc_free). Formats are defined next to the
 * Python encoders in vericcl/native/. Return codes: 0 ok, 1 semantic error
 * (message equals the Python reference), 2 malformed input or internal
 * failure, 3 allocation failure.
 */
#ifndef VERICCL_CORE_H
#define VERICCL_CORE_H

#include <stddef.h>
#include <stdint.h>

#define VERICCL_CORE_ABI_VERSION 1

#if defined(__GNUC__)
#define VC_API __attribute__((visibility("default")))
#else
#define VC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

VC_API int vc_abi_version(void);
VC_API int vc_predict(const uint8_t* input, size_t input_size, uint8_t** output, size_t* output_size);
VC_API void vc_free(void* pointer);

#ifdef __cplusplus
}
#endif

#endif
