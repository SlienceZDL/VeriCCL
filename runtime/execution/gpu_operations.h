#ifndef VERICCL_GPU_OPERATIONS_H
#define VERICCL_GPU_OPERATIONS_H
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>
void vericcl_scale_output(void*, size_t elements, ncclDataType_t, int divisor, cudaStream_t);
void vericcl_sample_timer(unsigned long long*, cudaStream_t);
void vericcl_fill_input(void*, size_t elements, ncclDataType_t, int rank, cudaStream_t);
unsigned long long vericcl_check_output(const void*, size_t elements, ncclDataType_t,
    int collective, int reduction, int rank, int ranks, int root, size_t input_elements,
    double atol, double rtol, cudaStream_t);
#endif
