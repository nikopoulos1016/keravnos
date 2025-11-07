#pragma once

#include "global.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void residual_add(    
    __half *output,
    const __half *input,
    const __half *skip,
    const int batch_size,
    const int sequence_length,
    const int num_dims
);

#ifdef __cplusplus
}
#endif



