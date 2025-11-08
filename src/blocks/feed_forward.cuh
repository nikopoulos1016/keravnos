#pragma once

#include "global.cuh"


#ifdef __cplusplus
extern "C" {
#endif

void feed_forward(    
    __half *output,
    __half *hidden,
    cublasHandle_t handle,
    const __half *input,
    const __half *weights,
    const __half *biases,
    const int layer_index,
    const int batch_size,
    const int sequence_length,
    const int num_dims,
    const int ff_mult
);

#ifdef __cplusplus
}
#endif

