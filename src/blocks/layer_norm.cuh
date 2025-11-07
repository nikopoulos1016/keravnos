#pragma once

#include "global.cuh"


#ifdef __cplusplus
extern "C" {
#endif

void layer_norm(    
    __half *output,
    const __half *input,
    const __half *ln_params,
    const int layer_index,
    const int batch_size,
    const int sequence_length,
    const int num_dims,
    const float epsilon = EPSILON
);

#ifdef __cplusplus
}
#endif

