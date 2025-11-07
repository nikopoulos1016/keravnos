#pragma once

#include "global.cuh"


#ifdef __cplusplus
extern "C" {
#endif

void embedding_input_vector(
    __half *input_embed,
    const int *token_ids,
    const __half *token_embed,
    const __half *pos_embed,
    const int batch_size,
    const int vocab_size,
    const int num_dims,
    const int sequence_length
);

#ifdef __cplusplus
}
#endif


