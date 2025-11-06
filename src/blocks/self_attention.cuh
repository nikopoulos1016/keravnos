#pragma once

#include "global.cuh"


#ifdef __cplusplus
extern "C" {
#endif

void selfattn_compute_qkv_projection(
    __half *out_qkv_matrix,
    cublasHandle_t &handle,
    const __half *input_embed,
    const __half *qkv_proj,
    const __half *qkv_proj_bias,
    const int num_dims,
    const int batch_size,
    const int sequence_length,
    const bool use_bias
);

void selfattn_compute_attention(
    __half *attn_scores,
    __half *context_layer,
    const __half *dropout_mask,
    const __half *q_matrix,
    const __half *k_matrix,
    const __half *v_matrix,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_dim,
    const float scale,
    const bool use_causal_mask,
    const float dropout
);

void selfattn_compute_output_projection(
    __half *output,
    cublasHandle_t handle,
    const __half *context,
    const __half *output_proj,
    const __half *output_proj_bias,
    const int batch_size,
    const int seq_len,
    const int n_dims,
    const int head_dim,
    const int n_heads,
    const bool use_bias
);

#ifdef __cplusplus
}
#endif 

