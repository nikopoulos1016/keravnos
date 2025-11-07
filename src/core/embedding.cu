#include "core/transformer.cuh"
#include "core/embedding.cuh"

__global__ void _embedding_input_vector(
    __half *out,
    const int *token_ids,
    const __half *token_embed,
    const __half *pos_embed,
    const int vocab_size,
    const int n_dims,
    const int seq_len
) {
    int b_ = blockIdx.x;
    int t_ = blockIdx.y;
    int d_ = threadIdx.x;

    if (d_ >= n_dims) return;

    int token_id_ = token_ids[b_ * seq_len + t_];
    if (token_id_ >= vocab_size) return;

    int tok_idx_ = token_id_ * n_dims + d_;
    int pos_idx_ = t_ * n_dims + d_;
    int out_idx_ = (b_ * seq_len + t_) * n_dims + d_;
    
    __half tok_ = token_embed[tok_idx_];
    __half pos_ = pos_embed[pos_idx_];
        
    out[out_idx_] = __hadd(tok_, pos_);
}

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
) {
    dim3 blocks_(batch_size, sequence_length);
    dim3 threads_(num_dims);

    _embedding_input_vector<<<blocks_, threads_>>>(
        input_embed, token_ids, token_embed, pos_embed,
        vocab_size, num_dims, sequence_length
    );
}

#ifdef __cplusplus
}
#endif

