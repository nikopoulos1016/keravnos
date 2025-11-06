#include "neural_network/embedding.cuh"
#include "transformer/transformer.cuh"

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

void embedding_input_vector(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    } 
    
    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *input_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_input_embed);
    int *token_ids_ = reinterpret_cast<int *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_ids);
    __half *token_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_embed);
    __half *pos_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_pos_embed);

    dim3 blocks_(header_._batch_size, header_._sequence_length);
    dim3 threads_(header_._num_dims);

    _embedding_input_vector<<<blocks_, threads_>>>(
        input_embed_, token_ids_, token_embed_, pos_embed_,
        header_._vocab_size, header_._num_dims, header_._sequence_length
    );
}

#ifdef __cplusplus
}
#endif

