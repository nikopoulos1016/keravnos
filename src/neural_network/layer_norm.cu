#include "transformer/transformer.cuh"
#include "neural_network/layer_norm.cuh"


__global__ void _layer_norm(
    __half *input_embed,
    const __half *ln_params,
    const int layer_idx,
    const int seq_len,
    const int n_dims,
    const float epsilon
) {
    int block_size_ = TILE_SIZE;
    
    int b_idx_ = blockIdx.y;
    int t_idx_ = blockIdx.x;
    int l_idx_ = threadIdx.x;
    int w_idx_ = threadIdx.y;
    int d_idx_ = w_idx_ * block_size_ + l_idx_;
    
    int n_warps_ = blockDim.x;
    int stride_ = n_dims;
    int ln_param_size_ = 4 * n_dims;

    int offset_x_ = (b_idx_ * seq_len + t_idx_) * n_dims;
    int offset_gamma_ = layer_idx * ln_param_size_;
    int offset_beta_  = layer_idx * ln_param_size_ + n_dims;

    float val_ = 0.0f;
    if (d_idx_ < n_dims) {
        val_ = __half2float(input_embed[offset_x_ + d_idx_]);
    }

    extern __shared__ float sm_buf_[];
    float *warp_means_ = &sm_buf_[0]; // [block_size_]

    // ----------------- compute mean ----------------- //

    float warpwise_sum_ = val_;
    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
        float other_ = __shfl_xor_sync(UINT32_MAX, warpwise_sum_, offset_);
        warpwise_sum_ += other_;
    }
    if (l_idx_ == 0) {
        warp_means_[w_idx_] = warpwise_sum_;
    }
    __syncthreads();

    float mean_ = 0.0f;
    if (w_idx_ == 0 && (l_idx_ < n_warps_)) {
        float m_ = warp_means_[l_idx_];

        for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
            float other_ = __shfl_xor_sync(UINT32_MAX, m_, offset_);
            m_ += other_;
        }
        if (l_idx_ == 0) {
            warp_means_[0] = m_ / n_dims;
        }
    }
    __syncthreads();
    mean_ = warp_means_[0];

    // ----------------- compute variance ----------------- //

    float diff_ = val_ - mean_;
    float warpwise_var_ = powf(diff_, 2.0f);

    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
        float other_ = __shfl_xor_sync(UINT32_MAX, warpwise_var_, offset_);
        warpwise_var_ += other_;
    }
    if (l_idx_ == 0) {
        warp_means_[w_idx_] = warpwise_var_;
    }
    __syncthreads();

    float var_ = 0.0f;
    if (w_idx_ == 0 && (l_idx_ < n_warps_)) {
        float v_ = warp_means_[l_idx_];
        
        for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
            float other_ = __shfl_xor_sync(UINT32_MAX, v_, offset_);
            v_ += other_;
        }
        if (l_idx_ == 0) {
            warp_means_[0] = v_ / n_dims;
        }
    }
    __syncthreads();

    var_ = warp_means_[0];
    float std_ = rsqrtf(var_ + epsilon);

    // ----------------- apply normalisation ----------------- //
    if (d_idx_ < n_dims) {
        float gamma_ = __half2float(ln_params[offset_gamma_ + d_idx_]);
        float beta_ = __half2float(ln_params[offset_beta_ + d_idx_]);

        float norm_ = gamma_ * ((val_ - mean_) * std_) + beta_;
        input_embed[offset_x_ + d_idx_] = __float2half(norm_); 
    }
}


#ifdef __cplusplus
extern "C" {
#endif

void layer_norm(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    } 
    
    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_input_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_input_embed);
    __half *dvc_ln_params_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_ln_params);

    dim3 blocks_(header_._sequence_length, header_._batch_size);
    dim3 threads_(TILE_SIZE, header_._num_dims / TILE_SIZE);

    _layer_norm<<<blocks_, threads_>>>(
        dvc_input_embed_, 
        dvc_ln_params_, 
        header_._current_layer_index, 
        header_._sequence_length, 
        header_._num_dims,
        EPSILON
    );
}

#ifdef __cplusplus
}
#endif