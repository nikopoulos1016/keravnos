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
    
    int n_lanes_ = blockDim.x;
    int n_warps_ = blockDim.y;
    int stride_ = n_dims;
    int ln_param_size_ = 4 * n_dims;
    int valid_ = (d_idx_ < n_dims);

    int offset_x_ = (b_idx_ * seq_len + t_idx_) * n_dims;
    int offset_gamma_ = layer_idx * ln_param_size_;
    int offset_beta_  = layer_idx * ln_param_size_ + n_dims;

    extern __shared__ float sm_buf_[];
    float *warp_buf_ = &sm_buf_[0];
    
    // ----------------- load input ----------------- //
    
    float val_ = valid_ ? __half2float(input_embed[offset_x_ + d_idx_]) : 0.0f;

    // ----------------- compute mean ----------------- //

    float warpwise_sum_ = val_;
    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
        float other_ = __shfl_down_sync(UINT32_MAX, warpwise_sum_, offset_);
        warpwise_sum_ += other_;
    }
    if (l_idx_ == 0) {
        int warp_start_ = w_idx_ * block_size_;
        int warp_width_ = std::min(block_size_, n_dims - warp_start_);
        warp_buf_[w_idx_] = warpwise_sum_ * (1.0f / warp_width_);
    }
    __syncthreads();

    float mean_ = 0.0f;
    if (w_idx_ == 0 && (l_idx_ < n_lanes_)) {
        float m_ = warp_buf_[l_idx_];

        for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
            float other_ = __shfl_down_sync(UINT32_MAX, m_, offset_);
            m_ += other_;
        }
        if (l_idx_ == 0) {
            warp_buf_[0] = m_ / n_dims;
        }
    }
    __syncthreads();
    mean_ = warp_buf_[0];

    // ----------------- compute variance ----------------- //

    float diff_ = val_ - mean_;
    float warpwise_var_ = powf(diff_, 2.0f);

    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
        float other_ = __shfl_down_sync(UINT32_MAX, warpwise_var_, offset_);
        warpwise_var_ += other_;
    }
    if (l_idx_ == 0) {
        int warp_start_ = w_idx_ * block_size_;
        int warp_width_ = std::min(block_size_, n_dims - warp_start_);
        warp_buf_[w_idx_] = warpwise_var_ * (1.0f / warp_width_);
    }
    __syncthreads();

    float var_ = 0.0f;
    if (w_idx_ == 0 && (l_idx_ < n_lanes_)) {
        float v_ = warp_buf_[l_idx_];
        
        for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
            float other_ = __shfl_down_sync(UINT32_MAX, v_, offset_);
            v_ += other_;
        }
        if (l_idx_ == 0) {
            warp_buf_[0] = v_ / n_dims;
        }
    }
    __syncthreads();

    var_ = warp_buf_[0];
    float std_ = rsqrtf(var_ + epsilon);

    // ----------------- apply normalisation ----------------- //
    if (valid_) {
        float gamma_ = __half2float(ln_params[offset_gamma_ + d_idx_]);
        float beta_ = __half2float(ln_params[offset_beta_ + d_idx_]);

        float norm_ = gamma_ * ((val_ - mean_) * std_) + beta_;
        input_embed[offset_x_ + d_idx_] = __float2half(norm_); 
    }
}


#ifdef __cplusplus
extern "C" {
#endif

void layer_norm(
    __half *input_embed,
    const __half *ln_params,
    const int layer_index,
    const int batch_size,
    const int sequence_length,
    const int num_dims,
    const float epsilon
) {
    std::size_t n_warps_ = (num_dims + TILE_SIZE - 1) / TILE_SIZE;

    dim3 blocks_(sequence_length, batch_size);
    dim3 threads_(TILE_SIZE, n_warps_);
    std::size_t shared_mem_ = n_warps_ * sizeof(float);

    _layer_norm<<<blocks_, threads_, shared_mem_>>>(
        input_embed, ln_params, 
        layer_index, sequence_length, num_dims, 
        epsilon
    );
}

#ifdef __cplusplus
}
#endif