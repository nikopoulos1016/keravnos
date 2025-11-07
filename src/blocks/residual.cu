#include "blocks/residual.cuh"


__global__ void _residual_add(
    __half *output,
    const __half *input,
    const __half *skip,
    const int seq_len,
    const int n_dims
) {    
    int b_idx_ = blockIdx.x;
    int t_idx_ = blockIdx.y;
    int dx_idx_ = threadIdx.x;
    int dy_idx_ = threadIdx.y;
    int d_idx_ = dy_idx_ * blockDim.x + dx_idx_;
    
    int idx_ = (b_idx_ * seq_len + t_idx_) * n_dims + d_idx_;

    if (d_idx_ < n_dims) {
        output[idx_] = __hadd(input[idx_], skip[idx_]);
    }
}


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
) {
    int wd_ = (num_dims + TILE_SIZE - 1) / TILE_SIZE;

    dim3 blocks_(batch_size, sequence_length);
    dim3 threads_(TILE_SIZE, wd_);

    _residual_add<<<blocks_, threads_>>>(output, input, skip, sequence_length, num_dims);
}

#ifdef __cplusplus
}
#endif




