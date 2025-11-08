#include "utils/math.cuh"
#include "blocks/feed_forward.cuh"


__global__ void _ffn_add_bias(
    __half *output,
    const __half *input,
    const __half *bias,
    const int total_rows,
    const int hidden_size
) {
    int row_ = blockIdx.y * blockDim.y + threadIdx.y;
    int col_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_ >= total_rows || col_ >= hidden_size) return;

    int idx_ = row_ * hidden_size + col_;
    output[idx_] = __hadd(input[idx_], bias[col_]);
}

__global__ void _ffn_apply_gelu(
    __half *output,
    const __half *input,
    const int total_rows,
    const int hidden_size
) {
    int row_ = blockIdx.y * blockDim.y + threadIdx.y;
    int col_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_ >= total_rows || col_ >= hidden_size) return;

    int idx_ = row_ * hidden_size + col_;
    output[idx_] = __float2half(utils_math_gelu(__half2float(input[idx_])));
}


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
) {
    float alpha_ = 1.0f;
    float beta_  = 0.0f;

    int offset_w1_ = 0;
    int offset_w2_ = num_dims * (num_dims * ff_mult);
    int offset_b1_ = 0;
    int offset_b2_ = num_dims * ff_mult;

    int total_rows_ = batch_size * sequence_length;
    int hidden_size_ = num_dims * ff_mult;
    int tile_y_ = (hidden_size_ + TILE_SIZE - 1) / TILE_SIZE;

    // x_norm2 @ W1
    {
        int m_ = batch_size * sequence_length;
        int n_ = 4 * num_dims;
        int k_ = num_dims;
        
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_, m_, k_,
            &alpha_,
            &weights[offset_w1_], CUDA_R_16F, n_,
            input, CUDA_R_16F, k_,
            &beta_,
            hidden, CUDA_R_16F, n_,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        );

        // + b1
        {
            dim3 blocks_(
                (hidden_size_ + TILE_SIZE - 1) / TILE_SIZE,
                (total_rows_ + tile_y_ - 1) / tile_y_
            );
            dim3 threads_(TILE_SIZE, tile_y_);
            _ffn_add_bias<<<blocks_, threads_>>>(hidden, hidden, &biases[offset_b1_], total_rows_, hidden_size_);
        }
    }

    // apply GELU
    {
        dim3 blocks_(
            (hidden_size_ + TILE_SIZE - 1) / TILE_SIZE,
            (total_rows_ + tile_y_ - 1) / tile_y_
        );
        dim3 threads_(TILE_SIZE, tile_y_);
        _ffn_apply_gelu<<<blocks_, threads_>>>(hidden, hidden, total_rows_, hidden_size_);
    }

    hidden_size_ = num_dims;
    tile_y_ = (hidden_size_ + TILE_SIZE - 1) / TILE_SIZE;

    // ff_hidden @ W2
    {
        int m_ = batch_size * sequence_length;
        int n_ = num_dims;
        int k_ = 4 * num_dims;
        
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            n_, m_, k_,
            &alpha_,
            &weights[offset_w2_], CUDA_R_16F, n_,
            hidden, CUDA_R_16F, k_,
            &beta_,
            output, CUDA_R_16F, n_,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
        );

        // + b2
        {
            dim3 blocks_(
                (hidden_size_ + TILE_SIZE - 1) / TILE_SIZE,
                (total_rows_ + tile_y_ - 1) / tile_y_
            );
            dim3 threads_(TILE_SIZE, tile_y_);
            _ffn_add_bias<<<blocks_, threads_>>>(output, output, &biases[offset_b2_], total_rows_, hidden_size_);
        }
    }
}

#ifdef __cplusplus
}
#endif
