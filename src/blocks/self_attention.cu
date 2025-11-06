#include "blocks/self_attention.cuh"


__global__ void _selfattn_add_qkv_projection_bias(
  __half *out_qkv_matrix,
  const __half *qkv_proj_bias,
  int total_rows,
  int hidden_size
) {
    int row_ = blockIdx.y * blockDim.y + threadIdx.y;
    int col_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_ >= total_rows || col_ >= hidden_size) return;

    int idx_ = row_ * hidden_size + col_;
    out_qkv_matrix[idx_] = __hadd(out_qkv_matrix[idx_], qkv_proj_bias[col_]);
}

__global__ void _selfattn_compute_attention(
    __half *attn_scores,
    __half *context_layer,
    const __half *dropout_mask,
    const __half *q_matrix,
    const __half *k_matrix,
    const __half *v_matrix,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int head_dim,
    const float scale,
    const bool use_causal_mask,
    const float dropout
) {
    int block_size_ = TILE_SIZE;

    int row_idx_ = threadIdx.y;
    int col_idx_ = threadIdx.x;

    int bh_idx_ = blockIdx.z;
    int b_idx_ = bh_idx_ / n_heads;
    int h_idx_ = bh_idx_ % n_heads;

    int tq_idx_ = blockIdx.y * block_size_ + row_idx_;
    int tk_idx_ = blockIdx.x * block_size_ + col_idx_;

    extern __shared__ float sm_buf_[];
    float *attn_tile_ = &sm_buf_[0]; // [block_size_, block_size_]
    float *v_tile_ = &sm_buf_[block_size_ * block_size_]; // [block_size_, head_dim]

    if (tq_idx_ >= seq_len || tk_idx_ >= seq_len) return;

    attn_tile_[row_idx_ * block_size_ + col_idx_] = 0.0f;
    __syncthreads();

    // ---------- compute attention score ---------- //

    float acc_ = 0.0f;

    // load tiles for Q and K matrices, compute dot product
    for (int hdi_ = 0; hdi_ < head_dim; ++hdi_) {
        float q_val_ = __half2float(q_matrix[((b_idx_ * n_heads + h_idx_) * seq_len + tq_idx_) * head_dim + hdi_]);
        float k_val_ = __half2float(k_matrix[((b_idx_ * n_heads + h_idx_) * seq_len + tk_idx_) * head_dim + hdi_]);
    
        acc_ += q_val_ * k_val_; 
    }

    // scale by sqrtf(head_dim)
    acc_ *= scale;    
    
    // apply causal mask
    if (use_causal_mask && (col_idx_ > row_idx_)) {
        acc_ = FP32_MASK_VAL;
    }

    attn_tile_[row_idx_ * block_size_ + col_idx_] = acc_;
    __syncthreads();
    
    // ---------- apply softmax + dropout ---------- //

    // max reduction
    float attn_val_ = attn_tile_[row_idx_ * block_size_ + col_idx_];
    float row_max_ = attn_val_;
    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
	    float other_ = __shfl_xor_sync(UINT32_MAX, row_max_, offset_);
	    row_max_ = fmaxf(row_max_, other_);
    }

    // compute exp (val - max)
    attn_val_ = __expf(attn_val_ - row_max_);

    // sum reduction
    float row_sum_ = attn_val_;
    for (int offset_ = block_size_ >> 1; offset_ > 0; offset_ >>= 1) {
	    float other_ = __shfl_xor_sync(UINT32_MAX, row_sum_, offset_);
	    row_sum_ += other_;
    }

    // apply normalisation by sum of row
    attn_val_ /= row_sum_;

    // apply dropout
    if (dropout > 0.0f) {
        int mask_idx_ = ((b_idx_ * n_heads + h_idx_) * seq_len + row_idx_) * seq_len + col_idx_;
        
        float keep_prob_ = 1.0f - dropout;
        float mask_ = __half2float(dropout_mask[mask_idx_]);

        attn_val_ *= mask_;
        attn_val_ /= keep_prob_;
    }

    attn_tile_[row_idx_ * block_size_ + col_idx_] = attn_val_;
    attn_scores[((b_idx_ * n_heads + h_idx_) * seq_len + row_idx_) * seq_len + col_idx_] = __float2half(attn_val_);
    __syncthreads();

    // ---------- compute weighted attention ---------- //
 
    // load tiles for V matrix
    for (int hdi_ = 0; hdi_ < head_dim; ++hdi_) {
        v_tile_[row_idx_ * head_dim + hdi_] = __half2float(v_matrix[(((b_idx_ * n_heads + h_idx_) * seq_len + tk_idx_) * head_dim) + hdi_]);
    }
    __syncthreads();

    // mat-mul attention weights with V matrix
    float ctx_val_ = 0.0f;
    for (int idx_ = 0; idx_ < block_size_; ++idx_) {
        ctx_val_ += attn_tile_[row_idx_ * block_size_ + idx_] * v_tile_[idx_ * head_dim + col_idx_];
    }

    context_layer[((b_idx_ * n_heads + h_idx_) * seq_len + row_idx_) * head_dim + col_idx_] = __float2half(ctx_val_);
    __syncthreads();
}

__global__ void _selfattn_add_output_projection_bias(
    __half *out_proj,
    const __half *proj_bias,
    const int n_dims,
    const int rows
) {
    int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_ >= rows * n_dims) return;

    int dim_ = idx_ % n_dims;
    out_proj[idx_] = __hadd(out_proj[idx_], proj_bias[dim_]);
}


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
) {
    int m_ = batch_size * sequence_length;
    int n_ = num_dims * 3;
    int k_ = num_dims;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_, m_, k_,
        &alpha_,
        qkv_proj, CUDA_R_16F, k_,
        input_embed, CUDA_R_16F, k_,
        &beta_,
        out_qkv_matrix, CUDA_R_16F, n_,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP 
    );

    if (use_bias) {
        int total_rows_ = batch_size * sequence_length;
        int hidden_size_ = num_dims * 3;
        
        dim3 blocks_(TILE_SIZE, NUM_THREADS / TILE_SIZE);
        dim3 grids_(
            (hidden_size_ + blocks_.x - 1) / blocks_.x,
            (total_rows_ + blocks_.y - 1) / blocks_.y
        );
        _selfattn_add_qkv_projection_bias<<<grids_, blocks_>>>(out_qkv_matrix, qkv_proj_bias, total_rows_, hidden_size_);
    }
}

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
) {
    dim3 blocks_(
        (sequence_length + TILE_SIZE - 1) / TILE_SIZE,  // keys
        (sequence_length + TILE_SIZE - 1) / TILE_SIZE,  // queries
        batch_size * num_heads
    );
    dim3 threads_(TILE_SIZE, TILE_SIZE);
    std::size_t shared_mem_ = ((TILE_SIZE * head_dim) + (TILE_SIZE * TILE_SIZE)) * sizeof(float);

    _selfattn_compute_attention<<<blocks_, threads_, shared_mem_>>>(
        attn_scores, context_layer,
        dropout_mask, q_matrix, k_matrix, v_matrix,
        batch_size, sequence_length, num_heads, head_dim, 
        scale, use_causal_mask, dropout
    );
}

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
) {
    int m_ = n_dims;
    int n_ = batch_size * seq_len;
    int k_ = n_heads * head_dim;
    float alpha_ = 1.0f;
    float beta_  = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        m_, n_, k_,
        &alpha_,
        output_proj, CUDA_R_16F, m_,
        context, CUDA_R_16F, k_,
        &beta_,
        output, CUDA_R_16F, m_,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (use_bias) {
        int total_ = batch_size * seq_len * n_dims;
        int blocks_ = (total_ + NUM_THREADS - 1) / NUM_THREADS;

        _selfattn_add_output_projection_bias<<<blocks_, NUM_THREADS>>>(output, output_proj_bias, n_dims, batch_size * seq_len);
    }
}

#ifdef __cplusplus
}
#endif 
