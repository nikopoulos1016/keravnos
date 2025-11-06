#include "memory/memory.cuh"
#include "utils/device.cuh"
#include "neural_network/embedding.cuh"
#include "blocks/self_attention.cuh"
#include "transformer/transformer.cuh"


void transformer_allocate_device_memory(
  Transformer *transformer,
  const int batch_size, 
  const int sequence_length, 
  const int vocab_size,
  const int num_dims,
  const int num_heads,
  const int num_layers,
  const int ff_mult,
  const bool verbose
) {
    if (num_dims % num_heads != 0) {
        if (verbose) KERAVNOS_PRINT_ERROR("num_dims of ", num_dims, " is not divisible by num_heads of ", num_heads);
        return;
    }

    TransformerHeader staging_header_ = {};

    // header info
    staging_header_._batch_size = batch_size;
    staging_header_._sequence_length = sequence_length;
    staging_header_._vocab_size = vocab_size;
    staging_header_._num_dims = num_dims;
    staging_header_._num_heads = num_heads;
    staging_header_._num_layers = num_layers;
    staging_header_._ff_multiplier = ff_mult;
    staging_header_._type_bytes = sizeof(__half);
    staging_header_._current_layer_index = 0;

    const int d_ff_ = num_dims * ff_mult;
    std::size_t offset_ = sizeof(TransformerHeader);

    auto advance_ = [&](std::size_t bytes) {
        offset_ = ALIGN_OFFSET(offset_, 256);
        std::size_t offset_current_ = offset_;
        offset_ += bytes;
        return offset_current_;
    };

    // global weights
    staging_header_._offset_token_embed = advance_(vocab_size * num_dims * sizeof(__half));
    staging_header_._offset_pos_embed = advance_(sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_token_ids = advance_(batch_size * sequence_length * sizeof(int));
    staging_header_._offset_input_embed = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_dropout = advance_(num_layers * batch_size * num_heads * sequence_length * sequence_length * sizeof(__half));
    staging_header_._offset_dropout_ffn = advance_(num_layers * batch_size * sequence_length * d_ff_ * sizeof(__half));

    // per-layer weights
    const std::size_t qkv_proj_size_ = 3 * num_dims * num_dims * sizeof(__half);
    const std::size_t qkv_bias_size_ = 3 * num_dims * sizeof(__half);
    const std::size_t out_proj_size_ = num_dims * num_dims * sizeof(__half);
    const std::size_t out_bias_size_ = num_dims * sizeof(__half);
    const std::size_t ffn_weight_size_ = 2 * num_dims * d_ff_ * sizeof(__half);
    const std::size_t ffn_bias_size_ = 2 * d_ff_ * sizeof(__half);
    const std::size_t ln_param_size_ = 4 * num_dims * sizeof(__half);  // 2 LayerNorms per layer
    
    staging_header_._offset_qkv_proj = advance_(num_layers * qkv_proj_size_);
    staging_header_._offset_qkv_proj_bias = advance_(num_layers * qkv_bias_size_);
    staging_header_._offset_out_proj = advance_(num_layers * out_proj_size_);
    staging_header_._offset_out_proj_bias = advance_(num_layers * out_bias_size_);
    staging_header_._offset_ffn_weights = advance_(num_layers * ffn_weight_size_);
    staging_header_._offset_ffn_bias = advance_(num_layers * ffn_bias_size_);
    staging_header_._offset_ln_params = advance_(num_layers * ln_param_size_);

    staging_header_._offset_qkv_proj_grad = advance_(num_layers * qkv_proj_size_);     
    staging_header_._offset_qkv_bias_grad = advance_(num_layers * qkv_bias_size_);     
    staging_header_._offset_out_proj_grad = advance_(num_layers * out_proj_size_);     
    staging_header_._offset_out_bias_grad = advance_(num_layers * out_bias_size_);     
    staging_header_._offset_ffn_weight_grad = advance_(num_layers * ffn_weight_size_);   
    staging_header_._offset_ffn_bias_grad = advance_(num_layers * ffn_bias_size_);     
    staging_header_._offset_ln_params_grad = advance_(num_layers * ln_param_size_);     
    staging_header_._offset_token_embed_grad = advance_(vocab_size * num_dims * sizeof(__half));
    staging_header_._offset_pos_embed_grad = advance_(sequence_length * num_dims * sizeof(__half));
    
    staging_header_._offset_qkv_proj_opt = advance_(2 * num_layers * qkv_proj_size_); 
    staging_header_._offset_qkv_bias_opt = advance_(2 * num_layers * qkv_bias_size_); 
    staging_header_._offset_out_proj_opt = advance_(2 * num_layers * out_proj_size_); 
    staging_header_._offset_out_bias_opt = advance_(2 * num_layers * out_bias_size_); 
    staging_header_._offset_ffn_weight_opt = advance_(2 * num_layers * ffn_weight_size_); 
    staging_header_._offset_ffn_bias_opt = advance_(2 * num_layers * ffn_bias_size_); 
    staging_header_._offset_ln_params_opt = advance_(2 * num_layers * ln_param_size_); 
    staging_header_._offset_token_embed_opt = advance_(2 * vocab_size * num_dims * sizeof(__half)); 
    staging_header_._offset_pos_embed_opt = advance_(2 * sequence_length * num_dims * sizeof(__half));

    // per-layer gradient temp buffers
    staging_header_._offset_grad_temp = advance_(num_layers * PER_LAYER_BUFFER_SIZE);
    
    // checkpoint fallback
    staging_header_._offset_x_in = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_x_norm = advance_(batch_size * sequence_length * num_dims * sizeof(__half));

    // shared activation
    staging_header_._offset_qkv_matrix = advance_(batch_size * sequence_length * (3 * num_dims) * sizeof(__half));
    staging_header_._offset_attn_scores = advance_(batch_size * num_heads * sequence_length * sequence_length * sizeof(__half));
    staging_header_._offset_context_layer = advance_(batch_size * num_heads * sequence_length * (num_dims / num_heads) * sizeof(__half));
    staging_header_._offset_ffn_input = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_ffn_hidden = advance_(batch_size * sequence_length * d_ff_ * sizeof(__half));
    staging_header_._offset_ffn_output = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_ln1_out = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_ln2_out = advance_(batch_size * sequence_length * num_dims * sizeof(__half));
    staging_header_._offset_output = advance_(batch_size * sequence_length * num_dims * sizeof(__half));

    staging_header_._mem_total = ALIGN_OFFSET(offset_, 256);
    staging_header_._allocated = true;

    transformer->_dvc_base = static_cast<__half *>(memory_device_allocate(staging_header_._mem_total, verbose));
    memory_copy_host_to_device(transformer->_dvc_base, &staging_header_, sizeof(TransformerHeader), verbose);

    if (verbose) {
        KERAVNOS_PRINT("transformer allocated device memory.");
        PY_PRINT("--------------------------");
        PY_PRINT("Total Bytes              : ", staging_header_._mem_total, " bytes");
        PY_PRINT("Num Layers               : ", num_layers);
        PY_PRINT("Embedding Dim            : ", num_dims);
        PY_PRINT("FFN Hidden Dim           : ", d_ff_);
        PY_PRINT("Shared QKV Matrix        : ", batch_size * sequence_length * 3 * num_dims * sizeof(__half), " bytes");
        PY_PRINT("Shared Attention Scores  : ", batch_size * num_heads * sequence_length * sequence_length * sizeof(__half), " bytes");
        PY_PRINT("Shared FFN Hidden        : ", batch_size * sequence_length * d_ff_ * sizeof(__half), " bytes");
        PY_PRINT("Shared FFN Output        : ", batch_size * sequence_length * num_dims * sizeof(__half), " bytes");
        PY_PRINT("Shared LayerNorm Output  : ", 2 * batch_size * sequence_length * num_dims * sizeof(__half), " bytes");
        PY_PRINT("--------------------------");
    }
}

void transformer_deallocate_device_memory(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    } 
    
    memory_device_deallocate(transformer->_dvc_base, verbose);
    memset(transformer, 0, sizeof(Transformer));
    if (verbose) KERAVNOS_PRINT("transformer deallocated device memory.");
}

void transformer_feed_token_ids(Transformer *transformer, const py::array_t<int, py::array::c_style | py::array::forcecast> token_ids, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    } 
    
    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    
    py::buffer_info buf_ = token_ids.request();
    if (buf_.ndim != 2) {
        if (verbose) KERAVNOS_PRINT_ERROR("token_ids must be 2D.");
        return;
    }
    if (buf_.shape[0] != header_._batch_size || buf_.shape[1] != header_._sequence_length) {
        if (verbose) KERAVNOS_PRINT_ERROR("token_ids shape mismatched: expected ", header_._batch_size, " x ", header_._sequence_length);
        return;
    }

    std::size_t count_ = buf_.size;
    int *hst_token_ids_ = static_cast<int *>(buf_.ptr);

    int *dvc_token_ids_ = reinterpret_cast<int *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_ids);
    memory_copy_host_to_device(dvc_token_ids_, hst_token_ids_, count_ * sizeof(int), verbose);
    
    if (verbose) KERAVNOS_PRINT("finished feeding token ids into transformer.");
    
    embedding_input_vector(transformer, verbose);
    if (verbose) KERAVNOS_PRINT("transformer input vector embedding completed.");
}

void transformer_generate_embedding_weights(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    
    __half *dvc_token_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_embed);
    __half *dvc_pos_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_pos_embed);
    __half *dvc_dropout_mask_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_dropout);

    const int base_ = 10000;
    utils_device_generate_random_half_dim2(dvc_token_embed_, header_._vocab_size, header_._num_dims, -1.0f, 1.0f);
    utils_device_generate_sinusodial_half(dvc_pos_embed_, header_._sequence_length, header_._num_dims, base_);
    utils_device_generate_boolean_half(dvc_dropout_mask_, header_._batch_size * header_._num_heads * header_._sequence_length * header_._sequence_length);
}

void transformer_generate_projection_weights(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj);
    __half *dvc_out_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj);

    const std::size_t fan_in_ = header_._num_dims;
    const std::size_t qkv_fan_out_ = header_._num_dims * 3;
    const std::size_t output_fan_out_ = header_._num_dims;
    
    const float qkv_limit_ = sqrt(6.0f / (fan_in_ + qkv_fan_out_));
    const float output_limit_ = sqrt(6.0f / (fan_in_ + output_fan_out_));
    utils_device_generate_random_half_dim2(dvc_qkv_proj_, fan_in_, qkv_fan_out_, -qkv_limit_, qkv_limit_);
    utils_device_generate_random_half_dim2(dvc_out_proj_, fan_in_, output_fan_out_, -output_limit_, output_limit_);
}

void transformer_generate_bias_weights(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj_bias);
    __half *dvc_out_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj_bias);    

    const std::size_t qkv_fan_out_ = header_._num_dims * 3;
    const std::size_t output_fan_out_ = header_._num_dims;
    const float limit_ = 0.0f;

    utils_device_generate_random_half_dim1(dvc_qkv_proj_bias_, qkv_fan_out_, -limit_, limit_);
    utils_device_generate_random_half_dim1(dvc_out_proj_bias_, output_fan_out_, -limit_, limit_);
}

void transformer_edit_input_embedding(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_input_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_input_embed);

    std::size_t num_elems_ = header_._batch_size * header_._sequence_length * header_._num_dims;
    utils_device_convert<__half, std::uint16_t><<<(num_elems_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_input_embed_, dvc_tensor, num_elems_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
}

void transformer_edit_qkv_projection(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj);
    
    std::size_t num_elems_ = (header_._num_dims * header_._num_dims) * 3;
    utils_device_convert<__half, std::uint16_t><<<(num_elems_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_qkv_proj_, dvc_tensor, num_elems_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
}

void transformer_edit_output_projection(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_out_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj);
    
    std::size_t num_elems_ = header_._num_dims * header_._num_dims;
    utils_device_convert<__half, std::uint16_t><<<(num_elems_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_out_proj_, dvc_tensor, num_elems_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
}

void transformer_edit_qkv_projection_bias(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj_bias);
    
    std::size_t num_elems_ = header_._num_dims * 3;
    utils_device_convert<__half, std::uint16_t><<<(num_elems_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_qkv_proj_bias_, dvc_tensor, num_elems_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
}

void transformer_edit_output_projection_bias(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_out_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj_bias);
    
    std::size_t num_elems_ = header_._num_dims;
    utils_device_convert<__half, std::uint16_t><<<(num_elems_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_out_proj_bias_, dvc_tensor, num_elems_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
}

void transformer_causal_self_attention(Transformer *transformer, const bool bias, const float dropout, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return;  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_matrix_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_matrix);
    __half *dvc_input_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_input_embed);
    __half *dvc_qkv_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj);
    __half *dvc_qkv_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj_bias);
    __half *dvc_attn_scores_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_attn_scores);
    __half *dvc_dropout_mask_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_dropout);
    __half *dvc_context_layer_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_context_layer);
    __half *dvc_out_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj);
    __half *dvc_out_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj_bias);
    __half *dvc_output_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_output);

    const int head_dim_ = header_._num_dims / header_._num_heads;
    const float scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const int qkv_stride_ = header_._batch_size * header_._sequence_length * header_._num_dims;
    const __half *dvc_q_matrix_ = dvc_qkv_matrix_;
    const __half *dvc_k_matrix_ = dvc_q_matrix_ + qkv_stride_;
    const __half *dvc_v_matrix_ = dvc_k_matrix_ + qkv_stride_;

    cublasHandle_t handle_;
    cublasStatus_t stat_ = cublasCreate(&handle_);
    if (stat_ != CUBLAS_STATUS_SUCCESS) {
		if (verbose) KERAVNOS_PRINT_ERROR("failed to create CUBLAS.");
		throw std::runtime_error("transformer_causal_self_attention failed - failed to create CUBLAS.");
	}

    // perform QKV projection
    selfattn_compute_qkv_projection(
        dvc_qkv_matrix_,
        handle_,
        dvc_input_embed_, dvc_qkv_proj_, dvc_qkv_proj_bias_,
        header_._num_dims, header_._batch_size, header_._sequence_length,
        bias
    );

    selfattn_compute_attention(
        dvc_attn_scores_, dvc_context_layer_,
        dvc_dropout_mask_, dvc_q_matrix_, dvc_k_matrix_, dvc_v_matrix_,
        header_._batch_size, header_._sequence_length, header_._num_heads, 
        head_dim_, 
        scale_, 
        true, // causal masking enabled
        0.0f // dropout
    );

    // compute output projection
    selfattn_compute_output_projection(
        dvc_output_,
        handle_,
        dvc_context_layer_, dvc_out_proj_, dvc_out_proj_bias_,
        header_._batch_size, header_._sequence_length, header_._num_dims, head_dim_, header_._num_heads,
        bias
    );

    stat_ = cublasDestroy(handle_);
    if (stat_ != CUBLAS_STATUS_SUCCESS) {
		if (verbose) KERAVNOS_PRINT_ERROR("failed to destroy CUBLAS.");
		throw std::runtime_error("transformer_causal_self_attention failed - failed to destroy CUBLAS.");
	}
}
 
TransformerHeader transformer_get_header(Transformer *transformer, const bool verbose) {
    TransformerHeader hst_header_ = {};    
    
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return hst_header_;  
    }

    TransformerHeader *dvc_header_ = reinterpret_cast<TransformerHeader *>(reinterpret_cast<char *>(transformer->_dvc_base));
    memory_copy_device_to_host(&hst_header_, dvc_header_, sizeof(TransformerHeader), verbose);

    return hst_header_;
}

py::array_t<int> transformer_get_token_ids(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);

    py::array_t<int> token_ids_({header_._batch_size, header_._sequence_length});    
    py::buffer_info buf_ = token_ids_.request();
    std::size_t count_ = buf_.size;
    int *hst_ptr_ = static_cast<int *>(buf_.ptr);

    int *dvc_token_ids_ = reinterpret_cast<int *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_ids);
    memory_copy_device_to_host(hst_ptr_, dvc_token_ids_, count_ * sizeof(int), verbose);

    return token_ids_;
}

py::array_t<float> transformer_get_token_embed(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_token_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_token_embed);

    std::size_t count_ = header_._vocab_size * header_._num_dims;
    float *dvc_fp32_out_ = static_cast<float *>(memory_device_allocate(count_ * sizeof(float), verbose));

    utils_device_convert<float, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_fp32_out_, dvc_token_embed_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<float> hst_fp32_out_(count_);
    memory_copy_device_to_host(hst_fp32_out_.data(), dvc_fp32_out_, count_ * sizeof(float), verbose);
    
    memory_device_deallocate(dvc_fp32_out_, verbose);
    
    py::array_t<float> result_({header_._vocab_size, header_._num_dims});
    std::memcpy(result_.mutable_data(), hst_fp32_out_.data(), count_ * sizeof(float));
    return result_;
}

py::array_t<float> transformer_get_pos_embed(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_pos_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_pos_embed);

    std::size_t count_ = header_._sequence_length * header_._num_dims;
    float *dvc_fp32_out_ = static_cast<float *>(memory_device_allocate(count_ * sizeof(float), verbose));

    utils_device_convert<float, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_fp32_out_, dvc_pos_embed_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<float> hst_fp32_out_(count_);
    memory_copy_device_to_host(hst_fp32_out_.data(), dvc_fp32_out_, count_ * sizeof(float), verbose);
    
    memory_device_deallocate(dvc_fp32_out_, verbose);
    
    py::array_t<float> result_({header_._vocab_size, header_._num_dims});
    std::memcpy(result_.mutable_data(), hst_fp32_out_.data(), count_ * sizeof(float));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_input_embedding(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_input_embed_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_input_embed);

    std::size_t count_ = header_._batch_size * header_._sequence_length * header_._num_dims;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));

    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_input_embed_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._batch_size, header_._sequence_length, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._sequence_length * header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);
    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_qkv_projection(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj);

    std::size_t count_ = (header_._num_dims * header_._num_dims) * 3;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));

    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_qkv_proj_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {3, header_._num_dims, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._num_dims * header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);
    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_output_projection(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_out_proj_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj);

    std::size_t count_ = header_._num_dims * header_._num_dims;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));

    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_out_proj_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._num_dims, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);
    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_qkv_projection_bias(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_proj_bias);

    std::size_t count_ = header_._num_dims * 3;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));

    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_qkv_proj_bias_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {3, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);

    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_output_projection_bias(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};  
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_out_proj_bias_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_out_proj_bias);

    std::size_t count_ = header_._num_dims;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));

    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_out_proj_bias_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);

    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._num_dims};
    py::array_t<std::uint16_t> result_(shape_);
    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_output(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_output_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_output);

    std::size_t count_ = header_._batch_size * header_._sequence_length * header_._num_dims;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));
    if (verbose) KERAVNOS_PRINT("allocated dvc_u16_out_");
    
    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_output_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
    if (verbose) KERAVNOS_PRINT("done convert.");
    
    std::vector<std::uint16_t> hst_u16_out_(count_);
    if (verbose) KERAVNOS_PRINT("should copy device to host.");
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._batch_size, header_._sequence_length, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._sequence_length * header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);

    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_qkv_matrix(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_qkv_mat_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_qkv_matrix);

    std::size_t count_ = header_._batch_size * header_._sequence_length * (header_._num_dims * 3);
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));
    
    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_qkv_mat_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
    
    std::vector<std::uint16_t> hst_u16_out_(count_);
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._batch_size, header_._sequence_length, 3, header_._num_dims};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._sequence_length * 3 * header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(3 * header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._num_dims * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);

    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_attention_scores(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_attn_scores_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_attn_scores);

    std::size_t count_ = header_._batch_size * header_._num_heads * header_._sequence_length * header_._sequence_length;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));
    if (verbose) KERAVNOS_PRINT("allocated dvc_u16_out_");
    
    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_attn_scores_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
    if (verbose) KERAVNOS_PRINT("done convert.");
    
    std::vector<std::uint16_t> hst_u16_out_(count_);
    if (verbose) KERAVNOS_PRINT("should copy device to host.");
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._batch_size, header_._num_heads, header_._sequence_length, header_._sequence_length};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._num_heads * header_._sequence_length * header_._sequence_length * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._sequence_length * header_._sequence_length * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._sequence_length * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);

    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}

py::array_t<std::uint16_t> transformer_get_context_layer(Transformer *transformer, const bool verbose) {
    if (!transformer) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer is null.");
        return {};
    }

    const TransformerHeader &header_ = transformer_get_header(transformer, verbose);
    __half *dvc_context_layer_ = reinterpret_cast<__half *>(reinterpret_cast<char *>(transformer->_dvc_base) + header_._offset_context_layer);

    std::size_t head_dim_ = header_._num_dims / header_._num_heads;
    std::size_t count_ = header_._batch_size * header_._num_heads * header_._sequence_length * head_dim_;
    std::uint16_t *dvc_u16_out_ = static_cast<std::uint16_t *>(memory_device_allocate(count_ * sizeof(std::uint16_t), verbose));
    if (verbose) KERAVNOS_PRINT("allocated dvc_u16_out_");
    
    utils_device_convert<std::uint16_t, __half><<<(count_ + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dvc_u16_out_, dvc_context_layer_, count_);
    CUDA_CHECK(cudaDeviceSynchronize(), verbose);
    if (verbose) KERAVNOS_PRINT("done convert.");
    
    std::vector<std::uint16_t> hst_u16_out_(count_);
    if (verbose) KERAVNOS_PRINT("should copy device to host.");
    memory_copy_device_to_host(hst_u16_out_.data(), dvc_u16_out_, count_ * sizeof(std::uint16_t), verbose);
    
    memory_device_deallocate(dvc_u16_out_, verbose);

    std::vector<std::size_t> shape_ = {header_._batch_size, header_._num_heads, header_._sequence_length, head_dim_};
    std::vector<std::size_t> strides_ = {
        static_cast<std::size_t>(header_._num_heads * header_._sequence_length * head_dim_ * sizeof(std::uint16_t)),
        static_cast<std::size_t>(header_._sequence_length * head_dim_ * sizeof(std::uint16_t)),
        static_cast<std::size_t>(head_dim_ * sizeof(std::uint16_t)),
        static_cast<std::size_t>(sizeof(std::uint16_t))
    };

    py::array_t<std::uint16_t> result_(shape_, strides_);

    std::memcpy(result_.mutable_data(), hst_u16_out_.data(), count_ * sizeof(std::uint16_t));
    return result_;
}
