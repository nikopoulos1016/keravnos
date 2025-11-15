#include "utils/device.cuh"
#include "core/memory.cuh"
#include "core/transformer.cuh"


Transformer *current_transformer = nullptr;
std::unordered_map<std::string, Transformer *> transformer_registry;

void keravnos_construct_transformer(
  const std::string name,
  const int batch_size, 
  const int sequence_length, 
  const int vocab_size,
  const int num_dims,
  const int num_heads,
  const int num_layers,
  const int ff_mult,
  const bool verbose
) {
    if (transformer_registry.find(name) != transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' already registered.");
        return;
    }

    Transformer *tr_ = new Transformer;
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' constructed.");
    
    transformer_allocate_device_memory(tr_, batch_size, sequence_length, vocab_size, num_dims, num_heads, num_layers, ff_mult, verbose);
    if (!tr_->_dvc_base) {
        if (verbose) KERAVNOS_PRINT_ERROR("failed to allocate device memory for transformer '", name, "'.");
        return;
    }
    if (verbose) KERAVNOS_PRINT("completed device memory allocation for transformer '", name, "'.");

    // generate embedding weights
    transformer_generate_embedding_weights(tr_, verbose);
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' generated embedding weights.");
    
    // generate projection weights
    transformer_generate_projection_weights(tr_, verbose);
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' generated projection weights.");
    
    // generate bias weights
    transformer_generate_bias_weights(tr_, verbose);
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' generated bias weights.");
    
    // construct cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&tr_->_cublas_handle));
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' constructed cuBLAS handle.");

    // add to registry
    transformer_registry[name] = tr_;
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' added to registry.");

    // set current transformer if null
    if (!current_transformer) {
        current_transformer = tr_;
        if (verbose) KERAVNOS_PRINT("transformer '", name, "' is set as current.");
    }
}

void keravnos_destruct_transformer(const std::string name, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return;
    }

    Transformer *tr_ = transformer_registry[name];

    CUBLAS_CHECK(cublasDestroy(tr_->_cublas_handle));
    if (verbose) KERAVNOS_PRINT("transformer '", name, "' destructed cuBLAS handle.");

    transformer_deallocate_device_memory(tr_, verbose);
    if (verbose) KERAVNOS_PRINT("completed device memory deallocation for transformer '", name, "'.");

    if (verbose) KERAVNOS_PRINT("destructed transformer '", name, "'.");
    transformer_registry.erase(name);

    if (current_transformer == tr_) {
        if (!transformer_registry.empty()) {
            current_transformer = transformer_registry.begin()->second;
            if (verbose) KERAVNOS_PRINT("'", transformer_registry.begin()->first, "' is set as current transformer.");
        } else {
            current_transformer = nullptr;
            if (verbose) KERAVNOS_PRINT("no transformer remaining. current transformer is set to null.");
        }
    }
}

void keravnos_feed_token_ids(const std::string name, const py::array_t<int, py::array::c_style | py::array::forcecast> token_ids, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return;
    }

    Transformer *tr_ = transformer_registry[name];
    transformer_feed_token_ids(tr_, token_ids, verbose);
}

void keravnos_edit_tensor(const std::string name, const std::string tensor_id, const py::array_t<std::uint16_t> input, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return;
    }

    Transformer *tr_ = transformer_registry[name];

    py::buffer_info input_info_ = input.request();
    const std::size_t num_elems_ = input_info_.size;
    const std::uint16_t* host_data_ = static_cast<std::uint16_t *>(input_info_.ptr);

    std::uint16_t *dvc_input_ = static_cast<std::uint16_t *>(memory_device_allocate(num_elems_ * sizeof(std::uint16_t), verbose));
    memory_copy_host_to_device(dvc_input_, host_data_, num_elems_ * sizeof(__half), verbose);

    if (tensor_id == "input_embed") {
        transformer_edit_input_embedding(tr_, dvc_input_, verbose);
    }
    else if (tensor_id == "qkv_proj") {
        transformer_edit_qkv_projection(tr_, dvc_input_, verbose);
    } 
    else if (tensor_id == "qkv_proj_bias") {
        transformer_edit_qkv_projection_bias(tr_, dvc_input_, verbose);
    }
    else if (tensor_id == "out_proj") {
        transformer_edit_output_projection(tr_, dvc_input_, verbose);
    } 
    else if (tensor_id == "out_proj_bias") {
        transformer_edit_output_projection_bias(tr_, dvc_input_, verbose);
    }
    else {
        if (verbose) KERAVNOS_PRINT_WARN("attempted to edit weights for transformer '", name, "' with unknown tensor id: '", tensor_id, "'.");
    }

    memory_device_deallocate(dvc_input_, verbose);
    if (verbose) KERAVNOS_PRINT("edit completed for tensor id '", tensor_id, "' of transformer '", name, "'.");
}

void keravnos_layer_forward(const std::string name, const int layer_index, const bool bias, const float dropout, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return;
    }

    Transformer *tr_ = transformer_registry[name];
    transformer_layer_forward(tr_, layer_index, bias, dropout, verbose);
}

void keravnos_forward(const std::string name, const bool bias, const float dropout, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return;
    }
    
    Transformer *tr_ = transformer_registry[name];
    transformer_forward(tr_, bias, dropout, verbose);

    if (verbose) KERAVNOS_PRINT("forward pass completed for transformer '", name, "'.");
}

py::dict keravnos_get_transformer_info(const std::string name, const bool verbose) {
    py::dict info_;

    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return info_;
    }

    Transformer *tr_ = transformer_registry[name];
    const TransformerHeader &header_ = transformer_get_header(tr_, verbose);

    const int d_ff_ = header_._num_dims * header_._ff_multiplier;

    // per-layer sizes
    const std::size_t qkv_proj_ = 3 * header_._num_dims * header_._num_dims * sizeof(__half);
    const std::size_t qkv_bias_ = 3 * header_._num_dims * sizeof(__half);
    const std::size_t out_proj_ = header_._num_dims * header_._num_dims * sizeof(__half);
    const std::size_t out_bias_ = header_._num_dims * sizeof(__half);
    const std::size_t ffn_weight_ = 2 * header_._num_dims * d_ff_ * sizeof(__half);
    const std::size_t ffn_bias_ = 2 * d_ff_ * sizeof(__half);
    const std::size_t ln_param_ = 4 * header_._num_dims * sizeof(__half);

    // global sizes
    const std::size_t token_embed_ = header_._vocab_size * header_._num_dims * sizeof(__half);
    const std::size_t pos_embed_ = header_._sequence_length * header_._num_dims * sizeof(__half);
    const std::size_t token_ids_ = header_._batch_size * header_._sequence_length * sizeof(int);
    const std::size_t input_embed_ = header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);
    const std::size_t dropout_attn_ = header_._num_layers * header_._batch_size * header_._num_heads * header_._sequence_length * header_._sequence_length * sizeof(__half);
    const std::size_t dropout_ffn_ = header_._num_layers * header_._batch_size * header_._sequence_length * d_ff_ * sizeof(__half);

    // shared activation
    const std::size_t shared_qkv_ = header_._batch_size * header_._sequence_length * 3 * header_._num_dims * sizeof(__half);
    const std::size_t shared_attn_scores_ = header_._batch_size * header_._num_heads * header_._sequence_length * header_._sequence_length * sizeof(__half);
    const std::size_t shared_ctx_ = header_._batch_size * header_._num_heads * header_._sequence_length * (header_._num_dims / header_._num_heads) * sizeof(__half);
    const std::size_t shared_ffn_in_ = header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);
    const std::size_t shared_ffn_hidden_ = header_._batch_size * header_._sequence_length * d_ff_ * sizeof(__half);
    const std::size_t shared_ffn_out_ = header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);
    const std::size_t shared_ln_ = 2 * header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);
    const std::size_t final_out_ = header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);

    // gradients
    const std::size_t grad_total_ = header_._num_layers * (qkv_proj_ + qkv_bias_ + out_proj_ + out_bias_ + ffn_weight_ + ffn_bias_ + ln_param_) + token_embed_ + pos_embed_;

    // optimisers (adam: 2x each)
    const std::size_t opt_total_ = 2 * header_._num_layers * (qkv_proj_ + qkv_bias_ + out_proj_ + out_bias_ + ffn_weight_ + ffn_bias_ + ln_param_) + 2 * (token_embed_ + pos_embed_);

    // grad temp buffers
    const std::size_t grad_temp_total_ = header_._num_layers * PER_LAYER_BUFFER_SIZE;

    // checkpoint scratch
    const std::size_t checkpoint_buf_ = 2 * header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half);

    // set info
    info_["name"] = name;
    info_["num_layers"] = header_._num_layers;
    info_["ff_multiplier"] = header_._ff_multiplier;
    info_["total_bytes"] = header_._mem_total;
    info_["batch_size"] = header_._batch_size;
    info_["sequence_length"] = header_._sequence_length;
    info_["vocab_size"] = header_._vocab_size;
    info_["embedding_dim"] = header_._num_dims;
    info_["ffn_hidden_dim"] = d_ff_;
    info_["num_heads"] = header_._num_heads;
    info_["header_size"] = sizeof(TransformerHeader);

    // parameter sizes
    info_["parameter_size"] = token_embed_ + pos_embed_ + header_._num_layers * (qkv_proj_ + qkv_bias_ + out_proj_ + out_bias_ + ffn_weight_ + ffn_bias_ + ln_param_);

    // gradients
    info_["gradient_size"] = grad_total_;

    // optimiser states
    info_["optimizer_state_size"] = opt_total_;

    // dropout masks
    info_["dropout_mask_size"] = dropout_attn_ + dropout_ffn_;

    // shared activation
    info_["activation_size"] =
        token_ids_ + input_embed_ +
        dropout_attn_ + dropout_ffn_ +
        shared_qkv_ + shared_attn_scores_ + shared_ctx_ +
        shared_ffn_in_ + shared_ffn_hidden_ + shared_ffn_out_ +
        shared_ln_ + final_out_;

    // checkpoint scratch
    info_["activation_checkpoint_buffer_size"] = checkpoint_buf_;

    // gradient temp
    info_["gradient_scratch_size"] = grad_temp_total_;

    if (verbose) {
        KERAVNOS_PRINT("transformer '", name, "' info:");
        PY_PRINT(info_);
    }

    return info_;
}

py::array_t<int> keravnos_get_token_ids(const std::string name, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return {};
    }

    Transformer *tr_ = transformer_registry[name];
    return transformer_get_token_ids(tr_, verbose);
}

py::array_t<float> keravnos_get_token_embed(const std::string name, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return {};
    }

    Transformer *tr_ = transformer_registry[name];
    return transformer_get_token_embed(tr_, verbose);
}

py::array_t<float> keravnos_get_pos_embed(const std::string name, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return {};
    }

    Transformer *tr_ = transformer_registry[name];
    return transformer_get_pos_embed(tr_, verbose);
}

py::array_t<std::uint16_t> keravnos_get_tensor(const std::string name, const std::string tensor_id, const bool verbose) {
    if (transformer_registry.find(name) == transformer_registry.end()) {
        if (verbose) KERAVNOS_PRINT_ERROR("transformer '", name, "' does not exist.");
        return {};
    }

    Transformer *tr_ = transformer_registry[name];

    if (tensor_id == "input_embed") {
        return transformer_get_input_embedding(tr_, verbose);
    }
    else if (tensor_id == "qkv_proj") {
        return transformer_get_qkv_projection(tr_, verbose);
    } 
    else if (tensor_id == "qkv_proj_bias") {
        return transformer_get_qkv_projection_bias(tr_, verbose);
    }
    else if (tensor_id == "out_proj") {
        return transformer_get_output_projection(tr_, verbose);
    } 
    else if (tensor_id == "out_proj_bias") {
        return transformer_get_output_projection_bias(tr_, verbose);
    }
    else if (tensor_id == "output") {
        return transformer_get_output(tr_, verbose);
    }
    else if (tensor_id == "qkv_matrix") {
        return transformer_get_qkv_matrix(tr_, verbose);
    } 
    else if (tensor_id == "attention_scores") {
        return transformer_get_attention_scores(tr_, verbose);
    }
    else if (tensor_id == "context_layer") {
        return transformer_get_context_layer(tr_, verbose);
    }
    else {
        if (verbose) KERAVNOS_PRINT_WARN("attempted to retrieve tensor for transformer '", name, "' with unknown tensor id: '", tensor_id, "'.");
        return {};
    }
}

// -------------------------------- interface -------------------------------- //

PYBIND11_MODULE(keravnos, m) {
    m.def(
        "construct",
        &keravnos_construct_transformer,
        py::arg("name"),
        py::arg("batch_size") = 2,
        py::arg("sequence_length") = 2048,
        py::arg("vocab_size") = 48000,
        py::arg("num_dims") = 768,
        py::arg("num_heads") = 12,
        py::arg("num_layers") = 6,
        py::arg("ff_multiplier") = 4,
        py::arg("verbose") = false
    );
  
    m.def(
        "destruct",
        &keravnos_destruct_transformer,
        py::arg("name"),
        py::arg("verbose") = false
    );

    m.def(
        "feed_token_ids",
        &keravnos_feed_token_ids,
        py::arg("name"),
        py::arg("token_ids"),
        py::arg("verbose") = false
    );

    m.def(
        "edit_tensor",
        &keravnos_edit_tensor,
        py::arg("name"),
        py::arg("tensor_id"),
        py::arg("input"),
        py::arg("verbose") = false
    );

    m.def(
        "layer_forward",
        &keravnos_layer_forward,
        py::arg("name"),
        py::arg("layer_index"),
        py::arg("bias") = false,
        py::arg("dropout") = 0.5,
        py::arg("verbose") = false
    );

    m.def(
        "forward",
        &keravnos_forward,
        py::arg("name"),
        py::arg("bias") = false,
        py::arg("dropout") = 0.5,
        py::arg("verbose") = false
    );

    m.def(
        "get_info",
        &keravnos_get_transformer_info,
        py::arg("name"),
        py::arg("verbose") = false
    );

    m.def(
        "get_token_ids",
        &keravnos_get_token_ids,
        py::arg("name"),
        py::arg("verbose") = false
    );

    m.def(
        "get_token_embed",
        &keravnos_get_token_embed,
        py::arg("name"),
        py::arg("verbose") = false
    );

    m.def(
        "get_pos_embed",
        &keravnos_get_pos_embed,
        py::arg("name"),
        py::arg("verbose") = false
    );

    m.def(
        "get_tensor",
        &keravnos_get_tensor,
        py::arg("name"),
        py::arg("tensor_id"),
        py::arg("verbose") = false
    );
}
 