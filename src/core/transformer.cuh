#pragma once

#include "global.cuh"

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
);
void transformer_deallocate_device_memory(Transformer *transformer, const bool verbose);

void transformer_feed_token_ids(Transformer *transformer, const py::array_t<int, py::array::c_style | py::array::forcecast> token_ids, const bool verbose);

void transformer_generate_embedding_weights(Transformer *transformer, const bool verbose); 
void transformer_generate_projection_weights(Transformer *transformer, const bool verbose);
void transformer_generate_bias_weights(Transformer *transformer, const bool verbose);

void transformer_edit_input_embedding(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose);
void transformer_edit_qkv_projection(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose);
void transformer_edit_output_projection(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose);
void transformer_edit_qkv_projection_bias(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose);
void transformer_edit_output_projection_bias(Transformer *transformer, const std::uint16_t *dvc_tensor, const bool verbose);

void transformer_layer_forward(Transformer *transformer, const int layer_index, const bool bias, const float dropout, const bool verbose);
void transformer_forward(Transformer *transformer, const bool bias, const float dropout, const bool verbose);

TransformerHeader transformer_get_header(Transformer *transformer, const bool verbose);
py::array_t<int> transformer_get_token_ids(Transformer *transformer, const bool verbose);
py::array_t<float> transformer_get_token_embed(Transformer *transformer, const bool verbose);
py::array_t<float> transformer_get_pos_embed(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_input_embedding(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_qkv_projection(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_output_projection(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_qkv_projection_bias(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_output_projection_bias(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_output(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_qkv_matrix(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_attention_scores(Transformer *transformer, const bool verbose);
py::array_t<std::uint16_t> transformer_get_context_layer(Transformer *transformer, const bool verbose);