#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nvml.h>

#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include <cublasLt.h>
#include <cublas_v2.h>

namespace py = pybind11;

// -------------------------------- macros -------------------------------- //

#define NUM_THREADS 256
#define TILE_SIZE 32
#define EPSILON 1e-5f
#define PER_LAYER_BUFFER_SIZE 16777216 // 16 MB
#define ALIGN_OFFSET(offset, n) (((offset) + ((n) - 1)) & ~((n) - 1))
#define FP16_INF __int2half_rn(0x7C00)
#define FP16_NINF __int2half_rn(0xFC00)
#define FP32_INF __int_as_float(0x7f800000)
#define FP32_NINF __int_as_float(0xff800000)
#define FP16_MAX_NEG -65504.0f
#define FP16_MAX_POS 65504.0f
#define FP16_MASK_VAL __float2half(-1e9f)
#define FP32_MASK_VAL -1e9f
#define PY_PRINT(...) py::print(__VA_ARGS__, py::arg("sep") = "")
#define BASE_NAME(filepath) std::string(filepath).substr(std::string(filepath).find_last_of("/\\") + 1)
#define KERAVNOS_PRINT(...) PY_PRINT("> keravnos (", BASE_NAME(__FILE__), ":", __LINE__, ") ", __VA_ARGS__)		
#define KERAVNOS_PRINT_WARN(...) PY_PRINT("> keravnos [warn] (", BASE_NAME(__FILE__), ":", __LINE__, ") ", __VA_ARGS__)		
#define KERAVNOS_PRINT_ERROR(...) PY_PRINT("> keravnos [error] (", BASE_NAME(__FILE__), ":", __LINE__, ") ", __VA_ARGS__)		
#define KERAVNOS_PRINT_CUDA(...) PY_PRINT("> keravnos [cuda] (", BASE_NAME(__FILE__), ":", __LINE__, ") ", __VA_ARGS__)		
#define KERAVNOS_PRINT_CUDA_ERROR(...) PY_PRINT("> keravnos [cuda error] (", BASE_NAME(__FILE__), ":", __LINE__, ") ", __VA_ARGS__)		
#define CUDA_CHECK(call, verbose) do {	\
    	cudaError_t err_ = (call);	\
    	if (err_ != cudaSuccess) {	\
    	    if (verbose) KERAVNOS_PRINT_CUDA_ERROR(#call, " — ", cudaGetErrorString(err_));	\
			std::ostringstream oss_;	\
			oss_ << "CUDA Error: " << #call << " — " << cudaGetErrorString(err_);	\
			throw std::runtime_error(oss_.str());	\
		}	\
	} while(0)
#define FORMAT_ADDRESS(ptr, as_name)	\
	std::string as_name;	\
	do {	\
		std::ostringstream oss_;	\
		oss_ << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr); \
		as_name = oss_.str();	\
	} while(0)
	
// -------------------------------- typedefs -------------------------------- //

typedef struct TransformerHeader {
	std::size_t   		_batch_size;
	std::size_t   		_sequence_length;
	std::size_t   		_vocab_size;
	std::size_t   		_num_dims;
	std::size_t   		_num_heads;
	std::size_t			_num_layers;
	std::size_t			_ff_multiplier;
	std::size_t   		_type_bytes;
	std::size_t			_current_layer_index;

	std::size_t   		_offset_token_embed;
	std::size_t   		_offset_pos_embed;
	std::size_t   		_offset_token_ids;
	std::size_t   		_offset_input_embed;
	std::size_t   		_offset_dropout;
	std::size_t   		_offset_dropout_ffn;

	std::size_t   		_offset_qkv_proj;
	std::size_t   		_offset_qkv_proj_bias;
	std::size_t   		_offset_out_proj;
	std::size_t   		_offset_out_proj_bias;
	std::size_t			_offset_ffn_weights;
	std::size_t			_offset_ffn_bias;
	std::size_t			_offset_ln_params;
	
	std::size_t			_offset_qkv_proj_grad;
	std::size_t			_offset_qkv_bias_grad;
	std::size_t			_offset_out_proj_grad;
	std::size_t			_offset_out_bias_grad;
	std::size_t			_offset_ffn_weight_grad;
	std::size_t			_offset_ffn_bias_grad;
	std::size_t			_offset_ln_params_grad;
	std::size_t			_offset_token_embed_grad;
	std::size_t			_offset_pos_embed_grad;

	std::size_t			_offset_qkv_proj_opt;
	std::size_t			_offset_qkv_bias_opt;
	std::size_t			_offset_out_proj_opt;
	std::size_t			_offset_out_bias_opt;
	std::size_t			_offset_ffn_weight_opt;
	std::size_t			_offset_ffn_bias_opt;
	std::size_t			_offset_ln_params_opt;
	std::size_t			_offset_token_embed_opt;
	std::size_t			_offset_pos_embed_opt;

	std::size_t			_offset_grad_temp;
	std::size_t			_offset_x_in;
	std::size_t			_offset_x_norm;

	std::size_t   		_offset_qkv_matrix;
	std::size_t   		_offset_attn_scores;
	std::size_t   		_offset_context_layer;
	std::size_t   		_offset_output;
	
	std::size_t			_offset_ffn_input;
	std::size_t			_offset_ffn_hidden;
	std::size_t			_offset_ffn_output;
	std::size_t			_offset_ln2_out;
	std::size_t			_offset_ln1_out;

	bool          		_allocated;
	std::size_t   		_mem_total;
} TransformerHeader;

typedef struct Transformer {
	__half				*_dvc_base;
} Transformer;


