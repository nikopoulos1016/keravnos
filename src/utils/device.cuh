#pragma once

#include "global.cuh"


template <typename D, typename S>
__device__ D _utils_device_convert(S x);

// fp16 -> fp32
template <>
__device__ __forceinline__ float _utils_device_convert<float, __half>(__half x) {
    return __half2float(x);
}

// fp32 -> fp16
template <>
__device__ __forceinline__ __half _utils_device_convert<__half, float>(float x) {
    return __float2half(x);
}

// fp16 -> uint16
template <>
__device__ __forceinline__ std::uint16_t _utils_device_convert<std::uint16_t, __half>(__half x) {
    return __half_as_ushort(x);
}

// uint16 -> fp16
template <>
__device__ __forceinline__ __half _utils_device_convert<__half, std::uint16_t>(std::uint16_t x) {
    return __ushort_as_half(x);
}

template <typename D, typename S>
__global__ void utils_device_convert(D *dst, const S *src, const int n) {
    int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_ >= n) return;
    
    dst[idx_] = _utils_device_convert<D, S>(src[idx_]);
}


#ifdef __cplusplus
extern "C" {
#endif

void utils_device_generate_random_half_dim1(__half *out, const int dx, const float min, const float max, const int num_threads = NUM_THREADS);
void utils_device_generate_random_half_dim2(__half *out, const int dx, const int dy, const float min, const float max, const int num_threads = NUM_THREADS);
void utils_device_generate_random_half_dim3(__half *out, const int dx, const int dy, const int dz, const float min, const float max, const int num_threads = NUM_THREADS);

void utils_device_generate_sinusodial_half(__half *out, const int dx, const int dy, const int base, const int num_threads = NUM_THREADS); 
void utils_device_generate_boolean_half(__half *out, const int n, const int num_threads = NUM_THREADS);

#ifdef __cplusplus
}
#endif
