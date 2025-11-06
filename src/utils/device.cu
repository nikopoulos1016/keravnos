#include "utils/device.cuh"


__global__ void _utils_device_generate_random_half(__half *out, const int n, const float min, const float max, const std::uint64_t seed) {
    int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_ >= n) return;

    if (min == max) {
        out[idx_] = __float2half(min);
        return;
    }

    int tile_id_ = idx_ / TILE_SIZE;
    int lane_ = idx_ % TILE_SIZE;

    extern __shared__ float sm_buf_[];
    float *tile_buf_ = &sm_buf_[(threadIdx.x / TILE_SIZE) * TILE_SIZE];

    if (lane_ == 0) {
        curandState state_;
        curand_init(seed, tile_id_, 0, &state_);

        #pragma unroll
        for (int i_ = 0; (i_ < TILE_SIZE) && (tile_id_ * TILE_SIZE + i_) < n; ++i_) {
            tile_buf_[i_] = min + curand_uniform(&state_) * (max - min);
        }
    }

    __syncthreads();

    if ((tile_id_ * TILE_SIZE + lane_) < n) {
        float val_ = tile_buf_[lane_];
        out[idx_] = __float2half(val_);
    }
}

__global__ void _utils_device_generate_sinusodial_half(__half *out, const int n, const int n_dims, const int base) {
    int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_ >= n) return;

    int pos_ = idx_ / n_dims;
    int i_   = idx_ % n_dims;

    float exp_ = -2.0f * floorf(static_cast<float>(i_) / 2.0f) / static_cast<float>(n_dims);
    float angle_ = pos_ * __powf(static_cast<float>(base), exp_);

    float val_ = (i_ % 2 == 0) ? __sinf(angle_) : __cosf(angle_);
    out[idx_] = __float2half(val_);
}

__global__ void _utils_device_generate_boolean_half(__half *out, const int n, const std::uint64_t seed) {
    int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_ >= n) return;
    
    int tile_id_ = idx_ / TILE_SIZE;
    int lane_ = idx_ % TILE_SIZE;
    
    extern __shared__ float sm_buf_[];
    float *tile_buf_ = &sm_buf_[(threadIdx.x / TILE_SIZE) * TILE_SIZE];

    if (lane_ == 0) {
        curandState state_;
        curand_init(seed, tile_id_, 0, &state_);

        #pragma unroll
        for (int i_ = 0; (i_ < TILE_SIZE) && (tile_id_ * TILE_SIZE + i_) < n; ++i_) {
            tile_buf_[i_] = curand_uniform(&state_);
        }
    }

    __syncthreads();

    if ((tile_id_ * TILE_SIZE + lane_) < n) {
        float val_ = tile_buf_[lane_];
        out[idx_] = __float2half(static_cast<float>(val_ >= 0.5f));
    }
}


#ifdef __cplusplus
extern "C" {
#endif

void utils_device_generate_random_half_dim1(__half *out, const int dx, const float min, const float max, const int num_threads) {
    const int n_ = dx;
    std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
    std::size_t shared_mem_ = (num_threads / TILE_SIZE) * TILE_SIZE * sizeof(float);
    _utils_device_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads, shared_mem_>>>(out, n_, min, max, seed_);
}

void utils_device_generate_random_half_dim2(__half *out, const int dx, const int dy, const float min, const float max, const int num_threads) {
    const int n_ = dx * dy;
    std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
    std::size_t shared_mem_ = (num_threads / TILE_SIZE) * TILE_SIZE * sizeof(float);
    _utils_device_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads, shared_mem_>>>(out, n_, min, max, seed_);
}

void utils_device_generate_random_half_dim3(__half *out, const int dx, const int dy, const int dz, const float min, const float max, const int num_threads) {
    const int n_ = dx * dy * dz;
    std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
    std::size_t shared_mem_ = (num_threads / TILE_SIZE) * TILE_SIZE * sizeof(float);
    _utils_device_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads, shared_mem_>>>(out, n_, min, max, seed_);
}

void utils_device_generate_sinusodial_half(__half *out, const int dx, const int dy, const int base, const int num_threads) {
    const int n_ = dx * dy;
    _utils_device_generate_sinusodial_half<<<(n_ + num_threads - 1) / num_threads, num_threads>>>(out, n_, dy, base);
}

void utils_device_generate_boolean_half(__half *out, const int n, const int num_threads) {
    std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
    std::size_t shared_mem_ = (num_threads / TILE_SIZE) * TILE_SIZE * sizeof(float);
    _utils_device_generate_boolean_half<<<(n + num_threads - 1) / num_threads, num_threads, shared_mem_>>>(out, n, seed_);
}

#ifdef __cplusplus
}
#endif
