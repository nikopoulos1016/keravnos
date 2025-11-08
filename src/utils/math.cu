#include "utils/math.cuh"


__device__ float utils_math_gelu(float x) {
    float k_ = __fsqrt_rn(2.0f / FP32_PI);
    float x3_ = x * x * x;

    return 0.5f * x * (1.0f + __tanhf(k_ * (x + 0.044715f * x3_)));
}

