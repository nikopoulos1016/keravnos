#pragma once

#include "global.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void layer_norm(Transformer *transformer, const bool verbose);

#ifdef __cplusplus
}
#endif

