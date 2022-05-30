#pragma once
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdint>

void invoke_custom_div(const float* src1, const float * src2, const float alpha, float* dest1,float* dest2, int64_t N, cudaStream_t stream);
