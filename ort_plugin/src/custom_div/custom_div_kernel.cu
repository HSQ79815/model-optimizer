#include "custom_div_kernel.h"

#define DivUp(a, b) (a + b - 1) / (b)

__global__ void custom_div(const float *src1, const float *src2, const float alpha, float *dest1, float *dest2, int64_t N)
{
    int x = threadIdx.x + blockIdx.x * blockDi

                                           m.x;
    if (x >= N)
        return;
    float s1 = src1[x];
    float s2 = src2[x];
    if (s1 == 0.f)
    {
        dest1[x] = 0.f;
        dest2[x] = 0.f;
    }
    else
    {
        dest1[x] = 1.f;
        dest2[x] = s2 * alpha / s1;
    }
}

void invoke_custom_div(const float *src1, const float *src2, const float alpha, float *dest1, float *dest2, int64_t N, cudaStream_t stream)
{
    constexpr int SIZE = 1024;
    dim3 block{SIZE};
    dim3 grid{DivUp(N, SIZE)};

    custom_div<<<grid, block, 0, stream>>>(src1, src2, alpha, dest1, dest2, N);
}