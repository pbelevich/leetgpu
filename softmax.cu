#include "solve.h"
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 1024

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float sdata[sizeof(float) * BLOCK_SIZE];
    int lane = threadIdx.x;
    float local_max = -FLT_MAX;
    for (int i = lane; i < N; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[lane] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sdata[lane] = fmaxf(sdata[lane], sdata[lane + stride]);
        }
        __syncthreads();
    }
    float global_max = sdata[0];

    float local_sum = 0.0;
    for (int i = lane; i < N; i += BLOCK_SIZE) {
        float e = __expf(input[i] - global_max);
        local_sum += e;
    }
    sdata[lane] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sdata[lane] = sdata[lane] + sdata[lane + stride];
        }
        __syncthreads();
    }
    float global_sum = sdata[0];

    for (int i = lane; i < N; i += BLOCK_SIZE) {
        float e = __expf(input[i] - global_max);
        output[i] = e / global_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = 1;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
