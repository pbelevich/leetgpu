#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    // Load input to shared memory
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    for (
        int j = tid;
        j < blockDim.x + kernel_size - 1 && j + blockIdx.x * blockDim.x < input_size;
        j += blockDim.x
    ) {
        shared[j] = input[j + blockIdx.x * blockDim.x];
    }
    // Load kernel to shared memory
    float* kernel_shared = shared + blockDim.x + kernel_size - 1;
    for (
        int j = tid;
        j < kernel_size;
        j += blockDim.x
    ) {
        kernel_shared[j] = kernel[j];
    }
    __syncthreads();
    // Calculate convolution and store it to global memory
    int output_size = input_size - kernel_size + 1;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < output_size) {
        float res = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            res += shared[tid + j] * kernel_shared[j];
        }
        output[i] = res;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 512;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t shmemBytes = ((threadsPerBlock + kernel_size - 1) + kernel_size) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shmemBytes>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
