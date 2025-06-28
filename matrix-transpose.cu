#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
