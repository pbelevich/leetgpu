#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N / 2) {
        auto temp = input[i];
        input[i] = input[N - 1 - i];
        input[N - 1 - i] = temp;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
