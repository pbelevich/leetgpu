#include "solve.h"
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N * N) {
        B[i] = A[i];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
} 
