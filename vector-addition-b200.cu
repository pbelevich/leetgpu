#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N / 2) {
        float2& c = reinterpret_cast<float2*>(C)[i];
        const float2& a = reinterpret_cast<const float2*>(A)[i];
        const float2& b = reinterpret_cast<const float2*>(B)[i];
        c = __fadd2_ru(a, b);
    }
    if (i == N / 2 && N % 2 == 1) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
