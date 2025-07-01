#include "solve.h"
#include <cuda_runtime.h>

#define BLOCKSIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    if (cRow * BLOCKSIZE + threadRow < M && cCol * BLOCKSIZE + threadCol < N) {
        C[threadRow * N + threadCol] = tmp;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE);
    dim3 blocksPerGrid((M + BLOCKSIZE - 1) / BLOCKSIZE,
                       (N + BLOCKSIZE - 1) / BLOCKSIZE);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
