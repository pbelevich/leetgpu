#include "solve.h"
#include <cuda_runtime.h>

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

#define WARP_SIZE 32

__global__ void reduction_kernel(const float* input, float* output, int N, int stride_factor) {
    extern __shared__ float smem[];
    auto tid = threadIdx.x;
    auto thread_per_block = blockDim.x;
    auto block_size = thread_per_block * stride_factor;
    auto block_start = blockIdx.x * block_size;
    
    // Stride reduction
    auto sum = 0.0;
    for (int i = 0; i < stride_factor; ++i) {
        auto idx = block_start + i * thread_per_block + tid;
        if (idx < N) {
            sum += input[idx];
        }
    }

    // Warp reduction
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }
    if (tid % WARP_SIZE == 0) {
        smem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    // Maybe the second warp reduction
    if (thread_per_block > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            sum = tid < cdiv(thread_per_block, WARP_SIZE) ? smem[tid] : 0.0;
            for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, s);
            }
            if (tid == 0) {
                smem[0] = sum;
            }
        }
        __syncthreads();
    }
    // Grid reduction
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    auto threads_per_block = 512;
    auto stride_factor = 8;
    dim3 threads(threads_per_block);
    dim3 blocks(cdiv(N, threads_per_block * stride_factor));
    auto smem_size = threads_per_block * sizeof(float);
    reduction_kernel<<<blocks, threads, smem_size>>>(input, output, N, stride_factor);
}
