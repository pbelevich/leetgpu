#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int block_size = blockDim.x;

    float local_row_max = -FLT_MAX;
    for (int c = tid; c < cols; c += block_size) {
        int idx = row * cols + c;
        local_row_max = fmaxf(local_row_max, x[idx]);
    }
    smem[tid] = local_row_max;
    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    float global_row_max = smem[0];

    float local_row_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        int idx = row * cols + c;
        float e = __expf(x[idx] - global_row_max);
        local_row_sum += e;
    }
    smem[tid] = local_row_sum;
    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float global_row_sum = smem[0];

    for (int c = tid; c < cols; c += block_size) {
        int idx = row * cols + c;
        y[idx] = __expf(x[idx] - global_row_max) / global_row_sum;
    }
}

uint32_t next_power_of_two(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

torch::Tensor softmax(const torch::Tensor& x) {
    auto x_flatten = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x_flatten);
    auto rows = x_flatten.size(0);
    auto cols = x_flatten.size(1);
    auto block_size = std::min(static_cast<uint32_t>(1024), next_power_of_two(cols));
    dim3 block(block_size);
    dim3 grid(rows);
    auto smem_size = block_size * sizeof(float);
    softmax_kernel<<<grid, block, smem_size>>>(x_flatten.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y.view_as(x);
}
