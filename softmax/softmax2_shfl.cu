#include <cfloat>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define WARP_SIZE 32

__global__ void softmax2_shfl_kernel(const float* x, float* y, int rows, int cols) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int block_size = blockDim.x;

    float local_row_max = -FLT_MAX;
    for (int c = tid; c < cols; c += block_size) {
        int idx = row * cols + c;
        local_row_max = fmaxf(local_row_max, x[idx]);
    }

    float val = local_row_max;
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
    }
    if (tid % WARP_SIZE == 0) {
        smem[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            val = tid < CEIL_DIV(block_size, WARP_SIZE) ? smem[tid] : -FLT_MAX;
            for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
            }
            if (tid == 0) smem[0] = val;
        }
    }
    __syncthreads();
    float global_row_max = smem[0];

    float local_row_sum = 0.0;
    for (int c = tid; c < cols; c += block_size) {
        int idx = row * cols + c;
        float e = __expf(x[idx] - global_row_max);
        local_row_sum += e;
    }

    val = local_row_sum;
    for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, s);
    }
    if (tid % WARP_SIZE == 0) {
        smem[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            val = tid < CEIL_DIV(block_size, WARP_SIZE) ? smem[tid] : 0.0;
            for (int s = WARP_SIZE >> 1; s > 0; s >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, s);
            }
            if (tid == 0) smem[0] = val;
        }
    }
    __syncthreads();
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

torch::Tensor softmax2_shfl(const torch::Tensor& x) {
    auto x_flatten = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x_flatten);
    auto rows = x_flatten.size(0);
    auto cols = x_flatten.size(1);
    auto block_size = std::min(static_cast<uint32_t>(1024), next_power_of_two(cols));
    dim3 block(block_size);
    dim3 grid(rows);
    auto smem_size = block_size * sizeof(float);
    softmax2_shfl_kernel<<<grid, block, smem_size>>>(x_flatten.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y.view_as(x);
}
