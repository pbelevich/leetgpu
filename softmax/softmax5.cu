#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <cassert>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline __device__ __host__ unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float *x, int rows, int cols, int coarseFactor, float *max_values) {
    extern __shared__ float smem[];
    auto tid = threadIdx.x;
    auto row = blockIdx.y;

    float max_value = -FLT_MAX;
    for (int i = 0; i < coarseFactor; i++) {
        auto col = blockIdx.x * (blockDim.x * coarseFactor) + tid + i * blockDim.x;
        if (col < cols) {
            max_value = fmaxf(max_value, x[(row * cols) + col]);
        }
    }
    smem[tid] = max_value;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(&max_values[row], smem[0]);
    }
}

__global__ void sum_kernel(const float *x, float* y, int rows, int cols, int coarseFactor, const float *max_values, float *sum_values) {
    extern __shared__ float smem[];
    auto tid = threadIdx.x;
    auto row = blockIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < coarseFactor; i++) {
        auto col = blockIdx.x * (blockDim.x * coarseFactor) + tid + i * blockDim.x;
        if (col < cols) {
            float val = expf(x[(row * cols) + col] - max_values[row]);
            y[(row * cols) + col] = val;
            sum += val;
        }
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sum_values[row], smem[0]);
    }
}

__global__ void softmax_kernel5(float *y, int rows, int cols, const float *sum_values) {
    auto tid = threadIdx.x;
    auto row = blockIdx.y;
    auto col = tid + blockIdx.x * blockDim.x;

    if (col < cols) {
        y[row * cols + col] /= sum_values[row];
    }
}

torch::Tensor softmax(const torch::Tensor& x) {
    auto x_flatten = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x_flatten);
    auto rows = x_flatten.size(0);
    auto cols = x_flatten.size(1);

    auto coarseFactor = 32;
    dim3 threads(1024);
    dim3 blocks(cdiv(cols, threads.x * coarseFactor), rows);
    auto smem_size = threads.x * sizeof(float);

    float* max_values = nullptr;
    cudaMalloc(&max_values, rows * sizeof(float));
    float* sum_values = nullptr;
    cudaMalloc(&sum_values, rows * sizeof(float));
    float* max_values_host = new float[rows];
    float* sum_values_host = new float[rows];
    for (int i = 0; i < rows; i++) {
        max_values_host[i] = -FLT_MAX;
        sum_values_host[i] = 0.0f;
    }
    cudaMemcpy(max_values, max_values_host, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sum_values, sum_values_host, rows * sizeof(float), cudaMemcpyHostToDevice);
    delete[] max_values_host;
    delete[] sum_values_host;

    max_kernel<<<blocks, threads, smem_size>>>(x.data_ptr<float>(), rows, cols, coarseFactor, max_values);    
    sum_kernel<<<blocks, threads, smem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols, coarseFactor, max_values, sum_values);
    auto blocks_softmax = dim3(cdiv(cols, threads.x), rows);
    softmax_kernel5<<<blocks_softmax, threads>>>(y.data_ptr<float>(), rows, cols, sum_values);

    cudaFree(sum_values);
    cudaFree(max_values);

    return y.view_as(x);
}