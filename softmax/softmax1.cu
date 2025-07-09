#include <cfloat>

__global__ void softmax1_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            max_val = max(max_val, x[i]);
        }
        float sum = 0;
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            sum += exp(x[i] - max_val);
        }
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            y[i] = exp(x[i] - max_val) / sum;
        }
    }
}

torch::Tensor softmax1(const torch::Tensor& x) {
    auto x_flatten = x.reshape({-1, x.size(-1)});
    auto y = torch::empty_like(x_flatten);
    dim3 block(1024);
    dim3 grid(cdiv(x_flatten.size(0), block.x));
    softmax1_kernel<<<grid, block>>>(x_flatten.data_ptr<float>(), y.data_ptr<float>(), x_flatten.size(0), x_flatten.size(1));
    return y.view_as(x);
}
