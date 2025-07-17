import triton
import triton.language as tl

@triton.jit
def triton_softmax_kernel(
    x,
    x_row_stride,
    y,
    y_row_stride,
    n_cols,
    block_size: tl.constexpr,
):
    x = x.to(tl.pointer_type(tl.float32))
    y = y.to(tl.pointer_type(tl.float32))

    row_idx = tl.program_id(0)
    x_row_start = x + row_idx * x_row_stride
    y_row_start = y + row_idx * y_row_stride
    current_max = float('-inf')
    current_sum = 0.0
    
    for block_offset in tl.range(0, n_cols, block_size):
        col_offsets = block_offset + tl.arange(0, block_size)
        block = tl.load(x_row_start + col_offsets, mask=col_offsets < n_cols, other=float('-inf'))
        new_max = tl.maximum(current_max, tl.max(block))
        current_sum *= tl.exp(current_max - new_max)
        current_sum += tl.sum(tl.exp(block - new_max))
        current_max = new_max

    for block_offset in tl.range(0, n_cols, block_size):
        col_offsets = block_offset + tl.arange(0, block_size)
        block = tl.load(x_row_start + col_offsets, mask=col_offsets < n_cols, other=float('-inf'))
        tl.store(y_row_start + col_offsets, tl.exp(block - current_max) / current_sum, mask=col_offsets < n_cols)


# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    rows, cols = 1, N
    block_size = 32 * 1024
    num_warps = 32
    grid = (rows, )
    triton_softmax_kernel[grid](input_ptr, cols, output_ptr, cols, cols, block_size, num_warps=num_warps)
