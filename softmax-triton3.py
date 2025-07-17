import triton
import triton.language as tl

@triton.jit
def max_kernel(x, x_row_stride, max_values, n_cols, block_size: tl.constexpr):
    x = x.to(tl.pointer_type(tl.float32))
    max_values = max_values.to(tl.pointer_type(tl.float32))

    row_idx = tl.program_id(0)
    x_row_start = x + row_idx * x_row_stride

    max_value = tl.full((), float('-inf'), dtype=tl.float32)
    for block_offset in tl.range(0, n_cols, block_size):
        col_offsets = block_offset + tl.arange(0, block_size)
        block = tl.load(x_row_start + col_offsets, mask=col_offsets < n_cols, other=float('-inf'))
        max_value = tl.maximum(max_value, tl.max(block))
    
    tl.store(max_values + row_idx, max_value)

@triton.jit
def sum_kernel(x, x_row_stride, max_values, sum_values, n_cols, block_size: tl.constexpr):
    x = x.to(tl.pointer_type(tl.float32))
    max_values = max_values.to(tl.pointer_type(tl.float32))
    sum_values = sum_values.to(tl.pointer_type(tl.float32))

    row_idx = tl.program_id(0)
    x_row_start = x + row_idx * x_row_stride

    max_value = tl.load(max_values + row_idx)

    sum_value = tl.full((), 0.0, dtype=tl.float32)
    for block_offset in tl.range(0, n_cols, block_size):
        col_offsets = block_offset + tl.arange(0, block_size)
        block = tl.load(x_row_start + col_offsets, mask=col_offsets < n_cols, other=float('-inf'))
        sum_value += tl.sum(tl.exp(block - max_value))
    
    tl.store(sum_values + row_idx, sum_value)
    

@triton.jit
def softmax_kernel(x, x_row_stride, max_values, sum_values, y, y_row_stride, n_cols, block_size: tl.constexpr):
    x = x.to(tl.pointer_type(tl.float32))
    y = y.to(tl.pointer_type(tl.float32))
    max_values = max_values.to(tl.pointer_type(tl.float32))
    sum_values = sum_values.to(tl.pointer_type(tl.float32))

    row_idx = tl.program_id(0)
    x_row_start = x + row_idx * x_row_stride
    y_row_start = y + row_idx * y_row_stride

    max_value = tl.load(max_values + row_idx)
    sum_value = tl.load(sum_values + row_idx)

    for block_offset in tl.range(0, n_cols, block_size):
        col_offsets = block_offset + tl.arange(0, block_size)
        block = tl.load(x_row_start + col_offsets, mask=col_offsets < n_cols, other=float('-inf'))
        tl.store(y_row_start + col_offsets, tl.exp(block - max_value) / sum_value, mask=col_offsets < n_cols)


def cudaEmpty(num_elements:int):
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), num_elements*4)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed, code {err}")
    return ptr.value

def cudaFree(ptr):
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    err = cudart.cudaFree(ptr)
    if err != 0:
        raise RuntimeError(f"cudaFree failed, code {err}")


# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    rows, cols = 1, N
    block_size = 64 * 1024
    num_warps = 32
    grid = (rows, )
    max_values = cudaEmpty(rows)
    max_kernel[grid](input_ptr, cols, max_values, cols, block_size, num_warps=num_warps)
    sum_values = cudaEmpty(rows)
    sum_kernel[grid](input_ptr, cols, max_values, sum_values, cols, block_size, num_warps=num_warps)
    softmax_kernel[grid](input_ptr, cols, max_values, sum_values, output_ptr, cols, cols, block_size, num_warps=num_warps)
