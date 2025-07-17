import triton
import triton.language as tl

@triton.jit
def partial_max_value_kernel(X, partial_max, cols, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    partial_max = partial_max.to(tl.pointer_type(tl.float32))

    row = tl.program_id(0)
    pid = tl.program_id(1)

    col_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * cols + col_offset, mask=col_offset < cols, other=-float('inf'))
    local_max = tl.max(x, axis=0)
    tl.store(partial_max + row * BLOCK_SIZE + pid, local_max)


@triton.jit
def partial_exp_sum_value_kernel(X, partial_sum, global_max, cols, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    partial_sum = partial_sum.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))

    row = tl.program_id(0)
    pid = tl.program_id(1)

    col_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * cols + col_offset, mask=col_offset < cols, other=-float('inf'))
    gmax = tl.load(global_max + row)
    x = tl.exp(x - gmax)
    sum_value = tl.sum(x, axis=0)
    tl.store(partial_sum + row * BLOCK_SIZE + pid, sum_value)


@triton.jit
def normalize_kernel(X, Y, global_max, global_sum, cols, BLOCK_SIZE: tl.constexpr):
    X = X.to(tl.pointer_type(tl.float32))
    Y = Y.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))
    global_sum = global_sum.to(tl.pointer_type(tl.float32))

    row = tl.program_id(0)
    pid = tl.program_id(1)

    col_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * cols + col_offset, mask=col_offset < cols, other=-float('inf'))
    gmax = tl.load(global_max + row)
    gsum = tl.load(global_sum + row)
    y = tl.exp(x - gmax) / gsum
    tl.store(Y + row * cols + col_offset, y, mask=col_offset < cols)


@triton.jit
def get_max_value(partial_max, global_max, BLOCK_SIZE: tl.constexpr):
    partial_max = partial_max.to(tl.pointer_type(tl.float32))
    global_max = global_max.to(tl.pointer_type(tl.float32))

    row = tl.program_id(0)

    col_offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(partial_max + row * BLOCK_SIZE + col_offset)
    local_max = tl.max(x, axis=0)
    tl.store(global_max + row, local_max)
    

@triton.jit
def get_sum_value(partial_sum, global_sum, BLOCK_SIZE: tl.constexpr):
    partial_sum = partial_sum.to(tl.pointer_type(tl.float32))
    global_sum = global_sum.to(tl.pointer_type(tl.float32))

    row = tl.program_id(0)

    col_offset = tl.arange(0, BLOCK_SIZE)
    x = tl.load(partial_sum + row * BLOCK_SIZE + col_offset)
    tl.store(global_sum + row, tl.sum(x))
    

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
    BLOCK_SIZE = 32 * 1024
    num_blocks = triton.cdiv(cols, BLOCK_SIZE)
    grid1 = (rows, num_blocks)
    grid2 = (rows, )
    partial_max = cudaEmpty(rows * BLOCK_SIZE)
    partial_sum = cudaEmpty(rows * BLOCK_SIZE)
    global_max = cudaEmpty(rows)
    global_sum = cudaEmpty(rows)
    partial_max_value_kernel[grid1](input_ptr, partial_max, cols, BLOCK_SIZE)
    get_max_value[grid2](partial_max, global_max, BLOCK_SIZE)
    partial_exp_sum_value_kernel[grid1](input_ptr, partial_sum, global_max, cols, BLOCK_SIZE)
    get_sum_value[grid2](partial_sum, global_sum, BLOCK_SIZE)
    normalize_kernel[grid1](input_ptr, output_ptr, global_max, global_sum, cols, BLOCK_SIZE)
    cudaFree(partial_max)
    cudaFree(partial_sum)
    cudaFree(global_max)
    cudaFree(global_sum)
