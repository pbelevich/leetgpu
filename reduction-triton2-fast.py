# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def sum_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offset, mask=offset < N, other=0.0)
    sum = tl.sum(x)
    tl.atomic_add(output_ptr, sum)

def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sum_kernel[grid](input_ptr, output_ptr, N, BLOCK_SIZE)
