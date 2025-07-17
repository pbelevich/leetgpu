# Very slow
import triton
import triton.language as tl

@triton.jit
def sum_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    sum = 0.0
    for block_offset in tl.range(0, N, BLOCK_SIZE):
        offset = block_offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(input_ptr + offset, mask=offset < N, other=0.0)
        sum += tl.sum(x)
    tl.store(output_ptr, sum)

def solve(input_ptr: int, output_ptr: int, N: int):
    block_size = 32 * 1024
    grid = (1,)
    sum_kernel[grid](input_ptr, output_ptr, N, block_size)
