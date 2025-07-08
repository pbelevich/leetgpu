# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    offsets = tl.arange(0, BLOCK_SIZE)

    row_max = tl.full((), -float("inf"), dtype=tl.float32)
    for b in range(0, (N + BLOCK_SIZE) // BLOCK_SIZE):
        idx = b * BLOCK_SIZE + offsets
        mask = idx < N
        x = tl.load(input_ptr + idx, mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x.to(tl.float32)))

    row_sum = tl.zeros((), dtype=tl.float32)
    for b in range(0, (N + BLOCK_SIZE) // BLOCK_SIZE):
        idx = b * BLOCK_SIZE + offsets
        mask = idx < N
        x = tl.load(input_ptr + idx, mask=mask, other=-float("inf"))
        x = tl.exp(x - row_max)
        row_sum += tl.sum(x)
        tl.store(output_ptr + idx, x, mask=mask)

    for b in range(0, (N + BLOCK_SIZE) // BLOCK_SIZE):
        idx = b * BLOCK_SIZE + offsets
        mask = idx < N
        y = tl.load(output_ptr + idx, mask=mask)
        y /= row_sum
        tl.store(output_ptr + idx, y, mask=mask)


# input_ptr, output_ptr are raw device pointers
def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    softmax_kernel[(1,)](input_ptr, output_ptr, N, BLOCK_SIZE)
