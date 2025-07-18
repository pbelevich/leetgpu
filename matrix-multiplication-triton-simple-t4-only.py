# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    BLOCK_SIZE: tl.constexpr,
):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    m_blk_idx = tl.program_id(0)
    m_idx = m_blk_idx * BLOCK_SIZE
    k_blk_idx = tl.program_id(1)
    k_idx = k_blk_idx * BLOCK_SIZE

    c_rows = m_idx + tl.arange(0, BLOCK_SIZE)
    c_cols = k_idx + tl.arange(0, BLOCK_SIZE)
    c_ptrs = c_ptr + c_rows[:, None] * stride_cm + c_cols[None, :] * stride_ck
    c_mask = (c_rows[:, None] < M) & (c_cols[None, :] < K)
    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for n_blk_idx in tl.range(0, tl.cdiv(N, BLOCK_SIZE)):
        n_idx = n_blk_idx * BLOCK_SIZE

        a_rows = c_rows
        a_cols = n_idx + tl.arange(0, BLOCK_SIZE)
        a_ptrs = a_ptr + a_rows[:, None] * stride_am + a_cols[None, :] * stride_an
        a_mask = (a_rows[:, None] < M) & (a_cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_rows = n_idx + tl.arange(0, BLOCK_SIZE)
        b_cols = c_cols
        b_ptrs = b_ptr + b_rows[:, None] * stride_bn + b_cols[None, :] * stride_bk
        b_mask = (b_rows[:, None] < N) & (b_cols[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        c = tl.dot(a, b, c)

    tl.store(c_ptrs, c, mask=c_mask)

   
# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE)) 
    matrix_multiplication_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE
    )
