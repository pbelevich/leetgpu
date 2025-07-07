import torch

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    max = torch.max(input)
    torch.exp(input - max, out=output)
    sum = torch.sum(output)
    torch.div(output, sum, out=output)
