import torch

import triton
import triton.language as tl

def add(x,y): 
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),) # creates a 1-d launch grid

    add_kernel[grid](x,y,output, n_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr+offsets, mask)
    y = tl.load(y_ptr+offsets, mask)
    output = x + y
    tl.store(output_ptr+offsets, output, mask)


torch.manual_seed(0)
size = 1000
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)
print(output_torch[:10])
print(output_triton[:10])

"""
The maximum difference between torch and triton is 0.0
tensor([1.3713, 1.3076, 0.4940, 1.2701, 1.2803, 1.1750, 1.1790, 1.4607, 0.3393,
        1.2689], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940, 1.2701, 1.2803, 1.1750, 1.1790, 1.4607, 0.3393,
        1.2689], device='cuda:0')
        
"""