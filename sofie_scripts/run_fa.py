import torch
from kernels import get_kernel

kernel = get_kernel("kernels-community/flash-attn4")
flash_attn = kernel.flash_attn_interface

device = "cuda"
B, T, H, D = 2, 512, 8, 64

# Tensors that need gradients so backward is traced
q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

def fwd(q, k, v):
    out, lse = flash_attn.flash_attn_func(q, k, v, softmax_scale=None, causal=True)
    return out.sum()

compiled_fwd = torch.compile(fwd, dynamic=False)

loss = compiled_fwd(q, k, v)
# loss = fwd(q, k, v)
loss.backward()
print("Done (no error).")
