from kernels import get_kernel, has_kernel
hf_kernel = "kernels-community/flash-attn4"
if has_kernel(hf_kernel):
    print("got it")
    kernel = get_kernel(hf_kernel).flash_attn_interface
else:
    print("lost it")
    
