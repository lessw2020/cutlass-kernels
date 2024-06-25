# core pytorch test for scaled mm
import vllm
from vllm._custom_ops import cutlass_scaled_mm
import torch

m, k, n = 16, 4096, 4096
dtype= torch.float8_e4m3fn 
out_dtype= torch.float16

a = torch.empty(m, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda')
bt = torch.empty(n, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda').t()
scale_a = torch.ones((1,)).to(dtype=torch.float32, device='cuda') # scale tensors on cpu results in faster time
scale_b = torch.ones((1,)).to(dtype=torch.float32, device='cuda') # scale tensors on cpu results in faster time

y = cutlass_scaled_mm(a, bt, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
