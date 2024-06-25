import contextlib
import functools
from typing import List, Optional, Tuple, Type

import torch


try:
    import pingpong._C
except ImportError as e:
    print("Failed to import from _C with %r", e)



def is_custom_op_supported(op_name: str) -> bool:
    op, overloads = torch._C._jit_get_operation(op_name)
    return op is not None


def hint_on_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AttributeError as e:
            msg = (
                "Error in calling custom op %s: %s\n"
                "Possibly you have built or installed an obsolete version of vllm.\n"
                "Please try a clean build and install of vllm,"
                "or remove old built files such as vllm/*cpython*.so and build/ ."
            )
            logger.error(msg, fn.__name__, e)
            raise e

    return wrapper


# cutlass
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: Type[torch.dtype]) -> torch.Tensor:
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b)
    return out


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)
    return output, scale



def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)
