//#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "registration.h"

#include <torch/library.h>
#include <torch/all.h>
#include <cstdint>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // custom ops

  
  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);
  ops.impl("cutlass_scaled_mm_supports_fp8", torch::kCUDA,
           &cutlass_scaled_mm_supports_fp8);

  
  // Compute FP8 quantized tensor for given scaling factor.
  /*ops.def(
      "static_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);
*/
  
}

/*
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute", &get_device_attribute);
  cuda_utils.impl("get_device_attribute", torch::kCUDA, &get_device_attribute);

}
*/


REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
