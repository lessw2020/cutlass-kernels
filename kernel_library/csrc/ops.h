#pragma once

#include <torch/library.h>
#include <torch/all.h>

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales);



void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                             torch::Tensor& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                              torch::Tensor& scale);


using fptr_t = int64_t;
