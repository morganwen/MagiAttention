# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused projection epilogues for Magi-DSA."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_scale_cast_kernel(
    input_ptr,
    output_ptr,
    element_count,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < element_count
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, values * SCALE, mask=mask)


def _launch_dsa_scale_cast(
    tensor: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    if not tensor.numel():
        return
    block = 2048
    _dsa_scale_cast_kernel[(triton.cdiv(tensor.numel(), block),)](
        tensor,
        output,
        tensor.numel(),
        SCALE=scale,
        BLOCK=block,
        num_warps=8,
    )


class _FusedDSAScaleCast(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.empty_like(tensor, dtype=output_dtype)
        _launch_dsa_scale_cast(tensor, output, scale)
        ctx.scale = scale
        ctx.input_dtype = tensor.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = torch.empty_like(grad_output, dtype=ctx.input_dtype)
        _launch_dsa_scale_cast(grad_output.contiguous(), grad_input, ctx.scale)
        return grad_input, None, None


def fused_dsa_scale_cast(
    tensor: torch.Tensor,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Scale a contiguous projection output and cast it in one launch."""

    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("fused DSA projection epilogue expects bfloat16 or float32")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("fused DSA projection epilogue has an invalid output dtype")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("fused DSA projection epilogue requires contiguous CUDA input")
    if not math.isfinite(scale):
        raise ValueError("fused DSA projection scale must be finite")
    return _FusedDSAScaleCast.apply(tensor, float(scale), output_dtype)


__all__ = ["fused_dsa_scale_cast"]
