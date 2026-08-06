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

"""Fused post-GEMM gated reduction for the ratio-4 CSA Compressor."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _csa_compressor_forward_kernel(
    projected_kv_ptr,
    projected_gate_ptr,
    ape_ptr,
    valid_rows_ptr,
    output_ptr,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dimension_block = tl.program_id(1)
    support_1d = tl.arange(0, 8)
    dimensions_1d = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    support = support_1d[:, None]
    dimensions = dimensions_1d[None, :]
    dimension_mask = dimensions < OUTPUT_DIM
    previous = support < 4
    branch_dimensions = tl.where(previous, dimensions, dimensions + OUTPUT_DIM)
    ape_rows = tl.where(previous, support, support - 4)
    row_offset = row * 8 * (2 * OUTPUT_DIM)
    offsets = row_offset + support * (2 * OUTPUT_DIM) + branch_dimensions
    values = tl.load(projected_kv_ptr + offsets, mask=dimension_mask, other=0.0).to(
        tl.float32
    )
    logits = tl.load(
        projected_gate_ptr + offsets, mask=dimension_mask, other=-float("inf")
    ).to(tl.float32)
    logits += tl.load(
        ape_ptr + ape_rows * (2 * OUTPUT_DIM) + branch_dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)
    valid_support = tl.load(valid_rows_ptr + row * 8 + support_1d).to(tl.int1)
    valid = valid_support[:, None] & dimension_mask
    valid_any = tl.sum(valid_support.to(tl.int32), axis=0) > 0
    logits = tl.where(valid, logits, -float("inf"))
    maximum = tl.where(valid_any, tl.max(logits, axis=0), 0.0)
    exponentials = tl.where(valid, tl.exp(logits - maximum[None, :]), 0.0)
    denominator = tl.sum(exponentials, axis=0)
    weights = tl.where(valid_any, exponentials / denominator[None, :], 0.0)
    compressed = tl.sum(values * weights, axis=0)
    tl.store(
        output_ptr + row * OUTPUT_DIM + dimensions_1d,
        compressed,
        mask=dimensions_1d < OUTPUT_DIM,
    )


@triton.jit
def _csa_compressor_backward_kernel(
    projected_kv_ptr,
    projected_gate_ptr,
    ape_ptr,
    valid_rows_ptr,
    grad_output_ptr,
    grad_projected_kv_ptr,
    grad_projected_gate_ptr,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dimension_block = tl.program_id(1)
    support_1d = tl.arange(0, 8)
    dimensions_1d = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    support = support_1d[:, None]
    dimensions = dimensions_1d[None, :]
    dimension_mask = dimensions < OUTPUT_DIM
    previous = support < 4
    branch_dimensions = tl.where(previous, dimensions, dimensions + OUTPUT_DIM)
    ape_rows = tl.where(previous, support, support - 4)
    row_offset = row * 8 * (2 * OUTPUT_DIM)
    selected_offsets = row_offset + support * (2 * OUTPUT_DIM) + branch_dimensions
    values = tl.load(
        projected_kv_ptr + selected_offsets, mask=dimension_mask, other=0.0
    ).to(tl.float32)
    logits = tl.load(
        projected_gate_ptr + selected_offsets,
        mask=dimension_mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits += tl.load(
        ape_ptr + ape_rows * (2 * OUTPUT_DIM) + branch_dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)
    valid_support = tl.load(valid_rows_ptr + row * 8 + support_1d).to(tl.int1)
    valid = valid_support[:, None] & dimension_mask
    valid_any = tl.sum(valid_support.to(tl.int32), axis=0) > 0
    logits = tl.where(valid, logits, -float("inf"))
    maximum = tl.where(valid_any, tl.max(logits, axis=0), 0.0)
    exponentials = tl.where(valid, tl.exp(logits - maximum[None, :]), 0.0)
    denominator = tl.sum(exponentials, axis=0)
    weights = tl.where(valid_any, exponentials / denominator[None, :], 0.0)
    compressed = tl.sum(values * weights, axis=0)
    grad_output = tl.load(
        grad_output_ptr + row * OUTPUT_DIM + dimensions_1d,
        mask=dimensions_1d < OUTPUT_DIM,
        other=0.0,
    ).to(tl.float32)
    grad_values = weights * grad_output[None, :]
    grad_logits = weights * (values - compressed[None, :]) * grad_output[None, :]

    first_half_offsets = row_offset + support * (2 * OUTPUT_DIM) + dimensions
    second_half_offsets = first_half_offsets + OUTPUT_DIM
    first_grad_values = tl.where(previous, grad_values, 0.0)
    second_grad_values = tl.where(previous, 0.0, grad_values)
    first_grad_logits = tl.where(previous, grad_logits, 0.0)
    second_grad_logits = tl.where(previous, 0.0, grad_logits)
    tl.store(
        grad_projected_kv_ptr + first_half_offsets,
        first_grad_values,
        mask=dimension_mask,
    )
    tl.store(
        grad_projected_kv_ptr + second_half_offsets,
        second_grad_values,
        mask=dimension_mask,
    )
    tl.store(
        grad_projected_gate_ptr + first_half_offsets,
        first_grad_logits,
        mask=dimension_mask,
    )
    tl.store(
        grad_projected_gate_ptr + second_half_offsets,
        second_grad_logits,
        mask=dimension_mask,
    )


@triton.jit
def _csa_compressor_ape_backward_kernel(
    grad_projected_gate_ptr,
    grad_ape_ptr,
    row_count,
    OUTPUT_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    ape_row = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimensions_1d = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dimensions = dimensions_1d[None, :]
    dimension_mask = dimensions < 2 * OUTPUT_DIM
    support = tl.where(dimensions < OUTPUT_DIM, ape_row, ape_row + 4)
    gradient = tl.zeros((BLOCK_D,), tl.float32)
    for row_start in range(0, row_count, BLOCK_ROWS):
        rows = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
        offsets = rows * 8 * (2 * OUTPUT_DIM) + support * (2 * OUTPUT_DIM) + dimensions
        values = tl.load(
            grad_projected_gate_ptr + offsets,
            mask=(rows < row_count) & dimension_mask,
            other=0.0,
        ).to(tl.float32)
        gradient += tl.sum(values, axis=0)
    tl.store(
        grad_ape_ptr + ape_row * (2 * OUTPUT_DIM) + dimensions_1d,
        gradient,
        mask=dimensions_1d < 2 * OUTPUT_DIM,
    )


def _validate_fused_csa_compressor_inputs(
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    ape: torch.Tensor,
    valid_rows: torch.Tensor,
    output_dim: int,
) -> None:
    if output_dim <= 0:
        raise ValueError("CSA Compressor output dimension must be positive")
    if projected_kv.ndim != 3 or projected_kv.shape[1:] != (8, 2 * output_dim):
        raise ValueError("CSA projected KV has an invalid shape")
    if projected_gate.shape != projected_kv.shape:
        raise ValueError("CSA projected gate must match projected KV")
    if ape.shape != (4, 2 * output_dim):
        raise ValueError("CSA Compressor APE has an invalid shape")
    if valid_rows.shape != projected_kv.shape[:2] or valid_rows.dtype != torch.bool:
        raise ValueError("CSA valid rows must be a matching bool mask")
    if projected_kv.dtype != torch.bfloat16 or projected_gate.dtype != torch.bfloat16:
        raise TypeError("CSA projected tensors must use bfloat16")
    if ape.dtype != torch.float32:
        raise TypeError("CSA Compressor APE must use float32")
    tensors = (projected_kv, projected_gate, ape, valid_rows)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused CSA Compressor inputs must be contiguous CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("fused CSA Compressor inputs must share one device")


def _launch_csa_compressor_forward(
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    ape: torch.Tensor,
    valid_rows: torch.Tensor,
    output: torch.Tensor,
    output_dim: int,
) -> None:
    rows = projected_kv.shape[0]
    if not rows:
        return
    block_d = min(128, triton.next_power_of_2(output_dim))
    grid = (rows, triton.cdiv(output_dim, block_d))
    _csa_compressor_forward_kernel[grid](
        projected_kv,
        projected_gate,
        ape,
        valid_rows,
        output,
        OUTPUT_DIM=output_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


def _launch_csa_compressor_backward(
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    ape: torch.Tensor,
    valid_rows: torch.Tensor,
    grad_output: torch.Tensor,
    grad_projected_kv: torch.Tensor,
    grad_projected_gate: torch.Tensor,
    grad_ape: torch.Tensor,
    output_dim: int,
) -> None:
    rows = projected_kv.shape[0]
    if not rows:
        grad_ape.zero_()
        return
    block_d = min(128, triton.next_power_of_2(output_dim))
    grid = (rows, triton.cdiv(output_dim, block_d))
    _csa_compressor_backward_kernel[grid](
        projected_kv,
        projected_gate,
        ape,
        valid_rows,
        grad_output,
        grad_projected_kv,
        grad_projected_gate,
        OUTPUT_DIM=output_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    ape_block_d = min(32, triton.next_power_of_2(2 * output_dim))
    ape_grid = (4, triton.cdiv(2 * output_dim, ape_block_d))
    _csa_compressor_ape_backward_kernel[ape_grid](
        grad_projected_gate,
        grad_ape,
        rows,
        OUTPUT_DIM=output_dim,
        BLOCK_ROWS=64,
        BLOCK_D=ape_block_d,
        num_warps=4,
    )


class _FusedCSACompressorReduction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        projected_kv: torch.Tensor,
        projected_gate: torch.Tensor,
        ape: torch.Tensor,
        valid_rows: torch.Tensor,
        output_dim: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.empty(
            (projected_kv.shape[0], output_dim),
            dtype=output_dtype,
            device=projected_kv.device,
        )
        _launch_csa_compressor_forward(
            projected_kv,
            projected_gate,
            ape,
            valid_rows,
            output,
            output_dim,
        )
        ctx.save_for_backward(projected_kv, projected_gate, ape, valid_rows)
        ctx.output_dim = output_dim
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        projected_kv, projected_gate, ape, valid_rows = ctx.saved_tensors
        grad_projected_kv = torch.empty_like(projected_kv)
        grad_projected_gate = torch.empty_like(projected_gate)
        grad_ape = torch.empty_like(ape)
        _launch_csa_compressor_backward(
            projected_kv,
            projected_gate,
            ape,
            valid_rows,
            grad_output.contiguous(),
            grad_projected_kv,
            grad_projected_gate,
            grad_ape,
            ctx.output_dim,
        )
        return grad_projected_kv, grad_projected_gate, grad_ape, None, None, None


def fused_csa_compressor_reduce(
    projected_kv: torch.Tensor,
    projected_gate: torch.Tensor,
    ape: torch.Tensor,
    valid_rows: torch.Tensor,
    output_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply ratio-4 overlap assembly, masked softmax, and pooling in one launch."""

    _validate_fused_csa_compressor_inputs(
        projected_kv,
        projected_gate,
        ape,
        valid_rows,
        output_dim,
    )
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("fused CSA Compressor output must use a floating-point dtype")
    return _FusedCSACompressorReduction.apply(
        projected_kv,
        projected_gate,
        ape,
        valid_rows,
        output_dim,
        output_dtype,
    )


__all__ = ["fused_csa_compressor_reduce"]
