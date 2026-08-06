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

"""Fused gradient postprocessing for the Magi-DSA Indexer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_mask_empty_indexer_gradients_kernel(
    grad_q_ptr,
    grad_weights_ptr,
    lengths_ptr,
    output_q_ptr,
    output_weights_ptr,
    q_row_width: tl.constexpr,
    weight_row_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    keep = tl.load(lengths_ptr + row) > 0
    q_mask = columns < q_row_width
    q = tl.load(
        grad_q_ptr + row * q_row_width + columns,
        mask=q_mask,
        other=0.0,
    )
    tl.store(
        output_q_ptr + row * q_row_width + columns,
        tl.where(keep, q, 0.0),
        mask=q_mask,
    )
    if block == 0:
        weight_mask = columns < weight_row_width
        weights = tl.load(
            grad_weights_ptr + row * weight_row_width + columns,
            mask=weight_mask,
            other=0.0,
        )
        tl.store(
            output_weights_ptr + row * weight_row_width + columns,
            tl.where(keep, weights, 0.0),
            mask=weight_mask,
        )


@triton.jit
def _dsa_scale_indexer_gradients_kernel(
    grad_q_ptr,
    grad_weights_ptr,
    grad_k_ptr,
    grad_loss_ptr,
    output_q_ptr,
    output_weights_ptr,
    output_k_ptr,
    q_count,
    weight_count,
    k_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    scale = tl.load(grad_loss_ptr).to(tl.float32)
    q_mask = offsets < q_count
    q = tl.load(grad_q_ptr + offsets, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(output_q_ptr + offsets, q * scale, mask=q_mask)
    weight_mask = offsets < weight_count
    weights = tl.load(grad_weights_ptr + offsets, mask=weight_mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_weights_ptr + offsets, weights * scale, mask=weight_mask)
    k_mask = offsets < k_count
    k = tl.load(grad_k_ptr + offsets, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(output_k_ptr + offsets, k * scale, mask=k_mask)


def _validate_gradient_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA")


def fused_dsa_mask_empty_indexer_gradients(
    grad_q: torch.Tensor,
    grad_weights: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask empty query rows in Q/weight gradients with one launch."""

    if grad_q.ndim != 3 or grad_weights.ndim != 2:
        raise ValueError("Indexer Q/weight gradients have invalid ranks")
    rows = grad_q.shape[0]
    if grad_weights.shape[0] != rows or grad_weights.shape[1] != grad_q.shape[1]:
        raise ValueError("Indexer Q/weight gradients have incompatible shapes")
    if lengths.shape != (rows,) or lengths.dtype != torch.int32:
        raise ValueError("Indexer gradient lengths must be a matching int32 vector")
    _validate_gradient_tensor("grad_q", grad_q)
    _validate_gradient_tensor("grad_weights", grad_weights)
    if not lengths.is_cuda or not lengths.is_contiguous():
        raise ValueError("Indexer gradient lengths must be contiguous CUDA")
    if len({grad_q.device, grad_weights.device, lengths.device}) != 1:
        raise ValueError("Indexer gradient tensors must share one device")

    output_q = torch.empty_like(grad_q)
    output_weights = torch.empty_like(grad_weights)
    if rows:
        q_row_width = grad_q.shape[1] * grad_q.shape[2]
        block = min(2048, triton.next_power_of_2(q_row_width))
        grid = (rows, triton.cdiv(q_row_width, block))
        _dsa_mask_empty_indexer_gradients_kernel[grid](
            grad_q,
            grad_weights,
            lengths,
            output_q,
            output_weights,
            q_row_width=q_row_width,
            weight_row_width=grad_weights.shape[1],
            BLOCK=block,
            num_warps=8,
        )
    return output_q, output_weights


def fused_dsa_scale_indexer_gradients(
    grad_q: torch.Tensor,
    grad_weights: torch.Tensor,
    grad_k: torch.Tensor,
    grad_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale three saved Indexer gradients by one device scalar in one launch."""

    for name, tensor in (
        ("grad_q", grad_q),
        ("grad_weights", grad_weights),
        ("grad_k", grad_k),
    ):
        _validate_gradient_tensor(name, tensor)
    if grad_loss.shape != () or grad_loss.dtype != torch.float32:
        raise ValueError("Indexer grad_loss must be an FP32 scalar")
    if not grad_loss.is_cuda or not grad_loss.is_contiguous():
        raise ValueError("Indexer grad_loss must be contiguous CUDA")
    if len({grad_q.device, grad_weights.device, grad_k.device, grad_loss.device}) != 1:
        raise ValueError("Indexer gradient tensors must share one device")

    output_q = torch.empty_like(grad_q)
    output_weights = torch.empty_like(grad_weights)
    output_k = torch.empty_like(grad_k)
    maximum_count = max(grad_q.numel(), grad_weights.numel(), grad_k.numel())
    if maximum_count:
        block = 2048
        _dsa_scale_indexer_gradients_kernel[(triton.cdiv(maximum_count, block),)](
            grad_q,
            grad_weights,
            grad_k,
            grad_loss,
            output_q,
            output_weights,
            output_k,
            grad_q.numel(),
            grad_weights.numel(),
            grad_k.numel(),
            BLOCK=block,
            num_warps=8,
        )
    return output_q, output_weights, output_k


__all__ = [
    "fused_dsa_mask_empty_indexer_gradients",
    "fused_dsa_scale_indexer_gradients",
]
