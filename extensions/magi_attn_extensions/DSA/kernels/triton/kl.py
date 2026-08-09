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

"""Fused selected-KL preprocessing and loss reduction for Magi-DSA."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_selected_kl_rows_kernel(
    target_ptr,
    predict_ptr,
    lengths_ptr,
    row_loss_ptr,
    LOSS_SCALE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PART_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    columns = part * BLOCK + tl.arange(0, BLOCK)
    column_mask = columns < TOPK_WIDTH
    length = tl.minimum(tl.maximum(tl.load(lengths_ptr + row), 0), TOPK_WIDTH)
    valid = column_mask & (columns < length)
    offsets = row * TOPK_WIDTH + columns
    target = tl.load(target_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    predict = tl.load(predict_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    minimum = 3.783505853677006e-44
    log_target = tl.maximum(
        tl.minimum(tl.log(tl.maximum(target, minimum)), 0.0), -100.0
    )
    log_predict = tl.maximum(
        tl.minimum(tl.log(tl.maximum(predict, minimum)), 0.0),
        -100.0,
    )
    pointwise = tl.where(valid, target * (log_target - log_predict), 0.0)
    tl.store(
        row_loss_ptr + row * PART_COUNT + part,
        tl.sum(pointwise) * LOSS_SCALE,
    )


@triton.jit
def _dsa_selected_kl_loss_kernel(
    row_loss_ptr,
    loss_ptr,
    row_count,
    BLOCK: tl.constexpr,
):
    loss = 0.0
    for row_start in range(0, row_count, BLOCK):
        rows = row_start + tl.arange(0, BLOCK)
        loss += tl.sum(
            tl.load(row_loss_ptr + rows, mask=rows < row_count, other=0.0).to(
                tl.float32
            )
        )
    tl.store(loss_ptr, loss)


def _validate_selected_kl_inputs(
    target: torch.Tensor,
    predict: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    if target.ndim != 2 or predict.shape != target.shape:
        raise ValueError("selected KL target/predict must be matching matrices")
    rows, topk_width = target.shape
    if topk_width <= 0 or topk_width > 1024:
        raise ValueError("selected KL width must be in [1, 1024]")
    if lengths.shape != (rows,):
        raise ValueError("selected KL row metadata has an invalid shape")
    if target.dtype != torch.float32 or predict.dtype != torch.float32:
        raise TypeError("selected KL floating-point inputs must use float32")
    if lengths.dtype != torch.int32:
        raise TypeError("selected KL lengths must use int32")
    tensors = (target, predict, lengths)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused selected KL requires contiguous CUDA inputs")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("fused selected KL inputs must share one CUDA device")


def fused_dsa_selected_kl_state(
    target: torch.Tensor,
    predict: torch.Tensor,
    lengths: torch.Tensor,
    loss_coeff: float,
) -> torch.Tensor:
    """Compute selected-only KL before cuDNN consumes target and predict."""

    _validate_selected_kl_inputs(target, predict, lengths)
    if not math.isfinite(loss_coeff):
        raise ValueError("selected KL loss coefficient must be finite")
    rows, topk_width = target.shape
    part_block = min(128, triton.next_power_of_2(topk_width))
    part_count = triton.cdiv(topk_width, part_block)
    row_loss = torch.empty(
        (rows, part_count), dtype=torch.float32, device=target.device
    )
    loss = torch.empty((), dtype=torch.float32, device=target.device)
    if not rows:
        loss.zero_()
        return loss

    _dsa_selected_kl_rows_kernel[(rows, part_count)](
        target,
        predict,
        lengths,
        row_loss,
        LOSS_SCALE=float(loss_coeff) / rows,
        TOPK_WIDTH=topk_width,
        PART_COUNT=part_count,
        BLOCK=part_block,
        num_warps=4,
    )
    _dsa_selected_kl_loss_kernel[(1,)](
        row_loss,
        loss,
        row_loss.numel(),
        BLOCK=1024,
        num_warps=8,
    )
    return loss


__all__ = ["fused_dsa_selected_kl_state"]
