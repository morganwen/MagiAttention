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

"""Low-overhead CUDA tensor diagnostics for explicit debug runs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_nonfinite_block_stats_kernel(
    input_ptr,
    count_ptr,
    max_abs_ptr,
    elements,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    is_nan = valid & (values != values)
    is_positive_inf = valid & (values == float("inf"))
    is_negative_inf = valid & (values == -float("inf"))
    is_finite = valid & ~is_nan & ~is_positive_inf & ~is_negative_inf
    count_offset = block * 3
    tl.store(count_ptr + count_offset, tl.sum(is_nan.to(tl.int32)))
    tl.store(
        count_ptr + count_offset + 1,
        tl.sum(is_positive_inf.to(tl.int32)),
    )
    tl.store(
        count_ptr + count_offset + 2,
        tl.sum(is_negative_inf.to(tl.int32)),
    )
    tl.store(
        max_abs_ptr + block,
        tl.max(tl.where(is_finite, tl.abs(values), 0.0)),
    )


@triton.jit
def _dsa_nonfinite_row_counts_kernel(
    input_ptr,
    output_ptr,
    row_width,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    nan_count = tl.zeros((), tl.int32)
    positive_inf_count = tl.zeros((), tl.int32)
    negative_inf_count = tl.zeros((), tl.int32)
    first_nan = row_width
    for column_start in range(0, row_width, BLOCK):
        columns = column_start + tl.arange(0, BLOCK)
        valid = columns < row_width
        values = tl.load(
            input_ptr + row * row_width + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        is_nan = valid & (values != values)
        is_positive_inf = valid & (values == float("inf"))
        is_negative_inf = valid & (values == -float("inf"))
        nan_count += tl.sum(is_nan.to(tl.int32))
        positive_inf_count += tl.sum(is_positive_inf.to(tl.int32))
        negative_inf_count += tl.sum(is_negative_inf.to(tl.int32))
        first_nan = tl.minimum(
            first_nan,
            tl.min(tl.where(is_nan, columns, row_width)),
        )
    output_offset = row * 4
    tl.store(output_ptr + output_offset, nan_count)
    tl.store(output_ptr + output_offset + 1, positive_inf_count)
    tl.store(output_ptr + output_offset + 2, negative_inf_count)
    tl.store(
        output_ptr + output_offset + 3,
        tl.where(first_nan < row_width, first_nan, -1),
    )


def dsa_nonfinite_row_counts(tensor: torch.Tensor) -> torch.Tensor:
    """Count NaN/+Inf/-Inf values and locate the first NaN in every row."""

    if tensor.ndim < 1:
        raise ValueError("DSA nonfinite diagnostics require at least one dimension")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("DSA nonfinite diagnostics require a floating-point tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("DSA nonfinite diagnostics require contiguous CUDA input")
    rows = tensor.shape[0]
    row_width = tensor.numel() // rows if rows else 0
    output = torch.empty((rows, 4), dtype=torch.int32, device=tensor.device)
    if rows and row_width:
        block = min(1024, triton.next_power_of_2(row_width))
        _dsa_nonfinite_row_counts_kernel[(rows,)](
            tensor,
            output,
            row_width,
            BLOCK=block,
            num_warps=8 if block >= 512 else 4,
        )
    elif rows:
        output.zero_()
        output[:, 3].fill_(-1)
    return output


def dsa_nonfinite_block_stats(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-block nonfinite counts and finite absolute maxima."""

    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("DSA nonfinite diagnostics require a floating-point tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("DSA nonfinite diagnostics require contiguous CUDA input")
    block = 1024
    blocks = triton.cdiv(tensor.numel(), block)
    counts = torch.empty((blocks, 3), dtype=torch.int32, device=tensor.device)
    max_abs = torch.empty((blocks,), dtype=torch.float32, device=tensor.device)
    if blocks:
        _dsa_nonfinite_block_stats_kernel[(blocks,)](
            tensor,
            counts,
            max_abs,
            tensor.numel(),
            BLOCK=block,
            num_warps=8,
        )
    return counts, max_abs


__all__ = ["dsa_nonfinite_block_stats", "dsa_nonfinite_row_counts"]
