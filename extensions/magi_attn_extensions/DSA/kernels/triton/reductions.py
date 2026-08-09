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

"""Fused floating-point reductions for the DeepSeek-V4 Indexer path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_row_logsumexp_kernel(
    scores_ptr,
    row_lengths_ptr,
    output_ptr,
    column_count,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_length = tl.minimum(tl.maximum(tl.load(row_lengths_ptr + row), 0), column_count)
    if row_length > 0:
        row_max = -float("inf")
        row_sum = 0.0
        for column_start in range(0, column_count, BLOCK_SIZE):
            columns = column_start + tl.arange(0, BLOCK_SIZE)
            valid = columns < row_length
            values = tl.load(
                scores_ptr + row * row_stride + columns,
                mask=valid,
                other=-float("inf"),
            ).to(tl.float32)
            new_max = tl.maximum(row_max, tl.max(values))
            row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
                tl.where(valid, tl.exp(values - new_max), 0.0)
            )
            row_max = new_max
        result = row_max + tl.log(row_sum)
    else:
        result = -float("inf")
    tl.store(output_ptr + row, result)


def fused_dsa_row_logsumexp(
    scores: torch.Tensor,
    row_lengths: torch.Tensor,
) -> torch.Tensor:
    """Reduce each valid FP32 score prefix to one detached FP32 LSE value."""

    if scores.ndim != 2:
        raise ValueError("DSA score workspace must be rank 2")
    rows, columns = scores.shape
    if columns <= 0:
        raise ValueError("DSA score workspace must have a non-empty column domain")
    if scores.dtype != torch.float32:
        raise TypeError("DSA score workspace must use float32")
    if row_lengths.shape != (rows,) or row_lengths.dtype != torch.int32:
        raise ValueError("DSA row lengths must be a matching int32 vector")
    if not scores.is_cuda or not row_lengths.is_cuda:
        raise ValueError("fused DSA row LSE requires CUDA tensors")
    if scores.device != row_lengths.device:
        raise ValueError("DSA score workspace and row lengths must share one device")
    if scores.stride(1) != 1 or not row_lengths.is_contiguous():
        raise ValueError("fused DSA row LSE requires contiguous rows and lengths")

    output = torch.empty((rows,), dtype=torch.float32, device=scores.device)
    if rows:
        block_size = min(1024, triton.next_power_of_2(columns))
        _dsa_row_logsumexp_kernel[(rows,)](
            scores,
            row_lengths,
            output,
            columns,
            scores.stride(0),
            BLOCK_SIZE=block_size,
            num_warps=8 if block_size >= 512 else 4,
        )
    return output


__all__ = ["fused_dsa_row_logsumexp"]
