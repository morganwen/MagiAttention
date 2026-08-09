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

"""Fused integer index construction for the DeepSeek-V4 CSA warm path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _finalize_dsa_topk_kernel(
    local_ids_ptr,
    seq_lens_ptr,
    block_offsets_ptr,
    global_ids_ptr,
    lengths_ptr,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    input_mask = columns < INPUT_WIDTH
    output_mask = columns < OUTPUT_WIDTH
    sequence_length = tl.load(seq_lens_ptr + row)
    length = tl.minimum(tl.maximum(sequence_length, 0), INPUT_WIDTH)
    local_ids = tl.load(
        local_ids_ptr + row * INPUT_WIDTH + columns,
        mask=input_mask,
        other=-1,
    )
    block_offset = tl.load(block_offsets_ptr + row)
    valid = output_mask & (columns < length) & (local_ids >= 0)
    global_ids = tl.where(valid, local_ids + block_offset, -1)
    tl.store(
        global_ids_ptr + row * OUTPUT_WIDTH + columns,
        global_ids,
        mask=output_mask,
    )
    tl.store(lengths_ptr + row, length)


@triton.jit
def _build_csa_indices_kernel(
    global_ids_ptr,
    topk_lengths_ptr,
    attention_map_ptr,
    indexer_map_ptr,
    window_base_ptr,
    window_lengths_ptr,
    attention_indices_ptr,
    attention_lengths_ptr,
    indexer_indices_ptr,
    attention_compressed_indices_ptr,
    raw_bank_rows,
    MAP_ROWS: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    WINDOW_WIDTH: tl.constexpr,
    ATTENTION_WIDTH: tl.constexpr,
    HAS_COMPRESSED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    topk_mask = columns < TOPK_WIDTH
    topk_length = tl.load(topk_lengths_ptr + row)
    effective_topk = tl.minimum(tl.maximum(topk_length, 0), TOPK_WIDTH)
    global_ids = tl.load(
        global_ids_ptr + row * TOPK_WIDTH + columns,
        mask=topk_mask,
        other=-1,
    )
    global_valid = (
        topk_mask
        & (columns < effective_topk)
        & (global_ids >= 0)
        & (global_ids < MAP_ROWS)
    )
    if HAS_COMPRESSED:
        attention_rows = tl.load(
            attention_map_ptr + global_ids,
            mask=global_valid,
            other=-1,
        )
        indexer_rows = tl.load(
            indexer_map_ptr + global_ids,
            mask=global_valid,
            other=-1,
        )
    else:
        attention_rows = tl.full((BLOCK,), -1, tl.int32)
        indexer_rows = tl.full((BLOCK,), -1, tl.int32)
    attention_rows = tl.where(global_valid, attention_rows, -1)
    indexer_rows = tl.where(global_valid, indexer_rows, -1)
    tl.store(
        indexer_indices_ptr + row * TOPK_WIDTH + columns,
        indexer_rows,
        mask=topk_mask,
    )
    tl.store(
        attention_compressed_indices_ptr + row * TOPK_WIDTH + columns,
        attention_rows,
        mask=topk_mask,
    )

    compressed_attention = tl.where(
        attention_rows >= 0,
        attention_rows + raw_bank_rows,
        -1,
    )
    # A raw window is a contiguous run in the WINDOW_KV consumer bank, so its
    # rows are computed from one base instead of read out of a resident table.
    window_columns = columns - TOPK_WIDTH
    window_length = tl.load(window_lengths_ptr + row)
    window_base = tl.load(window_base_ptr + row)
    window_valid = (
        (columns >= TOPK_WIDTH)
        & (columns < ATTENTION_WIDTH)
        & (window_columns < window_length)
        & (window_base >= 0)
    )
    window = tl.where(window_valid, window_base + window_columns, -1)
    attention = tl.where(topk_mask, compressed_attention, window)
    tl.store(
        attention_indices_ptr + row * ATTENTION_WIDTH + columns,
        attention,
        mask=columns < ATTENTION_WIDTH,
    )
    tl.store(attention_lengths_ptr + row, effective_topk + window_length)


def _validate_int32_cuda(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use int32")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA")


def finalize_dsa_topk(
    seq_lens: torch.Tensor,
    backend_local_ids: torch.Tensor,
    sample_block_offsets: torch.Tensor,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply backend-native global offsets and padding in one CUDA launch."""

    if backend_local_ids.ndim != 2:
        raise ValueError("backend Top-K IDs must be rank 2")
    rows, input_width = backend_local_ids.shape
    if seq_lens.shape != (rows,) or sample_block_offsets.shape != (rows,):
        raise ValueError("Top-K metadata has an invalid shape")
    if output_width < input_width or output_width <= 0 or output_width > 1024:
        raise ValueError("invalid finalized Top-K width")
    for name, tensor in (
        ("backend_local_ids", backend_local_ids),
        ("seq_lens", seq_lens),
        ("sample_block_offsets", sample_block_offsets),
    ):
        _validate_int32_cuda(name, tensor)
    if not (backend_local_ids.device == seq_lens.device == sample_block_offsets.device):
        raise ValueError("Top-K tensors must share one CUDA device")
    global_ids = torch.empty(
        (rows, output_width), dtype=torch.int32, device=backend_local_ids.device
    )
    lengths = torch.empty((rows,), dtype=torch.int32, device=backend_local_ids.device)
    if rows:
        block = triton.next_power_of_2(output_width)
        _finalize_dsa_topk_kernel[(rows,)](
            backend_local_ids,
            seq_lens,
            sample_block_offsets,
            global_ids,
            lengths,
            INPUT_WIDTH=input_width,
            OUTPUT_WIDTH=output_width,
            BLOCK=block,
            num_warps=8 if block >= 512 else 4,
        )
    return global_ids, lengths


def build_csa_index_tensors(
    topk_global_ids: torch.Tensor,
    topk_lengths: torch.Tensor,
    attention_global_to_consumer: torch.Tensor,
    indexer_global_to_consumer: torch.Tensor,
    window_base: torch.Tensor,
    window_lengths: torch.Tensor,
    window_width: int,
    raw_bank_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build FlashMLA and dual selected-KL indices in one CUDA launch."""

    if topk_global_ids.ndim != 2:
        raise ValueError("CSA Top-K ids must be rank 2")
    rows, topk_width = topk_global_ids.shape
    if window_base.shape != (rows,):
        raise ValueError("CSA window base has an invalid shape")
    if window_width <= 0:
        raise ValueError("CSA window width must be positive")
    attention_width = topk_width + window_width
    if attention_width <= 0 or attention_width > 2048:
        raise ValueError("CSA attention index width must fit one 2048-column block")
    if topk_lengths.shape != (rows,) or window_lengths.shape != (rows,):
        raise ValueError("CSA length metadata has an invalid shape")
    if attention_global_to_consumer.ndim != 1:
        raise ValueError("attention global-to-consumer map must be rank 1")
    if indexer_global_to_consumer.shape != attention_global_to_consumer.shape:
        raise ValueError("CSA KI/KV maps must span the same global ID domain")
    if raw_bank_rows < 0:
        raise ValueError("raw KV bank row count must be non-negative")
    tensors = (
        ("topk_global_ids", topk_global_ids),
        ("topk_lengths", topk_lengths),
        ("attention_global_to_consumer", attention_global_to_consumer),
        ("indexer_global_to_consumer", indexer_global_to_consumer),
        ("window_base", window_base),
        ("window_lengths", window_lengths),
    )
    for name, tensor in tensors:
        _validate_int32_cuda(name, tensor)
    devices = {tensor.device for _, tensor in tensors}
    if len(devices) != 1:
        raise ValueError("CSA index tensors must share one CUDA device")

    device = topk_global_ids.device
    attention_indices = torch.empty(
        (rows, attention_width), dtype=torch.int32, device=device
    )
    attention_lengths = torch.empty((rows,), dtype=torch.int32, device=device)
    indexer_indices = torch.empty_like(topk_global_ids)
    attention_compressed_indices = torch.empty_like(topk_global_ids)
    if rows:
        block = triton.next_power_of_2(attention_width)
        map_rows = attention_global_to_consumer.numel()
        _build_csa_indices_kernel[(rows,)](
            topk_global_ids,
            topk_lengths,
            attention_global_to_consumer,
            indexer_global_to_consumer,
            window_base,
            window_lengths,
            attention_indices,
            attention_lengths,
            indexer_indices,
            attention_compressed_indices,
            raw_bank_rows,
            MAP_ROWS=map_rows,
            TOPK_WIDTH=topk_width,
            WINDOW_WIDTH=window_width,
            ATTENTION_WIDTH=attention_width,
            HAS_COMPRESSED=map_rows > 0,
            BLOCK=block,
            num_warps=8 if block >= 512 else 4,
        )
    return (
        attention_indices,
        attention_lengths,
        indexer_indices,
        attention_compressed_indices,
    )


__all__ = [
    "build_csa_index_tensors",
    "finalize_dsa_topk",
]
