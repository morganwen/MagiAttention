# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Fused DeepSeek-V4 GPT-J RoPE and Indexer rotation.

The Megatron kernel rotates the trailing dimensions in place from a prebuilt
cos/sin table. Magi keeps the same token/head program mapping and adjacent-pair
math, but writes to a private output and evaluates the small YaRN frequency
vector in the kernel. The Indexer variant follows RoPE with a normalized
Walsh-Hadamard transform in registers, avoiding one launch per butterfly stage.
Both variants preserve reentrant autograd without mutating their inputs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
    ],
    key=["head_dim", "rope_dim", "head_count"],
)
@triton.jit
def _fused_dsa_rope_kernel(
    input_ptr,
    output_ptr,
    positions_ptr,
    inverse_frequencies_ptr,
    stride_input_token,
    stride_input_head,
    stride_input_dim,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    head_count: tl.constexpr,
    block_prefix: tl.constexpr,
    block_pairs: tl.constexpr,
    inverse: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_index = tl.program_id(axis=0)
    head_block = tl.program_id(axis=1)
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < head_count

    input_base = (
        input_ptr
        + token_index * stride_input_token
        + head_offsets[:, None] * stride_input_head
    )
    output_base = (
        output_ptr
        + token_index * stride_output_token
        + head_offsets[:, None] * stride_output_head
    )

    nope_dim: tl.constexpr = head_dim - rope_dim
    prefix_offsets = tl.arange(0, block_prefix)[None, :]
    prefix_mask = head_mask[:, None] & (prefix_offsets < nope_dim)
    prefix = tl.load(
        input_base + prefix_offsets * stride_input_dim,
        mask=prefix_mask,
        other=0.0,
    )
    tl.store(
        output_base + prefix_offsets * stride_output_dim,
        prefix,
        mask=prefix_mask,
    )

    pair_offsets = tl.arange(0, block_pairs)[None, :]
    pair_mask = head_mask[:, None] & (pair_offsets < rope_dim // 2)
    rotary_begin: tl.constexpr = head_dim - rope_dim
    even_offsets = rotary_begin + pair_offsets * 2
    odd_offsets = even_offsets + 1
    even = tl.load(
        input_base + even_offsets * stride_input_dim,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        input_base + odd_offsets * stride_input_dim,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    position = tl.load(positions_ptr + token_index).to(tl.float32)
    frequency = tl.load(
        inverse_frequencies_ptr + pair_offsets,
        mask=pair_offsets < rope_dim // 2,
        other=0.0,
    ).to(tl.float32)
    angle = position * frequency
    sine = tl.sin(angle)
    if inverse:
        sine = -sine
    cosine = tl.cos(angle)
    rotated_even = even * cosine - odd * sine
    rotated_odd = odd * cosine + even * sine
    tl.store(
        output_base + even_offsets * stride_output_dim,
        rotated_even,
        mask=pair_mask,
    )
    tl.store(
        output_base + odd_offsets * stride_output_dim,
        rotated_odd,
        mask=pair_mask,
    )


@triton.jit
def _register_rope(
    values,
    position,
    inverse_frequencies_ptr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_heads: tl.constexpr,
    inverse: tl.constexpr,
):
    pairs = tl.reshape(values, (block_heads, head_dim // 2, 2))
    even, odd = tl.split(pairs)
    pair_offsets = tl.arange(0, head_dim // 2)[None, :]
    rotary_pair_begin: tl.constexpr = (head_dim - rope_dim) // 2
    frequency_offsets = pair_offsets - rotary_pair_begin
    rotary_mask = pair_offsets >= rotary_pair_begin
    frequency = tl.load(
        inverse_frequencies_ptr + frequency_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    angle = position * frequency
    sine = tl.sin(angle)
    if inverse:
        sine = -sine
    cosine = tl.cos(angle)
    rotated_even = even * cosine - odd * sine
    rotated_odd = odd * cosine + even * sine
    output_even = tl.where(rotary_mask, rotated_even, even)
    output_odd = tl.where(rotary_mask, rotated_odd, odd)
    return tl.reshape(tl.join(output_even, output_odd), (block_heads, head_dim))


@triton.jit
def _register_normalized_hadamard(
    values,
    head_dim: tl.constexpr,
    log2_head_dim: tl.constexpr,
    block_heads: tl.constexpr,
):
    for stage in tl.static_range(0, log2_head_dim):
        grouped = tl.reshape(
            values,
            (
                block_heads,
                head_dim // (2 * (1 << stage)),
                2,
                1 << stage,
            ),
        )
        transposed = tl.permute(grouped, (0, 1, 3, 2))
        left, right = tl.split(transposed)
        combined = tl.join(left + right, left - right)
        values = tl.reshape(
            tl.permute(combined, (0, 1, 3, 2)),
            (block_heads, head_dim),
        )
    return values * (head_dim**-0.5)


@triton.jit
def _fused_dsa_rope_hadamard_kernel(
    input_ptr,
    output_ptr,
    positions_ptr,
    inverse_frequencies_ptr,
    stride_input_token,
    stride_input_head,
    stride_input_dim,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    head_count: tl.constexpr,
    log2_head_dim: tl.constexpr,
    inverse: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    ROUND_FP16: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_index = tl.program_id(axis=0)
    head_block = tl.program_id(axis=1)
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, head_dim)
    mask = (head_offsets[:, None] < head_count) & (dim_offsets[None, :] < head_dim)
    input_offsets = (
        token_index * stride_input_token
        + head_offsets[:, None] * stride_input_head
        + dim_offsets[None, :] * stride_input_dim
    )
    output_offsets = (
        token_index * stride_output_token
        + head_offsets[:, None] * stride_output_head
        + dim_offsets[None, :] * stride_output_dim
    )
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    if ROUND_BF16:
        values = values.to(tl.bfloat16).to(tl.float32)
    elif ROUND_FP16:
        values = values.to(tl.float16).to(tl.float32)
    position = tl.load(positions_ptr + token_index).to(tl.float32)

    if inverse:
        values = _register_normalized_hadamard(
            values,
            head_dim,
            log2_head_dim,
            BLOCK_H,
        )
        values = _register_rope(
            values,
            position,
            inverse_frequencies_ptr,
            head_dim,
            rope_dim,
            BLOCK_H,
            True,
        )
    else:
        values = _register_rope(
            values,
            position,
            inverse_frequencies_ptr,
            head_dim,
            rope_dim,
            BLOCK_H,
            False,
        )
        values = _register_normalized_hadamard(
            values,
            head_dim,
            log2_head_dim,
            BLOCK_H,
        )
    if ROUND_BF16:
        values = values.to(tl.bfloat16).to(tl.float32)
    elif ROUND_FP16:
        values = values.to(tl.float16).to(tl.float32)
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _launch_fused_dsa_rope(
    tensor: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool,
) -> None:
    if tensor.shape[0] == 0:
        return
    head_count = tensor.shape[1]
    head_dim = tensor.shape[2]
    block_prefix = triton.next_power_of_2(max(head_dim - rope_dim, 1))
    block_pairs = triton.next_power_of_2(rope_dim // 2)
    grid = lambda meta: (tensor.shape[0], triton.cdiv(head_count, meta["BLOCK_H"]))
    _fused_dsa_rope_kernel[grid](
        tensor,
        output,
        positions,
        inverse_frequencies,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim=head_dim,
        rope_dim=rope_dim,
        head_count=head_count,
        block_prefix=block_prefix,
        block_pairs=block_pairs,
        inverse=inverse,
    )


def _launch_fused_dsa_rope_hadamard(
    tensor: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool,
    round_dtype: torch.dtype | None = None,
) -> None:
    if tensor.shape[0] == 0:
        return
    head_count = tensor.shape[1]
    head_dim = tensor.shape[2]
    block_heads = min(8, triton.next_power_of_2(head_count))
    grid = (tensor.shape[0], triton.cdiv(head_count, block_heads))
    _fused_dsa_rope_hadamard_kernel[grid](
        tensor,
        output,
        positions,
        inverse_frequencies,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim=head_dim,
        rope_dim=rope_dim,
        head_count=head_count,
        log2_head_dim=head_dim.bit_length() - 1,
        inverse=inverse,
        ROUND_BF16=round_dtype == torch.bfloat16,
        ROUND_FP16=round_dtype == torch.float16,
        BLOCK_H=block_heads,
        num_warps=8 if block_heads == 8 else 4,
    )


class _FusedDSARoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
        inverse: bool,
    ) -> torch.Tensor:
        output = torch.empty_like(tensor, memory_format=torch.contiguous_format)
        _launch_fused_dsa_rope(
            tensor,
            output,
            positions,
            inverse_frequencies,
            rope_dim,
            inverse=inverse,
        )
        ctx.save_for_backward(positions, inverse_frequencies)
        ctx.rope_dim = rope_dim
        ctx.inverse = inverse
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        positions, inverse_frequencies = ctx.saved_tensors
        grad_input = torch.empty_like(
            grad_output, memory_format=torch.contiguous_format
        )
        _launch_fused_dsa_rope(
            grad_output,
            grad_input,
            positions,
            inverse_frequencies,
            ctx.rope_dim,
            inverse=not ctx.inverse,
        )
        return grad_input, None, None, None, None


class _FusedDSARoPEHadamard(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.empty_like(
            tensor,
            dtype=output_dtype,
            memory_format=torch.contiguous_format,
        )
        _launch_fused_dsa_rope_hadamard(
            tensor,
            output,
            positions,
            inverse_frequencies,
            rope_dim,
            inverse=False,
            round_dtype=output_dtype if tensor.dtype != output_dtype else None,
        )
        ctx.save_for_backward(positions, inverse_frequencies)
        ctx.rope_dim = rope_dim
        ctx.input_dtype = tensor.dtype
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        positions, inverse_frequencies = ctx.saved_tensors
        grad_input = torch.empty_like(
            grad_output,
            dtype=ctx.input_dtype,
            memory_format=torch.contiguous_format,
        )
        _launch_fused_dsa_rope_hadamard(
            grad_output,
            grad_input,
            positions,
            inverse_frequencies,
            ctx.rope_dim,
            inverse=True,
            round_dtype=(
                grad_output.dtype if grad_output.dtype != ctx.input_dtype else None
            ),
        )
        return grad_input, None, None, None, None


def fused_dsa_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply trailing GPT-J RoPE or its inverse into private output storage."""

    _validate_fused_dsa_rope_inputs(
        tensor,
        positions,
        inverse_frequencies,
        rope_dim,
    )
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be a bool")
    return _FusedDSARoPE.apply(
        tensor,
        positions,
        inverse_frequencies,
        rope_dim,
        inverse,
    )


def fused_dsa_rope_hadamard(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply Indexer RoPE then normalized Hadamard in one CUDA launch."""

    _validate_fused_dsa_rope_inputs(
        tensor,
        positions,
        inverse_frequencies,
        rope_dim,
    )
    head_dim = tensor.shape[-1]
    if head_dim & (head_dim - 1):
        raise ValueError("fused DSA Hadamard requires a power-of-two head dimension")
    resolved_output_dtype = tensor.dtype if output_dtype is None else output_dtype
    if resolved_output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("fused DSA RoPE output must use a floating-point dtype")
    if resolved_output_dtype != tensor.dtype and not (
        tensor.dtype == torch.float32
        and resolved_output_dtype in (torch.float16, torch.bfloat16)
    ):
        raise TypeError(
            "mixed-dtype fused DSA RoPE requires FP32 input and FP16/BF16 output"
        )
    return _FusedDSARoPEHadamard.apply(
        tensor,
        positions,
        inverse_frequencies,
        rope_dim,
        resolved_output_dtype,
    )


def _validate_fused_dsa_rope_inputs(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
) -> None:
    if tensor.ndim != 3:
        raise ValueError("fused DSA RoPE expects [tokens, heads, head_dim]")
    if positions.shape != (tensor.shape[0],):
        raise ValueError("positions must contain one value per tensor row")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must use int32 or int64")
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > tensor.shape[-1]:
        raise ValueError("invalid RoPE dimension")
    if inverse_frequencies.shape != (rope_dim // 2,):
        raise ValueError("inverse frequencies have an invalid shape")
    if inverse_frequencies.dtype != torch.float32:
        raise TypeError("inverse frequencies must use FP32")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("fused DSA RoPE requires a floating-point tensor")
    if not tensor.is_cuda or not positions.is_cuda or not inverse_frequencies.is_cuda:
        raise ValueError("fused DSA RoPE requires CUDA tensors")
    if not (tensor.device == positions.device == inverse_frequencies.device):
        raise ValueError("fused DSA RoPE tensors must share one CUDA device")
    if tensor.stride(-1) != 1:
        raise ValueError("fused DSA RoPE requires a contiguous head dimension")
    if not positions.is_contiguous() or not inverse_frequencies.is_contiguous():
        raise ValueError("positions and inverse frequencies must be contiguous")


__all__ = ["fused_dsa_rope", "fused_dsa_rope_hadamard"]
