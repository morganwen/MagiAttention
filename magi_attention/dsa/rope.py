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

"""Self-contained YaRN rotary embedding for compressed positions.

Follows the Megatron dsv4 "pure rotation" contract: the cos/sin table is
built with mscale fixed to 1.0 so no concentration factor enters the cache,
and compressed entry ``i`` takes the phase of original position
``i * ratio`` (strided slicing of a full-resolution table).

The rotation convention is GPT-NeoX style rotate-half, matching Megatron's
``apply_rotary_pos_emb`` unfused path. Verified against the Megatron
implementation in the parity tests; an externally built table can be
injected wherever exact sharing is required.
"""

import math

import torch

from .config import MagiDSAV4YarnConfig


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, orig_max_pos: int
) -> tuple:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, orig_max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, orig_max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, device) -> torch.Tensor:
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def build_yarn_freqs(
    rope_dim: int,
    total_len: int,
    yarn: MagiDSAV4YarnConfig,
    device: torch.device,
) -> torch.Tensor:
    """Full-resolution rotary phase table ``[total_len, rope_dim // 2]`` in FP32.

    Interpolated (yarn) and extrapolated frequencies are mixed by the linear
    ramp over the correction range; mscale is intentionally NOT applied.
    """
    half = rope_dim // 2
    inv_freq_extra = 1.0 / (
        yarn.rotary_base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    inv_freq_inter = inv_freq_extra / yarn.scaling_factor

    low, high = _yarn_find_correction_range(
        yarn.beta_fast,
        yarn.beta_slow,
        rope_dim,
        yarn.rotary_base,
        yarn.original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, half, device)
    inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask

    positions = torch.arange(total_len, dtype=torch.float32, device=device)
    return torch.outer(positions, inv_freq)  # [total_len, half]


def build_standard_freqs(
    rope_dim: int, total_len: int, rotary_base: float, device: torch.device
) -> torch.Tensor:
    """Plain RoPE phase table ``[total_len, rope_dim // 2]`` in FP32."""
    inv_freq = 1.0 / (
        rotary_base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    positions = torch.arange(total_len, dtype=torch.float32, device=device)
    return torch.outer(positions, inv_freq)


def apply_rope_last_dims(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """Rotate the last ``rope_dim`` dims of ``x`` by ``freqs``; keep the rest.

    MLA interleaved convention, matching Megatron's ``_apply_unfused_rope``
    with ``mla_rotary_interleaved=True`` and
    ``mla_output_remove_interleaving=True``: adjacent positional dims form
    complex pairs ``(x0, x1), (x2, x3), ...``; each pair is rotated in place,
    so ``out[2i] = x[2i] cos_i - x[2i+1] sin_i`` and
    ``out[2i+1] = x[2i+1] cos_i + x[2i] sin_i``.

    Numerics follow the reference: cos/sin are cast to the compute dtype
    before the multiply.

    ``x`` is ``[seq, ..., head_dim]`` with ``head_dim >= rope_dim``;
    ``freqs`` is ``[seq, rope_dim // 2]``.
    """
    seq = x.shape[0]
    assert freqs.shape[0] == seq, f"freqs length {freqs.shape[0]} != seq {seq}"
    nope = x[..., : x.shape[-1] - rope_dim]
    pos = x[..., x.shape[-1] - rope_dim :]

    shape = [seq] + [1] * (x.dim() - 2) + [rope_dim // 2]
    cos = freqs.cos().view(shape).to(x.dtype)
    sin = freqs.sin().view(shape).to(x.dtype)

    even = pos[..., 0::2]
    odd = pos[..., 1::2]
    out_even = even * cos - odd * sin
    out_odd = odd * cos + even * sin
    pos_out = torch.stack([out_even, out_odd], dim=-1).flatten(start_dim=-2)
    return torch.cat([nope, pos_out], dim=-1)


def strided_freqs_for_compressed(
    freqs_full: torch.Tensor, n_compressed: int, ratio: int
) -> torch.Tensor:
    """Compressed entry ``i`` takes the phase of original position ``i * ratio``."""
    return freqs_full[: n_compressed * ratio : ratio][:n_compressed]
