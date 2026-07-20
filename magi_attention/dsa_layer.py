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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.dsa_types import MagiDSAInput

if TYPE_CHECKING:
    from magi_attention.dsa_runtime_mgr import DsaExecutionHandle, MagiDSARuntimeMgr


def _yarn_inverse_frequencies(config: MagiDSAConfig) -> torch.Tensor:
    frequencies = 1.0 / (
        config.compress_rope_theta
        ** (torch.arange(0, config.rope_dim, 2, dtype=torch.float32) / config.rope_dim)
    )
    if config.original_seq_len <= 0:
        return frequencies

    def correction_dim(rotations: int) -> float:
        return (
            config.rope_dim
            * math.log(config.original_seq_len / (rotations * 2 * math.pi))
            / (2 * math.log(config.compress_rope_theta))
        )

    low = max(math.floor(correction_dim(config.rope_beta_fast)), 0)
    high = float(
        min(math.ceil(correction_dim(config.rope_beta_slow)), config.rope_dim - 1)
    )
    if low == high:
        high += 0.001
    ramp = (torch.arange(config.rope_dim // 2, dtype=torch.float32) - low) / (
        high - low
    )
    smooth = 1.0 - ramp.clamp(0.0, 1.0)
    return frequencies / config.rope_factor * (1.0 - smooth) + frequencies * smooth


def apply_dsa_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply sample-relative RoPE to the final dimensions without an in-place view."""

    if tensor.shape[0] != positions.numel():
        raise ValueError("positions must contain one value per tensor row")
    if tensor.shape[-1] < rope_dim or rope_dim % 2:
        raise ValueError("invalid RoPE dimension")
    prefix, rotary = tensor.split((tensor.shape[-1] - rope_dim, rope_dim), dim=-1)
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequencies.to(
        device=tensor.device
    ).unsqueeze(0)
    if inverse:
        angles = -angles
    cos = angles.cos()
    sin = angles.sin()
    for _ in range(tensor.ndim - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    even = rotary.float()[..., 0::2]
    odd = rotary.float()[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat(
        (prefix, rotated.flatten(-2).to(tensor.dtype)), dim=-1
    ).contiguous()


class DsaRMSNorm(nn.Module):
    """DeepSeek compressor RMSNorm with an FP32 learnable scale."""

    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.dimension = dimension
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension, dtype=torch.float32))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        dtype = tensor.dtype
        value = tensor.float()
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (value * self.weight).to(dtype)


class DsaCompressor(nn.Module):
    """Trainable CSA overlap or HCA non-overlap compressor."""

    def __init__(
        self, config: MagiDSAConfig, output_dim: int, *, overlap: bool
    ) -> None:
        super().__init__()
        if overlap and config.ratio != 4:
            raise ValueError("only ratio=4 CSA uses overlap compression")
        self.config = config
        self.output_dim = output_dim
        self.overlap = overlap
        branches = 2 if overlap else 1
        self.wkv = nn.Linear(
            config.hidden_size, branches * output_dim, bias=False, dtype=torch.float32
        )
        self.wgate = nn.Linear(
            config.hidden_size, branches * output_dim, bias=False, dtype=torch.float32
        )
        self.ape = nn.Parameter(
            torch.zeros(config.ratio, branches * output_dim, dtype=torch.float32)
        )
        self.norm = DsaRMSNorm(output_dim, config.norm_eps)
        self.register_buffer(
            "inverse_frequencies", _yarn_inverse_frequencies(config), persistent=False
        )
        self.reset_parameters()

    @property
    def support(self) -> int:
        return self.config.ratio * (2 if self.overlap else 1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wgate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ape)
        nn.init.ones_(self.norm.weight)

    def forward(
        self,
        packed_x: torch.Tensor,
        valid_rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if packed_x.ndim != 3 or packed_x.shape[1:] != (
            self.support,
            self.config.hidden_size,
        ):
            raise ValueError("packed_x has an invalid compressor shape")
        if valid_rows.shape != packed_x.shape[:2] or valid_rows.dtype != torch.bool:
            raise ValueError("valid_rows must be a bool mask over packed_x rows")
        if positions.shape != (packed_x.shape[0],):
            raise ValueError("positions must contain one compressed-block position")
        dtype = packed_x.dtype
        projected_kv = F.linear(packed_x.float(), self.wkv.weight)
        projected_gate = F.linear(packed_x.float(), self.wgate.weight)
        ratio = self.config.ratio
        if self.overlap:
            previous_kv = projected_kv[:, :ratio, : self.output_dim]
            current_kv = projected_kv[:, ratio:, self.output_dim :]
            previous_gate = (
                projected_gate[:, :ratio, : self.output_dim]
                + self.ape[None, :, : self.output_dim]
            )
            current_gate = (
                projected_gate[:, ratio:, self.output_dim :]
                + self.ape[None, :, self.output_dim :]
            )
            values = torch.cat((previous_kv, current_kv), dim=1)
            logits = torch.cat((previous_gate, current_gate), dim=1)
        else:
            values = projected_kv
            logits = projected_gate + self.ape.unsqueeze(0)
        logits = logits.masked_fill(~valid_rows.unsqueeze(-1), float("-inf"))
        compressed = (values * logits.softmax(dim=1)).sum(dim=1).to(dtype)
        compressed = self.norm(compressed)
        return apply_dsa_rope(
            compressed,
            positions,
            self.inverse_frequencies,
            self.config.rope_dim,
        )


class DsaIndexer(nn.Module):
    """CSA Indexer projections and its independently parameterized compressor."""

    def __init__(self, config: MagiDSAConfig) -> None:
        super().__init__()
        if config.ratio != 4:
            raise ValueError("DsaIndexer is only valid for ratio=4")
        self.config = config
        self.q_proj = nn.Linear(
            config.q_lora_rank,
            config.indexer_heads * config.indexer_head_dim,
            bias=False,
            dtype=torch.float32,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size,
            config.indexer_heads,
            bias=False,
            dtype=torch.float32,
        )
        self.compressor = DsaCompressor(config, config.indexer_head_dim, overlap=True)
        self.register_buffer(
            "inverse_frequencies", _yarn_inverse_frequencies(config), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.weights_proj.weight, mean=0.0, std=0.02)

    def project_queries(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        *,
        detach_trunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_x = x.detach() if detach_trunk else x
        branch_qr = qr.detach() if detach_trunk else qr
        q = F.linear(branch_qr.float(), self.q_proj.weight).view(
            qr.shape[0], self.config.indexer_heads, self.config.indexer_head_dim
        )
        q = q.to(qr.dtype)
        q = apply_dsa_rope(
            q,
            positions,
            self.inverse_frequencies,
            self.config.rope_dim,
        )
        weight_scale = (
            self.config.indexer_head_dim**-0.5 * self.config.indexer_heads**-0.5
        )
        weights = (
            F.linear(branch_x.float(), self.weights_proj.weight) * weight_scale
        ).to(x.dtype)
        return q, weights


class MagiDSALayer(nn.Module):
    """Model-side owner of every trainable Compressor and Indexer parameter."""

    def __init__(self, config: MagiDSAConfig) -> None:
        super().__init__()
        self.config = config
        self.compressor = (
            DsaCompressor(config, config.head_dim, overlap=config.ratio == 4)
            if config.has_compressor
            else None
        )
        self.indexer = DsaIndexer(config) if config.has_indexer else None

    def forward(
        self,
        dsa_input: MagiDSAInput,
        runtime: MagiDSARuntimeMgr,
        handle: DsaExecutionHandle,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = runtime.calc_dsa(self, dsa_input, handle)
        return result.output, result.kl


__all__ = [
    "DsaCompressor",
    "DsaIndexer",
    "DsaRMSNorm",
    "MagiDSALayer",
    "apply_dsa_rope",
]
