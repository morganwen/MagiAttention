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

"""Gated pooling compressor for the Magi_DSA V4 runtime.

Faithful pure-PyTorch port of the Megatron dsv4 ``Compressor``
(megatron/core/transformer/experimental_attention_variant/csa.py, commit
c6449f0b2): overlapping compression (coff=2) for ratio 4, non-overlapping
(coff=1) for ratio 128; per-block gated softmax pooling in FP32 with a
learnable intra-block position embedding; RMSNorm; strided-position RoPE;
optional Hadamard rotation for the indexer copy.

Arbitrary-length rule: only ``seqlen // ratio`` full blocks are pooled;
trailing tokens have no compressed entry and rely on the sliding window.
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import MagiDSAV4Config
from .rope import apply_rope_last_dims, build_yarn_freqs, strided_freqs_for_compressed


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation, matching the Megatron/DeepSeek reference exactly."""
    assert x.dtype == torch.bfloat16, f"rotate_activation expects bf16, got {x.dtype}"
    from fast_hadamard_transform import hadamard_transform

    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


class DSAv4RMSNorm(nn.Module):
    """RMSNorm with TransformerEngine-matching numerics.

    The whole normalization runs in FP32 — including the gamma multiply —
    with a single cast back to the input dtype at the end. torch.nn.RMSNorm
    rounds differently and diverges from the TE kernel by one bf16 ulp,
    which the top-k selection then amplifies into discrete differences.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        y = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class DSAv4Compressor(nn.Module):
    """Compress every ``ratio`` tokens into one entry via learned gated pooling.

    Args:
        config: runtime config (carries hidden_size, rope_dim, yarn params).
        head_dim: output entry dim (kv_dim for the main stream,
            indexer_dim for the indexer copy).
        rotate: apply Hadamard rotation to outputs (indexer copy only).
    """

    def __init__(
        self,
        config: MagiDSAV4Config,
        head_dim: int,
        rotate: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        assert config.compress_ratio > 1, "compressor requires ratio 4 or 128"
        self.config = config
        self.ratio = config.compress_ratio
        self.head_dim = head_dim
        self.rope_dim = config.rope_dim
        self.overlap = self.ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate

        proj_out = self.coff * head_dim
        self.linear_wkv = nn.Linear(config.hidden_size, proj_out, bias=False, dtype=dtype)
        self.linear_wgate = nn.Linear(config.hidden_size, proj_out, bias=False, dtype=dtype)
        # Intra-block position embedding, kept in FP32 like the reference.
        self.ape = nn.Parameter(torch.empty(self.ratio, proj_out, dtype=torch.float32))
        nn.init.normal_(self.ape, mean=0.0, std=0.02)
        self.norm = DSAv4RMSNorm(head_dim, eps=config.norm_eps)

        self._freqs_cache: Optional[torch.Tensor] = None
        self._freqs_cache_len: int = 0

    def _compressed_freqs(self, n_compressed: int, device: torch.device) -> torch.Tensor:
        total = n_compressed * self.ratio
        if self._freqs_cache is None or self._freqs_cache_len < total:
            self._freqs_cache = build_yarn_freqs(
                self.rope_dim, total, self.config.yarn, device
            )
            self._freqs_cache_len = total
        return strided_freqs_for_compressed(self._freqs_cache, n_compressed, self.ratio)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        """[n, ratio, b, coff*d] -> [n, 2*ratio, b, d]; block i's first slots
        take block i-1's give-away halves, block 0 keeps ``fill_value``."""
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        out = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        out[:, ratio:] = tensor[:, :, :, d:]
        out[1:, :ratio] = tensor[:-1, :, :, :d]
        return out

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """SBHD path. ``x``: (sq, b, hidden) -> (sq // ratio, b, head_dim),
        or None when ``sq < ratio``."""
        sq = x.size(0)
        if sq < self.ratio:
            return None

        kv = self.linear_wkv(x)
        score = self.linear_wgate(x)

        cutoff = (sq // self.ratio) * self.ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]
        n_compressed = cutoff // self.ratio

        b_dim = kv.size(1)
        kv = kv.view(n_compressed, self.ratio, b_dim, -1)
        score = score.view(n_compressed, self.ratio, b_dim, -1)
        # FP32 APE promotes the scores to FP32, exactly as in the reference.
        score = score + self.ape.view(1, self.ratio, 1, -1)
        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0.0)
            score = self._overlap_transform(score, fill_value=float("-inf"))
        weights = torch.softmax(score, dim=1, dtype=torch.float32).to(kv.dtype)
        kv = (kv * weights).sum(dim=1)  # [n_compressed, b, head_dim]
        kv = self.norm(kv.to(x.dtype))

        freqs = self._compressed_freqs(n_compressed, x.device)
        kv = apply_rope_last_dims(kv, freqs, self.rope_dim)

        if self.rotate:
            kv = rotate_activation(kv)
        return kv
