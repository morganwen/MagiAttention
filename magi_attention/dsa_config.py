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

from dataclasses import dataclass
from typing import Literal

DsaRatio = Literal[0, 4, 128]
DsaPlanPolicy = Literal["sequential", "indexer_balanced"]


@dataclass(frozen=True)
class MagiDSAConfig:
    """Frozen DeepSeek-V4 attention dimensions and execution constants.

    Tests may construct dimension-reduced instances, but release/profile entry points
    call :meth:`validate_release_contract` before allocating any workload tensors.
    """

    ratio: DsaRatio
    hidden_size: int = 4096
    q_lora_rank: int = 1024
    num_query_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    indexer_heads: int = 64
    indexer_head_dim: int = 128
    indexer_topk: int = 512
    window_size: int = 128
    norm_eps: float = 1e-6
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    indexer_atom_size: int = 128
    indexer_score_block: int = 128
    indexer_topk_block: int = 256
    kl_loss_coeff: float = 1.0

    def __post_init__(self) -> None:
        if self.ratio not in (0, 4, 128):
            raise ValueError("ratio must be one of 0, 4, or 128")
        positive = {
            "hidden_size": self.hidden_size,
            "q_lora_rank": self.q_lora_rank,
            "num_query_heads": self.num_query_heads,
            "head_dim": self.head_dim,
            "rope_dim": self.rope_dim,
            "indexer_heads": self.indexer_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_topk": self.indexer_topk,
            "window_size": self.window_size,
            "indexer_atom_size": self.indexer_atom_size,
            "indexer_score_block": self.indexer_score_block,
            "indexer_topk_block": self.indexer_topk_block,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rope_dim > self.head_dim or self.rope_dim > self.indexer_head_dim:
            raise ValueError("rope_dim must fit both attention and indexer heads")
        if self.rope_dim % 2:
            raise ValueError("rope_dim must be even")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.kl_loss_coeff < 0:
            raise ValueError("kl_loss_coeff must be non-negative")

    @property
    def has_compressor(self) -> bool:
        return self.ratio != 0

    @property
    def has_indexer(self) -> bool:
        return self.ratio == 4

    @property
    def compressor_support(self) -> int:
        if self.ratio == 0:
            return 0
        return self.ratio * (2 if self.ratio == 4 else 1)

    @property
    def attention_topk_capacity_per_sample(self) -> int:
        if self.ratio == 4:
            return self.window_size + self.indexer_topk
        return self.window_size

    def validate_release_contract(self) -> None:
        expected = {
            "hidden_size": 4096,
            "q_lora_rank": 1024,
            "num_query_heads": 64,
            "head_dim": 512,
            "rope_dim": 64,
            "indexer_heads": 64,
            "indexer_head_dim": 128,
            "indexer_topk": 512,
            "window_size": 128,
            "compress_rope_theta": 160000.0,
            "original_seq_len": 65536,
            "rope_factor": 16.0,
            "rope_beta_fast": 32,
            "rope_beta_slow": 1,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError("release config mismatch: " + ", ".join(mismatches))


__all__ = ["DsaPlanPolicy", "DsaRatio", "MagiDSAConfig"]
