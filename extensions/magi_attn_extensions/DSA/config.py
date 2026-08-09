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

DsaRatio = Literal[4, 128]

DSV4_PRO_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT = 8
_DSV4_PRO_CSA_HCA_PAIR: tuple[DsaRatio, DsaRatio] = (4, 128)
DSV4_PRO_MAIN_COMPRESS_RATIOS: tuple[DsaRatio, ...] = (
    128,
    128,
    *tuple(ratio for _ in range(29) for ratio in _DSV4_PRO_CSA_HCA_PAIR),
    4,
)
# The separate MTP block is window-only in the official model. The main stack
# this extension implements contains no window-only layer, so ``DsaRatio`` has
# no 0 member and the MTP ratio is carried as a plain int for provenance only.
DSV4_PRO_MTP_COMPRESS_RATIO: int = 0


@dataclass(frozen=True)
class DsaStructuralLayoutConfig:
    """Native Magi packed-global dispatch configuration for DSV4-Pro."""

    chunk_size: int = 512
    min_chunks_per_rank: int = 16
    uneven_shard: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.min_chunks_per_rank <= 0:
            raise ValueError("min_chunks_per_rank must be positive")
        if not self.uneven_shard:
            raise ValueError("DSV4-Pro structural dispatch requires uneven_shard=True")


@dataclass(frozen=True)
class MagiDSAConfig:
    """Frozen DeepSeek-V4 attention dimensions and execution constants.

    Tests may construct dimension-reduced instances, but release/profile entry points
    call :meth:`validate_release_contract` before allocating any workload tensors.
    """

    ratio: DsaRatio
    hidden_size: int = 7168
    q_lora_rank: int = 1536
    num_query_heads: int = 128
    head_dim: int = 512
    rope_dim: int = 64
    indexer_heads: int = 64
    indexer_head_dim: int = 128
    indexer_topk: int = 1024
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
    linear_weight_dtype: Literal["bfloat16"] = "bfloat16"
    accumulator_dtype: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        if self.ratio not in (4, 128):
            raise ValueError("ratio must be either 4 (CSA) or 128 (HCA)")
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
        if self.linear_weight_dtype != "bfloat16":
            raise ValueError("DSV4-Pro linear weights must use BF16")
        if self.accumulator_dtype != "float32":
            raise ValueError("DSV4-Pro reductions must accumulate in FP32")

    @property
    def has_indexer(self) -> bool:
        return self.ratio == 4

    @property
    def compressor_support(self) -> int:
        """Rows one compressed block reads: CSA reads B and A, HCA reads A."""

        return self.ratio * (2 if self.ratio == 4 else 1)

    @property
    def attention_topk_capacity_per_sample(self) -> int:
        if self.ratio == 4:
            return self.window_size + self.indexer_topk
        return self.window_size

    def validate_release_contract(self) -> None:
        expected = {
            "hidden_size": 7168,
            "q_lora_rank": 1536,
            "num_query_heads": 128,
            "head_dim": 512,
            "rope_dim": 64,
            "indexer_heads": 64,
            "indexer_head_dim": 128,
            "indexer_topk": 1024,
            "window_size": 128,
            "norm_eps": 1e-6,
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


@dataclass(frozen=True)
class MagiDSAProModelSpec:
    """Official DeepSeek-V4-Pro main-stack attention contract.

    The 61-layer Transformer main stack contains only HCA and CSA layers.  The
    separate MTP block is exposed explicitly but is not part of
    ``main_compress_ratios``.
    """

    source_revision: str = DSV4_PRO_REVISION
    main_compress_ratios: tuple[DsaRatio, ...] = DSV4_PRO_MAIN_COMPRESS_RATIOS
    mtp_compress_ratio: DsaRatio = DSV4_PRO_MTP_COMPRESS_RATIO
    activation_dtype: Literal["bfloat16"] = "bfloat16"
    linear_weight_dtype: Literal["bfloat16"] = "bfloat16"
    main_gradient_dtype: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        if self.source_revision != DSV4_PRO_REVISION:
            raise ValueError(
                "DeepSeek-V4-Pro source revision must match the frozen contract"
            )
        if self.main_compress_ratios != DSV4_PRO_MAIN_COMPRESS_RATIOS:
            raise ValueError(
                "DeepSeek-V4-Pro main attention schedule must contain the official "
                "61-layer ratio sequence"
            )
        if self.mtp_compress_ratio != DSV4_PRO_MTP_COMPRESS_RATIO:
            raise ValueError("DeepSeek-V4-Pro MTP attention ratio must be zero")
        if self.activation_dtype != "bfloat16":
            raise ValueError("DeepSeek-V4-Pro training activation dtype must be BF16")
        if self.linear_weight_dtype != "bfloat16":
            raise ValueError("DeepSeek-V4-Pro linear weights must use BF16")
        if self.main_gradient_dtype != "float32":
            raise ValueError(
                "DeepSeek-V4-Pro outer training main gradients must use FP32"
            )

    @property
    def main_layer_count(self) -> int:
        return len(self.main_compress_ratios)

    @property
    def csa_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer_id
            for layer_id, ratio in enumerate(self.main_compress_ratios)
            if ratio == 4
        )

    @property
    def hca_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer_id
            for layer_id, ratio in enumerate(self.main_compress_ratios)
            if ratio == 128
        )

    def make_layer_config(self, layer_id: int) -> MagiDSAConfig:
        if not 0 <= layer_id < self.main_layer_count:
            raise IndexError(f"main layer_id must be in [0, {self.main_layer_count})")
        config = MagiDSAConfig(
            ratio=self.main_compress_ratios[layer_id],
            linear_weight_dtype=self.linear_weight_dtype,
        )
        config.validate_release_contract()
        return config


__all__ = [
    "DSV4_PRO_MAIN_COMPRESS_RATIOS",
    "DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT",
    "DSV4_PRO_MTP_COMPRESS_RATIO",
    "DSV4_PRO_REVISION",
    "DsaRatio",
    "DsaStructuralLayoutConfig",
    "MagiDSAConfig",
    "MagiDSAProModelSpec",
]
