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

"""Configuration for the Magi_DSA V4 runtime.

Frozen per docs/magi_dsa_v4_design.md. Defaults follow the DeepSeek
V4-Flash recipe; validation rejects combinations outside the V1 contract.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

VALID_COMPRESS_RATIOS = (0, 4, 128)


@dataclass(frozen=True)
class MagiDSAYarnConfig:
    """YaRN rotary parameters for compressed-position RoPE.

    Matches the frozen V4-Flash recipe values; consumed by the compressor
    when embedding compressed entries at block positions.
    """

    rotary_base: float = 160000.0
    scaling_factor: float = 16.0
    original_max_position_embeddings: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 1.0


@dataclass(frozen=True)
class MagiDSAConfig:
    """Static configuration of one Magi_DSA V4 attention instance.

    One instance serves exactly one layer form, fixed by ``compress_ratio``:
    0 = sliding-window only, 4 = CSA (compressed + indexer top-k + window),
    128 = HCA (compressed dense + window).
    """

    # Layer form and model-side dims (required).
    compress_ratio: int
    hidden_size: int
    q_lora_rank: int
    softmax_scale: float

    # Main attention contract (V4-Flash defaults).
    num_heads: int = 64
    kv_dim: int = 512
    rope_dim: int = 64

    # Sliding window and sparse selection.
    window_size: int = 128
    topk: int = 512

    # Lightning indexer.
    indexer_heads: int = 64
    indexer_dim: int = 128

    # Compressor RMSNorm epsilon (V4-Flash recipe value).
    norm_eps: float = 1e-6

    # Indexer KL auxiliary loss.
    indexer_loss_coeff: float = 0.01
    use_sparse_loss: bool = True
    calculate_per_token_loss: bool = False

    # Compressed-position rotary.
    yarn: MagiDSAYarnConfig = field(default_factory=MagiDSAYarnConfig)

    # Backend selection: reference is the pure-PyTorch path.
    backend: Literal["reference", "kernel"] = "reference"

    # Deprecated compatibility field. The public runtime uses
    # DsaOverlapConfig's two independent overlap switches.
    overlap: bool = False

    # Numerics.
    params_dtype: Optional[str] = "bfloat16"

    def __post_init__(self) -> None:
        if self.compress_ratio not in VALID_COMPRESS_RATIOS:
            raise ValueError(
                f"compress_ratio must be one of {VALID_COMPRESS_RATIOS}, "
                f"got {self.compress_ratio}"
            )
        if self.kv_dim <= self.rope_dim:
            raise ValueError(
                f"kv_dim ({self.kv_dim}) must exceed rope_dim ({self.rope_dim})"
            )
        if self.indexer_dim & (self.indexer_dim - 1):
            raise ValueError(
                f"indexer_dim must be a power of two for the Hadamard rotation, "
                f"got {self.indexer_dim}"
            )
        if self.hidden_size <= 0 or self.q_lora_rank <= 0:
            raise ValueError("hidden_size and q_lora_rank must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.compress_ratio == 4 and self.topk <= 0:
            raise ValueError("topk must be positive for compress_ratio=4")
        if self.softmax_scale <= 0:
            raise ValueError("softmax_scale must be positive")

    @property
    def is_window_only(self) -> bool:
        return self.compress_ratio == 0

    @property
    def has_compressor(self) -> bool:
        return self.compress_ratio > 1

    @property
    def has_indexer(self) -> bool:
        return self.compress_ratio == 4

    @property
    def nope_dim(self) -> int:
        return self.kv_dim - self.rope_dim


# V4-spelled compatibility aliases remain available for callers that used the
# prototype config names. New serialization records the stable public names.
MagiDSAV4Config = MagiDSAConfig
MagiDSAV4YarnConfig = MagiDSAYarnConfig

__all__ = [
    "VALID_COMPRESS_RATIOS",
    "MagiDSAConfig",
    "MagiDSAYarnConfig",
    "MagiDSAV4Config",
    "MagiDSAV4YarnConfig",
]
