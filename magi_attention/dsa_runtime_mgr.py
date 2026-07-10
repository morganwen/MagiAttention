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

"""Static owner of Magi_DSA parameters and context-parallel plan state."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from magi_attention.experimental.dsa_v4 import MagiDSAV4, MagiDSAV4Config

if TYPE_CHECKING:
    from magi_attention.api.dsa_attn_interface import MagiDSAInput


@dataclass(frozen=True)
class DsaStaticPlan:
    """Step-1 plan skeleton, replaced by fragment plans in step 2."""

    cp_rank: int
    cp_size: int
    compress_ratio: int
    communication_ready: bool


class MagiDSARuntimeMgr(nn.Module):
    """Own one fixed-ratio Magi_DSA module and its static CP plan.

    Step 1 executes only the CP=1 packed path.  A manager constructed with a
    larger process group still creates a deterministic plan skeleton, but
    calculation raises before any collective is launched.
    """

    _FIXED_CONFIG = {
        "num_heads": 64,
        "kv_dim": 512,
        "rope_dim": 64,
        "window_size": 128,
        "topk": 512,
        "indexer_heads": 64,
        "indexer_dim": 128,
    }

    def __init__(
        self,
        config: MagiDSAV4Config,
        cp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.cp_group = cp_group

        if cp_group is None:
            cp_rank, cp_size = 0, 1
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized when cp_group is provided"
                )
            cp_rank = dist.get_rank(cp_group)
            cp_size = dist.get_world_size(cp_group)
            if cp_size not in (1, 2):
                raise ValueError(f"Magi_DSA V1 supports CP size 1 or 2, got {cp_size}")

        self.plan = DsaStaticPlan(
            cp_rank=cp_rank,
            cp_size=cp_size,
            compress_ratio=config.compress_ratio,
            communication_ready=cp_size == 1,
        )
        self.dsa_module = MagiDSAV4(config, dtype=torch.bfloat16)

    @classmethod
    def _validate_config(cls, config: MagiDSAV4Config) -> None:
        if not isinstance(config, MagiDSAV4Config):
            raise TypeError("config must be a MagiDSAV4Config")

        mismatches = [
            f"{name}={getattr(config, name)!r} (expected {expected!r})"
            for name, expected in cls._FIXED_CONFIG.items()
            if getattr(config, name) != expected
        ]
        if mismatches:
            raise ValueError(
                "config violates the fixed DeepSeek V4 DSA contract: "
                + ", ".join(mismatches)
            )
        if config.params_dtype != "bfloat16":
            raise ValueError(
                "Magi_DSA V1 parameters must be bfloat16, "
                f"got params_dtype={config.params_dtype!r}"
            )
        if config.backend not in ("reference", "kernel"):
            raise ValueError(
                f"backend must be 'reference' or 'kernel', got {config.backend!r}"
            )
        if not config.use_sparse_loss:
            raise ValueError("Magi_DSA V1 requires use_sparse_loss=True")
        if config.calculate_per_token_loss:
            raise ValueError(
                "calculate_per_token_loss must be False; calc_dsa returns a "
                "global-token-normalized KL contribution"
            )
        if config.yarn.rotary_base != 160000.0:
            raise ValueError(
                "compressed RoPE rotary_base must be 160000.0 for the frozen HF config"
            )
        if config.yarn.scaling_factor != 16.0:
            raise ValueError(
                "compressed RoPE scaling_factor must be 16.0 for the frozen HF config"
            )
        if config.yarn.original_max_position_embeddings != 65536:
            raise ValueError(
                "compressed RoPE original_max_position_embeddings must be 65536"
            )

    def validate_input(self, dsa_input: "MagiDSAInput") -> None:
        from magi_attention.api.dsa_attn_interface import (
            DsaPackedMeta,
            MagiDSAInput,
        )

        if not isinstance(dsa_input, MagiDSAInput):
            raise TypeError("dsa_input must be a MagiDSAInput")
        if not isinstance(dsa_input.packed_meta, DsaPackedMeta):
            raise TypeError("dsa_input.packed_meta must be a DsaPackedMeta")

        tensors = {
            "x": dsa_input.x,
            "qr": dsa_input.qr,
            "q": dsa_input.q,
            "latent_kv": dsa_input.latent_kv,
            "sink": dsa_input.sink,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        token_count = dsa_input.q.size(0) if dsa_input.q.ndim > 0 else 0
        expected_shapes = {
            "x": (token_count, self.config.hidden_size),
            "qr": (token_count, self.config.q_lora_rank),
            "q": (token_count, self.config.num_heads, self.config.kv_dim),
            "latent_kv": (token_count, self.config.kv_dim),
            "sink": (self.config.num_heads,),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(tensors[name].shape)
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")

        for name in ("x", "qr", "q", "latent_kv"):
            if tensors[name].dtype != torch.bfloat16:
                raise TypeError(
                    f"{name} must have dtype torch.bfloat16, got {tensors[name].dtype}"
                )
        if dsa_input.sink.dtype != torch.float32:
            raise TypeError(
                f"sink must have dtype torch.float32, got {dsa_input.sink.dtype}"
            )

        device = dsa_input.q.device
        if device.type != "cuda":
            raise ValueError(f"Magi_DSA V1 requires CUDA tensors, got device {device}")
        for name, tensor in tensors.items():
            if tensor.device != device:
                raise ValueError(
                    f"all DSA tensors must share device {device}; "
                    f"{name} is on {tensor.device}"
                )

        total_tokens = token_count if self.plan.cp_size == 1 else None
        dsa_input.packed_meta.validate(total_tokens=total_tokens)

    def calc_dsa(self, dsa_input: "MagiDSAInput") -> tuple[torch.Tensor, torch.Tensor]:
        from magi_attention.functional.dist_dsa import dist_dsa_func

        return dist_dsa_func(dsa_input, self)

    def forward(self, dsa_input: "MagiDSAInput") -> tuple[torch.Tensor, torch.Tensor]:
        return self.calc_dsa(dsa_input)


__all__ = ["DsaStaticPlan", "MagiDSARuntimeMgr"]
