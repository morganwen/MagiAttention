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
from threading import RLock
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from magi_attention.dsa import MagiDSAConfig, MagiDSAV4
from magi_attention.meta.collection.dsa_meta import DsaDispatchPlan
from magi_attention.meta.solver.dsa_calibration import (
    CALIBRATION_ID,
    calibration_manifest,
    get_dsa_cost_model,
)
from magi_attention.meta.solver.dsa_solver import DsaPlanSolver

if TYPE_CHECKING:
    from magi_attention.api.dsa_attn_interface import DsaPackedMeta, MagiDSAInput
    from magi_attention.functional.dist_dsa import DsaForwardPlan


@dataclass(frozen=True)
class DsaStaticPlan:
    """Runtime-wide facts; packed-input fragment plans are cached separately."""

    cp_rank: int
    cp_size: int
    compress_ratio: int
    communication_ready: bool


@dataclass(frozen=True)
class DsaOverlapConfig:
    """Independent step-7 communication/compute overlap switches.

    The switches are deliberately runtime properties rather than properties of
    the mathematical DSA layer.  This lets the performance harness execute the
    complete 2x2 ablation without changing parameters or numerical semantics.
    """

    compressed_cast_indexer: bool = True
    dki_reduce_sparse_backward: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("compressed_cast_indexer", self.compressed_cast_indexer),
            ("dki_reduce_sparse_backward", self.dki_reduce_sparse_backward),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")


class MagiDSARuntimeMgr(nn.Module):
    """Own one fixed-ratio Magi_DSA module and its static CP plan.

    CP=1 executes the packed attention path directly.  Distributed CP caches immutable
    fragment, communication and device-remap plans per packed layout while all
    invocation-owned tensors and collective work stay outside the manager.

    Full-module deepcopy/pickle is supported for CP=1. A distributed process
    group is external runtime state and is not pickleable by PyTorch; checkpoint
    it with ``state_dict`` and construct a new manager bound to the target group.
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
        config: MagiDSAConfig,
        cp_group: dist.ProcessGroup | None = None,
        *,
        dispatch_policy: str = "balanced",
        overlap_config: DsaOverlapConfig | None = None,
    ) -> None:
        super().__init__()
        self._validate_config(config)
        if dispatch_policy not in ("sequential", "balanced"):
            raise ValueError("dispatch_policy must be 'sequential' or 'balanced'")
        if overlap_config is not None and not isinstance(
            overlap_config, DsaOverlapConfig
        ):
            raise TypeError("overlap_config must be a DsaOverlapConfig")
        self.config = config
        self.cp_group = cp_group
        self.dispatch_policy = dispatch_policy
        self.overlap_config = overlap_config or DsaOverlapConfig()

        if cp_group is None:
            cp_rank, cp_size = 0, 1
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized when cp_group is provided"
                )
            cp_rank = dist.get_rank(cp_group)
            cp_size = dist.get_world_size(cp_group)
            if cp_size not in (1, 2, 8):
                raise ValueError(
                    f"Magi_DSA V1 supports CP sizes 1, 2, or 8; got {cp_size}"
                )

        self.plan = DsaStaticPlan(
            cp_rank=cp_rank,
            cp_size=cp_size,
            compress_ratio=config.compress_ratio,
            communication_ready=cp_size in (1, 2, 8),
        )
        token_memory_bytes = 2 * (
            config.hidden_size
            + config.q_lora_rank
            + 2 * config.num_heads * config.kv_dim
            + config.kv_dim
        )
        # One logical compressed row carries KV for both compressed layer
        # forms, plus Ki only for the ratio=4 Indexer form.  The dispatch model
        # accounts separately for the owner result, packed send buffer, remote
        # receive buffer and globally reordered resident tensor.
        compressed_block_memory_bytes = 2 * config.kv_dim
        if config.compress_ratio == 4:
            compressed_block_memory_bytes += 2 * config.indexer_dim
        remote_row_memory_bytes = 2 * max(config.hidden_size, config.kv_dim)
        calibrated_cost_model = get_dsa_cost_model(
            config.compress_ratio,
            token_memory_bytes=token_memory_bytes,
            compressed_block_memory_bytes=compressed_block_memory_bytes,
            remote_row_memory_bytes=remote_row_memory_bytes,
        )
        self._plan_solver = DsaPlanSolver(
            alignment=128,
            window_size=config.window_size,
            cost_model=calibrated_cost_model,
        )
        self.dsa_module = MagiDSAV4(config, dtype=torch.bfloat16)
        self._forward_plan_cache: dict[tuple[str, int], "DsaForwardPlan"] = {}
        # Plan construction may broadcast host objects and materialize device
        # maps.  Serialize cache misses while keeping all invocation-owned CUDA
        # tensors, events and collective work outside the runtime manager.
        self._plan_lock = RLock()

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_plan_lock", None)
        # Device communication maps contain process-group-bound state and are
        # cheap to rematerialize after deepcopy/load. Host solver plans remain.
        state["_forward_plan_cache"] = {}
        return state

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        self._plan_lock = RLock()

    @property
    def plan_cache_size(self) -> int:
        return self._plan_solver.cache_size

    @property
    def forward_plan_cache_size(self) -> int:
        return len(self._forward_plan_cache)

    @property
    def solver_calibration_id(self) -> str:
        return CALIBRATION_ID

    @property
    def solver_calibration(self) -> dict[str, object]:
        return calibration_manifest()

    @property
    def solver_cost_model(self):
        """Frozen predictor used for plans created by this runtime."""

        return self._plan_solver.cost_model

    def get_dispatch_plan(
        self,
        packed_meta: "DsaPackedMeta",
        *,
        policy: str | None = None,
    ) -> DsaDispatchPlan:
        """Return the cached immutable plan for one packed sample layout."""

        from magi_attention.api.dsa_attn_interface import DsaPackedMeta

        if not isinstance(packed_meta, DsaPackedMeta):
            raise TypeError("packed_meta must be a DsaPackedMeta")
        packed_meta.validate()
        policy = self.dispatch_policy if policy is None else policy
        if policy not in ("sequential", "balanced"):
            raise ValueError("policy must be 'sequential' or 'balanced'")
        bounds = packed_meta.cu_seqlens.detach().cpu().tolist()
        sample_lengths = tuple(
            int(end - begin) for begin, end in zip(bounds, bounds[1:])
        )
        with self._plan_lock:
            if self.cp_group is None:
                return self._plan_solver.solve(
                    sample_lengths,
                    self.plan.cp_size,
                    self.config.compress_ratio,
                    policy=policy,
                )
            return self._plan_solver.solve_distributed(
                sample_lengths,
                self.config.compress_ratio,
                self.cp_group,
                policy=policy,
            )

    def get_forward_plan(
        self,
        dispatch_plan: DsaDispatchPlan,
        device: torch.device | str | int,
    ) -> "DsaForwardPlan":
        """Return cached CP communication and packing metadata for one device."""

        if self.plan.cp_size == 1:
            raise RuntimeError("CP=1 does not need a distributed forward plan")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"DSA forward plans require CUDA, got {resolved}")
        device_index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        key = (dispatch_plan.plan_hash, device_index)
        with self._plan_lock:
            cached = self._forward_plan_cache.get(key)
            if cached is None:
                from magi_attention.functional.dist_dsa import build_dsa_forward_plan

                cached = build_dsa_forward_plan(
                    dispatch_plan,
                    self.plan.cp_rank,
                    self.cp_group,
                    torch.device("cuda", device_index),
                )
                self._forward_plan_cache[key] = cached
            return cached

    @classmethod
    def _validate_config(cls, config: MagiDSAConfig) -> None:
        if not isinstance(config, MagiDSAConfig):
            raise TypeError("config must be a MagiDSAConfig")

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
        from magi_attention.api.dsa_attn_interface import DsaPackedMeta, MagiDSAInput

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


__all__ = ["DsaOverlapConfig", "DsaStaticPlan", "MagiDSARuntimeMgr"]
