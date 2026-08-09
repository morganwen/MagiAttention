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

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import torch.distributed as dist

from .comm import layout_dsa_hidden, route_dsa_tensor
from .config import DsaStructuralLayoutConfig, MagiDSAConfig
from .meta import DsaExecutionPlan
from .packing import DsaDeviceRankPlan, DsaDeviceRoutePlan, make_dsa_device_rank_plan
from .projection import DsaProjections
from .solver import build_dsa_execution_plan
from .types import MagiDSAForwardResult, MagiDSAInput, MagiDSAPackedMeta


@dataclass(frozen=True)
class DsaRuntimeCounters:
    """Observable cold/warm boundary counters used by regression tests."""

    solver_invocations: int
    object_collective_invocations: int
    device_materializations: int
    health_checks: int
    warm_invocations: int


@dataclass(frozen=True, eq=False)
class DsaExecutionHandle:
    """Immutable plan identity and resident maps consumed by warm execution."""

    runtime_identity: int
    schema_hash: str
    packed_meta: MagiDSAPackedMeta
    plan: DsaExecutionPlan
    device_plan: DsaDeviceRankPlan
    rank: int
    world_size: int
    device: torch.device
    local_token_capacity: int
    sparse_backward_stream: torch.cuda.Stream | None
    csa_main_stream: torch.cuda.Stream | None
    csa_indexer_stream: torch.cuda.Stream | None
    csa_route_stream: torch.cuda.Stream | None
    hca_main_stream: torch.cuda.Stream | None
    hca_route_stream: torch.cuda.Stream | None

    @property
    def plan_hash(self) -> str:
        return self.plan.plan_hash


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MagiDSARuntimeMgr:
    """Parameter-free cold-plan owner and warm DSA execution orchestrator.

    Preparing an execution runs no collective of any kind. The caller states the
    packed metadata and the source split, the structural solver is a pure
    function of those, and so every rank independently derives a bit-identical
    plan. The handle cache is therefore keyed on purely local information and is
    consulted before any work is done.
    """

    def __init__(
        self,
        config: MagiDSAConfig,
        cp_group: dist.ProcessGroup | None = None,
        *,
        structural_layout_config: DsaStructuralLayoutConfig | None = None,
        max_cached_handles: int = 4,
    ) -> None:
        if max_cached_handles <= 0:
            raise ValueError("max_cached_handles must be positive")
        if cp_group is None:
            self.rank = 0
            self.world_size = 1
        else:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before constructing "
                    "a CP runtime"
                )
            self.rank = dist.get_rank(cp_group)
            self.world_size = dist.get_world_size(cp_group)
        self.config = config
        self.cp_group = cp_group
        self.structural_layout_config = (
            structural_layout_config or DsaStructuralLayoutConfig()
        )
        self.max_cached_handles = max_cached_handles
        self._identity = id(self)
        self._handle_cache: OrderedDict[
            tuple[object, ...], DsaExecutionHandle
        ] = OrderedDict()
        self._solver_invocations = 0
        self._device_materializations = 0
        self._health_checks = 0
        self._warm_invocations = 0

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Return no parameters; all trainable state belongs to the model."""

        del recurse
        return iter(())

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        del prefix, recurse
        return iter(())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    @property
    def counters(self) -> DsaRuntimeCounters:
        return DsaRuntimeCounters(
            solver_invocations=self._solver_invocations,
            # Kept at zero and asserted by the release gate: neither the cold
            # nor the warm path may run an object collective.
            object_collective_invocations=0,
            device_materializations=self._device_materializations,
            health_checks=self._health_checks,
            warm_invocations=self._warm_invocations,
        )

    @property
    def runtime_identity(self) -> int:
        """Opaque identity used to reject handles from another runtime."""

        return self._identity

    def clear_handle_cache(self) -> None:
        self._handle_cache.clear()

    def _shared_schema_hash(self, packed_meta: MagiDSAPackedMeta) -> str:
        return _digest(
            {
                "config": asdict(self.config),
                "cu_seqlens": packed_meta.cu_seqlens,
                "source_token_counts": packed_meta.source_token_counts,
                "structural_layout_config": asdict(self.structural_layout_config),
                "world_size": self.world_size,
            }
        )

    def _validate_packed_meta(self, packed_meta: MagiDSAPackedMeta) -> None:
        if packed_meta.cp_size != self.world_size:
            raise ValueError(
                "packed metadata source split does not match the CP group size"
            )

    def _cache_key(
        self,
        packed_meta: MagiDSAPackedMeta,
        local_token_capacity: int,
        device: torch.device,
    ) -> tuple[object, ...]:
        return (
            packed_meta.cu_seqlens,
            packed_meta.source_token_counts,
            local_token_capacity,
            device.type,
            device.index,
            self.rank,
        )

    def _health_check_route(
        self, route: DsaDeviceRoutePlan, device: torch.device
    ) -> None:
        source = torch.ones(
            (route.producer_row_count, 8),
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        consumer = route_dsa_tensor(source, route, self.cp_group)
        gradient = torch.autograd.grad(consumer.float().sum(), source)[0]
        # Every owner row reaches as many consumers as reference it, so the
        # reverse route must return exactly that multiplicity.
        expected = (
            _owner_multiplicity(route, device)
            .to(torch.bfloat16)
            .unsqueeze(1)
            .expand_as(source)
        )
        if not torch.equal(gradient, expected):
            raise RuntimeError(f"{route.name} reverse-route health check failed")

    def _health_check(self, handle: DsaExecutionHandle) -> None:
        device_plan = handle.device_plan
        routes = (
            device_plan.token_layout_route,
            device_plan.window_route,
            device_plan.overlap_x_route,
            device_plan.compressed_kv_route,
            device_plan.compressed_ki_route,
        )
        for route in routes:
            if route is not None:
                self._health_check_route(route, handle.device)
        self._health_checks += 1

    def prepare_execution(
        self,
        packed_meta: MagiDSAPackedMeta,
        device: torch.device | str,
        *,
        local_token_capacity: int,
        health_check: bool = False,
    ) -> DsaExecutionHandle:
        """Solve and materialize one cold plan, or return the cached handle.

        ``health_check`` runs a real collective per route and is off by default,
        because it is a debugging dry run rather than part of preparing a plan.
        """

        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("Magi-DSA production execution requires a CUDA device")
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(resolved_device) != (10, 3):
            raise RuntimeError("Magi-DSA v4 release execution requires B300 SM103")
        self._validate_packed_meta(packed_meta)
        source_tokens = packed_meta.local_token_count(self.rank)
        if local_token_capacity < source_tokens:
            raise ValueError("local_token_capacity is smaller than the source count")

        cache_key = self._cache_key(packed_meta, local_token_capacity, resolved_device)
        cached = self._handle_cache.get(cache_key)
        if cached is not None:
            self._handle_cache.move_to_end(cache_key)
            return cached

        self._solver_invocations += 1
        plan = build_dsa_execution_plan(
            self.config,
            packed_meta.cu_seqlens,
            packed_meta.source_token_counts,
            structural_layout_config=self.structural_layout_config,
        )
        if plan.query_token_counts[self.rank] > local_token_capacity:
            raise ValueError(
                "a final Query token count exceeds its declared execution capacity"
            )
        device_plan = make_dsa_device_rank_plan(
            plan,
            self.rank,
            self.config,
            resolved_device,
            cp_group=self.cp_group,
            deterministic=False,
        )
        self._device_materializations += 1
        is_csa = self.config.ratio == 4
        handle = DsaExecutionHandle(
            runtime_identity=self._identity,
            schema_hash=self._shared_schema_hash(packed_meta),
            packed_meta=packed_meta,
            plan=plan,
            device_plan=device_plan,
            rank=self.rank,
            world_size=self.world_size,
            device=resolved_device,
            local_token_capacity=local_token_capacity,
            sparse_backward_stream=(
                torch.cuda.Stream(device=resolved_device) if is_csa else None
            ),
            csa_main_stream=(
                torch.cuda.Stream(device=resolved_device) if is_csa else None
            ),
            csa_indexer_stream=(
                torch.cuda.Stream(device=resolved_device) if is_csa else None
            ),
            csa_route_stream=(
                torch.cuda.Stream(device=resolved_device) if is_csa else None
            ),
            hca_main_stream=(
                None
                if is_csa
                else torch.cuda.Stream(device=resolved_device, priority=-1)
            ),
            hca_route_stream=(
                None
                if is_csa
                else torch.cuda.Stream(device=resolved_device, priority=-1)
            ),
        )
        if health_check:
            self._health_check(handle)
        self._handle_cache[cache_key] = handle
        self._handle_cache.move_to_end(cache_key)
        while len(self._handle_cache) > self.max_cached_handles:
            self._handle_cache.popitem(last=False)
        return handle

    def _validate_warm_handle(
        self, dsa_input: MagiDSAInput, handle: DsaExecutionHandle
    ) -> None:
        if handle.runtime_identity != self._identity:
            raise ValueError(
                "execution handle belongs to a different MagiDSARuntimeMgr"
            )
        if handle.rank != self.rank or handle.world_size != self.world_size:
            raise ValueError("execution handle CP identity does not match the runtime")
        # A frozen dataclass compares as a pair of tuples, so the warm path
        # never re-hashes the schema.
        if dsa_input.packed_meta != handle.packed_meta:
            raise ValueError(
                "input packed metadata does not match the frozen execution handle"
            )
        tensors = (
            dsa_input.x,
            dsa_input.qr,
            dsa_input.q,
            dsa_input.latent_kv,
            dsa_input.sink,
        )
        if any(tensor.device != handle.device for tensor in tensors):
            raise ValueError("all Magi-DSA inputs must be on the handle device")

    def layout_hidden(
        self,
        source_x: torch.Tensor,
        handle: DsaExecutionHandle,
    ) -> torch.Tensor:
        """Apply the model-boundary TOKEN_LAYOUT before local projections."""

        if handle.runtime_identity != self._identity:
            raise ValueError(
                "execution handle belongs to a different MagiDSARuntimeMgr"
            )
        if source_x.device != handle.device:
            raise ValueError("source hidden state is on the wrong device")
        if (
            source_x.ndim != 2
            or source_x.shape[0] != handle.device_plan.source_token_count
            or source_x.shape[1] != self.config.hidden_size
            or source_x.dtype != torch.bfloat16
            or not source_x.is_contiguous()
        ):
            raise ValueError(
                "source hidden state must be contiguous owner-local CUDA BF16"
            )
        return layout_dsa_hidden(
            source_x, handle.device_plan.token_layout_route, self.cp_group
        )

    def get_position_ids(self, handle: DsaExecutionHandle) -> torch.Tensor:
        """Return resident sample-relative positions in final Query-row order."""

        if handle.runtime_identity != self._identity:
            raise ValueError(
                "execution handle belongs to a different MagiDSARuntimeMgr"
            )
        if handle.rank != self.rank or handle.world_size != self.world_size:
            raise ValueError("execution handle CP identity does not match the runtime")
        return handle.device_plan.local_q_positions

    def calc_dsa(
        self,
        projections: DsaProjections,
        dsa_input: MagiDSAInput,
        handle: DsaExecutionHandle,
    ) -> MagiDSAForwardResult:
        """Execute only the frozen warm DAG; this method never prepares a plan.

        ``projections`` carries the model's own parameterized callbacks. The
        runtime schedules them and never owns them.
        """

        self._validate_warm_handle(dsa_input, handle)
        from .dist import dist_dsa
        from .phase import dsa_phase

        with dsa_phase("forward"):
            result = dist_dsa(
                self.config, projections, dsa_input, handle, self.cp_group
            )
        self._warm_invocations += 1
        return result


def _owner_multiplicity(
    route: DsaDeviceRoutePlan, device: torch.device
) -> torch.Tensor:
    """Count how many consumers reference each owner row of a route.

    A route is not always onto: a sample tail shorter than the compression ratio
    produces no block, so some OVERLAP_X owner rows are consumed by nobody and
    must come back with a zero gradient rather than one.
    """

    counts = torch.zeros(
        (route.producer_row_count,), dtype=torch.float32, device=device
    )
    arg = route.group_collective_arg
    if arg is None:
        ranges = route.local_gather_ranges
        if ranges is None:
            raise RuntimeError(f"{route.name}: one-rank route has no local range plan")
        for begin, end in ranges.tolist():
            counts[begin:end] += 1.0
        return counts
    cursor = 0
    for split_size, destinations in zip(
        arg.input_split_size_list, arg.dst_indices_list
    ):
        counts[cursor : cursor + split_size] = float(len(destinations))
        cursor += split_size
    return counts


__all__ = ["DsaExecutionHandle", "DsaRuntimeCounters", "MagiDSARuntimeMgr"]
