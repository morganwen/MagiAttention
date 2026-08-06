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
from typing import TYPE_CHECKING, Iterator

import torch
import torch.distributed as dist

from magi_attention.dsa_config import (
    DsaPlanPolicy,
    DsaSharedLayoutConfig,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
)
from magi_attention.dsa_types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)
from magi_attention.functional.dsa_comm import layout_dsa_hidden, route_dsa_tensor
from magi_attention.functional.dsa_packing import (
    DsaDeviceRankPlan,
    DsaDeviceRoutePlan,
    make_dsa_device_rank_plan,
)
from magi_attention.meta.collection.dsa_meta import DsaExecutionPlan
from magi_attention.meta.solver.dsa_solver import (
    build_dsa_execution_plan,
    validate_dsa_execution_plan,
)

if TYPE_CHECKING:
    from magi_attention.dsa_layer import MagiDSALayer


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
    """Parameter-free cold-plan owner and warm DSA execution orchestrator."""

    def __init__(
        self,
        config: MagiDSAConfig,
        cp_group: dist.ProcessGroup | None = None,
        *,
        policy: DsaPlanPolicy = "indexer_balanced",
        shared_layout_config: DsaSharedLayoutConfig | None = None,
        structural_layout_config: DsaStructuralLayoutConfig | None = None,
        max_cached_handles: int = 4,
    ) -> None:
        if policy not in (
            "sequential",
            "indexer_balanced",
            "shared_greedy",
            "structural_balanced",
        ):
            raise ValueError(
                "DSA policy must be sequential, indexer_balanced, shared_greedy, "
                "or structural_balanced"
            )
        if policy == "shared_greedy" and shared_layout_config is None:
            raise ValueError("shared_greedy requires an explicit shared_layout_config")
        if policy != "shared_greedy" and shared_layout_config is not None:
            raise ValueError("shared_layout_config is only valid for shared_greedy")
        if policy == "structural_balanced" and structural_layout_config is None:
            raise ValueError(
                "structural_balanced requires an explicit structural_layout_config"
            )
        if policy != "structural_balanced" and structural_layout_config is not None:
            raise ValueError(
                "structural_layout_config is only valid for structural_balanced"
            )
        if max_cached_handles <= 0:
            raise ValueError("max_cached_handles must be positive")
        if cp_group is None:
            self.rank = 0
            self.world_size = 1
        else:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before constructing a CP runtime"
                )
            self.rank = dist.get_rank(cp_group)
            self.world_size = dist.get_world_size(cp_group)
        self.config = config
        self.cp_group = cp_group
        self.policy = policy
        self.shared_layout_config = shared_layout_config
        self.structural_layout_config = structural_layout_config
        self.max_cached_handles = max_cached_handles
        self._identity = id(self)
        self._handle_cache: OrderedDict[
            tuple[object, ...], DsaExecutionHandle
        ] = OrderedDict()
        self._solver_invocations = 0
        self._object_collective_invocations = 0
        self._device_materializations = 0
        self._health_checks = 0
        self._warm_invocations = 0

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Return no parameters; all trainable state belongs to MagiDSALayer."""

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
            object_collective_invocations=self._object_collective_invocations,
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
                "policy": self.policy,
                "shared_layout_config": (
                    None
                    if self.shared_layout_config is None
                    else asdict(self.shared_layout_config)
                ),
                "structural_layout_config": (
                    None
                    if self.structural_layout_config is None
                    else asdict(self.structural_layout_config)
                ),
                "world_size": self.world_size,
            }
        )

    def _collect_owner_layout(
        self,
        packed_meta: MagiDSAPackedMeta,
        local_token_capacity: int,
        schema_hash: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        local_record = (
            schema_hash,
            int(packed_meta.local_token_count),
            int(local_token_capacity),
        )
        records: list[tuple[str, int, int] | None]
        if self.cp_group is None:
            records = [local_record]
        else:
            records = [None] * self.world_size
            dist.all_gather_object(records, local_record, group=self.cp_group)
            self._object_collective_invocations += 1
        concrete = [record for record in records if record is not None]
        if len(concrete) != self.world_size:
            raise RuntimeError(
                "owner-layout collective returned an incomplete rank table"
            )
        if any(record[0] != schema_hash for record in concrete):
            raise ValueError(
                "DSA config, packed metadata, or policy differs across CP ranks"
            )
        local_counts = tuple(record[1] for record in concrete)
        capacities = tuple(record[2] for record in concrete)
        if any(count > capacity for count, capacity in zip(local_counts, capacities)):
            raise ValueError(
                "a local token count exceeds its declared execution capacity"
            )
        if sum(local_counts) != packed_meta.cu_seqlens[-1]:
            raise ValueError(
                "owner-local counts do not cover the global packed token count"
            )
        return local_counts, capacities

    def _build_and_broadcast_plan(
        self,
        packed_meta: MagiDSAPackedMeta,
        local_counts: tuple[int, ...],
    ) -> DsaExecutionPlan:
        if self.cp_group is None:
            self._solver_invocations += 1
            return build_dsa_execution_plan(
                self.config,
                packed_meta.cu_seqlens,
                local_counts,
                policy=self.policy,
                shared_layout_config=self.shared_layout_config,
                structural_layout_config=self.structural_layout_config,
            )
        payload: list[DsaExecutionPlan | None]
        if self.rank == 0:
            self._solver_invocations += 1
            payload = [
                build_dsa_execution_plan(
                    self.config,
                    packed_meta.cu_seqlens,
                    local_counts,
                    policy=self.policy,
                    shared_layout_config=self.shared_layout_config,
                    structural_layout_config=self.structural_layout_config,
                )
            ]
        else:
            payload = [None]
        dist.broadcast_object_list(payload, src=0, group=self.cp_group)
        self._object_collective_invocations += 1
        plan = payload[0]
        if not isinstance(plan, DsaExecutionPlan):
            raise RuntimeError("rank 0 did not broadcast a DSA execution plan")
        validate_dsa_execution_plan(plan)
        return plan

    def _cache_key(
        self,
        schema_hash: str,
        local_counts: tuple[int, ...],
        capacities: tuple[int, ...],
        device: torch.device,
    ) -> tuple[object, ...]:
        return (
            schema_hash,
            local_counts,
            capacities,
            device.type,
            device.index,
            self.rank,
        )

    def _health_check_route(
        self,
        route: DsaDeviceRoutePlan,
    ) -> None:
        source = torch.ones(
            (route.producer_row_count, 8),
            dtype=torch.bfloat16,
            device=route.consumer_global_rows.device,
            requires_grad=True,
        )
        consumer = route_dsa_tensor(source, route, self.cp_group)
        gradient = torch.autograd.grad(consumer.float().sum(), source)[0]
        multiplicity = (
            route.owner_reduce.row_offsets[1:] - route.owner_reduce.row_offsets[:-1]
        )
        expected = multiplicity.to(torch.bfloat16).unsqueeze(1).expand_as(source)
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
                self._health_check_route(route)
        torch.cuda.synchronize(handle.device)
        self._health_checks += 1

    def prepare_execution(
        self,
        packed_meta: MagiDSAPackedMeta,
        device: torch.device | str,
        *,
        local_token_capacity: int,
        health_check: bool = True,
    ) -> DsaExecutionHandle:
        """Collect, solve, broadcast, materialize, and optionally dry-run a cold plan."""

        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("Magi-DSA production execution requires a CUDA device")
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(resolved_device) != (10, 3):
            raise RuntimeError("Magi-DSA v4 release execution requires B300 SM103")
        if local_token_capacity < packed_meta.local_token_count:
            raise ValueError("local_token_capacity is smaller than local_token_count")
        schema_hash = self._shared_schema_hash(packed_meta)
        local_counts, capacities = self._collect_owner_layout(
            packed_meta,
            local_token_capacity,
            schema_hash,
        )
        cache_key = self._cache_key(
            schema_hash, local_counts, capacities, resolved_device
        )
        cached = self._handle_cache.get(cache_key)
        if cached is not None:
            self._handle_cache.move_to_end(cache_key)
            return cached

        plan = self._build_and_broadcast_plan(packed_meta, local_counts)
        if any(
            query_count > capacity
            for query_count, capacity in zip(plan.query_token_counts, capacities)
        ):
            raise ValueError(
                "a final Query token count exceeds its declared execution capacity"
            )
        device_plan = make_dsa_device_rank_plan(
            plan, self.rank, self.config, resolved_device
        )
        self._device_materializations += 1
        handle = DsaExecutionHandle(
            runtime_identity=self._identity,
            schema_hash=schema_hash,
            plan=plan,
            device_plan=device_plan,
            rank=self.rank,
            world_size=self.world_size,
            device=resolved_device,
            local_token_capacity=local_token_capacity,
            sparse_backward_stream=(
                torch.cuda.Stream(device=resolved_device)
                if self.config.ratio == 4
                else None
            ),
            csa_main_stream=(
                torch.cuda.Stream(device=resolved_device)
                if self.config.ratio == 4
                else None
            ),
            csa_indexer_stream=(
                torch.cuda.Stream(device=resolved_device)
                if self.config.ratio == 4
                else None
            ),
            csa_route_stream=(
                torch.cuda.Stream(device=resolved_device)
                if self.config.ratio == 4
                else None
            ),
            hca_main_stream=(
                torch.cuda.Stream(device=resolved_device, priority=-1)
                if self.config.ratio == 128
                else None
            ),
            hca_route_stream=(
                torch.cuda.Stream(device=resolved_device, priority=-1)
                if self.config.ratio == 128
                else None
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
        if self._shared_schema_hash(dsa_input.packed_meta) != handle.schema_hash:
            raise ValueError(
                "input packed metadata does not match the frozen execution handle"
            )
        if (
            dsa_input.packed_meta.local_token_count
            != handle.device_plan.source_token_count
        ):
            raise ValueError(
                "input source token count does not match the frozen rank plan"
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
        route = handle.device_plan.token_layout_route
        if route is None:
            if (
                handle.device_plan.source_token_count
                != handle.device_plan.local_token_count
            ):
                raise RuntimeError("a changed Query layout is missing TOKEN_LAYOUT")
            return source_x
        return layout_dsa_hidden(source_x, route, self.cp_group)

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
        layer: MagiDSALayer,
        dsa_input: MagiDSAInput,
        handle: DsaExecutionHandle,
    ) -> MagiDSAForwardResult:
        """Execute only the frozen warm DAG; this method never prepares a plan."""

        self._validate_warm_handle(dsa_input, handle)
        if layer.config != self.config:
            raise ValueError("MagiDSALayer config does not match its runtime")
        from magi_attention.functional.dist_dsa import dist_dsa
        from magi_attention.functional.dsa_phase import dsa_phase

        with dsa_phase("forward"):
            result = dist_dsa(layer, dsa_input, handle, self.cp_group)
        self._warm_invocations += 1
        return result


__all__ = ["DsaExecutionHandle", "DsaRuntimeCounters", "MagiDSARuntimeMgr"]
