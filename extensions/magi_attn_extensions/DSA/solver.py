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

import bisect
import hashlib
import heapq
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import cast

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._make_dispatch_meta import make_dispatch_meta_from_qk_ranges
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    MinHeapDispatchAlg,
)

from .config import (
    DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT,
    DsaPlanPolicy,
    DsaSharedLayoutConfig,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
)
from .meta import (
    DsaCompressionBlock,
    DsaExecutionPlan,
    DsaFragmentSpec,
    DsaGroupCollectiveArg,
    DsaLayoutMetrics,
    DsaLayoutRankCost,
    DsaQueryFragment,
    DsaRankPlan,
    DsaRequiredKRange,
    DsaRouteRankPlan,
    DsaStructuralLayoutMetrics,
    DsaStructuralRankCost,
    DsaTypedRoutePlan,
)

_SHARED_SCORE_WEIGHT = 8
_SHARED_TOPK_WEIGHT = 1
_SHARED_KI_PACK_WEIGHT = 32
_SHARED_HCA_QUERY_WEIGHT = 256
_SHARED_HCA_ROUTE_WEIGHT = 2048
_SHARED_HCA_PEER_WEIGHT = 4096
_SHARED_SWAP_CANDIDATES_PER_RANK = 4
_SHARED_SOLVER_SCHEME = "deterministic_greedy_local_improve_v1"
_SHARED_COST_MODEL_VERSION = "b300_sm103_structural_proxy_v1"
_STRUCTURAL_SOLVER_SCHEME = "magi_min_heap_packed_global_v1"
_STRUCTURAL_COST_MODEL_VERSION = "native_causal_attn_slice_area_v1"
_BF16_BYTES = 2
_INT32_BYTES = 4


def _validate_cu_seqlens(cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in cu_seqlens)
    if len(values) < 2 or values[0] != 0:
        raise ValueError(
            "cu_seqlens must start at zero and describe at least one sample"
        )
    if any(end < begin for begin, end in zip(values, values[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    return values


def _validate_source_counts(
    source_counts: Sequence[int], total_tokens: int
) -> tuple[int, ...]:
    values = tuple(int(value) for value in source_counts)
    if not values:
        raise ValueError("source_token_counts must contain at least one rank")
    if any(value < 0 for value in values):
        raise ValueError("source token counts must be non-negative")
    if sum(values) != total_tokens:
        raise ValueError(
            f"source token counts sum to {sum(values)}, expected {total_tokens}"
        )
    return values


def _prefix_offsets(counts: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return tuple(offsets)


def _owner_of_global_row(global_row: int, offsets: tuple[int, ...]) -> int:
    if not 0 <= global_row < offsets[-1]:
        raise ValueError(f"global row {global_row} is outside the source layout")
    rank = bisect.bisect_right(offsets, global_row) - 1
    if rank == len(offsets) - 1:
        rank -= 1
    if not offsets[rank] <= global_row < offsets[rank + 1]:
        raise ValueError(f"global row {global_row} has no non-empty source owner")
    return rank


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _csa_fragment_cost(config: MagiDSAConfig, spec: DsaFragmentSpec) -> tuple[int, int]:
    score_cost = 0
    topk_cost = 0
    for position in range(spec.q_begin, spec.q_end):
        visible = (position + 1) // 4
        score_cost += max(
            config.indexer_score_block,
            _round_up(visible, config.indexer_score_block),
        )
        topk_cost += max(
            config.indexer_topk_block,
            _round_up(visible, config.indexer_topk_block),
        )
    return score_cost, topk_cost


def _fragment_cost(config: MagiDSAConfig, spec: DsaFragmentSpec) -> tuple[int, int]:
    if config.ratio != 4:
        return 0, 0
    return _csa_fragment_cost(config, spec)


def _hca_fragment_cost(spec: DsaFragmentSpec) -> int:
    cost = 0
    for position in range(spec.q_begin, spec.q_end):
        visible = min(position + 1, 128) + (position + 1) // 128
        cost += (visible + 63) // 64
    return cost


def _balanced_query_counts(total_tokens: int, cp_size: int) -> tuple[int, ...]:
    base, extra = divmod(total_tokens, cp_size)
    return tuple(base + int(rank < extra) for rank in range(cp_size))


def _sequential_fragment_specs(
    cu_seqlens: tuple[int, ...], query_counts: tuple[int, ...]
) -> tuple[tuple[DsaFragmentSpec, ...], ...]:
    offsets = _prefix_offsets(query_counts)
    result: list[tuple[DsaFragmentSpec, ...]] = []
    for shard_begin, shard_end in zip(offsets, offsets[1:]):
        specs: list[DsaFragmentSpec] = []
        for sample_id, (sample_begin, sample_end) in enumerate(
            zip(cu_seqlens, cu_seqlens[1:])
        ):
            begin = max(shard_begin, sample_begin)
            end = min(shard_end, sample_end)
            if begin < end:
                specs.append(
                    DsaFragmentSpec(
                        sample_id=sample_id,
                        q_begin=begin - sample_begin,
                        q_end=end - sample_begin,
                    )
                )
        result.append(tuple(specs))
    return tuple(result)


def _make_indexer_atoms(
    config: MagiDSAConfig, cu_seqlens: tuple[int, ...]
) -> tuple[DsaFragmentSpec, ...]:
    atoms: list[DsaFragmentSpec] = []
    for sample_id, (sample_begin, sample_end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        sample_length = sample_end - sample_begin
        cursor = 0
        while cursor < sample_length:
            end = min(cursor + config.indexer_atom_size, sample_length)
            atoms.append(DsaFragmentSpec(sample_id, cursor, end))
            cursor = end
    return tuple(atoms)


def _merge_adjacent_specs(
    specs: Sequence[DsaFragmentSpec],
) -> tuple[DsaFragmentSpec, ...]:
    merged: list[DsaFragmentSpec] = []
    for spec in sorted(
        specs, key=lambda item: (item.sample_id, item.q_begin, item.q_end)
    ):
        if (
            merged
            and merged[-1].sample_id == spec.sample_id
            and merged[-1].q_end == spec.q_begin
        ):
            previous = merged[-1]
            merged[-1] = DsaFragmentSpec(
                previous.sample_id, previous.q_begin, spec.q_end
            )
        else:
            merged.append(spec)
    return tuple(merged)


def _csa_required_prefix_lengths(
    fragments: Sequence[DsaFragmentSpec | DsaQueryFragment],
) -> tuple[tuple[int, int], ...]:
    """Deduplicate CSA required K by keeping one longest prefix per sample."""

    required_lengths: dict[int, int] = {}
    for fragment in fragments:
        prefix_length = fragment.q_end // 4
        required_lengths[fragment.sample_id] = max(
            required_lengths.get(fragment.sample_id, 0),
            prefix_length,
        )
    return tuple(
        (sample_id, prefix_length)
        for sample_id, prefix_length in sorted(required_lengths.items())
        if prefix_length > 0
    )


def _csa_required_k_ranges(
    fragments: Sequence[DsaFragmentSpec | DsaQueryFragment],
    sample_block_offsets: tuple[int, ...],
) -> tuple[DsaRequiredKRange, ...]:
    """Resolve longest per-sample prefixes into global compressed-block ranges."""

    return tuple(
        DsaRequiredKRange(
            sample_id=sample_id,
            global_begin=sample_block_offsets[sample_id],
            global_end=sample_block_offsets[sample_id] + prefix_length,
        )
        for sample_id, prefix_length in _csa_required_prefix_lengths(fragments)
    )


def _native_causal_chunk_area(
    cu_seqlens: tuple[int, ...],
    chunk_begin: int,
    chunk_end: int,
) -> int:
    area = 0
    for sample_begin, sample_end in zip(cu_seqlens, cu_seqlens[1:]):
        q_begin = max(sample_begin, chunk_begin)
        q_end = min(sample_end, chunk_end)
        if q_begin >= q_end:
            continue
        relative_begin = q_begin - sample_begin
        relative_end = q_end - sample_begin
        query_count = relative_end - relative_begin
        area += (relative_begin + 1 + relative_end) * query_count // 2
    return area


@lru_cache(maxsize=32)
def _assign_structural_minheap(
    cu_seqlens: tuple[int, ...],
    source_counts: tuple[int, ...],
    solver_config: DsaStructuralLayoutConfig,
) -> tuple[tuple[tuple[DsaFragmentSpec, ...], ...], DsaStructuralLayoutMetrics,]:
    """Reuse native Magi causal-area MinHeap for one shared Query layout."""

    total_tokens = cu_seqlens[-1]
    cp_size = len(source_counts)
    if total_tokens < cp_size:
        raise ValueError(
            "structural_balanced requires at least one Query token per CP rank"
        )
    auto_chunk_size = max(
        1,
        (total_tokens + solver_config.min_chunks_per_rank * cp_size - 1)
        // (solver_config.min_chunks_per_rank * cp_size),
    )
    chunk_size = min(solver_config.chunk_size, auto_chunk_size)

    partitions: tuple[tuple[int, ...], ...]
    if cp_size == 1:
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        partitions = (tuple(range(num_chunks)),)
        chunk_actual_sizes: tuple[int, ...] | None = tuple(
            min(chunk_size, total_tokens - chunk_id * chunk_size)
            for chunk_id in range(num_chunks)
        )
        resolved_chunk_size = chunk_size
    else:
        nonempty_ranges = [
            (begin, end)
            for begin, end in zip(cu_seqlens, cu_seqlens[1:])
            if begin < end
        ]
        q_ranges = AttnRanges.from_ranges(nonempty_ranges)
        k_ranges = AttnRanges.from_ranges(nonempty_ranges)
        dispatch_config = DispatchConfig(
            chunk_size=chunk_size,
            uneven_shard=solver_config.uneven_shard,
            alg=MinHeapDispatchAlg(),
        )
        dispatch_meta, _ = make_dispatch_meta_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.CAUSAL] * len(nonempty_ranges),
            total_seqlen_q=total_tokens,
            total_seqlen_k=total_tokens,
            chunk_size=chunk_size,
            cp_size=cp_size,
            cp_rank=0,
            dispatch_config=dispatch_config,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            uneven_shard=solver_config.uneven_shard,
        )
        partitions = tuple(tuple(partition) for partition in dispatch_meta.partitions)
        chunk_actual_sizes = (
            None
            if dispatch_meta.chunk_actual_sizes is None
            else tuple(dispatch_meta.chunk_actual_sizes)
        )
        resolved_chunk_size = dispatch_meta.chunk_size
        num_chunks = dispatch_meta.num_chunks

    specs_per_rank: list[tuple[DsaFragmentSpec, ...]] = []
    rank_costs: list[DsaStructuralRankCost] = []
    for rank, partition in enumerate(partitions):
        rank_specs: list[DsaFragmentSpec] = []
        native_area = 0
        query_tokens = 0
        for chunk_id in partition:
            chunk_begin = chunk_id * resolved_chunk_size
            chunk_length = (
                resolved_chunk_size
                if chunk_actual_sizes is None
                else chunk_actual_sizes[chunk_id]
            )
            chunk_end = chunk_begin + chunk_length
            query_tokens += chunk_length
            native_area += _native_causal_chunk_area(
                cu_seqlens,
                chunk_begin,
                chunk_end,
            )
            for sample_id, (sample_begin, sample_end) in enumerate(
                zip(cu_seqlens, cu_seqlens[1:])
            ):
                global_begin = max(chunk_begin, sample_begin)
                global_end = min(chunk_end, sample_end)
                if global_begin < global_end:
                    rank_specs.append(
                        DsaFragmentSpec(
                            sample_id=sample_id,
                            q_begin=global_begin - sample_begin,
                            q_end=global_end - sample_begin,
                        )
                    )
        merged_specs = _merge_adjacent_specs(rank_specs)
        csa_packed_indexer_k_rows = sum(spec.q_end // 4 for spec in merged_specs)
        csa_unique_indexer_k_rows = sum(
            prefix_length
            for _, prefix_length in _csa_required_prefix_lengths(merged_specs)
        )
        specs_per_rank.append(merged_specs)
        rank_costs.append(
            DsaStructuralRankCost(
                rank=rank,
                native_causal_area=native_area,
                query_tokens=query_tokens,
                chunk_count=len(partition),
                fragment_count=len(merged_specs),
                csa_unique_indexer_k_rows=csa_unique_indexer_k_rows,
                csa_packed_indexer_k_rows=csa_packed_indexer_k_rows,
                csa_duplicate_indexer_k_rows=(
                    csa_packed_indexer_k_rows - csa_unique_indexer_k_rows
                ),
            )
        )

    return tuple(specs_per_rank), DsaStructuralLayoutMetrics(
        solver_scheme=_STRUCTURAL_SOLVER_SCHEME,
        cost_model_version=_STRUCTURAL_COST_MODEL_VERSION,
        chunk_size=resolved_chunk_size,
        num_chunks=num_chunks,
        uneven_shard=solver_config.uneven_shard,
        rank_costs=tuple(rank_costs),
    )


@dataclass(frozen=True)
class _SharedAtom:
    spec: DsaFragmentSpec
    score_cost: int
    topk_cost: int
    hca_query_cost: int
    indexer_weight: int
    remote_rows_by_rank: tuple[int, ...]


@dataclass(frozen=True)
class _SharedProblem:
    atoms: tuple[_SharedAtom, ...]
    hca_blocks: tuple[tuple[int, int, int], ...]
    sample_count: int
    cp_size: int
    indexer_head_dim: int
    total_csa_blocks: int


@dataclass(frozen=True)
class _SharedIndexerCost:
    score_cost: int
    topk_cost: int
    packed_ki_rows: int
    unique_ki_rows: int
    modeled_ki_bytes: int
    indexer_cost: int
    fragment_count: int
    local_query_count: int


def _make_shared_problem(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    source_counts: tuple[int, ...],
) -> _SharedProblem:
    source_offsets = _prefix_offsets(source_counts)
    atoms: list[_SharedAtom] = []
    sample_atom_indices: list[list[int]] = [[] for _ in range(len(cu_seqlens) - 1)]
    for spec in _make_indexer_atoms(config, cu_seqlens):
        score_cost, topk_cost = _csa_fragment_cost(config, spec)
        hca_query_cost = _hca_fragment_cost(spec)
        isolated_prefix_rows = spec.q_end // 4
        sample_begin = cu_seqlens[spec.sample_id]
        global_begin = sample_begin + spec.q_begin
        global_end = sample_begin + spec.q_end
        remote_rows = tuple(
            spec.length
            - max(
                0,
                min(global_end, source_offsets[rank + 1])
                - max(global_begin, source_offsets[rank]),
            )
            for rank in range(len(source_counts))
        )
        atom_index = len(atoms)
        atoms.append(
            _SharedAtom(
                spec=spec,
                score_cost=score_cost,
                topk_cost=topk_cost,
                hca_query_cost=hca_query_cost,
                indexer_weight=(
                    _SHARED_SCORE_WEIGHT * score_cost
                    + _SHARED_TOPK_WEIGHT * topk_cost
                    + _SHARED_KI_PACK_WEIGHT * isolated_prefix_rows
                ),
                remote_rows_by_rank=remote_rows,
            )
        )
        sample_atom_indices[spec.sample_id].append(atom_index)

    hca_blocks: list[tuple[int, int, int]] = []
    for sample_id, atom_indices in enumerate(sample_atom_indices):
        sample_length = cu_seqlens[sample_id + 1] - cu_seqlens[sample_id]
        atom_cursor = 0
        for block_id in range(sample_length // 128):
            block_end = 128 * (block_id + 1)
            endpoint = block_end - 1
            while atoms[atom_indices[atom_cursor]].spec.q_end <= endpoint:
                atom_cursor += 1
            atom_index = atom_indices[atom_cursor]
            atom_spec = atoms[atom_index].spec
            if not atom_spec.q_begin <= endpoint < atom_spec.q_end:
                raise RuntimeError("HCA block endpoint is missing from shared atoms")
            hca_blocks.append((sample_id, block_end, atom_index))
    return _SharedProblem(
        atoms=tuple(atoms),
        hca_blocks=tuple(hca_blocks),
        sample_count=len(cu_seqlens) - 1,
        cp_size=len(source_counts),
        indexer_head_dim=config.indexer_head_dim,
        total_csa_blocks=sum(
            (sample_end - sample_begin) // 4
            for sample_begin, sample_end in zip(cu_seqlens, cu_seqlens[1:])
        ),
    )


def _shared_rank_indexer_cost(
    problem: _SharedProblem,
    atom_indices: Sequence[int],
    *,
    workspace_reserve_bytes: int,
) -> _SharedIndexerCost:
    score_cost = sum(problem.atoms[index].score_cost for index in atom_indices)
    topk_cost = sum(problem.atoms[index].topk_cost for index in atom_indices)
    fragments = _merge_adjacent_specs(
        [problem.atoms[index].spec for index in atom_indices]
    )
    packed_ki_rows = sum(fragment.q_end // 4 for fragment in fragments)
    sample_max_ki_rows = [0] * problem.sample_count
    for fragment in fragments:
        sample_max_ki_rows[fragment.sample_id] = max(
            sample_max_ki_rows[fragment.sample_id], fragment.q_end // 4
        )
    unique_ki_rows = sum(sample_max_ki_rows)
    local_query_count = sum(fragment.length for fragment in fragments)
    fragment_count = len(fragments)

    # The layout-dependent byte ledger covers the BF16 grouped-K materialization
    # and every CUDA int32 tensor in DsaDeviceIndexerMap. The Indexer selection is
    # no-grad, so there is no same-shaped grouped-K backward gradient. Backend
    # scratch and allocator headroom are supplied explicitly by the caller.
    grouped_ki_bytes = packed_ki_rows * problem.indexer_head_dim * _BF16_BYTES
    packed_row_map_bytes = 2 * packed_ki_rows * _INT32_BYTES
    unique_row_map_bytes = (unique_ki_rows + 1) * _INT32_BYTES
    global_row_map_bytes = problem.total_csa_blocks * _INT32_BYTES
    grouped_sequence_map_bytes = (3 * fragment_count + 2) * _INT32_BYTES
    query_map_bytes = 2 * local_query_count * _INT32_BYTES
    modeled_ki_bytes = (
        grouped_ki_bytes
        + packed_row_map_bytes
        + unique_row_map_bytes
        + global_row_map_bytes
        + grouped_sequence_map_bytes
        + query_map_bytes
        + workspace_reserve_bytes
    )
    indexer_cost = (
        _SHARED_SCORE_WEIGHT * score_cost
        + _SHARED_TOPK_WEIGHT * topk_cost
        + _SHARED_KI_PACK_WEIGHT * packed_ki_rows
    )
    return _SharedIndexerCost(
        score_cost=score_cost,
        topk_cost=topk_cost,
        packed_ki_rows=packed_ki_rows,
        unique_ki_rows=unique_ki_rows,
        modeled_ki_bytes=modeled_ki_bytes,
        indexer_cost=indexer_cost,
        fragment_count=fragment_count,
        local_query_count=local_query_count,
    )


def _shared_specs_per_rank(
    problem: _SharedProblem,
    owners: Sequence[int],
) -> tuple[tuple[DsaFragmentSpec, ...], ...]:
    if len(owners) != len(problem.atoms):
        raise ValueError("shared owner table has the wrong atom count")
    specs: list[list[DsaFragmentSpec]] = [[] for _ in range(problem.cp_size)]
    for atom, owner in zip(problem.atoms, owners):
        if not 0 <= owner < problem.cp_size:
            raise ValueError("shared owner table contains an invalid rank")
        specs[owner].append(atom.spec)
    return tuple(_merge_adjacent_specs(rank_specs) for rank_specs in specs)


def _evaluate_shared_layout(
    problem: _SharedProblem,
    owners: Sequence[int],
    *,
    solver_config: DsaSharedLayoutConfig | None,
) -> DsaLayoutMetrics | None:
    specs_per_rank = _shared_specs_per_rank(problem, owners)
    rank_atom_indices: list[list[int]] = [[] for _ in range(problem.cp_size)]
    hca_query_costs = [0] * problem.cp_size
    remote_rows = [0] * problem.cp_size
    max_positions = [[-1] * problem.cp_size for _ in range(problem.sample_count)]
    for atom_index, (atom, owner) in enumerate(zip(problem.atoms, owners)):
        rank_atom_indices[owner].append(atom_index)
        hca_query_costs[owner] += atom.hca_query_cost
        remote_rows[owner] += atom.remote_rows_by_rank[owner]
        max_positions[atom.spec.sample_id][owner] = max(
            max_positions[atom.spec.sample_id][owner], atom.spec.q_end - 1
        )

    indexer_rows: list[_SharedIndexerCost] = []
    for atom_indices in rank_atom_indices:
        values = _shared_rank_indexer_cost(
            problem,
            atom_indices,
            workspace_reserve_bytes=(
                0 if solver_config is None else solver_config.ki_workspace_reserve_bytes
            ),
        )
        if (
            solver_config is not None
            and values.modeled_ki_bytes > solver_config.ki_memory_budget_bytes
        ):
            return None
        indexer_rows.append(values)

    route_matrix = [[0] * problem.cp_size for _ in range(problem.cp_size)]
    for sample_id, block_end, producer_atom in problem.hca_blocks:
        producer = owners[producer_atom]
        endpoint = block_end - 1
        for consumer in range(problem.cp_size):
            if max_positions[sample_id][consumer] >= endpoint:
                route_matrix[consumer][producer] += 1

    rank_costs: list[DsaLayoutRankCost] = []
    for rank in range(problem.cp_size):
        indexer_row = indexer_rows[rank]
        send_rows = sum(route_matrix[rank])
        recv_rows = sum(row[rank] for row in route_matrix)
        max_peer_rows = max(
            (*route_matrix[rank], *(row[rank] for row in route_matrix)),
            default=0,
        )
        hca_cost = (
            _SHARED_HCA_QUERY_WEIGHT * hca_query_costs[rank]
            + _SHARED_HCA_ROUTE_WEIGHT * (send_rows + recv_rows)
            + _SHARED_HCA_PEER_WEIGHT * max_peer_rows
        )
        rank_costs.append(
            DsaLayoutRankCost(
                rank=rank,
                indexer_score_cost=indexer_row.score_cost,
                indexer_topk_cost=indexer_row.topk_cost,
                packed_ki_rows=indexer_row.packed_ki_rows,
                unique_ki_rows=indexer_row.unique_ki_rows,
                modeled_ki_bytes=indexer_row.modeled_ki_bytes,
                indexer_cost=indexer_row.indexer_cost,
                hca_query_cost=hca_query_costs[rank],
                hca_send_rows=send_rows,
                hca_recv_rows=recv_rows,
                hca_max_peer_rows=max_peer_rows,
                hca_cost=hca_cost,
                token_layout_remote_rows=remote_rows[rank],
                fragment_count=len(specs_per_rank[rank]),
            )
        )
    key = (
        max(cost.indexer_cost for cost in rank_costs),
        max(cost.hca_cost for cost in rank_costs),
        sum(cost.token_layout_remote_rows for cost in rank_costs),
        sum(cost.fragment_count for cost in rank_costs),
    )
    return DsaLayoutMetrics(
        solver_scheme=_SHARED_SOLVER_SCHEME,
        cost_model_version=_SHARED_COST_MODEL_VERSION,
        key=key,
        rank_costs=tuple(rank_costs),
        candidate_evaluations=0,
        improvement_steps=0,
        stop_reason="evaluation_only",
    )


def _best_shared_neighbor(
    problem: _SharedProblem,
    owners: tuple[int, ...],
    current: DsaLayoutMetrics,
    solver_config: DsaSharedLayoutConfig,
    focus_ranks: set[int],
    *,
    require_indexer_improvement: bool,
) -> tuple[tuple[tuple[int, ...], DsaLayoutMetrics] | None, int]:
    best_owners: tuple[int, ...] | None = None
    best_metrics: DsaLayoutMetrics | None = None
    seen: set[tuple[int, ...]] = set()
    candidate_evaluations = 0
    focused_atoms = [
        index for index, owner in enumerate(owners) if owner in focus_ranks
    ]

    def consider(candidate: tuple[int, ...]) -> None:
        nonlocal best_owners, best_metrics, candidate_evaluations
        if candidate in seen:
            return
        seen.add(candidate)
        candidate_evaluations += 1
        metrics = _evaluate_shared_layout(
            problem,
            candidate,
            solver_config=solver_config,
        )
        if metrics is None or metrics.key >= current.key:
            return
        if require_indexer_improvement and metrics.key[0] >= current.key[0]:
            return
        if best_metrics is None or metrics.key < best_metrics.key:
            best_owners = candidate
            best_metrics = metrics

    for atom_index in focused_atoms:
        source = owners[atom_index]
        for destination in range(problem.cp_size):
            if destination == source:
                continue
            moved = list(owners)
            moved[atom_index] = destination
            consider(tuple(moved))

            target_atoms = [
                target for target, owner in enumerate(owners) if owner == destination
            ]
            target_atoms.sort(
                key=lambda target: (
                    abs(
                        problem.atoms[target].indexer_weight
                        - problem.atoms[atom_index].indexer_weight
                    ),
                    abs(
                        problem.atoms[target].hca_query_cost
                        - problem.atoms[atom_index].hca_query_cost
                    ),
                    problem.atoms[target].spec.sample_id,
                    problem.atoms[target].spec.q_begin,
                    target,
                )
            )
            for target in target_atoms[:_SHARED_SWAP_CANDIDATES_PER_RANK]:
                swapped = list(owners)
                swapped[atom_index] = destination
                swapped[target] = source
                consider(tuple(swapped))

    if best_owners is None or best_metrics is None:
        return None, candidate_evaluations
    return (best_owners, best_metrics), candidate_evaluations


@lru_cache(maxsize=16)
def _assign_shared_greedy(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    source_counts: tuple[int, ...],
    solver_config: DsaSharedLayoutConfig,
) -> tuple[tuple[tuple[DsaFragmentSpec, ...], ...], DsaLayoutMetrics]:
    problem = _make_shared_problem(
        config,
        cu_seqlens,
        source_counts,
    )
    ordered_atoms = sorted(
        range(len(problem.atoms)),
        key=lambda index: (
            -problem.atoms[index].indexer_weight,
            -problem.atoms[index].score_cost,
            -problem.atoms[index].topk_cost,
            -problem.atoms[index].spec.length,
            problem.atoms[index].spec.sample_id,
            problem.atoms[index].spec.q_begin,
            index,
        ),
    )
    owners = [-1] * len(problem.atoms)
    rank_atoms: list[list[int]] = [[] for _ in range(problem.cp_size)]
    empty_indexer = _shared_rank_indexer_cost(
        problem,
        (),
        workspace_reserve_bytes=solver_config.ki_workspace_reserve_bytes,
    )
    rank_indexer = [empty_indexer] * problem.cp_size
    candidate_evaluations = 0
    for atom_index in ordered_atoms:
        candidates: list[tuple[tuple[int, ...], int, _SharedIndexerCost]] = []
        for rank in range(problem.cp_size):
            candidate_evaluations += 1
            values = _shared_rank_indexer_cost(
                problem,
                (*rank_atoms[rank], atom_index),
                workspace_reserve_bytes=solver_config.ki_workspace_reserve_bytes,
            )
            if values.modeled_ki_bytes > solver_config.ki_memory_budget_bytes:
                continue
            next_costs = [row.indexer_cost for row in rank_indexer]
            next_costs[rank] = values.indexer_cost
            next_memory = [row.modeled_ki_bytes for row in rank_indexer]
            next_memory[rank] = values.modeled_ki_bytes
            next_fragments = [row.fragment_count for row in rank_indexer]
            next_fragments[rank] = values.fragment_count
            objective = (
                max(next_costs),
                max(next_memory),
                sum(next_fragments),
                values.indexer_cost,
                rank,
            )
            candidates.append((objective, rank, values))
        if not candidates:
            spec = problem.atoms[atom_index].spec
            raise RuntimeError(
                "shared_greedy found no feasible rank for band "
                f"sample={spec.sample_id} range=[{spec.q_begin},{spec.q_end}); "
                "this is a search failure, not a proof that the layout is infeasible"
            )
        _, destination, values = min(candidates, key=lambda item: item[0])
        owners[atom_index] = destination
        rank_atoms[destination].append(atom_index)
        rank_indexer[destination] = values

    owner_tuple = tuple(owners)
    metrics = _evaluate_shared_layout(
        problem,
        owner_tuple,
        solver_config=solver_config,
    )
    candidate_evaluations += 1
    if metrics is None:
        raise RuntimeError("shared_greedy constructed an over-budget Query layout")

    improvement_steps = 0
    stop_reason = "pass_limit"
    for _ in range(solver_config.local_improvement_passes):
        indexer_max = metrics.key[0]
        indexer_focus = {
            cost.rank for cost in metrics.rank_costs if cost.indexer_cost == indexer_max
        }
        neighbor, evaluated = _best_shared_neighbor(
            problem,
            owner_tuple,
            metrics,
            solver_config,
            indexer_focus,
            require_indexer_improvement=True,
        )
        candidate_evaluations += evaluated
        if neighbor is None:
            hca_max = metrics.key[1]
            hca_focus = {
                cost.rank for cost in metrics.rank_costs if cost.hca_cost == hca_max
            }
            neighbor, evaluated = _best_shared_neighbor(
                problem,
                owner_tuple,
                metrics,
                solver_config,
                hca_focus,
                require_indexer_improvement=False,
            )
            candidate_evaluations += evaluated
        if neighbor is None:
            stop_reason = "local_optimum"
            break
        owner_tuple, metrics = neighbor
        improvement_steps += 1

    metrics = replace(
        metrics,
        candidate_evaluations=candidate_evaluations,
        improvement_steps=improvement_steps,
        stop_reason=stop_reason,
    )

    return _shared_specs_per_rank(problem, owner_tuple), metrics


def _assign_indexer_atoms(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    query_counts: tuple[int, ...],
) -> tuple[tuple[DsaFragmentSpec, ...], ...]:
    """Assign cost-heavy atoms first while filling exact per-rank token counts."""

    cp_size = len(query_counts)
    atoms = _make_indexer_atoms(config, cu_seqlens)
    total_score = sum(_fragment_cost(config, atom)[0] for atom in atoms)
    total_topk = sum(_fragment_cost(config, atom)[1] for atom in atoms)
    score_target = max(total_score / cp_size, 1.0)
    topk_target = max(total_topk / cp_size, 1.0)

    pending: list[tuple[float, int, int, int, int, int, int, DsaFragmentSpec]] = []
    serial = 0

    def push(spec: DsaFragmentSpec) -> None:
        nonlocal serial
        score_cost, topk_cost = _fragment_cost(config, spec)
        heapq.heappush(
            pending,
            (
                -max(score_cost / score_target, topk_cost / topk_target),
                -score_cost,
                -topk_cost,
                -spec.length,
                spec.sample_id,
                spec.q_begin,
                serial,
                spec,
            ),
        )
        serial += 1

    for atom in atoms:
        push(atom)

    assignment: list[list[DsaFragmentSpec]] = [[] for _ in range(cp_size)]
    remaining = list(query_counts)
    score_load = [0] * cp_size
    topk_load = [0] * cp_size
    while pending:
        *_, spec = heapq.heappop(pending)
        score_cost, topk_cost = _fragment_cost(config, spec)
        candidates = [rank for rank in range(cp_size) if remaining[rank] >= spec.length]
        if not candidates:
            split_length = max(remaining)
            if not 0 < split_length < spec.length:
                raise RuntimeError("unable to split an Indexer atom for exact capacity")
            split = spec.q_end - split_length
            push(DsaFragmentSpec(spec.sample_id, spec.q_begin, split))
            push(DsaFragmentSpec(spec.sample_id, split, spec.q_end))
            continue

        def objective(rank: int) -> tuple[float, float, float, int, int]:
            next_score = score_load[rank] + score_cost
            next_topk = topk_load[rank] + topk_cost
            return (
                max(next_score / score_target, next_topk / topk_target),
                next_score / score_target + next_topk / topk_target,
                max(next_score, next_topk),
                len(assignment[rank]),
                rank,
            )

        destination = min(candidates, key=objective)
        assignment[destination].append(spec)
        remaining[destination] -= spec.length
        score_load[destination] += score_cost
        topk_load[destination] += topk_cost

    if any(remaining):
        raise RuntimeError(f"incomplete final Query assignment: {remaining}")
    return tuple(_merge_adjacent_specs(specs) for specs in assignment)


def _resolve_query_fragments(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    specs_per_rank: tuple[tuple[DsaFragmentSpec, ...], ...],
    *,
    include_csa_cost: bool = False,
) -> tuple[tuple[DsaQueryFragment, ...], ...]:
    intervals_per_sample: list[list[tuple[int, int]]] = [
        [] for _ in range(len(cu_seqlens) - 1)
    ]
    resolved: list[tuple[DsaQueryFragment, ...]] = []
    for rank, specs in enumerate(specs_per_rank):
        local_begin = 0
        rank_fragments: list[DsaQueryFragment] = []
        for spec in sorted(
            specs, key=lambda item: (item.sample_id, item.q_begin, item.q_end)
        ):
            if not 0 <= spec.sample_id < len(cu_seqlens) - 1:
                raise ValueError(f"invalid sample id in Query fragment: {spec}")
            sample_begin = cu_seqlens[spec.sample_id]
            sample_length = cu_seqlens[spec.sample_id + 1] - sample_begin
            if not 0 <= spec.q_begin < spec.q_end <= sample_length:
                raise ValueError(f"invalid Query fragment: {spec}")
            score_cost, topk_cost = (
                _csa_fragment_cost(config, spec)
                if include_csa_cost
                else _fragment_cost(config, spec)
            )
            rank_fragments.append(
                DsaQueryFragment(
                    sample_id=spec.sample_id,
                    rank=rank,
                    q_begin=spec.q_begin,
                    q_end=spec.q_end,
                    global_begin=sample_begin + spec.q_begin,
                    global_end=sample_begin + spec.q_end,
                    local_begin=local_begin,
                    sample_global_begin=sample_begin,
                    score_cost=score_cost,
                    topk_cost=topk_cost,
                )
            )
            intervals_per_sample[spec.sample_id].append((spec.q_begin, spec.q_end))
            local_begin += spec.length
        resolved.append(tuple(rank_fragments))

    for sample_id, intervals in enumerate(intervals_per_sample):
        sample_length = cu_seqlens[sample_id + 1] - cu_seqlens[sample_id]
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError(
                    f"sample {sample_id} Query fragments overlap or leave a gap at {cursor}"
                )
            cursor = end
        if cursor != sample_length:
            raise ValueError(
                f"sample {sample_id} Query fragments cover {cursor} of {sample_length} tokens"
            )
    return tuple(resolved)


def _query_producers(
    fragments_per_rank: tuple[tuple[DsaQueryFragment, ...], ...],
    query_counts: tuple[int, ...],
    total_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    owners = [-1] * total_tokens
    local_rows = [-1] * total_tokens
    for rank, fragments in enumerate(fragments_per_rank):
        for fragment in fragments:
            for offset in range(fragment.length):
                global_row = fragment.global_begin + offset
                local_row = fragment.local_begin + offset
                if owners[global_row] >= 0:
                    raise ValueError("final Query fragments overlap")
                owners[global_row] = rank
                local_rows[global_row] = local_row
        if sum(fragment.length for fragment in fragments) != query_counts[rank]:
            raise ValueError("final Query fragments do not match rank capacity")
    if any(owner < 0 for owner in owners):
        raise ValueError("final Query fragments do not cover every token")
    return tuple(owners), tuple(local_rows)


def _source_producers(
    source_counts: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    owners: list[int] = []
    local_rows: list[int] = []
    for rank, count in enumerate(source_counts):
        owners.extend([rank] * count)
        local_rows.extend(range(count))
    return tuple(owners), tuple(local_rows)


def _build_compression_blocks(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    query_owners: tuple[int, ...],
    cp_size: int,
) -> tuple[tuple[DsaCompressionBlock, ...], tuple[int, ...], tuple[int, ...]]:
    if config.ratio == 0:
        zeros = tuple(0 for _ in range(len(cu_seqlens) - 1))
        return (), zeros, zeros

    blocks: list[DsaCompressionBlock] = []
    sample_offsets: list[int] = []
    sample_counts: list[int] = []
    producer_counts = [0] * cp_size
    ratio = config.ratio
    for sample_id, (sample_begin, sample_end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        sample_offsets.append(len(blocks))
        block_count = (sample_end - sample_begin) // ratio
        sample_counts.append(block_count)
        for sample_block_id in range(block_count):
            global_begin = sample_begin + sample_block_id * ratio
            global_end = global_begin + ratio
            producer_rank = query_owners[global_end - 1]
            producer_local_index = producer_counts[producer_rank]
            producer_counts[producer_rank] += 1
            if ratio == 4:
                previous = (
                    tuple(range(global_begin - ratio, global_begin))
                    if sample_block_id > 0
                    else (-1,) * ratio
                )
                source_rows = (*previous, *range(global_begin, global_end))
            else:
                source_rows = tuple(range(global_begin, global_end))
            blocks.append(
                DsaCompressionBlock(
                    global_block_id=len(blocks),
                    sample_id=sample_id,
                    sample_block_id=sample_block_id,
                    global_begin=global_begin,
                    global_end=global_end,
                    producer_rank=producer_rank,
                    producer_local_index=producer_local_index,
                    position=sample_block_id * ratio,
                    source_global_rows=tuple(source_rows),
                )
            )
    return tuple(blocks), tuple(sample_offsets), tuple(sample_counts)


def _compressed_producers(
    blocks: tuple[DsaCompressionBlock, ...], cp_size: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    owners = tuple(block.producer_rank for block in blocks)
    local_rows = tuple(block.producer_local_index for block in blocks)
    counts = [0] * cp_size
    for block in blocks:
        counts[block.producer_rank] = max(
            counts[block.producer_rank], block.producer_local_index + 1
        )
    return owners, local_rows, tuple(counts)


def _build_route(
    name: str,
    producer_owner: tuple[int, ...],
    producer_local_row: tuple[int, ...],
    producer_row_counts: tuple[int, ...],
    consumer_rows: tuple[tuple[int, ...], ...],
) -> DsaTypedRoutePlan:
    cp_size = len(producer_row_counts)
    if len(consumer_rows) != cp_size:
        raise ValueError(f"{name}: consumer row table has the wrong CP size")
    row_count = len(producer_owner)
    if len(producer_local_row) != row_count:
        raise ValueError(f"{name}: producer maps have different lengths")
    for global_row, (owner, local_row) in enumerate(
        zip(producer_owner, producer_local_row)
    ):
        if not 0 <= owner < cp_size:
            raise ValueError(f"{name}: row {global_row} has invalid owner {owner}")
        if not 0 <= local_row < producer_row_counts[owner]:
            raise ValueError(
                f"{name}: row {global_row} has invalid owner-local row {local_row}"
            )

    producer_global_rows: list[list[int]] = [
        [-1] * producer_row_count for producer_row_count in producer_row_counts
    ]
    for global_row, (owner, local_row) in enumerate(
        zip(producer_owner, producer_local_row)
    ):
        if producer_global_rows[owner][local_row] >= 0:
            raise ValueError(
                f"{name}: owner {owner} local row {local_row} has two producers"
            )
        producer_global_rows[owner][local_row] = global_row
    if any(global_row < 0 for rows in producer_global_rows for global_row in rows):
        raise ValueError(f"{name}: producer-local rows are not a complete bijection")

    normalized_consumers: list[tuple[int, ...]] = []
    destinations_per_global_row: list[list[int]] = [[] for _ in range(row_count)]
    for rank, rank_rows in enumerate(consumer_rows):
        normalized = tuple(int(row) for row in rank_rows)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name}: rank {rank} consumer rows are not unique")
        if any(row < 0 or row >= row_count for row in normalized):
            raise ValueError(f"{name}: rank {rank} consumes an invalid global row")
        normalized_consumers.append(normalized)
        for global_row in normalized:
            destinations_per_global_row[global_row].append(rank)

    group_args: list[DsaGroupCollectiveArg] = []
    for rank in range(cp_size):
        local_destination_sets = tuple(
            tuple(destinations_per_global_row[global_row])
            for global_row in producer_global_rows[rank]
        )
        input_split_sizes: list[int] = []
        dst_indices_list: list[tuple[int, ...]] = []
        for destinations in local_destination_sets:
            if dst_indices_list and destinations == dst_indices_list[-1]:
                input_split_sizes[-1] += 1
            else:
                input_split_sizes.append(1)
                dst_indices_list.append(destinations)

        consumer_sources = tuple(
            producer_owner[global_row] for global_row in normalized_consumers[rank]
        )
        output_split_sizes: list[int] = []
        src_index_list: list[int] = []
        for source in consumer_sources:
            if src_index_list and source == src_index_list[-1]:
                output_split_sizes[-1] += 1
            else:
                output_split_sizes.append(1)
                src_index_list.append(source)

        for source in range(cp_size):
            source_local_rows = [
                producer_local_row[global_row]
                for global_row in normalized_consumers[rank]
                if producer_owner[global_row] == source
            ]
            if source_local_rows != sorted(source_local_rows):
                raise ValueError(
                    f"{name}: rank {rank} requests source {source} rows out of "
                    "producer-buffer order"
                )

        group_args.append(
            DsaGroupCollectiveArg(
                rank=rank,
                world_size=cp_size,
                input_split_size_list=tuple(input_split_sizes),
                output_split_size_list=tuple(output_split_sizes),
                dst_indices_list=tuple(dst_indices_list),
                src_index_list=tuple(src_index_list),
            )
        )

    send_counts = [[0] * cp_size for _ in range(cp_size)]
    recv_counts = [[0] * cp_size for _ in range(cp_size)]
    send_rows: list[list[int]] = [[] for _ in range(cp_size)]
    received_rows: list[list[int]] = [[] for _ in range(cp_size)]
    for source, group_arg in enumerate(group_args):
        input_segments: list[tuple[int, int, tuple[int, ...]]] = []
        local_begin = 0
        for split_size, destinations in zip(
            group_arg.input_split_size_list,
            group_arg.dst_indices_list,
        ):
            local_end = local_begin + split_size
            input_segments.append((local_begin, local_end, destinations))
            local_begin = local_end
        if local_begin != producer_row_counts[source]:
            raise ValueError(f"{name}: GroupCollectiveArg does not cover producer rows")

        for destination in range(cp_size):
            destination_rows: list[int] = []
            for begin, end, destinations in input_segments:
                if destination in destinations:
                    destination_rows.extend(range(begin, end))
            send_counts[source][destination] = len(destination_rows)
            send_rows[source].extend(destination_rows)

    for destination, group_arg in enumerate(group_args):
        for source in range(cp_size):
            expected_receive_count = sum(
                split_size
                for split_size, split_source in zip(
                    group_arg.output_split_size_list,
                    group_arg.src_index_list,
                )
                if split_source == source
            )
            if send_counts[source][destination] != expected_receive_count:
                raise ValueError(
                    f"{name}: GroupCollectiveArg send/receive counts disagree for "
                    f"{source}->{destination}"
                )
            recv_counts[destination][source] = expected_receive_count
            source_send_begin = sum(send_counts[source][:destination])
            source_send_end = source_send_begin + expected_receive_count
            received_rows[destination].extend(
                producer_global_rows[source][local_row]
                for local_row in send_rows[source][source_send_begin:source_send_end]
            )

    rank_plans: list[DsaRouteRankPlan] = []
    for rank in range(cp_size):
        received_position = {
            row: index for index, row in enumerate(received_rows[rank])
        }
        consumer_from_received = tuple(
            received_position[row] for row in normalized_consumers[rank]
        )
        received_from_consumer_list = [-1] * len(received_rows[rank])
        for consumer_index, received_index in enumerate(consumer_from_received):
            if received_from_consumer_list[received_index] != -1:
                raise ValueError(
                    f"{name}: rank {rank} receive permutation is not invertible"
                )
            received_from_consumer_list[received_index] = consumer_index
        if any(index < 0 for index in received_from_consumer_list):
            raise ValueError(f"{name}: rank {rank} receive permutation has a gap")

        occurrences: list[list[int]] = [[] for _ in range(producer_row_counts[rank])]
        for reverse_source, local_row in enumerate(send_rows[rank]):
            occurrences[local_row].append(reverse_source)
        row_offsets = [0]
        reverse_source_rows: list[int] = []
        for row_occurrences in occurrences:
            reverse_source_rows.extend(row_occurrences)
            row_offsets.append(len(reverse_source_rows))
        rank_plans.append(
            DsaRouteRankPlan(
                rank=rank,
                producer_row_count=producer_row_counts[rank],
                group_collective_arg=group_args[rank],
                send_counts=tuple(send_counts[rank]),
                recv_counts=tuple(recv_counts[rank]),
                send_source_rows=tuple(send_rows[rank]),
                received_global_rows=tuple(received_rows[rank]),
                consumer_global_rows=normalized_consumers[rank],
                consumer_from_received=consumer_from_received,
                received_from_consumer=tuple(received_from_consumer_list),
                reverse_row_offsets=tuple(row_offsets),
                reverse_source_rows=tuple(reverse_source_rows),
            )
        )
    return DsaTypedRoutePlan(name=name, rank_plans=tuple(rank_plans))


def _query_metadata(
    fragments: tuple[DsaQueryFragment, ...], local_count: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    sample_ids = [-1] * local_count
    positions = [-1] * local_count
    global_rows = [-1] * local_count
    for fragment in fragments:
        for offset in range(fragment.length):
            local_row = fragment.local_begin + offset
            sample_ids[local_row] = fragment.sample_id
            positions[local_row] = fragment.q_begin + offset
            global_rows[local_row] = fragment.global_begin + offset
    if any(value < 0 for value in (*sample_ids, *positions, *global_rows)):
        raise ValueError("Query fragments do not cover every local row exactly once")
    return tuple(sample_ids), tuple(positions), tuple(global_rows)


def _sample_blocks(
    blocks: tuple[DsaCompressionBlock, ...], sample_count: int
) -> tuple[tuple[DsaCompressionBlock, ...], ...]:
    grouped: list[list[DsaCompressionBlock]] = [[] for _ in range(sample_count)]
    for block in blocks:
        grouped[block.sample_id].append(block)
    return tuple(tuple(sample_rows) for sample_rows in grouped)


def _hash_plan_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dsa_execution_plan(
    config: MagiDSAConfig,
    cu_seqlens: Sequence[int],
    source_token_counts: Sequence[int],
    *,
    policy: DsaPlanPolicy = "indexer_balanced",
    shared_layout_config: DsaSharedLayoutConfig | None = None,
    structural_layout_config: DsaStructuralLayoutConfig | None = None,
) -> DsaExecutionPlan:
    """Build a source-layout to final-Query-layout DSA execution plan."""

    if policy not in (
        "sequential",
        "indexer_balanced",
        "shared_greedy",
        "structural_balanced",
    ):
        raise ValueError(f"unsupported DSA policy {policy!r}")
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
    cu = _validate_cu_seqlens(cu_seqlens)
    source_counts = _validate_source_counts(source_token_counts, cu[-1])
    cp_size = len(source_counts)

    layout_metrics: DsaLayoutMetrics | DsaStructuralLayoutMetrics | None = None
    if policy == "shared_greedy":
        assert shared_layout_config is not None
        shared_cost_config = replace(config, ratio=4)
        specs_per_rank, layout_metrics = _assign_shared_greedy(
            shared_cost_config,
            cu,
            source_counts,
            shared_layout_config,
        )
        query_counts = tuple(
            sum(spec.length for spec in specs) for specs in specs_per_rank
        )
    elif policy == "structural_balanced":
        assert structural_layout_config is not None
        specs_per_rank, layout_metrics = _assign_structural_minheap(
            cu,
            source_counts,
            structural_layout_config,
        )
        query_counts = tuple(
            sum(spec.length for spec in specs) for specs in specs_per_rank
        )
    elif config.ratio == 4:
        query_counts = _balanced_query_counts(cu[-1], cp_size)
        specs_per_rank = (
            _sequential_fragment_specs(cu, query_counts)
            if policy == "sequential"
            else _assign_indexer_atoms(config, cu, query_counts)
        )
    else:
        query_counts = source_counts
        specs_per_rank = _sequential_fragment_specs(cu, query_counts)
    query_layout_hash = _hash_plan_payload(
        [[asdict(spec) for spec in specs] for specs in specs_per_rank]
    )
    query_fragments = _resolve_query_fragments(
        config,
        cu,
        specs_per_rank,
        include_csa_cost=policy in ("shared_greedy", "structural_balanced"),
    )
    query_owners, query_local_rows = _query_producers(
        query_fragments, query_counts, cu[-1]
    )
    source_owners, source_local_rows = _source_producers(source_counts)
    source_offsets = _prefix_offsets(source_counts)

    blocks, sample_block_offsets, sample_block_counts = _build_compression_blocks(
        config, cu, query_owners, cp_size
    )
    blocks_by_sample = _sample_blocks(blocks, len(cu) - 1)
    compressed_owners, compressed_local_rows, compressed_counts = _compressed_producers(
        blocks, cp_size
    )
    indexer_required_ranges = tuple(
        (
            _csa_required_k_ranges(fragments, sample_block_offsets)
            if config.ratio == 4
            else ()
        )
        for fragments in query_fragments
    )

    layout_consumers = tuple(
        tuple(
            global_row
            for fragment in fragments
            for global_row in range(fragment.global_begin, fragment.global_end)
        )
        for fragments in query_fragments
    )
    token_layout = (
        _build_route(
            "TOKEN_LAYOUT",
            source_owners,
            source_local_rows,
            source_counts,
            layout_consumers,
        )
        if config.ratio == 4 or policy in ("shared_greedy", "structural_balanced")
        else None
    )

    window_consumers: list[tuple[int, ...]] = []
    overlap_consumers: list[tuple[int, ...]] = []
    compressed_kv_consumers: list[tuple[int, ...]] = []
    compressed_ki_consumers: list[tuple[int, ...]] = []
    for rank, fragments in enumerate(query_fragments):
        window_rows: set[int] = set()
        compressed_rows: set[int] = set()
        for fragment in fragments:
            sample_begin = cu[fragment.sample_id]
            window_begin = sample_begin + max(
                0, fragment.q_begin - config.window_size + 1
            )
            window_rows.update(range(window_begin, fragment.global_end))
            if config.ratio and config.ratio != 4:
                visible = fragment.q_end // config.ratio
                compressed_rows.update(
                    block.global_block_id
                    for block in blocks_by_sample[fragment.sample_id][:visible]
                )
        if config.ratio == 4:
            compressed_rows.update(
                global_block
                for required_range in indexer_required_ranges[rank]
                for global_block in range(
                    required_range.global_begin,
                    required_range.global_end,
                )
            )
        window_consumers.append(
            tuple(sorted(window_rows, key=lambda row: (query_owners[row], row)))
        )
        compressed_physical_rows = tuple(
            sorted(compressed_rows, key=lambda row: (compressed_owners[row], row))
        )
        compressed_kv_consumers.append(compressed_physical_rows)
        compressed_ki_consumers.append(compressed_physical_rows)
        overlap_rows = {
            source_row
            for block in blocks
            if block.producer_rank == rank
            for source_row in block.source_global_rows
            if source_row >= 0
        }
        overlap_order = (
            tuple(sorted(overlap_rows))
            if config.ratio == 128
            else tuple(sorted(overlap_rows, key=lambda row: (query_owners[row], row)))
        )
        overlap_consumers.append(overlap_order)

    window_route = _build_route(
        "WINDOW_KV",
        query_owners,
        query_local_rows,
        query_counts,
        tuple(window_consumers),
    )
    if config.ratio:
        overlap_route = _build_route(
            "OVERLAP_X",
            query_owners,
            query_local_rows,
            query_counts,
            tuple(overlap_consumers),
        )
        compressed_kv_route = _build_route(
            "COMPRESSED_KV",
            compressed_owners,
            compressed_local_rows,
            compressed_counts,
            tuple(compressed_kv_consumers),
        )
    else:
        overlap_route = None
        compressed_kv_route = None
    compressed_ki_route = (
        _build_route(
            "COMPRESSED_KI",
            compressed_owners,
            compressed_local_rows,
            compressed_counts,
            tuple(compressed_ki_consumers),
        )
        if config.ratio == 4
        else None
    )

    rank_plans: list[DsaRankPlan] = []
    for rank in range(cp_size):
        fragments = query_fragments[rank]
        local_blocks = tuple(block for block in blocks if block.producer_rank == rank)
        local_sample_ids, local_positions, local_global_rows = _query_metadata(
            fragments, query_counts[rank]
        )
        if overlap_route is not None:
            overlap_position = {
                global_row: index
                for index, global_row in enumerate(
                    overlap_route.rank_plans[rank].consumer_global_rows
                )
            }
            compression_source = tuple(
                -1 if source_row < 0 else overlap_position[source_row]
                for block in local_blocks
                for source_row in block.source_global_rows
            )
        else:
            compression_source = ()

        q_cu = [0]
        k_cu = [0]
        q_offsets: list[int] = []
        q_sample_offsets: list[int] = []
        seq_lens: list[int] = []
        max_q = 0
        max_k = 0
        score_cost = sum(fragment.score_cost for fragment in fragments)
        topk_cost = sum(fragment.topk_cost for fragment in fragments)
        if config.ratio == 4:
            for fragment in fragments:
                q_cu.append(q_cu[-1] + fragment.length)
                k_length = fragment.q_end // config.ratio
                k_cu.append(k_cu[-1] + k_length)
                q_offsets.append(fragment.q_begin)
                q_sample_offsets.extend(
                    [sample_block_offsets[fragment.sample_id]] * fragment.length
                )
                seq_lens.extend(
                    (position + 1) // config.ratio
                    for position in range(fragment.q_begin, fragment.q_end)
                )
                max_q = max(max_q, fragment.length)
                max_k = max(max_k, k_length)

        rank_plans.append(
            DsaRankPlan(
                rank=rank,
                source_token_count=source_counts[rank],
                local_token_count=query_counts[rank],
                source_global_begin=source_offsets[rank],
                source_global_end=source_offsets[rank + 1],
                query_fragments=fragments,
                local_query_global_rows=local_global_rows,
                produced_blocks=local_blocks,
                local_q_sample_ids=local_sample_ids,
                local_q_positions=local_positions,
                sample_block_offsets=sample_block_offsets,
                sample_block_counts=sample_block_counts,
                indexer_required_k_ranges=indexer_required_ranges[rank],
                compression_source_from_overlap=compression_source,
                indexer_q_cu_seqlens=tuple(q_cu),
                indexer_k_cu_seqlens=tuple(k_cu),
                indexer_q_causal_offsets=tuple(q_offsets),
                indexer_q_sample_block_offsets=tuple(q_sample_offsets),
                indexer_seq_lens=tuple(seq_lens),
                indexer_max_seqlen_q=max_q,
                indexer_logical_max_seqlen_k=max_k,
                indexer_backend_max_seqlen_k=(
                    0
                    if max_k == 0
                    else (
                        (max_k + DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT - 1)
                        // DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
                        * DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
                    )
                ),
                token_layout_route=(
                    None if token_layout is None else token_layout.rank_plans[rank]
                ),
                window_route=window_route.rank_plans[rank],
                overlap_x_route=(
                    None if overlap_route is None else overlap_route.rank_plans[rank]
                ),
                compressed_kv_route=(
                    None
                    if compressed_kv_route is None
                    else compressed_kv_route.rank_plans[rank]
                ),
                compressed_ki_route=(
                    None
                    if compressed_ki_route is None
                    else compressed_ki_route.rank_plans[rank]
                ),
                predicted_score_cost=score_cost,
                predicted_topk_cost=topk_cost,
            )
        )

    collective_order: tuple[str, ...]
    boundary_collective_order: tuple[str, ...]
    if config.ratio == 0:
        collective_order = ("WINDOW_KV",)
        boundary_collective_order = (
            ("TOKEN_LAYOUT",)
            if policy in ("shared_greedy", "structural_balanced")
            else ()
        )
    elif config.ratio == 4:
        collective_order = (
            "WINDOW_KV",
            "OVERLAP_X",
            "COMPRESSED_KI",
            "COMPRESSED_KV",
        )
        boundary_collective_order = ("TOKEN_LAYOUT",)
    else:
        collective_order = ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV")
        boundary_collective_order = (
            ("TOKEN_LAYOUT",)
            if policy in ("shared_greedy", "structural_balanced")
            else ()
        )

    payload = {
        "cu_seqlens": cu,
        "source_token_counts": source_counts,
        "query_token_counts": query_counts,
        "ratio": config.ratio,
        "policy": policy,
        "compressed_blocks": [asdict(block) for block in blocks],
        "rank_plans": [asdict(rank_plan) for rank_plan in rank_plans],
        "collective_order": collective_order,
        "boundary_collective_order": boundary_collective_order,
        "query_layout_hash": query_layout_hash,
        "layout_metrics": (None if layout_metrics is None else asdict(layout_metrics)),
        "shared_layout_config": (
            None if shared_layout_config is None else asdict(shared_layout_config)
        ),
        "structural_layout_config": (
            None
            if structural_layout_config is None
            else asdict(structural_layout_config)
        ),
    }
    plan = DsaExecutionPlan(
        cu_seqlens=cu,
        source_token_counts=source_counts,
        query_token_counts=query_counts,
        ratio=config.ratio,
        policy=policy,
        compressed_blocks=blocks,
        rank_plans=tuple(rank_plans),
        collective_order=collective_order,
        boundary_collective_order=boundary_collective_order,
        query_layout_hash=query_layout_hash,
        layout_metrics=layout_metrics,
        shared_layout_config=shared_layout_config,
        structural_layout_config=structural_layout_config,
        plan_hash=_hash_plan_payload(payload),
    )
    validate_dsa_execution_plan(plan)
    return plan


def _validate_routes(plan: DsaExecutionPlan, field: str) -> None:
    routes = [getattr(rank_plan, field) for rank_plan in plan.rank_plans]
    if all(route is None for route in routes):
        return
    if any(route is None for route in routes):
        raise ValueError(f"{field} must be present or absent on every rank")
    concrete = [route for route in routes if route is not None]
    for source, source_route in enumerate(concrete):
        if source_route.rank != source:
            raise ValueError(f"{field} rank metadata mismatch")
        group_arg = source_route.group_collective_arg
        if group_arg.rank != source or group_arg.world_size != plan.cp_size:
            raise ValueError(f"{field} GroupCollectiveArg rank metadata mismatch")
        if group_arg.deterministic or group_arg.split_alignment != 1:
            raise ValueError(f"{field} GroupCollectiveArg execution flags changed")
        if len(group_arg.input_split_size_list) != len(group_arg.dst_indices_list):
            raise ValueError(f"{field} GroupCollectiveArg input fields disagree")
        if len(group_arg.output_split_size_list) != len(group_arg.src_index_list):
            raise ValueError(f"{field} GroupCollectiveArg output fields disagree")
        if any(split_size <= 0 for split_size in group_arg.input_split_size_list):
            raise ValueError(f"{field} GroupCollectiveArg has an empty input split")
        if any(split_size <= 0 for split_size in group_arg.output_split_size_list):
            raise ValueError(f"{field} GroupCollectiveArg has an empty output split")
        if sum(group_arg.input_split_size_list) != source_route.producer_row_count:
            raise ValueError(f"{field} GroupCollectiveArg misses producer rows")
        if sum(group_arg.output_split_size_list) != len(
            source_route.consumer_global_rows
        ):
            raise ValueError(f"{field} GroupCollectiveArg misses consumer rows")
        if any(
            tuple(sorted(set(destinations))) != destinations
            or any(not 0 <= destination < plan.cp_size for destination in destinations)
            for destinations in group_arg.dst_indices_list
        ):
            raise ValueError(f"{field} GroupCollectiveArg has invalid destinations")
        if any(not 0 <= rank < plan.cp_size for rank in group_arg.src_index_list):
            raise ValueError(f"{field} GroupCollectiveArg has an invalid source")
        if any(
            left == right
            for left, right in zip(
                group_arg.dst_indices_list,
                group_arg.dst_indices_list[1:],
            )
        ):
            raise ValueError(f"{field} GroupCollectiveArg input RLE is not canonical")
        if any(
            left == right
            for left, right in zip(
                group_arg.src_index_list,
                group_arg.src_index_list[1:],
            )
        ):
            raise ValueError(f"{field} GroupCollectiveArg output RLE is not canonical")

        expected_send_counts = [0] * plan.cp_size
        expected_send_rows: list[int] = []
        input_segments: list[tuple[int, int, tuple[int, ...]]] = []
        local_begin = 0
        for split_size, destinations in zip(
            group_arg.input_split_size_list,
            group_arg.dst_indices_list,
        ):
            local_end = local_begin + split_size
            input_segments.append((local_begin, local_end, destinations))
            local_begin = local_end
        for destination in range(plan.cp_size):
            for begin, end, destinations in input_segments:
                if destination in destinations:
                    expected_send_counts[destination] += end - begin
                    expected_send_rows.extend(range(begin, end))
        if tuple(expected_send_counts) != source_route.send_counts:
            raise ValueError(f"{field} send counts do not match GroupCollectiveArg")
        if tuple(expected_send_rows) != source_route.send_source_rows:
            raise ValueError(f"{field} send buffer does not match GroupCollectiveArg")

        expected_recv_counts = tuple(
            sum(
                split_size
                for split_size, split_source in zip(
                    group_arg.output_split_size_list,
                    group_arg.src_index_list,
                )
                if split_source == peer
            )
            for peer in range(plan.cp_size)
        )
        if expected_recv_counts != source_route.recv_counts:
            raise ValueError(f"{field} receive counts do not match GroupCollectiveArg")
        if source_route.send_row_count != len(source_route.send_source_rows):
            raise ValueError(f"{field} send rows do not match split counts")
        if source_route.received_row_count != len(source_route.received_global_rows):
            raise ValueError(f"{field} receive rows do not match split counts")
        if len(source_route.consumer_from_received) != len(
            source_route.consumer_global_rows
        ):
            raise ValueError(f"{field} consumer permutation has an invalid length")
        if len(source_route.received_from_consumer) != len(
            source_route.received_global_rows
        ):
            raise ValueError(f"{field} inverse permutation has an invalid length")
        if len(set(source_route.received_global_rows)) != len(
            source_route.received_global_rows
        ) or len(set(source_route.consumer_global_rows)) != len(
            source_route.consumer_global_rows
        ):
            raise ValueError(f"{field} route contains duplicate consumer rows")
        if sorted(source_route.consumer_from_received) != list(
            range(len(source_route.received_global_rows))
        ):
            raise ValueError(f"{field} consumer permutation is not bijective")
        if source_route.consumer_global_rows != tuple(
            source_route.received_global_rows[received_row]
            for received_row in source_route.consumer_from_received
        ):
            raise ValueError(f"{field} consumer permutation changes global rows")
        for consumer_row, received_row in enumerate(
            source_route.consumer_from_received
        ):
            if source_route.received_from_consumer[received_row] != consumer_row:
                raise ValueError(f"{field} receive permutations are not inverses")
        for destination, send_count in enumerate(source_route.send_counts):
            if send_count != concrete[destination].recv_counts[source]:
                raise ValueError(f"{field} send/receive counts are asymmetric")
        if len(source_route.reverse_row_offsets) != source_route.producer_row_count + 1:
            raise ValueError(f"{field} reverse CSR row count is invalid")
        if (
            not source_route.reverse_row_offsets
            or source_route.reverse_row_offsets[0] != 0
        ):
            raise ValueError(f"{field} reverse CSR offsets do not start at zero")
        if any(
            end < begin
            for begin, end in zip(
                source_route.reverse_row_offsets,
                source_route.reverse_row_offsets[1:],
            )
        ):
            raise ValueError(f"{field} reverse CSR offsets are not monotonic")
        if source_route.reverse_row_offsets[-1] != len(
            source_route.reverse_source_rows
        ):
            raise ValueError(f"{field} reverse CSR item count is invalid")
        if sorted(source_route.reverse_source_rows) != list(
            range(source_route.send_row_count)
        ):
            raise ValueError(f"{field} reverse CSR misses a routed contribution")
        expected_occurrences: list[list[int]] = [
            [] for _ in range(source_route.producer_row_count)
        ]
        for source_row, producer_row in enumerate(source_route.send_source_rows):
            expected_occurrences[producer_row].append(source_row)
        expected_offsets = [0]
        expected_reverse_rows: list[int] = []
        for occurrences in expected_occurrences:
            expected_reverse_rows.extend(occurrences)
            expected_offsets.append(len(expected_reverse_rows))
        if source_route.reverse_row_offsets != tuple(
            expected_offsets
        ) or source_route.reverse_source_rows != tuple(expected_reverse_rows):
            raise ValueError(f"{field} reverse CSR does not lower the send buffer")


def validate_dsa_execution_plan(plan: DsaExecutionPlan) -> None:
    """Reject coverage, layout-bijection, route-symmetry, and DAG errors."""

    if len(plan.plan_hash) != 64:
        raise ValueError("plan hash must be a SHA-256 hex digest")
    if len(plan.query_layout_hash) != 64:
        raise ValueError("query layout hash must be a SHA-256 hex digest")
    if len(plan.rank_plans) != plan.cp_size:
        raise ValueError("rank plan count does not match CP size")
    if sum(plan.source_token_counts) != plan.total_tokens:
        raise ValueError("source layout does not cover the packed token count")
    if sum(plan.query_token_counts) != plan.total_tokens:
        raise ValueError("final Query layout does not cover the packed token count")
    if (
        plan.ratio == 4
        and plan.policy not in ("shared_greedy", "structural_balanced")
        and plan.query_token_counts
    ):
        if max(plan.query_token_counts) - min(plan.query_token_counts) > 1:
            raise ValueError("CSA final Query token capacities differ by more than one")

    covered = [0] * plan.total_tokens
    for rank, rank_plan in enumerate(plan.rank_plans):
        if rank_plan.rank != rank:
            raise ValueError("rank plan has an invalid rank")
        if rank_plan.source_token_count != plan.source_token_counts[rank]:
            raise ValueError("rank source-token count does not match the global plan")
        if rank_plan.local_token_count != plan.query_token_counts[rank]:
            raise ValueError("rank Query-token count does not match the global plan")
        local_cursor = 0
        expanded_rows: list[int] = []
        for fragment in rank_plan.query_fragments:
            if fragment.rank != rank or fragment.local_begin != local_cursor:
                raise ValueError("Query fragment local concatenation is invalid")
            local_cursor += fragment.length
            for global_row in range(fragment.global_begin, fragment.global_end):
                covered[global_row] += 1
                expanded_rows.append(global_row)
        if local_cursor != rank_plan.local_token_count:
            raise ValueError("Query fragments do not fill the rank token capacity")
        if tuple(expanded_rows) != rank_plan.local_query_global_rows:
            raise ValueError("rank local Query global-row table is inconsistent")
        if plan.ratio == 4 and rank_plan.indexer_token_count != local_cursor:
            raise ValueError(
                "grouped Indexer metadata does not cover every local Query"
            )
        if plan.ratio == 4:
            expected_q_cu = [0]
            expected_k_cu = [0]
            expected_q_offsets: list[int] = []
            expected_sample_offsets: list[int] = []
            expected_seq_lens: list[int] = []
            for fragment in rank_plan.query_fragments:
                expected_q_cu.append(expected_q_cu[-1] + fragment.length)
                expected_k_cu.append(expected_k_cu[-1] + fragment.q_end // 4)
                expected_q_offsets.append(fragment.q_begin)
                expected_sample_offsets.extend(
                    [rank_plan.sample_block_offsets[fragment.sample_id]]
                    * fragment.length
                )
                expected_seq_lens.extend(
                    (position + 1) // 4
                    for position in range(fragment.q_begin, fragment.q_end)
                )
            logical_max_k = max(
                (fragment.q_end // 4 for fragment in rank_plan.query_fragments),
                default=0,
            )
            backend_max_k = (
                0
                if logical_max_k == 0
                else (logical_max_k + DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT - 1)
                // DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
                * DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
            )
            if rank_plan.indexer_q_cu_seqlens != tuple(expected_q_cu):
                raise ValueError("grouped Indexer Q cu_seqlens are invalid")
            if rank_plan.indexer_k_cu_seqlens != tuple(expected_k_cu):
                raise ValueError("grouped Indexer K cu_seqlens are invalid")
            if rank_plan.indexer_q_causal_offsets != tuple(expected_q_offsets):
                raise ValueError("grouped Indexer causal offsets are invalid")
            if rank_plan.indexer_q_sample_block_offsets != tuple(
                expected_sample_offsets
            ):
                raise ValueError("grouped Indexer sample offsets are invalid")
            if rank_plan.indexer_seq_lens != tuple(expected_seq_lens):
                raise ValueError("grouped Indexer logical row lengths are invalid")
            if rank_plan.indexer_max_seqlen_q != max(
                (fragment.length for fragment in rank_plan.query_fragments),
                default=0,
            ):
                raise ValueError("grouped Indexer max Q length is invalid")
            if rank_plan.indexer_logical_max_seqlen_k != logical_max_k:
                raise ValueError("grouped Indexer logical max K length is invalid")
            if rank_plan.indexer_backend_max_seqlen_k != backend_max_k:
                raise ValueError("grouped Indexer backend max K width is invalid")
        elif any(
            (
                rank_plan.indexer_token_count,
                rank_plan.packed_indexer_k_count,
                len(rank_plan.indexer_q_causal_offsets),
                len(rank_plan.indexer_q_sample_block_offsets),
                len(rank_plan.indexer_seq_lens),
                rank_plan.indexer_max_seqlen_q,
                rank_plan.indexer_logical_max_seqlen_k,
                rank_plan.indexer_backend_max_seqlen_k,
            )
        ):
            raise ValueError("non-CSA plan contains grouped Indexer metadata")
        expected_required_ranges = (
            _csa_required_k_ranges(
                rank_plan.query_fragments,
                rank_plan.sample_block_offsets,
            )
            if plan.ratio == 4
            else ()
        )
        if rank_plan.indexer_required_k_ranges != expected_required_ranges:
            raise ValueError(
                "CSA required K ranges are not longest-prefix deduplicated"
            )
        if plan.ratio == 4:
            if rank_plan.compressed_ki_route is None:
                raise ValueError("CSA plan has no COMPRESSED_KI route")
            required_rows = tuple(
                global_block
                for required_range in expected_required_ranges
                for global_block in range(
                    required_range.global_begin,
                    required_range.global_end,
                )
            )
            if sorted(rank_plan.compressed_ki_route.consumer_global_rows) != list(
                required_rows
            ):
                raise ValueError("COMPRESSED_KI does not match CSA required K ranges")
    if any(count != 1 for count in covered):
        raise ValueError("Query fragments must cover every token exactly once")
    expected_layout_hash = _hash_plan_payload(
        [
            [
                asdict(
                    DsaFragmentSpec(
                        fragment.sample_id,
                        fragment.q_begin,
                        fragment.q_end,
                    )
                )
                for fragment in rank_plan.query_fragments
            ]
            for rank_plan in plan.rank_plans
        ]
    )
    if plan.query_layout_hash != expected_layout_hash:
        raise ValueError("query layout hash does not match final Query fragments")

    expected_collectives = {
        0: ("WINDOW_KV",),
        4: ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV"),
        128: ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
    }
    if plan.collective_order != expected_collectives[plan.ratio]:
        raise ValueError("DSA typed collective order does not match the ratio contract")
    expected_boundary = (
        ("TOKEN_LAYOUT",)
        if plan.ratio == 4 or plan.policy in ("shared_greedy", "structural_balanced")
        else ()
    )
    if plan.boundary_collective_order != expected_boundary:
        raise ValueError("DSA boundary collective order is invalid")

    for field in (
        "token_layout_route",
        "window_route",
        "overlap_x_route",
        "compressed_kv_route",
        "compressed_ki_route",
    ):
        _validate_routes(plan, field)

    if plan.ratio == 4 or plan.policy in (
        "shared_greedy",
        "structural_balanced",
    ):
        layout_rows = [
            row
            for rank_plan in plan.rank_plans
            for row in cast(
                DsaRouteRankPlan, rank_plan.token_layout_route
            ).consumer_global_rows
            if rank_plan.token_layout_route is not None
        ]
        if sorted(layout_rows) != list(range(plan.total_tokens)):
            raise ValueError("TOKEN_LAYOUT must be a global Query-token bijection")
    elif any(rank_plan.token_layout_route is not None for rank_plan in plan.rank_plans):
        raise ValueError("TOKEN_LAYOUT is absent from source-contiguous plans")

    if plan.policy == "shared_greedy":
        if (
            plan.shared_layout_config is None
            or plan.structural_layout_config is not None
            or not isinstance(plan.layout_metrics, DsaLayoutMetrics)
        ):
            raise ValueError("shared_greedy is missing its solver config or metrics")
        if plan.layout_metrics.solver_scheme != _SHARED_SOLVER_SCHEME:
            raise ValueError("shared layout has an unknown solver scheme")
        if plan.layout_metrics.cost_model_version != _SHARED_COST_MODEL_VERSION:
            raise ValueError("shared layout has an unknown cost model version")
        if plan.layout_metrics.candidate_evaluations <= 0:
            raise ValueError("shared layout did not record candidate evaluations")
        if (
            not 0
            <= plan.layout_metrics.improvement_steps
            <= (plan.shared_layout_config.local_improvement_passes)
        ):
            raise ValueError("shared layout improvement count is invalid")
        if plan.layout_metrics.stop_reason not in ("local_optimum", "pass_limit"):
            raise ValueError("shared layout has an invalid stop reason")
        if len(plan.layout_metrics.rank_costs) != plan.cp_size:
            raise ValueError("shared layout metrics have the wrong rank count")
        for rank, (rank_plan, rank_cost) in enumerate(
            zip(plan.rank_plans, plan.layout_metrics.rank_costs)
        ):
            if rank_cost.rank != rank:
                raise ValueError("shared layout rank cost has an invalid rank")
            if (
                rank_cost.indexer_score_cost != rank_plan.predicted_score_cost
                or rank_cost.indexer_topk_cost != rank_plan.predicted_topk_cost
            ):
                raise ValueError(
                    "shared layout Indexer costs do not match the rank plan"
                )
            if (
                rank_cost.modeled_ki_bytes
                > plan.shared_layout_config.ki_memory_budget_bytes
            ):
                raise ValueError("shared layout exceeds the KI memory budget")
            if (
                rank_cost.modeled_ki_bytes
                < plan.shared_layout_config.ki_workspace_reserve_bytes
            ):
                raise ValueError("shared layout KI memory is below its reserve")
            if rank_cost.fragment_count != len(rank_plan.query_fragments):
                raise ValueError("shared layout fragment count is inconsistent")
        expected_key = (
            max(cost.indexer_cost for cost in plan.layout_metrics.rank_costs),
            max(cost.hca_cost for cost in plan.layout_metrics.rank_costs),
            sum(
                cost.token_layout_remote_rows for cost in plan.layout_metrics.rank_costs
            ),
            sum(cost.fragment_count for cost in plan.layout_metrics.rank_costs),
        )
        if plan.layout_metrics.key != expected_key:
            raise ValueError("shared layout key does not match its rank costs")
    elif plan.policy == "structural_balanced":
        if (
            plan.shared_layout_config is not None
            or plan.structural_layout_config is None
            or not isinstance(plan.layout_metrics, DsaStructuralLayoutMetrics)
        ):
            raise ValueError(
                "structural_balanced is missing its solver config or metrics"
            )
        metrics = plan.layout_metrics
        solver_config = plan.structural_layout_config
        if metrics.solver_scheme != _STRUCTURAL_SOLVER_SCHEME:
            raise ValueError("structural layout has an unknown solver scheme")
        if metrics.cost_model_version != _STRUCTURAL_COST_MODEL_VERSION:
            raise ValueError("structural layout has an unknown cost model version")
        if not 0 < metrics.chunk_size <= solver_config.chunk_size:
            raise ValueError("structural layout resolved an invalid chunk size")
        if metrics.uneven_shard != solver_config.uneven_shard:
            raise ValueError("structural uneven-shard metadata is inconsistent")
        if len(metrics.rank_costs) != plan.cp_size:
            raise ValueError("structural layout metrics have the wrong rank count")
        if sum(cost.chunk_count for cost in metrics.rank_costs) != metrics.num_chunks:
            raise ValueError("structural layout chunk accounting is incomplete")
        chunk_counts = [cost.chunk_count for cost in metrics.rank_costs]
        if max(chunk_counts) - min(chunk_counts) > 1:
            raise ValueError("structural MinHeap chunk counts differ by more than one")
        for rank, (rank_plan, structural_rank_cost) in enumerate(
            zip(plan.rank_plans, metrics.rank_costs)
        ):
            if structural_rank_cost.rank != rank:
                raise ValueError("structural layout rank cost has an invalid rank")
            if structural_rank_cost.query_tokens != rank_plan.local_token_count:
                raise ValueError("structural layout Query token count is inconsistent")
            if structural_rank_cost.fragment_count != len(rank_plan.query_fragments):
                raise ValueError("structural layout fragment count is inconsistent")
            if structural_rank_cost.native_causal_area <= 0:
                raise ValueError("structural layout assigned a non-positive workload")
            expected_packed_rows = sum(
                fragment.q_end // 4 for fragment in rank_plan.query_fragments
            )
            expected_unique_rows = sum(
                prefix_length
                for _, prefix_length in _csa_required_prefix_lengths(
                    rank_plan.query_fragments
                )
            )
            if (
                structural_rank_cost.csa_packed_indexer_k_rows != expected_packed_rows
                or structural_rank_cost.csa_unique_indexer_k_rows
                != expected_unique_rows
                or structural_rank_cost.csa_duplicate_indexer_k_rows
                != expected_packed_rows - expected_unique_rows
            ):
                raise ValueError("structural CSA Indexer K audit is inconsistent")
    elif (
        plan.shared_layout_config is not None
        or plan.structural_layout_config is not None
        or plan.layout_metrics is not None
    ):
        raise ValueError("legacy plans must not contain shared solver state")

    for rank, rank_plan in enumerate(plan.rank_plans):
        expected_blocks = [
            block for block in plan.compressed_blocks if block.producer_rank == rank
        ]
        if list(rank_plan.produced_blocks) != expected_blocks:
            raise ValueError("rank compressed-block producer table is inconsistent")
        if [block.producer_local_index for block in expected_blocks] != list(
            range(len(expected_blocks))
        ):
            raise ValueError("rank compressed-block local rows are not contiguous")


__all__ = ["build_dsa_execution_plan", "validate_dsa_execution_plan"]
