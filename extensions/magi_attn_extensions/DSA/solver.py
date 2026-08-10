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

"""Cold planner for Magi-DSA.

The planner does two things and delegates everything else to MagiAttention Core:

- it picks one ratio-independent Query layout with the native causal-area
  MinHeap dispatch and reads the resulting fragments off the native
  ``AttnBucket``; and
- it states each typed payload route as owner and consumer ranges, leaving the
  lowering to splits and rank routes to Core's group-collective planner.

Everything the planner emits is sized by fragments and samples. There is no
per-row index table, so the plan is small enough for every rank to rebuild
deterministically rather than receive over an object collective.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from functools import lru_cache

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._make_dispatch_meta import (
    make_bucket_per_rank_from_qk_ranges,
    make_dispatch_meta_from_qk_ranges,
)
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    MinHeapDispatchAlg,
)

from .config import (
    DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
)
from .meta import (
    DsaExecutionPlan,
    DsaFragmentSpec,
    DsaInterval,
    DsaQueryFragment,
    DsaRankPlan,
    DsaRequiredKRange,
    DsaRoutePlan,
    DsaStructuralLayoutMetrics,
    DsaStructuralRankCost,
)

_STRUCTURAL_SOLVER_SCHEME = "magi_min_heap_packed_global_v1"
_STRUCTURAL_COST_MODEL_VERSION = "native_causal_attn_slice_area_v1"


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


def _merge_intervals(intervals: Sequence[DsaInterval]) -> tuple[DsaInterval, ...]:
    """Sort and coalesce half-open intervals into the minimal range list."""

    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if begin >= end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return tuple((begin, end) for begin, end in merged)


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


def _sample_of_global_row(cu_seqlens: tuple[int, ...], global_row: int) -> int:
    import bisect

    return bisect.bisect_right(cu_seqlens, global_row) - 1


def _required_block_prefix_lengths(
    fragments: Sequence[DsaFragmentSpec | DsaQueryFragment],
    ratio: int,
) -> tuple[tuple[int, int], ...]:
    """Keep one longest causal-visible compressed prefix per sample.

    A rank often holds several fragments of the same sample. Their compressed
    prefixes are nested, so only the longest one has to be communicated; the
    shorter ones are packed locally out of the same received rows.
    """

    required_lengths: dict[int, int] = {}
    for fragment in fragments:
        prefix_length = fragment.q_end // ratio
        required_lengths[fragment.sample_id] = max(
            required_lengths.get(fragment.sample_id, 0),
            prefix_length,
        )
    return tuple(
        (sample_id, prefix_length)
        for sample_id, prefix_length in sorted(required_lengths.items())
        if prefix_length > 0
    )


def _required_block_ranges(
    fragments: Sequence[DsaFragmentSpec | DsaQueryFragment],
    ratio: int,
    sample_block_offsets: tuple[int, ...],
) -> tuple[DsaRequiredKRange, ...]:
    return tuple(
        DsaRequiredKRange(
            sample_id=sample_id,
            global_begin=sample_block_offsets[sample_id],
            global_end=sample_block_offsets[sample_id] + prefix_length,
        )
        for sample_id, prefix_length in _required_block_prefix_lengths(fragments, ratio)
    )


@lru_cache(maxsize=32)
def _assign_structural_layout(
    cu_seqlens: tuple[int, ...],
    cp_size: int,
    solver_config: DsaStructuralLayoutConfig,
) -> tuple[tuple[tuple[DsaFragmentSpec, ...], ...], DsaStructuralLayoutMetrics]:
    """Pick one shared Query layout with the native causal-area MinHeap.

    The objective is ratio independent, so CSA and HCA are guaranteed the same
    fragments. Chunk assignment, the chunk-by-sample split and the causal area
    all come from Core; this function only relabels the resulting native
    ``AttnSlice`` list into sample-relative fragments.
    """

    total_tokens = cu_seqlens[-1]
    if total_tokens < cp_size:
        raise ValueError(
            "structural_balanced requires at least one Query token per CP rank"
        )
    nonempty_ranges = [
        [begin, end] for begin, end in zip(cu_seqlens, cu_seqlens[1:]) if begin < end
    ]
    q_ranges = AttnRanges.from_ranges(nonempty_ranges)
    k_ranges = AttnRanges.from_ranges(nonempty_ranges)
    mask_types = [AttnMaskType.CAUSAL] * len(nonempty_ranges)

    auto_chunk_size = max(
        1,
        (total_tokens + solver_config.min_chunks_per_rank * cp_size - 1)
        // (solver_config.min_chunks_per_rank * cp_size),
    )
    chunk_size = min(solver_config.chunk_size, auto_chunk_size)
    dispatch_config = DispatchConfig(
        chunk_size=chunk_size,
        uneven_shard=solver_config.uneven_shard,
        alg=MinHeapDispatchAlg(),
    )
    dispatch_meta, _ = make_dispatch_meta_from_qk_ranges(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=mask_types,
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
    bucket_per_rank = make_bucket_per_rank_from_qk_ranges(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=mask_types,
        dispatch_meta=dispatch_meta,
    )

    specs_per_rank: list[tuple[DsaFragmentSpec, ...]] = []
    rank_costs: list[DsaStructuralRankCost] = []
    for rank, bucket in enumerate(bucket_per_rank):
        rank_specs: list[DsaFragmentSpec] = []
        for attn_slice in bucket.attn_slices:
            q_range = attn_slice.q_range
            k_range = attn_slice.k_range
            assert q_range is not None and k_range is not None
            # A causal self-attention slice always starts its K at the sample
            # begin, so the sample origin is read straight off the native slice.
            sample_begin = k_range.start
            sample_id = _sample_of_global_row(cu_seqlens, sample_begin)
            rank_specs.append(
                DsaFragmentSpec(
                    sample_id=sample_id,
                    q_begin=q_range.start - sample_begin,
                    q_end=q_range.end - sample_begin,
                )
            )
        merged_specs = _merge_adjacent_specs(rank_specs)
        packed_rows = sum(spec.q_end // 4 for spec in merged_specs)
        unique_rows = sum(
            prefix for _, prefix in _required_block_prefix_lengths(merged_specs, 4)
        )
        specs_per_rank.append(merged_specs)
        rank_costs.append(
            DsaStructuralRankCost(
                rank=rank,
                native_causal_area=bucket.area,
                query_tokens=sum(spec.length for spec in merged_specs),
                chunk_count=len(bucket.q_chunks),
                fragment_count=len(merged_specs),
                csa_unique_indexer_k_rows=unique_rows,
                csa_packed_indexer_k_rows=packed_rows,
                csa_duplicate_indexer_k_rows=packed_rows - unique_rows,
            )
        )

    return tuple(specs_per_rank), DsaStructuralLayoutMetrics(
        solver_scheme=_STRUCTURAL_SOLVER_SCHEME,
        cost_model_version=_STRUCTURAL_COST_MODEL_VERSION,
        chunk_size=dispatch_meta.chunk_size,
        num_chunks=dispatch_meta.num_chunks,
        uneven_shard=solver_config.uneven_shard,
        rank_costs=tuple(rank_costs),
    )


def _resolve_query_fragments(
    cu_seqlens: tuple[int, ...],
    specs_per_rank: tuple[tuple[DsaFragmentSpec, ...], ...],
) -> tuple[tuple[DsaQueryFragment, ...], ...]:
    """Give every fragment its global rows and its rank-local row offset."""

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
                    f"sample {sample_id} Query fragments overlap or leave a gap "
                    f"at {cursor}"
                )
            cursor = end
        if cursor != sample_length:
            raise ValueError(
                f"sample {sample_id} Query fragments cover {cursor} of "
                f"{sample_length} tokens"
            )
    return tuple(resolved)


def _sample_block_layout(
    cu_seqlens: tuple[int, ...], ratio: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return per-sample compressed-block offsets and counts.

    A sample only produces a compressed row for each complete ``ratio``-token
    group, so a short tail contributes nothing.
    """

    offsets: list[int] = []
    counts: list[int] = []
    cursor = 0
    for sample_begin, sample_end in zip(cu_seqlens, cu_seqlens[1:]):
        offsets.append(cursor)
        count = (sample_end - sample_begin) // ratio
        counts.append(count)
        cursor += count
    return tuple(offsets), tuple(counts)


def _block_owner_ranges(
    cu_seqlens: tuple[int, ...],
    ratio: int,
    sample_block_offsets: tuple[int, ...],
    sample_block_counts: tuple[int, ...],
    query_fragments: tuple[tuple[DsaQueryFragment, ...], ...],
    cp_size: int,
) -> tuple[tuple[DsaInterval, ...], ...]:
    """Assign each compressed block to the Query owner of its last token.

    Blocks are visited in ascending global block id, so the per-rank range list
    is also that rank's Compressor output-buffer order.
    """

    # Intervals are collected per sample and merged per sample, so a produced
    # block range never straddles a sample boundary. Downstream code reads the
    # sample off a range's first block, which is only sound because of this.
    owner_intervals: list[dict[int, list[DsaInterval]]] = [
        {} for _ in range(cp_size)
    ]
    for rank, fragments in enumerate(query_fragments):
        for fragment in fragments:
            sample_offset = sample_block_offsets[fragment.sample_id]
            count = sample_block_counts[fragment.sample_id]
            # Block j ends at sample position (j + 1) * ratio - 1, and it is
            # owned here exactly when that last token lies in [q_begin, q_end).
            # That makes the first owned block q_begin // ratio, not
            # ceil(q_begin / ratio): a fragment starting mid-group still owns
            # the group it finishes.
            first = fragment.q_begin // ratio
            last = fragment.q_end // ratio
            begin = min(first, count)
            end = min(last, count)
            if begin < end:
                owner_intervals[rank].setdefault(fragment.sample_id, []).append(
                    (sample_offset + begin, sample_offset + end)
                )
    return tuple(
        tuple(
            interval
            for sample_id in sorted(per_sample)
            for interval in _merge_intervals(per_sample[sample_id])
        )
        for per_sample in owner_intervals
    )


def _window_consumer_ranges(
    fragments: tuple[DsaQueryFragment, ...],
    window_size: int,
) -> tuple[DsaInterval, ...]:
    """Raw-window rows a rank needs: one causal tail per fragment."""

    return _merge_intervals(
        [
            (
                max(
                    fragment.sample_global_begin,
                    fragment.global_begin - window_size + 1,
                ),
                fragment.global_end,
            )
            for fragment in fragments
        ]
    )


def _overlap_consumer_ranges(
    cu_seqlens: tuple[int, ...],
    ratio: int,
    overlap: bool,
    sample_block_offsets: tuple[int, ...],
    owned_block_ranges: tuple[DsaInterval, ...],
) -> tuple[DsaInterval, ...]:
    """Compressor support rows a rank needs for the blocks it produces.

    CSA reads the previous group as well as the current one, so a rank whose
    first local block does not start a sample still needs one remote group.
    """

    intervals: list[DsaInterval] = []
    for block_begin, block_end in owned_block_ranges:
        sample_id = _sample_of_block(sample_block_offsets, block_begin)
        sample_global_begin = cu_seqlens[sample_id]
        sample_block_begin = sample_block_offsets[sample_id]
        first_local = block_begin - sample_block_begin
        last_local = block_end - sample_block_begin
        support_begin = first_local * ratio - (ratio if overlap else 0)
        intervals.append(
            (
                sample_global_begin + max(0, support_begin),
                sample_global_begin + last_local * ratio,
            )
        )
    return _merge_intervals(intervals)


def _sample_of_block(sample_block_offsets: tuple[int, ...], global_block: int) -> int:
    import bisect

    return bisect.bisect_right(sample_block_offsets, global_block) - 1


def _block_prefix_consumer_ranges(
    fragments: tuple[DsaQueryFragment, ...],
    ratio: int,
    sample_block_offsets: tuple[int, ...],
) -> tuple[DsaInterval, ...]:
    return _merge_intervals(
        [
            (required.global_begin, required.global_end)
            for required in _required_block_ranges(
                fragments, ratio, sample_block_offsets
            )
        ]
    )


def _fragment_global_ranges(
    fragments: tuple[DsaQueryFragment, ...],
) -> tuple[DsaInterval, ...]:
    return _merge_intervals(
        [(fragment.global_begin, fragment.global_end) for fragment in fragments]
    )


def _indexer_metadata(
    fragments: tuple[DsaQueryFragment, ...],
    ratio: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int, int]:
    """Grouped Indexer varlen metadata, one entry per fragment.

    ``k_cu_seqlens`` describes the packed grouped-K the cuDNN Indexer consumes,
    where each fragment gets its own causal-visible prefix. Per-query sequence
    lengths are not built here; they are expanded on the device from these
    fragment bounds.
    """

    q_cu = [0]
    k_cu = [0]
    q_offsets: list[int] = []
    max_q = 0
    max_k = 0
    for fragment in fragments:
        q_cu.append(q_cu[-1] + fragment.length)
        k_length = fragment.q_end // ratio
        k_cu.append(k_cu[-1] + k_length)
        q_offsets.append(fragment.q_begin)
        max_q = max(max_q, fragment.length)
        max_k = max(max_k, k_length)
    backend_max_k = (
        0
        if max_k == 0
        else (
            (max_k + DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT - 1)
            // DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
            * DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
        )
    )
    return tuple(q_cu), tuple(k_cu), tuple(q_offsets), max_q, max_k, backend_max_k


def _hash_plan_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dsa_execution_plan(
    config: MagiDSAConfig,
    cu_seqlens: Sequence[int],
    source_token_counts: Sequence[int],
    *,
    structural_layout_config: DsaStructuralLayoutConfig | None = None,
) -> DsaExecutionPlan:
    """Build the source-layout to final-Query-layout DSA execution plan.

    The result is a pure function of its arguments, so every CP rank calls this
    and gets a bit-identical plan. That is what removes the plan broadcast.
    """

    solver_config = structural_layout_config or DsaStructuralLayoutConfig()
    cu = _validate_cu_seqlens(cu_seqlens)
    source_counts = _validate_source_counts(source_token_counts, cu[-1])
    cp_size = len(source_counts)
    ratio = config.ratio

    specs_per_rank, layout_metrics = _assign_structural_layout(
        cu, cp_size, solver_config
    )
    query_counts = tuple(
        sum(spec.length for spec in specs) for specs in specs_per_rank
    )
    query_layout_hash = _hash_plan_payload(
        [[asdict(spec) for spec in specs] for specs in specs_per_rank]
    )
    query_fragments = _resolve_query_fragments(cu, specs_per_rank)

    sample_block_offsets, sample_block_counts = _sample_block_layout(cu, ratio)
    total_blocks = sum(sample_block_counts)
    owned_block_ranges = _block_owner_ranges(
        cu,
        ratio,
        sample_block_offsets,
        sample_block_counts,
        query_fragments,
        cp_size,
    )

    source_offsets = _prefix_offsets(source_counts)
    source_ranges = tuple(
        ((source_offsets[rank], source_offsets[rank + 1]),) if source_counts[rank] else ()
        for rank in range(cp_size)
    )
    query_ranges = tuple(
        _fragment_global_ranges(fragments) for fragments in query_fragments
    )

    token_layout_route = DsaRoutePlan(
        name="TOKEN_LAYOUT",
        owner_ranges_per_rank=source_ranges,
        consumer_ranges_per_rank=query_ranges,
    )
    window_route = DsaRoutePlan(
        name="WINDOW_KV",
        owner_ranges_per_rank=query_ranges,
        consumer_ranges_per_rank=tuple(
            _window_consumer_ranges(fragments, config.window_size)
            for fragments in query_fragments
        ),
    )
    overlap_x_route = DsaRoutePlan(
        name="OVERLAP_X",
        owner_ranges_per_rank=query_ranges,
        consumer_ranges_per_rank=tuple(
            _overlap_consumer_ranges(
                cu,
                ratio,
                ratio == 4,
                sample_block_offsets,
                owned_block_ranges[rank],
            )
            for rank in range(cp_size)
        ),
    )
    compressed_prefix_ranges = tuple(
        _block_prefix_consumer_ranges(fragments, ratio, sample_block_offsets)
        for fragments in query_fragments
    )
    compressed_kv_route = DsaRoutePlan(
        name="COMPRESSED_KV",
        owner_ranges_per_rank=owned_block_ranges,
        consumer_ranges_per_rank=compressed_prefix_ranges,
    )
    # CSA needs the same causal-visible prefix twice, once as Indexer K and once
    # as attention KV. They are separate collectives on separate payload widths,
    # but they share one range plan.
    compressed_ki_route = (
        DsaRoutePlan(
            name="COMPRESSED_KI",
            owner_ranges_per_rank=owned_block_ranges,
            consumer_ranges_per_rank=compressed_prefix_ranges,
        )
        if ratio == 4
        else None
    )

    rank_plans: list[DsaRankPlan] = []
    for rank in range(cp_size):
        fragments = query_fragments[rank]
        if ratio == 4:
            q_cu, k_cu, q_offsets, max_q, max_k, backend_max_k = _indexer_metadata(
                fragments, ratio
            )
            required = _required_block_ranges(fragments, ratio, sample_block_offsets)
        else:
            q_cu, k_cu, q_offsets = (0,), (0,), ()
            max_q = max_k = backend_max_k = 0
            required = ()
        rank_plans.append(
            DsaRankPlan(
                rank=rank,
                source_token_count=source_counts[rank],
                local_token_count=query_counts[rank],
                source_global_begin=source_offsets[rank],
                source_global_end=source_offsets[rank + 1],
                query_fragments=fragments,
                produced_block_ranges=owned_block_ranges[rank],
                sample_block_offsets=sample_block_offsets,
                sample_block_counts=sample_block_counts,
                indexer_required_k_ranges=required,
                indexer_q_cu_seqlens=q_cu,
                indexer_k_cu_seqlens=k_cu,
                indexer_q_causal_offsets=q_offsets,
                indexer_max_seqlen_q=max_q,
                indexer_logical_max_seqlen_k=max_k,
                indexer_backend_max_seqlen_k=backend_max_k,
            )
        )

    if ratio == 4:
        collective_order = ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV")
    else:
        collective_order = ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV")

    payload = {
        "cu_seqlens": cu,
        "source_token_counts": source_counts,
        "query_token_counts": query_counts,
        "ratio": ratio,
        "total_compressed_blocks": total_blocks,
        "rank_plans": [asdict(rank_plan) for rank_plan in rank_plans],
        "routes": [
            asdict(route)
            for route in (
                token_layout_route,
                window_route,
                overlap_x_route,
                compressed_kv_route,
                compressed_ki_route,
            )
            if route is not None
        ],
        "collective_order": collective_order,
        "boundary_collective_order": ("TOKEN_LAYOUT",),
        "query_layout_hash": query_layout_hash,
        "layout_metrics": asdict(layout_metrics),
        "structural_layout_config": asdict(solver_config),
    }
    plan = DsaExecutionPlan(
        cu_seqlens=cu,
        source_token_counts=source_counts,
        query_token_counts=query_counts,
        ratio=ratio,
        total_compressed_blocks=total_blocks,
        rank_plans=tuple(rank_plans),
        token_layout_route=token_layout_route,
        window_route=window_route,
        overlap_x_route=overlap_x_route,
        compressed_kv_route=compressed_kv_route,
        compressed_ki_route=compressed_ki_route,
        collective_order=collective_order,
        boundary_collective_order=("TOKEN_LAYOUT",),
        query_layout_hash=query_layout_hash,
        layout_metrics=layout_metrics,
        structural_layout_config=solver_config,
        plan_hash=_hash_plan_payload(payload),
    )
    validate_dsa_execution_plan(plan)
    return plan


def validate_dsa_execution_plan(plan: DsaExecutionPlan) -> None:
    """Check the invariants the warm path and the device maps rely on.

    Route split and rank-route consistency is Core's job and is covered by
    Core's own tests, so this only checks what the planner itself asserts:
    exact Query coverage, producer-buffer alignment, and causal visibility of
    every routed prefix.
    """

    cp_size = plan.cp_size
    if len(plan.rank_plans) != cp_size or len(plan.query_token_counts) != cp_size:
        raise ValueError("plan rank tables have inconsistent CP sizes")
    if sum(plan.query_token_counts) != plan.total_tokens:
        raise ValueError("final Query counts do not cover the packed tokens")
    if sum(plan.source_token_counts) != plan.total_tokens:
        raise ValueError("source counts do not cover the packed tokens")

    covered: list[tuple[int, int]] = []
    produced: list[tuple[int, int]] = []
    for rank, rank_plan in enumerate(plan.rank_plans):
        if rank_plan.rank != rank:
            raise ValueError("rank plan is stored at the wrong index")
        if rank_plan.local_token_count != plan.query_token_counts[rank]:
            raise ValueError("rank plan Query count disagrees with the plan")
        local_cursor = 0
        for fragment in rank_plan.query_fragments:
            if fragment.rank != rank:
                raise ValueError("Query fragment carries the wrong rank")
            if fragment.local_begin != local_cursor:
                raise ValueError("Query fragments are not densely packed locally")
            local_cursor += fragment.length
            covered.append((fragment.global_begin, fragment.global_end))
        if local_cursor != rank_plan.local_token_count:
            raise ValueError("Query fragments do not fill the local token count")
        produced.extend(rank_plan.produced_block_ranges)

        # The routed owner rows must equal this rank's producer buffer, or the
        # group collective would send rows that the producer never wrote.
        if plan.window_route.owner_row_count(rank) != rank_plan.local_token_count:
            raise ValueError("WINDOW_KV owner rows do not match the Query buffer")
        if plan.token_layout_route.owner_row_count(rank) != rank_plan.source_token_count:
            raise ValueError("TOKEN_LAYOUT owner rows do not match the source buffer")
        if plan.token_layout_route.consumer_row_count(rank) != (
            rank_plan.local_token_count
        ):
            raise ValueError("TOKEN_LAYOUT consumer rows do not match the Query buffer")
        if plan.compressed_kv_route.owner_row_count(rank) != (
            rank_plan.produced_block_count
        ):
            raise ValueError("COMPRESSED_KV owner rows do not match produced blocks")

    if _merge_intervals(covered) != ((0, plan.total_tokens),):
        raise ValueError("final Query fragments do not exactly cover every token")
    if plan.total_compressed_blocks and _merge_intervals(produced) != (
        (0, plan.total_compressed_blocks),
    ):
        raise ValueError("compressed blocks do not have exactly one producer each")

    for rank_plan in plan.rank_plans:
        for required in rank_plan.indexer_required_k_ranges:
            sample_begin = rank_plan.sample_block_offsets[required.sample_id]
            sample_count = rank_plan.sample_block_counts[required.sample_id]
            if required.global_begin != sample_begin:
                raise ValueError("an Indexer prefix does not start at its sample")
            if required.global_end > sample_begin + sample_count:
                raise ValueError("an Indexer prefix runs past its sample")


__all__ = [
    "build_dsa_execution_plan",
    "validate_dsa_execution_plan",
]
