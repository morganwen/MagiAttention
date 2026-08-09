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

"""Cold-plan data model for Magi-DSA.

Every structure here is sized by fragments and samples, never by tokens or
compressed rows. A route is a pair of range tables, and MagiAttention Core
lowers those ranges to concrete splits and rank routes through
``DynamicAttnSolver._calc_group_collective_arg_from_ranges``. Keeping the plan
range-shaped is what lets each rank rebuild it deterministically instead of
receiving it over an object collective.
"""

from __future__ import annotations

from dataclasses import dataclass

from magi_attention.common import AttnRanges

from .config import DsaRatio, DsaStructuralLayoutConfig

# One global half-open interval, kept as a plain tuple so a plan stays trivially
# hashable, comparable and serializable without importing torch.
DsaInterval = tuple[int, int]


@dataclass(frozen=True)
class DsaFragmentSpec:
    """Sample-relative query interval assigned by the cold planner."""

    sample_id: int
    q_begin: int
    q_end: int

    @property
    def length(self) -> int:
        return self.q_end - self.q_begin


@dataclass(frozen=True)
class DsaQueryFragment:
    """Resolved final-owner query interval in rank-local concatenation order."""

    sample_id: int
    rank: int
    q_begin: int
    q_end: int
    global_begin: int
    global_end: int
    local_begin: int
    sample_global_begin: int

    @property
    def length(self) -> int:
        return self.q_end - self.q_begin


@dataclass(frozen=True)
class DsaRequiredKRange:
    """One consumer/sample longest compressed-K prefix in global block order."""

    sample_id: int
    global_begin: int
    global_end: int

    @property
    def length(self) -> int:
        return self.global_end - self.global_begin


@dataclass(frozen=True)
class DsaRoutePlan:
    """One typed payload route expressed purely as owner and consumer ranges.

    ``owner_ranges_per_rank[r]`` are the global rows rank ``r`` produces, in
    ascending order, which is also its producer-buffer order.
    ``consumer_ranges_per_rank[r]`` are the global rows rank ``r`` needs; the
    Core group-cast delivers them in that same ascending order, so no consumer
    unpermute is needed on either side of the collective.
    """

    name: str
    owner_ranges_per_rank: tuple[tuple[DsaInterval, ...], ...]
    consumer_ranges_per_rank: tuple[tuple[DsaInterval, ...], ...]

    def __post_init__(self) -> None:
        if len(self.owner_ranges_per_rank) != len(self.consumer_ranges_per_rank):
            raise ValueError(f"{self.name}: route range tables have different CP sizes")
        for kind, table in (
            ("owner", self.owner_ranges_per_rank),
            ("consumer", self.consumer_ranges_per_rank),
        ):
            for rank, ranges in enumerate(table):
                cursor = -1
                for begin, end in ranges:
                    if begin >= end:
                        raise ValueError(f"{self.name}: empty {kind} range on {rank}")
                    if begin < cursor:
                        raise ValueError(
                            f"{self.name}: {kind} ranges on {rank} are not sorted "
                            "and merged"
                        )
                    cursor = end

    @property
    def cp_size(self) -> int:
        return len(self.owner_ranges_per_rank)

    def owner_row_count(self, rank: int) -> int:
        return sum(end - begin for begin, end in self.owner_ranges_per_rank[rank])

    def consumer_row_count(self, rank: int) -> int:
        return sum(end - begin for begin, end in self.consumer_ranges_per_rank[rank])

    def owner_attn_ranges(self) -> list[AttnRanges]:
        return [
            AttnRanges.from_ranges([list(interval) for interval in ranges])
            for ranges in self.owner_ranges_per_rank
        ]

    def consumer_attn_ranges(self) -> list[AttnRanges]:
        return [
            AttnRanges.from_ranges([list(interval) for interval in ranges])
            for ranges in self.consumer_ranges_per_rank
        ]


@dataclass(frozen=True)
class DsaStructuralRankCost:
    """Native objective and non-objective CSA packing audit for one CP rank."""

    rank: int
    native_causal_area: int
    query_tokens: int
    chunk_count: int
    fragment_count: int
    csa_unique_indexer_k_rows: int
    csa_packed_indexer_k_rows: int
    csa_duplicate_indexer_k_rows: int


@dataclass(frozen=True)
class DsaStructuralLayoutMetrics:
    """Auditable output of Magi MinHeap packed-global dispatch."""

    solver_scheme: str
    cost_model_version: str
    chunk_size: int
    num_chunks: int
    uneven_shard: bool
    rank_costs: tuple[DsaStructuralRankCost, ...]


@dataclass(frozen=True)
class DsaRankPlan:
    """Cold-plan metadata for one CP rank, sized by fragments and samples.

    Nothing here grows with the token count. Per-token device metadata such as
    positions, sample ids and Indexer sequence lengths is expanded on the device
    from ``query_fragments`` during cold materialization.
    """

    rank: int
    source_token_count: int
    local_token_count: int
    query_fragments: tuple[DsaQueryFragment, ...]
    # Global compressed-block ids this rank produces, ascending, which is also
    # its Compressor output-buffer order.
    produced_block_ranges: tuple[DsaInterval, ...]
    sample_block_offsets: tuple[int, ...]
    sample_block_counts: tuple[int, ...]
    indexer_required_k_ranges: tuple[DsaRequiredKRange, ...]
    indexer_q_cu_seqlens: tuple[int, ...]
    indexer_k_cu_seqlens: tuple[int, ...]
    indexer_q_causal_offsets: tuple[int, ...]
    indexer_max_seqlen_q: int
    indexer_logical_max_seqlen_k: int
    indexer_backend_max_seqlen_k: int

    @property
    def indexer_token_count(self) -> int:
        return self.indexer_q_cu_seqlens[-1] if self.indexer_q_cu_seqlens else 0

    @property
    def packed_indexer_k_count(self) -> int:
        return self.indexer_k_cu_seqlens[-1] if self.indexer_k_cu_seqlens else 0

    @property
    def produced_block_count(self) -> int:
        return sum(end - begin for begin, end in self.produced_block_ranges)


@dataclass(frozen=True)
class DsaExecutionPlan:
    """Immutable global plan every CP rank rebuilds deterministically."""

    cu_seqlens: tuple[int, ...]
    source_token_counts: tuple[int, ...]
    query_token_counts: tuple[int, ...]
    ratio: DsaRatio
    total_compressed_blocks: int
    rank_plans: tuple[DsaRankPlan, ...]
    token_layout_route: DsaRoutePlan
    window_route: DsaRoutePlan
    overlap_x_route: DsaRoutePlan
    compressed_kv_route: DsaRoutePlan
    compressed_ki_route: DsaRoutePlan | None
    collective_order: tuple[str, ...]
    boundary_collective_order: tuple[str, ...]
    query_layout_hash: str
    layout_metrics: DsaStructuralLayoutMetrics
    structural_layout_config: DsaStructuralLayoutConfig
    plan_hash: str

    @property
    def cp_size(self) -> int:
        return len(self.source_token_counts)

    @property
    def total_tokens(self) -> int:
        return self.cu_seqlens[-1]

    def routes(self) -> tuple[DsaRoutePlan, ...]:
        ordered = (
            self.token_layout_route,
            self.window_route,
            self.overlap_x_route,
            self.compressed_kv_route,
            self.compressed_ki_route,
        )
        return tuple(route for route in ordered if route is not None)


__all__ = [
    "DsaExecutionPlan",
    "DsaFragmentSpec",
    "DsaInterval",
    "DsaQueryFragment",
    "DsaRankPlan",
    "DsaRequiredKRange",
    "DsaRoutePlan",
    "DsaStructuralLayoutMetrics",
    "DsaStructuralRankCost",
]
