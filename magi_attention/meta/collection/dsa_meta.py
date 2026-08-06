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

from magi_attention.dsa_config import (
    DsaPlanPolicy,
    DsaRatio,
    DsaSharedLayoutConfig,
    DsaStructuralLayoutConfig,
)


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
    score_cost: int
    topk_cost: int

    @property
    def length(self) -> int:
        return self.q_end - self.q_begin


@dataclass(frozen=True)
class DsaCompressionBlock:
    """One complete sample-local compression block and its unique producer."""

    global_block_id: int
    sample_id: int
    sample_block_id: int
    global_begin: int
    global_end: int
    producer_rank: int
    producer_local_index: int
    position: int
    source_global_rows: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.global_end - self.global_begin


@dataclass(frozen=True)
class DsaRequiredKRange:
    """One consumer/doc longest compressed-K prefix in global block order."""

    sample_id: int
    global_begin: int
    global_end: int

    @property
    def length(self) -> int:
        return self.global_end - self.global_begin


@dataclass(frozen=True)
class DsaGroupCollectiveArg:
    """Serializable planner contract lowered to typed All2AllV plus local CSR.

    This mirrors the four layout fields of Magi-MSA ``GroupCollectiveArg`` but
    intentionally owns no process group and selects no communication primitive.
    """

    rank: int
    world_size: int
    input_split_size_list: tuple[int, ...]
    output_split_size_list: tuple[int, ...]
    dst_indices_list: tuple[tuple[int, ...], ...]
    src_index_list: tuple[int, ...]
    deterministic: bool = False
    split_alignment: int = 1


@dataclass(frozen=True)
class DsaRouteRankPlan:
    """Static unique-row All2AllV and reverse-CSR metadata for one rank."""

    rank: int
    producer_row_count: int
    group_collective_arg: DsaGroupCollectiveArg
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_source_rows: tuple[int, ...]
    received_global_rows: tuple[int, ...]
    consumer_global_rows: tuple[int, ...]
    consumer_from_received: tuple[int, ...]
    received_from_consumer: tuple[int, ...]
    reverse_row_offsets: tuple[int, ...]
    reverse_source_rows: tuple[int, ...]

    @property
    def send_row_count(self) -> int:
        return sum(self.send_counts)

    @property
    def received_row_count(self) -> int:
        return sum(self.recv_counts)


@dataclass(frozen=True)
class DsaTypedRoutePlan:
    """All-rank metadata for one typed payload collective."""

    name: str
    rank_plans: tuple[DsaRouteRankPlan, ...]

    @property
    def cp_size(self) -> int:
        return len(self.rank_plans)


@dataclass(frozen=True)
class DsaLayoutRankCost:
    """Deterministic structural cost assigned to one final Query owner."""

    rank: int
    indexer_score_cost: int
    indexer_topk_cost: int
    packed_ki_rows: int
    unique_ki_rows: int
    modeled_ki_bytes: int
    indexer_cost: int
    hca_query_cost: int
    hca_send_rows: int
    hca_recv_rows: int
    hca_max_peer_rows: int
    hca_cost: int
    token_layout_remote_rows: int
    fragment_count: int


@dataclass(frozen=True)
class DsaLayoutMetrics:
    """Global lexicographic key and per-rank costs for one Query layout."""

    solver_scheme: str
    cost_model_version: str
    key: tuple[int, int, int, int]
    rank_costs: tuple[DsaLayoutRankCost, ...]
    candidate_evaluations: int
    improvement_steps: int
    stop_reason: str


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
    """Compact cold-plan metadata materialized once by a CP rank."""

    rank: int
    source_token_count: int
    local_token_count: int
    source_global_begin: int
    source_global_end: int
    query_fragments: tuple[DsaQueryFragment, ...]
    local_query_global_rows: tuple[int, ...]
    produced_blocks: tuple[DsaCompressionBlock, ...]
    local_q_sample_ids: tuple[int, ...]
    local_q_positions: tuple[int, ...]
    sample_block_offsets: tuple[int, ...]
    sample_block_counts: tuple[int, ...]
    indexer_required_k_ranges: tuple[DsaRequiredKRange, ...]
    compression_source_from_overlap: tuple[int, ...]
    indexer_q_cu_seqlens: tuple[int, ...]
    indexer_k_cu_seqlens: tuple[int, ...]
    indexer_q_causal_offsets: tuple[int, ...]
    indexer_q_sample_block_offsets: tuple[int, ...]
    indexer_seq_lens: tuple[int, ...]
    indexer_max_seqlen_q: int
    indexer_logical_max_seqlen_k: int
    indexer_backend_max_seqlen_k: int
    token_layout_route: DsaRouteRankPlan | None
    window_route: DsaRouteRankPlan
    overlap_x_route: DsaRouteRankPlan | None
    compressed_kv_route: DsaRouteRankPlan | None
    compressed_ki_route: DsaRouteRankPlan | None
    predicted_score_cost: int
    predicted_topk_cost: int

    @property
    def indexer_token_count(self) -> int:
        return self.indexer_q_cu_seqlens[-1] if self.indexer_q_cu_seqlens else 0

    @property
    def packed_indexer_k_count(self) -> int:
        return self.indexer_k_cu_seqlens[-1] if self.indexer_k_cu_seqlens else 0


@dataclass(frozen=True)
class DsaExecutionPlan:
    """Immutable global plan shared by sequential and balanced execution."""

    cu_seqlens: tuple[int, ...]
    source_token_counts: tuple[int, ...]
    query_token_counts: tuple[int, ...]
    ratio: DsaRatio
    policy: DsaPlanPolicy
    compressed_blocks: tuple[DsaCompressionBlock, ...]
    rank_plans: tuple[DsaRankPlan, ...]
    collective_order: tuple[str, ...]
    boundary_collective_order: tuple[str, ...]
    query_layout_hash: str
    layout_metrics: DsaLayoutMetrics | DsaStructuralLayoutMetrics | None
    shared_layout_config: DsaSharedLayoutConfig | None
    structural_layout_config: DsaStructuralLayoutConfig | None
    plan_hash: str

    @property
    def cp_size(self) -> int:
        return len(self.source_token_counts)

    @property
    def total_tokens(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def total_compressed_blocks(self) -> int:
        return len(self.compressed_blocks)


__all__ = [
    "DsaCompressionBlock",
    "DsaExecutionPlan",
    "DsaFragmentSpec",
    "DsaGroupCollectiveArg",
    "DsaLayoutMetrics",
    "DsaLayoutRankCost",
    "DsaStructuralLayoutMetrics",
    "DsaStructuralRankCost",
    "DsaQueryFragment",
    "DsaRankPlan",
    "DsaRequiredKRange",
    "DsaRouteRankPlan",
    "DsaTypedRoutePlan",
]
