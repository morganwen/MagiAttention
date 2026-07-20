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

from magi_attention.dsa_config import DsaPlanPolicy, DsaRatio


@dataclass(frozen=True)
class DsaOwnerFragment:
    """Sample-relative interval that remains on its original CP owner."""

    sample_id: int
    owner_rank: int
    q_begin: int
    q_end: int
    global_begin: int
    global_end: int
    owner_local_begin: int
    sample_global_begin: int

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
    owner_rank: int
    owner_local_index: int
    position: int
    source_global_rows: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.global_end - self.global_begin


@dataclass(frozen=True)
class DsaIndexerFragment:
    """One query atom assigned to an Indexer worker."""

    sample_id: int
    q_begin: int
    q_end: int
    global_begin: int
    global_end: int
    owner_rank: int
    owner_local_begin: int
    worker_rank: int
    worker_local_begin: int
    score_cost: int
    topk_cost: int

    @property
    def length(self) -> int:
        return self.q_end - self.q_begin

    @property
    def visible_block_count(self) -> int:
        return self.q_end // 4


@dataclass(frozen=True)
class DsaRouteRankPlan:
    """Static unique-row All2AllV and reverse-CSR metadata for one rank."""

    rank: int
    producer_row_count: int
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
class DsaRankPlan:
    """Compact cold-plan metadata materialized once by a CP rank."""

    rank: int
    local_token_count: int
    local_global_begin: int
    local_global_end: int
    owner_fragments: tuple[DsaOwnerFragment, ...]
    owned_blocks: tuple[DsaCompressionBlock, ...]
    worker_fragments: tuple[DsaIndexerFragment, ...]
    local_q_sample_ids: tuple[int, ...]
    local_q_positions: tuple[int, ...]
    sample_block_offsets: tuple[int, ...]
    sample_block_counts: tuple[int, ...]
    compression_source_from_overlap: tuple[int, ...]
    indexer_q_cu_seqlens: tuple[int, ...]
    indexer_k_cu_seqlens: tuple[int, ...]
    indexer_q_causal_offsets: tuple[int, ...]
    indexer_q_sample_block_offsets: tuple[int, ...]
    indexer_seq_lens: tuple[int, ...]
    indexer_max_seqlen_q: int
    indexer_max_seqlen_k: int
    window_route: DsaRouteRankPlan
    overlap_x_route: DsaRouteRankPlan | None
    compressed_kv_route: DsaRouteRankPlan | None
    compressed_ki_route: DsaRouteRankPlan | None
    indexer_qw_route: DsaRouteRankPlan | None
    predicted_score_cost: int
    predicted_topk_cost: int

    @property
    def worker_token_count(self) -> int:
        return self.indexer_q_cu_seqlens[-1] if self.indexer_q_cu_seqlens else 0

    @property
    def packed_indexer_k_count(self) -> int:
        return self.indexer_k_cu_seqlens[-1] if self.indexer_k_cu_seqlens else 0


@dataclass(frozen=True)
class DsaExecutionPlan:
    """Immutable global plan shared by sequential and balanced execution."""

    cu_seqlens: tuple[int, ...]
    local_token_counts: tuple[int, ...]
    ratio: DsaRatio
    policy: DsaPlanPolicy
    compressed_blocks: tuple[DsaCompressionBlock, ...]
    rank_plans: tuple[DsaRankPlan, ...]
    collective_order: tuple[str, ...]
    plan_hash: str

    @property
    def cp_size(self) -> int:
        return len(self.local_token_counts)

    @property
    def total_tokens(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def total_compressed_blocks(self) -> int:
        return len(self.compressed_blocks)


__all__ = [
    "DsaCompressionBlock",
    "DsaExecutionPlan",
    "DsaIndexerFragment",
    "DsaOwnerFragment",
    "DsaRankPlan",
    "DsaRouteRankPlan",
    "DsaTypedRoutePlan",
]
