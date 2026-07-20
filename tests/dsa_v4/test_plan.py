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

import pytest

from magi_attention.dsa_config import DsaRatio, MagiDSAConfig
from magi_attention.meta.solver.dsa_solver import build_dsa_execution_plan


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_owner_plan_covers_packed_ragged_batch_with_empty_rank(
    ratio: DsaRatio,
) -> None:
    config = MagiDSAConfig(ratio=ratio)
    plan = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 3, 132, 132, 391),
        local_token_counts=(50, 0, 79, 130, 132),
        policy="indexer_balanced",
    )

    assert plan.total_tokens == 391
    covered = [0] * plan.total_tokens
    for rank_plan in plan.rank_plans:
        assert (
            rank_plan.local_global_end - rank_plan.local_global_begin
            == rank_plan.local_token_count
        )
        for fragment in rank_plan.owner_fragments:
            for token in range(fragment.global_begin, fragment.global_end):
                covered[token] += 1
    assert covered == [1] * plan.total_tokens
    assert plan.rank_plans[1].local_token_count == 0
    assert plan.rank_plans[1].owner_fragments == ()


def test_csa_blocks_use_previous_b_and_current_a_rows_and_drop_tail() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 11, 20),
        local_token_counts=(5, 6, 9),
        policy="sequential",
    )

    assert [
        (block.sample_id, block.sample_block_id) for block in plan.compressed_blocks
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert plan.compressed_blocks[0].source_global_rows == (-1, -1, -1, -1, 0, 1, 2, 3)
    assert plan.compressed_blocks[1].source_global_rows == (0, 1, 2, 3, 4, 5, 6, 7)
    assert plan.compressed_blocks[2].source_global_rows == (
        -1,
        -1,
        -1,
        -1,
        11,
        12,
        13,
        14,
    )
    assert plan.compressed_blocks[-1].global_end == 19


def test_hca_blocks_are_non_overlapping_and_owned_by_last_token() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=128),
        cu_seqlens=(0, 257),
        local_token_counts=(64, 64, 64, 65),
    )

    assert len(plan.compressed_blocks) == 2
    assert plan.compressed_blocks[0].source_global_rows == tuple(range(128))
    assert plan.compressed_blocks[0].owner_rank == 1
    assert plan.compressed_blocks[1].source_global_rows == tuple(range(128, 256))
    assert plan.compressed_blocks[1].owner_rank == 3


def _relative_range(values: list[int]) -> float:
    return (max(values) - min(values)) / (sum(values) / len(values))


def test_frozen_128k_balanced_solver_meets_both_predicted_cost_targets() -> None:
    config = MagiDSAConfig(ratio=4)
    counts = (16384,) * 8
    sequential = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 131072),
        local_token_counts=counts,
        policy="sequential",
    )
    balanced = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 131072),
        local_token_counts=counts,
        policy="indexer_balanced",
    )

    sequential_score = [rank.predicted_score_cost for rank in sequential.rank_plans]
    sequential_topk = [rank.predicted_topk_cost for rank in sequential.rank_plans]
    balanced_score = [rank.predicted_score_cost for rank in balanced.rank_plans]
    balanced_topk = [rank.predicted_topk_cost for rank in balanced.rank_plans]
    assert _relative_range(sequential_score) > 0.5
    assert _relative_range(sequential_topk) > 0.5
    assert _relative_range(balanced_score) <= 0.01
    assert _relative_range(balanced_topk) <= 0.01
    assert sum(rank.worker_token_count for rank in balanced.rank_plans) == 131072


def test_plan_and_routes_are_deterministic_and_exactly_invertible() -> None:
    config = MagiDSAConfig(
        ratio=4,
        hidden_size=32,
        q_lora_rank=16,
        num_query_heads=4,
        head_dim=16,
        rope_dim=4,
        indexer_heads=4,
        indexer_head_dim=8,
        indexer_topk=4,
        window_size=4,
        indexer_atom_size=4,
    )
    arguments = ((0, 7, 18, 31), (5, 0, 8, 9, 9))
    first = build_dsa_execution_plan(config, *arguments)
    second = build_dsa_execution_plan(config, *arguments)
    assert first == second
    assert first.plan_hash == second.plan_hash

    for rank_plan in first.rank_plans:
        route = rank_plan.indexer_qw_route
        assert route is not None
        assert sorted(route.consumer_from_received) == list(
            range(route.received_row_count)
        )
        for consumer_index, received_index in enumerate(route.consumer_from_received):
            assert route.received_from_consumer[received_index] == consumer_index
        assert len(route.reverse_row_offsets) == route.producer_row_count + 1
        assert route.reverse_row_offsets[-1] == route.send_row_count


@pytest.mark.parametrize(
    ("cu_seqlens", "local_counts", "message"),
    [
        ((1, 2), (1,), "start at zero"),
        ((0, 3, 2), (2,), "nondecreasing"),
        ((0, 3), (1, 1), "expected 3"),
        ((0, 3), (4, -1), "non-negative"),
    ],
)
def test_invalid_plan_inputs_fail_before_building_routes(
    cu_seqlens: tuple[int, ...],
    local_counts: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_dsa_execution_plan(MagiDSAConfig(ratio=4), cu_seqlens, local_counts)
