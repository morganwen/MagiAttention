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

"""Cold-plan invariants for the range-shaped Magi-DSA planner."""

from __future__ import annotations

import pytest
from magi_attn_extensions.DSA.config import (
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAProModelSpec,
)
from magi_attn_extensions.DSA.meta import DsaRoutePlan
from magi_attn_extensions.DSA.solver import (
    build_dsa_execution_plan,
    validate_dsa_execution_plan,
)

SOLVER = DsaStructuralLayoutConfig(chunk_size=128, min_chunks_per_rank=4)


def _split(total: int, cp_size: int) -> tuple[int, ...]:
    base, extra = divmod(total, cp_size)
    return tuple(base + int(rank < extra) for rank in range(cp_size))


def _plan(ratio: int, cu: tuple[int, ...], cp_size: int):
    return build_dsa_execution_plan(
        MagiDSAConfig(ratio=ratio),
        cu,
        _split(cu[-1], cp_size),
        structural_layout_config=SOLVER,
    )


CASES = [
    ((0, 1024), 1),
    ((0, 1024), 4),
    ((0, 1030, 2570), 1),
    ((0, 1030, 2570), 4),
    ((0, 1030, 2570), 8),
    ((0, 512, 1024, 4096), 8),
]


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_plan_validates_and_covers_every_token(ratio, cu, cp_size):
    plan = _plan(ratio, cu, cp_size)
    validate_dsa_execution_plan(plan)
    assert sum(plan.query_token_counts) == cu[-1]
    covered: list[tuple[int, int]] = []
    for rank_plan in plan.rank_plans:
        for fragment in rank_plan.query_fragments:
            covered.append((fragment.global_begin, fragment.global_end))
    covered.sort()
    cursor = 0
    for begin, end in covered:
        assert begin == cursor, "Query fragments overlap or leave a gap"
        cursor = end
    assert cursor == cu[-1]


@pytest.mark.parametrize("cu,cp_size", CASES)
def test_csa_and_hca_share_one_query_layout(cu, cp_size):
    """The structural objective is ratio independent, so both plans must agree."""

    csa = _plan(4, cu, cp_size)
    hca = _plan(128, cu, cp_size)
    assert csa.query_layout_hash == hca.query_layout_hash
    assert csa.query_token_counts == hca.query_token_counts
    assert csa.token_layout_route == hca.token_layout_route
    for csa_rank, hca_rank in zip(csa.rank_plans, hca.rank_plans):
        assert csa_rank.query_fragments == hca_rank.query_fragments


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_plan_is_a_pure_function_of_its_inputs(ratio, cu, cp_size):
    """Every rank rebuilds the plan locally, so it must be reproducible."""

    first = _plan(ratio, cu, cp_size)
    second = _plan(ratio, cu, cp_size)
    assert first.plan_hash == second.plan_hash
    assert first.rank_plans == second.rank_plans


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_every_compressed_block_has_exactly_one_producer(ratio, cu, cp_size):
    plan = _plan(ratio, cu, cp_size)
    owned: list[tuple[int, int]] = []
    for rank_plan in plan.rank_plans:
        owned.extend(rank_plan.produced_block_ranges)
    owned.sort()
    cursor = 0
    for begin, end in owned:
        assert begin == cursor, "a compressed block has zero or two producers"
        cursor = end
    assert cursor == plan.total_compressed_blocks


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_block_producer_is_the_owner_of_its_last_token(ratio, cu, cp_size):
    """A group is compressed by whoever holds the token that completes it."""

    plan = _plan(ratio, cu, cp_size)
    owner_of_row: dict[int, int] = {}
    for rank_plan in plan.rank_plans:
        for fragment in rank_plan.query_fragments:
            for row in range(fragment.global_begin, fragment.global_end):
                owner_of_row[row] = rank_plan.rank
    for rank_plan in plan.rank_plans:
        for block_begin, block_end in rank_plan.produced_block_ranges:
            for global_block in range(block_begin, block_end):
                sample_id = max(
                    index
                    for index, offset in enumerate(rank_plan.sample_block_offsets)
                    if offset <= global_block
                )
                local_block = global_block - rank_plan.sample_block_offsets[sample_id]
                last_token = plan.cu_seqlens[sample_id] + (local_block + 1) * ratio - 1
                assert owner_of_row[last_token] == rank_plan.rank


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_routes_are_sorted_merged_ranges(ratio, cu, cp_size):
    """Core lowers ranges, so a route must never carry an unmerged range list."""

    plan = _plan(ratio, cu, cp_size)
    for route in plan.routes():
        assert isinstance(route, DsaRoutePlan)
        for table in (route.owner_ranges_per_rank, route.consumer_ranges_per_rank):
            for ranges in table:
                for (_, end), (next_begin, _) in zip(ranges, ranges[1:]):
                    # Sorted and non-overlapping. Compressed-block owner ranges
                    # are additionally split at sample boundaries, so touching
                    # ranges are legal there and only overlap is a defect.
                    assert end <= next_begin


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cu,cp_size", CASES)
def test_owner_rows_match_the_producer_buffers(ratio, cu, cp_size):
    plan = _plan(ratio, cu, cp_size)
    for rank, rank_plan in enumerate(plan.rank_plans):
        assert (
            plan.token_layout_route.owner_row_count(rank)
            == rank_plan.source_token_count
        )
        assert plan.window_route.owner_row_count(rank) == rank_plan.local_token_count
        assert plan.overlap_x_route.owner_row_count(rank) == (
            rank_plan.local_token_count
        )
        assert plan.compressed_kv_route.owner_row_count(rank) == (
            rank_plan.produced_block_count
        )


@pytest.mark.parametrize("cu,cp_size", CASES)
def test_csa_indexer_prefix_is_deduplicated_per_sample(cu, cp_size):
    """Several fragments of one sample share one longest routed prefix."""

    plan = _plan(4, cu, cp_size)
    for rank, rank_plan in enumerate(plan.rank_plans):
        longest: dict[int, int] = {}
        for fragment in rank_plan.query_fragments:
            longest[fragment.sample_id] = max(
                longest.get(fragment.sample_id, 0), fragment.q_end // 4
            )
        expected = sum(longest.values())
        assert plan.compressed_ki_route.consumer_row_count(rank) == expected
        # The packed grouped-K is what the Indexer actually consumes, and it is
        # allowed to be larger than the routed unique prefix.
        assert rank_plan.packed_indexer_k_count >= expected


@pytest.mark.parametrize("cu,cp_size", CASES)
def test_hca_has_no_indexer_route_or_metadata(cu, cp_size):
    plan = _plan(128, cu, cp_size)
    assert plan.compressed_ki_route is None
    for rank_plan in plan.rank_plans:
        assert rank_plan.indexer_required_k_ranges == ()
        assert rank_plan.packed_indexer_k_count == 0


@pytest.mark.parametrize("ratio", [4, 128])
def test_window_route_covers_every_local_causal_window(ratio):
    plan = _plan(ratio, (0, 1030, 2570), 4)
    for rank, rank_plan in enumerate(plan.rank_plans):
        received = set()
        for begin, end in plan.window_route.consumer_ranges_per_rank[rank]:
            received.update(range(begin, end))
        for fragment in rank_plan.query_fragments:
            for row in range(fragment.global_begin, fragment.global_end):
                position = row - fragment.sample_global_begin
                window = min(position + 1, 128)
                for offset in range(window):
                    assert row - offset in received


def test_plan_is_sized_by_fragments_not_tokens():
    """The plan must not grow a per-token or per-row table.

    Uses the production solver config, whose ``min_chunks_per_rank`` bounds the
    chunk count independently of the token count.
    """

    def build(total: int):
        return build_dsa_execution_plan(
            MagiDSAConfig(ratio=4),
            (0, total),
            _split(total, 4),
            structural_layout_config=DsaStructuralLayoutConfig(),
        )

    small = build(8192)
    large = build(65536)
    small_ranges = sum(
        len(ranges)
        for route in small.routes()
        for ranges in route.consumer_ranges_per_rank
    )
    large_ranges = sum(
        len(ranges)
        for route in large.routes()
        for ranges in route.consumer_ranges_per_rank
    )
    # Eight times the tokens must not cost eight times the plan.
    assert large_ranges < small_ranges * 3, (small_ranges, large_ranges)


def test_source_counts_must_cover_the_packed_tokens():
    with pytest.raises(ValueError, match="source token counts sum"):
        build_dsa_execution_plan(
            MagiDSAConfig(ratio=4),
            (0, 1024),
            (512, 256),
            structural_layout_config=SOLVER,
        )


def test_pro_model_spec_contains_only_csa_and_hca():
    spec = MagiDSAProModelSpec()
    assert spec.main_layer_count == 61
    assert len(spec.csa_layer_ids) == 30
    assert len(spec.hca_layer_ids) == 31
    assert set(spec.main_compress_ratios) == {4, 128}


def test_window_only_ratio_is_rejected():
    with pytest.raises(ValueError, match="ratio must be either 4"):
        MagiDSAConfig(ratio=0)  # type: ignore[arg-type]
