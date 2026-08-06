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

from magi_attention.dsa_config import (
    DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT,
    DSV4_PRO_MAIN_COMPRESS_RATIOS,
    DSV4_PRO_REVISION,
    DsaRatio,
    DsaSharedLayoutConfig,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAProModelSpec,
)
from magi_attention.meta.collection.dsa_meta import (
    DsaExecutionPlan,
    DsaLayoutMetrics,
    DsaStructuralLayoutMetrics,
)
from magi_attention.meta.solver.dsa_solver import _build_route, build_dsa_execution_plan


def test_dsv4_pro_main_schedule_matches_the_official_31_hca_30_csa_stack() -> None:
    spec = MagiDSAProModelSpec()

    assert spec.source_revision == DSV4_PRO_REVISION
    assert spec.main_compress_ratios == DSV4_PRO_MAIN_COMPRESS_RATIOS
    assert spec.main_layer_count == 61
    assert spec.hca_layer_ids == (0, 1, *range(3, 60, 2))
    assert spec.csa_layer_ids == tuple(range(2, 61, 2))
    assert len(spec.hca_layer_ids) == 31
    assert len(spec.csa_layer_ids) == 30
    assert set(spec.main_compress_ratios) == {4, 128}
    assert spec.mtp_compress_ratio == 0
    assert spec.activation_dtype == "bfloat16"
    assert spec.linear_weight_dtype == "bfloat16"
    assert spec.main_gradient_dtype == "float32"


@pytest.mark.parametrize(
    ("layer_id", "ratio"),
    ((0, 128), (1, 128), (2, 4), (59, 128), (60, 4)),
)
def test_dsv4_pro_layer_config_uses_official_dimensions(
    layer_id: int,
    ratio: DsaRatio,
) -> None:
    config = MagiDSAProModelSpec().make_layer_config(layer_id)

    assert config.ratio == ratio
    assert config.hidden_size == 7168
    assert config.q_lora_rank == 1536
    assert config.num_query_heads == 128
    assert config.head_dim == 512
    assert config.indexer_heads == 64
    assert config.indexer_head_dim == 128
    assert config.indexer_topk == 1024
    assert config.linear_weight_dtype == "bfloat16"
    assert config.accumulator_dtype == "float32"
    config.validate_release_contract()


def test_release_contract_rejects_nonofficial_compressor_epsilon() -> None:
    with pytest.raises(ValueError, match="norm_eps"):
        MagiDSAConfig(ratio=4, norm_eps=1e-5).validate_release_contract()


def test_csa_indexer_score_workspace_is_aligned_without_changing_lengths() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 37),
        source_token_counts=(37,),
        policy="sequential",
    )
    rank_plan = plan.rank_plans[0]

    assert rank_plan.indexer_logical_max_seqlen_k == 9
    assert rank_plan.indexer_backend_max_seqlen_k == 16
    assert (
        rank_plan.indexer_backend_max_seqlen_k % DSV4_PRO_INDEXER_SCORE_ROW_ALIGNMENT
        == 0
    )
    assert rank_plan.indexer_k_cu_seqlens == (0, 9)
    assert rank_plan.indexer_seq_lens[-1] == 9


def test_precision_contract_rejects_nonofficial_parameter_or_accumulator_dtype() -> (
    None
):
    with pytest.raises(ValueError, match="linear weights"):
        MagiDSAConfig(
            ratio=4,
            linear_weight_dtype="float32",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="accumulate"):
        MagiDSAConfig(
            ratio=4,
            accumulator_dtype="bfloat16",  # type: ignore[arg-type]
        )


def test_structural_minheap_builds_one_native_layout_for_csa_and_hca() -> None:
    structural_config = DsaStructuralLayoutConfig(
        chunk_size=512,
        min_chunks_per_rank=16,
    )
    ratios: tuple[DsaRatio, ...] = (4, 128)
    plans = [
        build_dsa_execution_plan(
            MagiDSAConfig(ratio=ratio),
            cu_seqlens=(0, 3, 132, 132, 391),
            source_token_counts=(50, 0, 79, 130, 132),
            policy="structural_balanced",
            structural_layout_config=structural_config,
        )
        for ratio in ratios
    ]

    assert len({plan.query_layout_hash for plan in plans}) == 1
    assert len({plan.query_token_counts for plan in plans}) == 1
    assert len({_query_layout_signature(plan) for plan in plans}) == 1
    for plan in plans:
        assert plan.boundary_collective_order == ("TOKEN_LAYOUT",)
        assert plan.shared_layout_config is None
        assert plan.structural_layout_config == structural_config
        assert isinstance(plan.layout_metrics, DsaStructuralLayoutMetrics)
        assert plan.layout_metrics.solver_scheme == "magi_min_heap_packed_global_v1"
        assert (
            plan.layout_metrics.cost_model_version == "native_causal_attn_slice_area_v1"
        )
        assert plan.layout_metrics.chunk_size == 5
        assert plan.layout_metrics.num_chunks == 79
        assert (
            sum(rank_cost.query_tokens for rank_cost in plan.layout_metrics.rank_costs)
            == 391
        )
        assert all(
            rank_cost.csa_packed_indexer_k_rows
            == rank_cost.csa_unique_indexer_k_rows
            + rank_cost.csa_duplicate_indexer_k_rows
            for rank_cost in plan.layout_metrics.rank_costs
        )
        assert (
            max(rank_cost.chunk_count for rank_cost in plan.layout_metrics.rank_costs)
            - min(rank_cost.chunk_count for rank_cost in plan.layout_metrics.rank_costs)
            <= 1
        )


def test_structural_minheap_cp1_covers_every_logical_chunk_and_query() -> None:
    tokens = 257
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, tokens),
        source_token_counts=(tokens,),
        policy="structural_balanced",
        structural_layout_config=DsaStructuralLayoutConfig(
            chunk_size=512,
            min_chunks_per_rank=16,
        ),
    )

    assert isinstance(plan.layout_metrics, DsaStructuralLayoutMetrics)
    assert plan.layout_metrics.chunk_size == 17
    assert plan.layout_metrics.num_chunks == 16
    assert plan.query_token_counts == (tokens,)
    fragments = plan.rank_plans[0].query_fragments
    assert sum(fragment.q_end - fragment.q_begin for fragment in fragments) == tokens
    assert fragments[0].sample_id == 0 and fragments[0].q_begin == 0
    assert fragments[-1].sample_id == 0 and fragments[-1].q_end == tokens
    rank_cost = plan.layout_metrics.rank_costs[0]
    assert rank_cost.chunk_count == 16
    assert rank_cost.query_tokens == tokens
    assert rank_cost.native_causal_area == tokens * (tokens + 1) // 2


def test_structural_minheap_balances_native_causal_area_without_device_weights() -> (
    None
):
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 131072),
        source_token_counts=(16384,) * 8,
        policy="structural_balanced",
        structural_layout_config=DsaStructuralLayoutConfig(),
    )

    assert isinstance(plan.layout_metrics, DsaStructuralLayoutMetrics)
    metrics = plan.layout_metrics
    areas = [rank_cost.native_causal_area for rank_cost in metrics.rank_costs]
    assert metrics.chunk_size == 512
    assert metrics.num_chunks == 256
    assert _relative_range(areas) <= 0.01
    assert all(rank_cost.chunk_count == 32 for rank_cost in metrics.rank_costs)
    assert any(
        rank_cost.csa_duplicate_indexer_k_rows > 0 for rank_cost in metrics.rank_costs
    )


def test_csa_required_ranges_keep_one_longest_prefix_per_consumer_sample() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 257, 600),
        source_token_counts=(150, 150, 150, 150),
        policy="structural_balanced",
        structural_layout_config=DsaStructuralLayoutConfig(
            chunk_size=64,
            min_chunks_per_rank=2,
        ),
    )

    saw_duplicate_prefix = False
    for rank_plan in plan.rank_plans:
        expected_lengths: dict[int, int] = {}
        for fragment in rank_plan.query_fragments:
            prefix_length = fragment.q_end // 4
            saw_duplicate_prefix |= (
                prefix_length > 0 and fragment.sample_id in expected_lengths
            )
            expected_lengths[fragment.sample_id] = max(
                expected_lengths.get(fragment.sample_id, 0),
                prefix_length,
            )
        expected_lengths = {
            sample_id: length
            for sample_id, length in expected_lengths.items()
            if length > 0
        }
        actual_ranges = rank_plan.indexer_required_k_ranges
        assert tuple(item.sample_id for item in actual_ranges) == tuple(
            sorted(expected_lengths)
        )
        assert tuple(item.length for item in actual_ranges) == tuple(
            expected_lengths[sample_id] for sample_id in sorted(expected_lengths)
        )
        for item in actual_ranges:
            assert item.global_begin == rank_plan.sample_block_offsets[item.sample_id]
            assert (
                item.global_end == item.global_begin + expected_lengths[item.sample_id]
            )

        required_rows = tuple(
            row
            for item in actual_ranges
            for row in range(item.global_begin, item.global_end)
        )
        assert rank_plan.compressed_ki_route is not None
        assert sorted(rank_plan.compressed_ki_route.consumer_global_rows) == list(
            required_rows
        )
    assert saw_duplicate_prefix


def test_group_collective_arg_is_the_canonical_typed_route_lowering() -> None:
    route = _build_route(
        "TEST",
        producer_owner=(0, 1, 0, 2, 0, 0, 0),
        producer_local_row=(0, 0, 1, 0, 2, 3, 4),
        producer_row_counts=(5, 1, 1),
        consumer_rows=((0, 1, 2, 3, 5), (4,), (0, 2, 5)),
    )

    rank0 = route.rank_plans[0]
    assert rank0.group_collective_arg.input_split_size_list == (2, 1, 1, 1)
    assert rank0.group_collective_arg.dst_indices_list == (
        (0, 2),
        (1,),
        (0, 2),
        (),
    )
    assert rank0.send_counts == (3, 1, 3)
    assert rank0.send_source_rows == (0, 1, 3, 2, 0, 1, 3)
    assert rank0.reverse_row_offsets == (0, 2, 4, 5, 7, 7)
    assert rank0.reverse_source_rows == (0, 4, 1, 5, 3, 2, 6)

    consumer0 = route.rank_plans[0]
    assert consumer0.group_collective_arg.output_split_size_list == (1, 1, 1, 1, 1)
    assert consumer0.group_collective_arg.src_index_list == (0, 1, 0, 2, 0)
    assert consumer0.received_global_rows == (0, 2, 5, 1, 3)
    assert consumer0.consumer_from_received == (0, 3, 1, 4, 2)
    assert consumer0.received_from_consumer == (0, 2, 4, 1, 3)


def test_group_collective_arg_rejects_consumer_rows_out_of_producer_order() -> None:
    with pytest.raises(ValueError, match="out of producer-buffer order"):
        _build_route(
            "TEST",
            producer_owner=(0, 0),
            producer_local_row=(0, 1),
            producer_row_counts=(2, 0),
            consumer_rows=((1, 0), ()),
        )


def test_structural_core_routes_use_backend_consumable_receive_order() -> None:
    for ratio in (4, 128):
        plan = build_dsa_execution_plan(
            MagiDSAConfig(ratio=ratio),
            cu_seqlens=(0, 3, 132, 391),
            source_token_counts=(50, 0, 79, 130, 132),
            policy="structural_balanced",
            structural_layout_config=DsaStructuralLayoutConfig(),
        )
        for rank_plan in plan.rank_plans:
            routes = [
                rank_plan.window_route,
                rank_plan.compressed_kv_route,
                rank_plan.compressed_ki_route,
            ]
            if ratio == 4:
                routes.append(rank_plan.overlap_x_route)
            for route in routes:
                if route is None:
                    continue
                rows = tuple(range(len(route.received_global_rows)))
                assert route.consumer_global_rows == route.received_global_rows
                assert route.consumer_from_received == rows
                assert route.received_from_consumer == rows

            if ratio == 128:
                assert rank_plan.compression_source_from_overlap == tuple(
                    range(len(rank_plan.compression_source_from_overlap))
                )


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_query_plan_covers_packed_ragged_batch_with_empty_source_rank(
    ratio: DsaRatio,
) -> None:
    config = MagiDSAConfig(ratio=ratio)
    plan = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 3, 132, 132, 391),
        source_token_counts=(50, 0, 79, 130, 132),
        policy="indexer_balanced",
    )

    assert plan.total_tokens == 391
    covered = [0] * plan.total_tokens
    for rank_plan in plan.rank_plans:
        assert (
            rank_plan.source_global_end - rank_plan.source_global_begin
            == rank_plan.source_token_count
        )
        assert len(rank_plan.local_query_global_rows) == rank_plan.local_token_count
        for fragment in rank_plan.query_fragments:
            for token in range(fragment.global_begin, fragment.global_end):
                covered[token] += 1
    assert covered == [1] * plan.total_tokens
    assert plan.rank_plans[1].source_token_count == 0
    if ratio == 4:
        assert plan.rank_plans[1].local_token_count > 0
        assert plan.rank_plans[1].query_fragments
    else:
        assert plan.rank_plans[1].local_token_count == 0
        assert plan.rank_plans[1].query_fragments == ()


def test_csa_blocks_use_previous_b_and_current_a_rows_and_drop_tail() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 11, 20),
        source_token_counts=(5, 6, 9),
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
        source_token_counts=(64, 64, 64, 65),
    )

    assert len(plan.compressed_blocks) == 2
    assert plan.compressed_blocks[0].source_global_rows == tuple(range(128))
    assert plan.compressed_blocks[0].producer_rank == 1
    assert plan.compressed_blocks[1].source_global_rows == tuple(range(128, 256))
    assert plan.compressed_blocks[1].producer_rank == 3
    assert plan.collective_order == (
        "OVERLAP_X",
        "WINDOW_KV",
        "COMPRESSED_KV",
    )


def _relative_range(values: list[int]) -> float:
    return (max(values) - min(values)) / (sum(values) / len(values))


def test_frozen_128k_balanced_solver_meets_both_predicted_cost_targets() -> None:
    config = MagiDSAConfig(ratio=4)
    counts = (16384,) * 8
    sequential = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 131072),
        source_token_counts=counts,
        policy="sequential",
    )
    balanced = build_dsa_execution_plan(
        config,
        cu_seqlens=(0, 131072),
        source_token_counts=counts,
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
    assert sum(rank.indexer_token_count for rank in balanced.rank_plans) == 131072
    assert balanced.collective_order == (
        "WINDOW_KV",
        "OVERLAP_X",
        "COMPRESSED_KI",
        "COMPRESSED_KV",
    )
    assert balanced.boundary_collective_order == ("TOKEN_LAYOUT",)


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
        route = rank_plan.token_layout_route
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


def _query_layout_signature(
    plan: DsaExecutionPlan,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(
        tuple(
            (fragment.sample_id, fragment.q_begin, fragment.q_end)
            for fragment in rank_plan.query_fragments
        )
        for rank_plan in plan.rank_plans
    )


def test_shared_greedy_builds_one_query_layout_for_all_attention_ratios() -> None:
    solver_config = DsaSharedLayoutConfig(
        ki_memory_budget_bytes=1 << 30,
        ki_workspace_reserve_bytes=64 << 20,
        local_improvement_passes=2,
    )
    ratios: tuple[DsaRatio, ...] = (0, 4, 128)
    plans = [
        build_dsa_execution_plan(
            MagiDSAConfig(ratio=ratio),
            cu_seqlens=(0, 3, 132, 391),
            source_token_counts=(50, 0, 79, 130, 132),
            policy="shared_greedy",
            shared_layout_config=solver_config,
        )
        for ratio in ratios
    ]

    assert len({plan.query_layout_hash for plan in plans}) == 1
    assert len({plan.query_token_counts for plan in plans}) == 1
    assert len({_query_layout_signature(plan) for plan in plans}) == 1
    assert len({plan.layout_metrics for plan in plans}) == 1
    for plan in plans:
        assert plan.boundary_collective_order == ("TOKEN_LAYOUT",)
        assert all(
            rank_plan.token_layout_route is not None for rank_plan in plan.rank_plans
        )
        assert isinstance(plan.layout_metrics, DsaLayoutMetrics)
        assert plan.shared_layout_config == solver_config
        assert plan.layout_metrics.solver_scheme.startswith(
            "deterministic_greedy_local_improve"
        )
        assert plan.layout_metrics.candidate_evaluations > 0
        assert plan.layout_metrics.stop_reason in ("local_optimum", "pass_limit")
        assert all(
            solver_config.ki_workspace_reserve_bytes
            <= rank_cost.modeled_ki_bytes
            <= solver_config.ki_memory_budget_bytes
            for rank_cost in plan.layout_metrics.rank_costs
        )


def test_shared_greedy_local_improvement_never_worsens_the_lexicographic_key() -> None:
    greedy = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 4096),
        source_token_counts=(1024, 1024, 1024, 1024),
        policy="shared_greedy",
        shared_layout_config=DsaSharedLayoutConfig(
            ki_memory_budget_bytes=1 << 30,
            ki_workspace_reserve_bytes=64 << 20,
            local_improvement_passes=0,
        ),
    )
    improved = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 4096),
        source_token_counts=(1024, 1024, 1024, 1024),
        policy="shared_greedy",
        shared_layout_config=DsaSharedLayoutConfig(
            ki_memory_budget_bytes=1 << 30,
            ki_workspace_reserve_bytes=64 << 20,
            local_improvement_passes=4,
        ),
    )

    assert isinstance(greedy.layout_metrics, DsaLayoutMetrics)
    assert isinstance(improved.layout_metrics, DsaLayoutMetrics)
    assert improved.layout_metrics.key <= greedy.layout_metrics.key


def test_shared_greedy_uses_fixed_ki_pack_weight_and_memory_budget() -> None:
    solver_config = DsaSharedLayoutConfig(
        ki_memory_budget_bytes=1 << 30,
        ki_workspace_reserve_bytes=64 << 20,
        local_improvement_passes=4,
    )

    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 4096),
        source_token_counts=(1024, 1024, 1024, 1024),
        policy="shared_greedy",
        shared_layout_config=solver_config,
    )

    assert isinstance(plan.layout_metrics, DsaLayoutMetrics)
    assert plan.layout_metrics.solver_scheme == (
        "deterministic_greedy_local_improve_v1"
    )
    assert plan.layout_metrics.cost_model_version == ("b300_sm103_structural_proxy_v1")
    for rank_cost in plan.layout_metrics.rank_costs:
        assert rank_cost.indexer_cost == (
            8 * rank_cost.indexer_score_cost
            + rank_cost.indexer_topk_cost
            + 32 * rank_cost.packed_ki_rows
        )
        assert rank_cost.modeled_ki_bytes <= solver_config.ki_memory_budget_bytes


def test_shared_greedy_requires_an_explicit_memory_budget() -> None:
    with pytest.raises(ValueError, match="explicit shared_layout_config"):
        build_dsa_execution_plan(
            MagiDSAConfig(ratio=4),
            cu_seqlens=(0, 128),
            source_token_counts=(64, 64),
            policy="shared_greedy",
        )
    with pytest.raises(RuntimeError, match="search failure"):
        build_dsa_execution_plan(
            MagiDSAConfig(ratio=4),
            cu_seqlens=(0, 128),
            source_token_counts=(64, 64),
            policy="shared_greedy",
            shared_layout_config=DsaSharedLayoutConfig(
                ki_memory_budget_bytes=1,
                ki_workspace_reserve_bytes=0,
                local_improvement_passes=0,
            ),
        )


def test_shared_layout_rejects_an_invalid_workspace_reserve() -> None:
    with pytest.raises(ValueError, match="smaller than"):
        DsaSharedLayoutConfig(
            ki_memory_budget_bytes=1024,
            ki_workspace_reserve_bytes=1024,
        )
