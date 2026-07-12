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

"""Cost, determinism, constraints and selection tests for the DSA solver."""

import random
import time
from unittest.mock import patch

import pytest

from magi_attention.meta.collection.dsa_meta import DsaFragmentSpec
from magi_attention.meta.solver.dsa_dispatch import (
    DsaCostModel,
    format_plan_comparison,
    fragment_indexer_cost,
    indexer_prefix_cost,
)
from magi_attention.meta.solver.dsa_solver import (
    DsaPlanSolver,
    DsaSolverConstraints,
    solve_dsa_plan,
)


def _fragment_tuples(plan):
    return tuple(
        tuple((fragment.q_begin, fragment.q_end) for fragment in rank.fragments)
        for rank in plan.ranks
    )


def _ragged_performance_lengths(sample_count, seed):
    rng = random.Random(seed)
    minimum_length = 129
    weights = [rng.randrange(1, 1001) for _ in range(sample_count)]
    remaining = 196608 - minimum_length * sample_count
    weight_sum = sum(weights)
    lengths = [minimum_length + remaining * weight // weight_sum for weight in weights]
    for index in range(196608 - sum(lengths)):
        lengths[index % sample_count] += 1
    return lengths


class TestDsaIndexerCost:
    def test_visible_blocks_are_sample_relative(self):
        assert [indexer_prefix_cost(length) for length in range(9)] == [
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            6,
        ]
        left = DsaFragmentSpec(0, 0, 128)
        late = DsaFragmentSpec(0, 128, 256)
        reset = DsaFragmentSpec(1, 0, 128)
        assert fragment_indexer_cost(late, 4) > fragment_indexer_cost(left, 4)
        assert fragment_indexer_cost(reset, 4) == fragment_indexer_cost(left, 4)

    @pytest.mark.parametrize("ratio", [0, 128])
    def test_non_indexer_layer_has_zero_indexer_cost(self, ratio):
        assert fragment_indexer_cost(DsaFragmentSpec(0, 128, 256), ratio) == 0


class TestDsaBalancedSolver:
    def test_canonical_noncontiguous_example_balances_scan_cost(self):
        sequential = solve_dsa_plan([1024], 2, 4, policy="sequential")
        balanced = solve_dsa_plan([1024], 2, 4, policy="balanced")

        expected = (
            ((0, 256), (768, 1024)),
            ((256, 768),),
        )
        actual = _fragment_tuples(balanced)
        assert actual == expected or actual == expected[::-1]
        assert balanced.max_rank_indexer_cost < sequential.max_rank_indexer_cost
        assert balanced.ranks[0].indexer_cost == balanced.ranks[1].indexer_cost

    def test_cp8_long_sample_balances_indexer_with_bounded_fragments(self):
        sequential = solve_dsa_plan([196608], 8, 4, policy="sequential")
        with patch(
            "magi_attention.meta.solver.dsa_solver._refine_moves_and_swaps"
        ) as refine:
            balanced = solve_dsa_plan([196608], 8, 4, policy="balanced")

        # LPT alternates nearly every atom at this scale and cannot satisfy the
        # 64-fragment budget.  Guard against bringing back its quadratic,
        # no-progress local search on the performance shape.
        refine.assert_not_called()
        assert _fragment_tuples(balanced) == (
            ((0, 12288), (184320, 196608)),
            ((12288, 24576), (172032, 184320)),
            ((24576, 36864), (159744, 172032)),
            ((36864, 49152), (147456, 159744)),
            ((49152, 61440), (135168, 147456)),
            ((61440, 73728), (122880, 135168)),
            ((73728, 86016), (110592, 122880)),
            ((86016, 110592),),
        )
        assert [rank.token_count for rank in balanced.ranks] == [24576] * 8
        assert max(rank.fragment_count for rank in balanced.ranks) <= 64

        mean_indexer_cost = sum(rank.indexer_cost for rank in balanced.ranks) / 8
        assert balanced.max_rank_indexer_cost / mean_indexer_cost <= 1.01
        assert balanced.max_rank_indexer_cost * 5 < sequential.max_rank_indexer_cost * 3

    def test_cp8_equal_length_packed_scale_skips_quadratic_refinement(self):
        started = time.perf_counter()
        with patch(
            "magi_attention.meta.solver.dsa_solver._refine_moves_and_swaps"
        ) as refine:
            balanced = solve_dsa_plan([3072] * 64, 8, 4, policy="balanced")
        elapsed = time.perf_counter() - started

        refine.assert_not_called()
        assert elapsed < 10.0
        assert balanced.plan_hash == (
            "09ea6a1f5984463eb6b289d4abac7517" "a1286c91a79ff4047c2156914b16d6b6"
        )
        assert [rank.token_count for rank in balanced.ranks] == [24576] * 8
        assert [rank.fragment_count for rank in balanced.ranks] == [8] * 8
        assert len({rank.indexer_cost for rank in balanced.ranks}) == 1

    @pytest.mark.parametrize(
        ("sample_count", "seed", "expected_hash"),
        [
            (
                20,
                62,
                "554ec4c0aee58219ffe63a0e7d7a4e927a2601b7747c6491877512c0ec00140e",
            ),
            (
                80,
                122,
                "e562a834b05026ccdaa63a42dcfb2608fac0497be6c44fb57509e1576c6224d8",
            ),
        ],
    )
    def test_cp8_ragged_packed_scale_is_balanced_and_bounded(
        self, sample_count, seed, expected_hash
    ):
        lengths = _ragged_performance_lengths(sample_count, seed)
        assert sum(lengths) == 196608
        sequential = solve_dsa_plan(lengths, 8, 4, policy="sequential")

        started = time.perf_counter()
        with patch(
            "magi_attention.meta.solver.dsa_solver._refine_moves_and_swaps"
        ) as refine:
            balanced = solve_dsa_plan(lengths, 8, 4, policy="balanced")
        elapsed = time.perf_counter() - started

        refine.assert_not_called()
        assert elapsed < 10.0
        assert balanced.plan_hash == expected_hash
        assert max(rank.fragment_count for rank in balanced.ranks) <= 64
        assert max(abs(rank.token_count - 24576) for rank in balanced.ranks) <= 256

        mean_indexer_cost = sum(rank.indexer_cost for rank in balanced.ranks) / 8
        assert balanced.max_rank_indexer_cost / mean_indexer_cost <= 1.01
        assert balanced.max_rank_indexer_cost < sequential.max_rank_indexer_cost

    @pytest.mark.parametrize("ratio", [0, 4, 128])
    def test_packed_plan_is_deterministic_for_all_layer_forms(self, ratio):
        first = solve_dsa_plan([17, 513, 129, 0, 260], 2, ratio)
        second = solve_dsa_plan([17, 513, 129, 0, 260], 2, ratio)
        assert first == second
        assert first.plan_hash == second.plan_hash
        assert len(first.plan_hash) == 64

    def test_solver_cache_returns_same_immutable_plan(self):
        solver = DsaPlanSolver()
        first = solver.solve([257, 769], 2, 4)
        second = solver.solve([257, 769], 2, 4)
        assert first is second
        assert solver.cache_size == 1
        solver.solve([257, 769], 2, 4, policy="sequential")
        assert solver.cache_size == 2
        solver.clear_cache()
        assert solver.cache_size == 0

    def test_report_compares_readable_sequential_and_balanced_loads(self):
        solver = DsaPlanSolver()
        sequential = solver.solve([1024], 2, 4, policy="sequential")
        balanced = solver.solve([1024], 2, 4)
        report = format_plan_comparison(sequential, balanced)
        assert "sequential plan" in report
        assert "balanced plan" in report
        assert "max/mean-1=" in report
        assert "predicted-e2e" in report

    def test_token_fragment_and_memory_constraints_are_enforced(self):
        with pytest.raises(ValueError, match="token/fragment"):
            solve_dsa_plan(
                [512],
                2,
                4,
                constraints=DsaSolverConstraints(max_tokens_per_rank=127),
            )
        with pytest.raises(ValueError, match="fragment"):
            solve_dsa_plan(
                [128, 128, 128],
                2,
                4,
                constraints=DsaSolverConstraints(max_fragments_per_rank=1),
            )
        with pytest.raises(ValueError, match="memory"):
            solve_dsa_plan(
                [256],
                2,
                4,
                constraints=DsaSolverConstraints(max_memory_bytes=1),
                cost_model=DsaCostModel(token_memory_bytes=2),
            )

    def test_empty_rank_and_empty_packed_batch(self):
        plan = solve_dsa_plan([7], 2, 4)
        assert sorted(rank.token_count for rank in plan.ranks) == [0, 7]
        empty = solve_dsa_plan([0, 0], 2, 128)
        assert [rank.token_count for rank in empty.ranks] == [0, 0]

    def test_rank_zero_solves_and_broadcasts_one_plan(self):
        solver = DsaPlanSolver()
        group = object()
        with (
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_rank", return_value=0
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_global_rank",
                return_value=7,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.broadcast_object_list"
            ) as broadcast,
        ):
            plan = solver.solve_distributed([1024], 4, group)
        assert plan.cp_size == 2
        broadcast.assert_called_once()
        assert broadcast.call_args.kwargs == {"src": 7, "group": group}

    def test_distributed_nonzero_rank_accepts_root_plan(self):
        root_plan = solve_dsa_plan([1024], 2, 4)
        solver = DsaPlanSolver()
        group = object()

        def receive(objects, **_kwargs):
            objects[0] = root_plan

        with (
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_rank", return_value=1
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.get_global_rank",
                return_value=0,
            ),
            patch(
                "magi_attention.meta.solver.dsa_solver.dist.broadcast_object_list",
                side_effect=receive,
            ),
        ):
            received = solver.solve_distributed([1024], 4, group)
        assert received == root_plan
        assert solver.cache_size == 1
