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
