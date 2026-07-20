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

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.meta.solver.dsa_solver import build_dsa_execution_plan


def test_all_typed_routes_have_symmetric_counts_and_unique_rows() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4, indexer_atom_size=8),
        cu_seqlens=(0, 17, 49, 90),
        local_token_counts=(0, 13, 27, 50),
        policy="indexer_balanced",
    )
    route_names = (
        "window_route",
        "overlap_x_route",
        "compressed_kv_route",
        "compressed_ki_route",
        "indexer_qw_route",
    )
    for route_name in route_names:
        routes = [getattr(rank, route_name) for rank in plan.rank_plans]
        assert all(route is not None for route in routes)
        for source, source_route in enumerate(routes):
            assert source_route is not None
            assert len(set(source_route.consumer_global_rows)) == len(
                source_route.consumer_global_rows
            )
            for destination, destination_route in enumerate(routes):
                assert destination_route is not None
                assert (
                    source_route.send_counts[destination]
                    == destination_route.recv_counts[source]
                )


def test_indexer_query_route_is_a_global_bijection() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4, indexer_atom_size=4),
        cu_seqlens=(0, 9, 21),
        local_token_counts=(7, 0, 5, 9),
        policy="indexer_balanced",
    )
    consumed: list[int] = []
    for rank_plan in plan.rank_plans:
        route = rank_plan.indexer_qw_route
        assert route is not None
        consumed.extend(route.consumer_global_rows)
    assert sorted(consumed) == list(range(plan.total_tokens))


def test_zero_row_rank_still_has_every_collective_route() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 256),
        local_token_counts=(128, 0, 128),
        policy="indexer_balanced",
    )
    empty = plan.rank_plans[1]
    assert empty.local_token_count == 0
    assert empty.window_route is not None
    assert empty.overlap_x_route is not None
    assert empty.compressed_kv_route is not None
    assert empty.compressed_ki_route is not None
    assert empty.indexer_qw_route is not None
    assert len(empty.window_route.send_counts) == 3
    assert len(empty.window_route.recv_counts) == 3
