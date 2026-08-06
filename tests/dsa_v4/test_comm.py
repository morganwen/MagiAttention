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

import inspect

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.functional import dsa_comm
from magi_attention.meta.solver.dsa_solver import build_dsa_execution_plan


def test_all_typed_routes_have_symmetric_counts_and_unique_rows() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4, indexer_atom_size=8),
        cu_seqlens=(0, 17, 49, 90),
        source_token_counts=(0, 13, 27, 50),
        policy="indexer_balanced",
    )
    route_names = (
        "window_route",
        "overlap_x_route",
        "compressed_kv_route",
        "compressed_ki_route",
    )
    for route_name in route_names:
        routes = [getattr(rank, route_name) for rank in plan.rank_plans]
        assert all(route is not None for route in routes)
        for source, source_route in enumerate(routes):
            assert source_route is not None
            group_arg = source_route.group_collective_arg
            assert group_arg.rank == source
            assert group_arg.world_size == plan.cp_size
            assert (
                sum(group_arg.input_split_size_list) == source_route.producer_row_count
            )
            assert sum(group_arg.output_split_size_list) == len(
                source_route.consumer_global_rows
            )

            input_segments: list[tuple[int, int, tuple[int, ...]]] = []
            local_begin = 0
            for split_size, destinations in zip(
                group_arg.input_split_size_list,
                group_arg.dst_indices_list,
            ):
                local_end = local_begin + split_size
                input_segments.append((local_begin, local_end, destinations))
                local_begin = local_end
            rebuilt_send_rows: list[int] = []
            rebuilt_send_counts: list[int] = []
            for destination in range(plan.cp_size):
                destination_rows = [
                    row
                    for begin, end, destinations in input_segments
                    if destination in destinations
                    for row in range(begin, end)
                ]
                rebuilt_send_counts.append(len(destination_rows))
                rebuilt_send_rows.extend(destination_rows)
            assert tuple(rebuilt_send_counts) == source_route.send_counts
            assert tuple(rebuilt_send_rows) == source_route.send_source_rows

            rebuilt_recv_counts = tuple(
                sum(
                    split_size
                    for split_size, split_source in zip(
                        group_arg.output_split_size_list,
                        group_arg.src_index_list,
                    )
                    if split_source == peer
                )
                for peer in range(plan.cp_size)
            )
            assert rebuilt_recv_counts == source_route.recv_counts
            assert len(set(source_route.consumer_global_rows)) == len(
                source_route.consumer_global_rows
            )
            for destination, destination_route in enumerate(routes):
                assert destination_route is not None
                assert (
                    source_route.send_counts[destination]
                    == destination_route.recv_counts[source]
                )


def test_token_layout_route_is_a_global_bijection() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4, indexer_atom_size=4),
        cu_seqlens=(0, 9, 21),
        source_token_counts=(7, 0, 5, 9),
        policy="indexer_balanced",
    )
    consumed: list[int] = []
    for rank_plan in plan.rank_plans:
        route = rank_plan.token_layout_route
        assert route is not None
        consumed.extend(route.consumer_global_rows)
    assert sorted(consumed) == list(range(plan.total_tokens))


def test_zero_row_rank_still_has_every_collective_route() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 256),
        source_token_counts=(128, 0, 128),
        policy="indexer_balanced",
    )
    empty = plan.rank_plans[1]
    assert empty.source_token_count == 0
    assert empty.local_token_count > 0
    assert empty.window_route is not None
    assert empty.overlap_x_route is not None
    assert empty.compressed_kv_route is not None
    assert empty.compressed_ki_route is not None
    assert empty.token_layout_route is not None
    assert len(empty.window_route.send_counts) == 3
    assert len(empty.window_route.recv_counts) == 3
    assert empty.token_layout_route.group_collective_arg.input_split_size_list == ()


def test_dsa_group_collective_data_plane_remains_direct_all2all_v() -> None:
    source = inspect.getsource(dsa_comm)

    assert "all2all_v(" in source
    assert "group_cast(" not in source
    assert "group_reduce(" not in source
    assert "grpcoll" not in source
