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

import torch

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.functional.dsa_packing import (
    copy_dsa_rows_reference,
    make_dsa_copy_map,
    make_dsa_reduce_map,
    make_dsa_route_maps,
    reduce_dsa_rows_reference,
)
from magi_attention.meta.solver.dsa_solver import build_dsa_execution_plan


def test_copy_and_reduce_maps_cover_duplicates_and_empty_rows() -> None:
    source = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    copy_map = make_dsa_copy_map((4, 1, 1, 3), source_row_count=5)
    packed = copy_dsa_rows_reference(source, copy_map)
    assert torch.equal(packed, source[torch.tensor([4, 1, 1, 3])])

    reduce_map = make_dsa_reduce_map((2, 0, 2, 2), destination_row_count=4)
    reduced = reduce_dsa_rows_reference(packed, reduce_map)
    expected = torch.stack(
        (packed[1], torch.zeros(4), packed[0] + packed[2] + packed[3], torch.zeros(4))
    )
    assert torch.equal(reduced, expected)


def test_route_maps_reconstruct_consumers_and_reduce_owner_gradients() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(
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
        ),
        cu_seqlens=(0, 7, 18, 31),
        local_token_counts=(5, 0, 8, 9, 9),
        policy="indexer_balanced",
    )
    routes = [rank.indexer_qw_route for rank in plan.rank_plans]
    assert all(route is not None for route in routes)
    concrete = [route for route in routes if route is not None]
    maps = [make_dsa_route_maps(route) for route in concrete]
    producers = [
        torch.arange(route.producer_row_count, dtype=torch.float32).unsqueeze(1)
        + plan.rank_plans[rank].local_global_begin
        for rank, route in enumerate(concrete)
    ]
    packed_send = [
        copy_dsa_rows_reference(source, route_maps.send_pack)
        for source, route_maps in zip(producers, maps)
    ]

    consumers: list[torch.Tensor] = []
    for destination, route in enumerate(concrete):
        chunks = []
        for source, source_route in enumerate(concrete):
            source_begin = sum(source_route.send_counts[:destination])
            count = source_route.send_counts[destination]
            chunks.append(packed_send[source].narrow(0, source_begin, count))
        received = torch.cat(chunks, dim=0)
        consumers.append(
            copy_dsa_rows_reference(received, maps[destination].consumer_pack)
        )
        expected = torch.tensor(
            route.consumer_global_rows, dtype=torch.float32
        ).unsqueeze(1)
        assert torch.equal(consumers[-1], expected)

    consumer_grads = [
        consumer + destination + 1 for destination, consumer in enumerate(consumers)
    ]
    reverse_received = [
        copy_dsa_rows_reference(gradient, route_maps.received_pack)
        for gradient, route_maps in zip(consumer_grads, maps)
    ]
    for source, source_route in enumerate(concrete):
        chunks = []
        for destination, destination_route in enumerate(concrete):
            begin = sum(destination_route.recv_counts[:source])
            count = destination_route.recv_counts[source]
            chunks.append(reverse_received[destination].narrow(0, begin, count))
        reverse_send = torch.cat(chunks, dim=0)
        actual = reduce_dsa_rows_reference(reverse_send, maps[source].owner_reduce)
        expected = torch.zeros_like(producers[source])
        for destination, destination_route in enumerate(concrete):
            for consumer_row, global_row in enumerate(
                destination_route.consumer_global_rows
            ):
                owner_begin = plan.rank_plans[source].local_global_begin
                owner_end = plan.rank_plans[source].local_global_end
                if owner_begin <= global_row < owner_end:
                    expected[global_row - owner_begin] += consumer_grads[destination][
                        consumer_row
                    ]
        assert torch.equal(actual, expected)


def test_empty_rank_route_maps_keep_zero_sized_permutations_valid() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 256),
        local_token_counts=(128, 0, 128),
        policy="indexer_balanced",
    )
    route = plan.rank_plans[1].indexer_qw_route
    assert route is not None
    maps = make_dsa_route_maps(route)
    source = torch.empty((0, 8), dtype=torch.float32)
    packed = copy_dsa_rows_reference(source, maps.send_pack)
    assert packed.shape == (0, 8)
    reduced = reduce_dsa_rows_reference(packed, maps.owner_reduce)
    assert reduced.shape == (0, 8)
