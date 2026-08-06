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

from magi_attention.dsa_config import DsaStructuralLayoutConfig, MagiDSAConfig
from magi_attention.functional.dsa_packing import (
    _make_indexer_map,
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
        source_token_counts=(5, 0, 8, 9, 9),
        policy="indexer_balanced",
    )
    routes = [rank.token_layout_route for rank in plan.rank_plans]
    assert all(route is not None for route in routes)
    concrete = [route for route in routes if route is not None]
    maps = [make_dsa_route_maps(route) for route in concrete]
    producers = [
        torch.arange(route.producer_row_count, dtype=torch.float32).unsqueeze(1)
        + plan.rank_plans[rank].source_global_begin
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
                owner_begin = plan.rank_plans[source].source_global_begin
                owner_end = plan.rank_plans[source].source_global_end
                if owner_begin <= global_row < owner_end:
                    expected[global_row - owner_begin] += consumer_grads[destination][
                        consumer_row
                    ]
        assert torch.equal(actual, expected)


def test_empty_rank_route_maps_keep_zero_sized_permutations_valid() -> None:
    plan = build_dsa_execution_plan(
        MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 256),
        source_token_counts=(128, 0, 128),
        policy="indexer_balanced",
    )
    route = plan.rank_plans[1].token_layout_route
    assert route is not None
    maps = make_dsa_route_maps(route)
    source = torch.empty((0, 8), dtype=torch.float32)
    packed = copy_dsa_rows_reference(source, maps.send_pack)
    assert packed.shape == (0, 8)
    reduced = reduce_dsa_rows_reference(packed, maps.owner_reduce)
    assert reduced.shape == (0, 8)


def test_csa_indexer_k_pack_materializes_each_fragment_from_one_unique_bank() -> None:
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
    rank_plan = plan.rank_plans[0]
    indexer_map = _make_indexer_map(rank_plan, torch.device("cpu"))

    assert indexer_map is not None
    assert not hasattr(indexer_map, "k_unpack")
    assert rank_plan.compressed_ki_route is not None
    unique_global_rows = torch.tensor(
        rank_plan.compressed_ki_route.consumer_global_rows,
        dtype=torch.int64,
    )
    packed_global_rows = unique_global_rows[indexer_map.k_pack.source_rows.long()]
    expected_global_rows = torch.tensor(
        [
            rank_plan.sample_block_offsets[fragment.sample_id] + block_offset
            for fragment in rank_plan.query_fragments
            for block_offset in range(fragment.q_end // 4)
        ],
        dtype=torch.int64,
    )
    assert torch.equal(packed_global_rows, expected_global_rows)
    assert packed_global_rows.numel() == rank_plan.packed_indexer_k_count
    assert packed_global_rows.numel() > unique_global_rows.numel()
