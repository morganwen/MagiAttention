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

"""Device materialization of the range-shaped cold plan."""

from __future__ import annotations

import pytest
import torch
from magi_attn_extensions.DSA.config import DsaStructuralLayoutConfig, MagiDSAConfig
from magi_attn_extensions.DSA.packing import (
    gather_compressor_support,
    make_dsa_device_rank_plan,
)
from magi_attn_extensions.DSA.solver import build_dsa_execution_plan

SOLVER = DsaStructuralLayoutConfig(chunk_size=128, min_chunks_per_rank=4)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="device plans require CUDA"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _plan(ratio: int, cu=(0, 1030, 2570)):
    return build_dsa_execution_plan(
        MagiDSAConfig(ratio=ratio), cu, (cu[-1],), structural_layout_config=SOLVER
    )


@pytest.mark.parametrize("ratio", [4, 128])
def test_device_plan_expands_positions_from_fragments(ratio):
    """Per-token metadata is derived on the device, never shipped in the plan."""

    config = MagiDSAConfig(ratio=ratio)
    plan = _plan(ratio)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    rank_plan = plan.rank_plans[0]

    expected_samples: list[int] = []
    expected_positions: list[int] = []
    for fragment in rank_plan.query_fragments:
        expected_samples.extend([fragment.sample_id] * fragment.length)
        expected_positions.extend(range(fragment.q_begin, fragment.q_end))
    assert device_plan.local_q_sample_ids.tolist() == expected_samples
    assert device_plan.local_q_positions.tolist() == expected_positions


@pytest.mark.parametrize("ratio", [4, 128])
def test_global_to_consumer_matches_the_consumer_ranges(ratio):
    config = MagiDSAConfig(ratio=ratio)
    plan = _plan(ratio)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    for route, plan_route in (
        (device_plan.window_route, plan.window_route),
        (device_plan.compressed_kv_route, plan.compressed_kv_route),
    ):
        table = route.global_to_consumer.tolist()
        cursor = 0
        for begin, end in plan_route.consumer_ranges_per_rank[0]:
            for global_row in range(begin, end):
                assert table[global_row] == cursor
                cursor += 1
        assert cursor == route.consumer_row_count
        assert sum(1 for value in table if value >= 0) == cursor


@pytest.mark.parametrize("ratio", [4, 128])
def test_attention_map_window_run_is_contiguous_and_causal(ratio):
    config = MagiDSAConfig(ratio=ratio)
    plan = _plan(ratio)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    attention = device_plan.attention
    positions = device_plan.local_q_positions.tolist()
    lengths = attention.window_length.tolist()
    bases = attention.window_base.tolist()
    table = device_plan.window_route.global_to_consumer.tolist()
    sample_ids = device_plan.local_q_sample_ids.tolist()
    for row, (position, length, base) in enumerate(zip(positions, lengths, bases)):
        assert length == min(position + 1, config.window_size)
        sample_begin = plan.cu_seqlens[sample_ids[row]]
        first_global = sample_begin + position - length + 1
        assert base == table[first_global]
        # The whole window must be one contiguous run in the consumer bank.
        for offset in range(length):
            assert table[first_global + offset] == base + offset


def test_compressed_prefix_run_is_contiguous():
    config = MagiDSAConfig(ratio=4)
    plan = _plan(4)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    attention = device_plan.attention
    table = device_plan.compressed_kv_route.global_to_consumer.tolist()
    sample_ids = device_plan.local_q_sample_ids.tolist()
    positions = device_plan.local_q_positions.tolist()
    offsets = plan.rank_plans[0].sample_block_offsets
    for row, (sample_id, position) in enumerate(zip(sample_ids, positions)):
        visible = (position + 1) // config.ratio
        if not visible:
            continue
        base = int(attention.compressed_base[row])
        sample_block_begin = offsets[sample_id]
        assert base == table[sample_block_begin]
        for offset in range(visible):
            assert table[sample_block_begin + offset] == base + offset


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_support_gather_matches_the_group_rows(ratio):
    """The gathered support must be exactly the rows each block compresses."""

    config = MagiDSAConfig(ratio=ratio)
    plan = _plan(ratio)
    device = _device()
    device_plan = make_dsa_device_rank_plan(plan, 0, config, device)
    rank_plan = plan.rank_plans[0]

    consumer_rows: list[int] = []
    for begin, end in plan.overlap_x_route.consumer_ranges_per_rank[0]:
        consumer_rows.extend(range(begin, end))
    overlap_x = torch.tensor(
        consumer_rows, dtype=torch.float32, device=device
    ).unsqueeze(1)

    packed = gather_compressor_support(
        overlap_x, device_plan.compression, config.compressor_support
    )
    valid = device_plan.compression.valid_rows
    assert packed.shape == (rank_plan.produced_block_count, config.compressor_support, 1)

    block_ids = [
        block
        for begin, end in rank_plan.produced_block_ranges
        for block in range(begin, end)
    ]
    for index, global_block in enumerate(block_ids):
        sample_id = max(
            i for i, off in enumerate(rank_plan.sample_block_offsets)
            if off <= global_block
        )
        local_block = global_block - rank_plan.sample_block_offsets[sample_id]
        group_begin = plan.cu_seqlens[sample_id] + local_block * ratio
        for column in range(config.compressor_support):
            if not bool(valid[index, column]):
                continue
            if ratio == 4:
                # CSA reads the previous group and then the current one.
                expected = group_begin - ratio + column
            else:
                expected = group_begin + column
            assert int(packed[index, column, 0]) == expected


def test_csa_first_block_of_a_sample_masks_its_missing_previous_group():
    config = MagiDSAConfig(ratio=4)
    plan = _plan(4)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    rank_plan = plan.rank_plans[0]
    valid = device_plan.compression.valid_rows
    block_ids = [
        block
        for begin, end in rank_plan.produced_block_ranges
        for block in range(begin, end)
    ]
    seen_first = False
    for index, global_block in enumerate(block_ids):
        is_sample_first = global_block in rank_plan.sample_block_offsets
        if is_sample_first:
            seen_first = True
            assert not bool(valid[index, :4].any()), "leading group must be masked"
            assert bool(valid[index, 4:].all())
        else:
            assert bool(valid[index].all())
    assert seen_first, "the CP1 case must contain a sample-initial block"


def test_indexer_gather_ranges_cover_the_grouped_prefixes():
    config = MagiDSAConfig(ratio=4)
    plan = _plan(4)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    indexer = device_plan.indexer
    assert indexer is not None
    rank_plan = plan.rank_plans[0]
    ranges = indexer.k_gather_ranges.tolist()
    assert len(ranges) == len(rank_plan.query_fragments)
    total = 0
    for (begin, end), fragment in zip(ranges, rank_plan.query_fragments):
        assert end - begin == fragment.q_end // config.ratio
        total += end - begin
    assert total == indexer.packed_k_rows == rank_plan.packed_indexer_k_count


def test_hca_device_plan_has_no_indexer():
    config = MagiDSAConfig(ratio=128)
    plan = _plan(128)
    device_plan = make_dsa_device_rank_plan(plan, 0, config, _device())
    assert device_plan.indexer is None
    assert device_plan.compressed_ki_route is None
