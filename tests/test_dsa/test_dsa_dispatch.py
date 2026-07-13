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

"""Fragment, owner, restore and transfer tests for the DSA static plan."""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from magi_attention.functional.dist_dsa import _canonical_query_tiles
from magi_attention.meta.collection.dsa_meta import (
    DsaCompressedBlockSpec,
    DsaFragmentSpec,
    DsaTransferSpec,
)
from magi_attention.meta.solver.dsa_dispatch import (
    DsaCostModel,
    build_dsa_dispatch_plan,
    make_sequential_plan,
    validate_dsa_dispatch_plan,
)
from magi_attention.meta.solver.dsa_solver import DsaPlanSolver


def _owner_at(plan, sample_id, position):
    for rank in plan.ranks:
        for fragment in rank.fragments:
            if (
                fragment.sample_id == sample_id
                and fragment.q_begin <= position < fragment.q_end
            ):
                return rank.rank
    raise AssertionError(f"no owner for sample={sample_id}, position={position}")


class TestDsaFragmentPlan:
    def test_fragment_rejects_invalid_coordinates(self):
        with pytest.raises(ValueError, match="sample_id"):
            DsaFragmentSpec(-1, 0, 128)
        with pytest.raises(ValueError, match="q_begin"):
            DsaFragmentSpec(0, -1, 128)
        with pytest.raises(ValueError, match="non-empty"):
            DsaFragmentSpec(0, 128, 128)

    def test_sequential_plan_is_sample_relative_aligned_and_contiguous(self):
        plan = make_sequential_plan([193, 17, 518], 2, 4)
        validate_dsa_dispatch_plan(plan)
        assert sum(rank.token_count for rank in plan.ranks) == 728
        for rank in plan.ranks:
            for fragment in rank.fragments:
                sample_length = plan.sample_lengths[fragment.sample_id]
                assert fragment.q_begin == 0 or fragment.q_begin % 128 == 0
                assert fragment.q_end == sample_length or fragment.q_end % 128 == 0

        owners = [
            _owner_at(plan, sample_id, position)
            for sample_id, length in enumerate(plan.sample_lengths)
            for position in range(length)
        ]
        assert owners == sorted(owners)

    def test_canonical_query_tiles_split_merged_fragment_and_tail(self):
        plan = build_dsa_dispatch_plan(
            [500],
            [[DsaFragmentSpec(0, 0, 500)]],
            compress_ratio=4,
            policy="sequential",
        )
        forward_plan = SimpleNamespace(
            dispatch_plan=plan,
            local_token_count=500,
        )
        runtime = SimpleNamespace(plan=SimpleNamespace(cp_rank=0))
        tiles = _canonical_query_tiles(forward_plan, runtime)

        assert tiles == (
            (DsaFragmentSpec(0, 0, 128), 0, 128),
            (DsaFragmentSpec(0, 128, 256), 128, 256),
            (DsaFragmentSpec(0, 256, 384), 256, 384),
            (DsaFragmentSpec(0, 384, 500), 384, 500),
        )

    def test_canonical_query_tiles_are_policy_invariant(self):
        sequential = make_sequential_plan([500], 2, 4)
        balanced = build_dsa_dispatch_plan(
            [500],
            [
                [DsaFragmentSpec(0, 0, 128), DsaFragmentSpec(0, 256, 384)],
                [DsaFragmentSpec(0, 128, 256), DsaFragmentSpec(0, 384, 500)],
            ],
            compress_ratio=4,
            policy="balanced",
        )

        def global_tiles(plan):
            result = []
            for rank in range(plan.cp_size):
                forward_plan = SimpleNamespace(
                    dispatch_plan=plan,
                    local_token_count=plan.ranks[rank].token_count,
                )
                runtime = SimpleNamespace(plan=SimpleNamespace(cp_rank=rank))
                result.extend(
                    tile
                    for tile, _local_begin, _local_end in _canonical_query_tiles(
                        forward_plan, runtime
                    )
                )
            return tuple(sorted(result))

        expected = (
            DsaFragmentSpec(0, 0, 128),
            DsaFragmentSpec(0, 128, 256),
            DsaFragmentSpec(0, 256, 384),
            DsaFragmentSpec(0, 384, 500),
        )
        assert global_tiles(sequential) == expected
        assert global_tiles(balanced) == expected

    def test_noncontiguous_plan_restore_map_and_transfers(self):
        plan = build_dsa_dispatch_plan(
            [1024],
            [
                [DsaFragmentSpec(0, 0, 256), DsaFragmentSpec(0, 768, 1024)],
                [DsaFragmentSpec(0, 256, 768)],
            ],
            compress_ratio=4,
            policy="balanced",
        )

        assert plan.restore_rows()[:3] == ((0, 0), (0, 1), (0, 2))
        assert plan.restore_rows()[255:258] == ((0, 255), (1, 0), (1, 1))
        assert plan.restore_rows()[767:770] == (
            (1, 511),
            (0, 256),
            (0, 257),
        )
        assert plan.window_transfers == (
            DsaTransferSpec(0, 1, 0, 129, 256),
            DsaTransferSpec(1, 0, 0, 641, 768),
        )
        assert plan.overlap_transfers == (
            DsaTransferSpec(0, 1, 0, 252, 256),
            DsaTransferSpec(1, 0, 0, 764, 768),
        )

    @pytest.mark.parametrize("ratio", [4, 128])
    def test_compressed_block_owner_uses_last_token_and_drops_tail(self, ratio):
        lengths = [ratio * 2 + ratio - 1, ratio + 1]
        plan = make_sequential_plan(lengths, 2, ratio)
        assert len(plan.compressed_blocks) == 3
        for logical_id, block in enumerate(plan.compressed_blocks):
            last_token = (block.sample_block_id + 1) * ratio - 1
            assert block.owner_rank == _owner_at(plan, block.sample_id, last_token)
            assert block.logical_block_id == logical_id

    def test_empty_samples_and_empty_rank_are_explicit(self):
        plan = make_sequential_plan([0, 13, 0], 4, 0)
        validate_dsa_dispatch_plan(plan)
        assert sum(not rank.fragments for rank in plan.ranks) >= 3
        assert plan.total_tokens == 13
        assert plan.compressed_blocks == ()
        assert len(plan.restore_rows()) == 13

    @pytest.mark.parametrize(
        "per_rank",
        [
            [[DsaFragmentSpec(0, 0, 128)], [DsaFragmentSpec(0, 256, 384)]],
            [[DsaFragmentSpec(0, 0, 256)], [DsaFragmentSpec(0, 128, 384)]],
            [[DsaFragmentSpec(0, 0, 127)], [DsaFragmentSpec(0, 127, 384)]],
        ],
    )
    def test_hole_overlap_and_unaligned_cut_are_rejected(self, per_rank):
        with pytest.raises(ValueError):
            build_dsa_dispatch_plan([384], per_rank, compress_ratio=0, policy="bad")

    def test_validator_rejects_corrupted_derived_tables(self):
        plan = make_sequential_plan([256], 2, 4)
        with pytest.raises(ValueError, match="window transfer"):
            validate_dsa_dispatch_plan(replace(plan, window_transfers=()))

        first = plan.compressed_blocks[0]
        wrong_owner = 1 - first.owner_rank
        corrupted_blocks = (
            DsaCompressedBlockSpec(
                first.logical_block_id,
                first.sample_id,
                first.sample_block_id,
                wrong_owner,
            ),
        ) + plan.compressed_blocks[1:]
        with pytest.raises(ValueError, match="block owner"):
            validate_dsa_dispatch_plan(
                replace(plan, compressed_blocks=corrupted_blocks)
            )

    def test_1000_random_packed_balanced_plans(self):
        rng = random.Random(20260709)
        solver = DsaPlanSolver()
        for _ in range(1000):
            sample_lengths = [rng.randrange(0, 641) for _ in range(rng.randrange(1, 6))]
            cp_size = rng.randrange(1, 5)
            ratio = rng.choice((0, 4, 128))
            plan = solver.solve(sample_lengths, cp_size, ratio)
            validate_dsa_dispatch_plan(plan)
            assert plan.total_tokens == sum(sample_lengths)
            assert len(plan.restore_rows()) == sum(sample_lengths)
            assert plan is solver.solve(sample_lengths, cp_size, ratio)
            if ratio == 4:
                sequential = solver.solve(
                    sample_lengths, cp_size, ratio, policy="sequential"
                )
                assert plan.max_rank_indexer_cost <= sequential.max_rank_indexer_cost


class TestDsaTransferPlan:
    def test_window_rows_are_unique_and_never_cross_samples(self):
        plan = build_dsa_dispatch_plan(
            [256, 256],
            [
                [DsaFragmentSpec(0, 0, 128), DsaFragmentSpec(1, 128, 256)],
                [DsaFragmentSpec(0, 128, 256), DsaFragmentSpec(1, 0, 128)],
            ],
            compress_ratio=0,
            policy="balanced",
        )
        assert plan.window_transfers == (
            DsaTransferSpec(0, 1, 0, 1, 128),
            DsaTransferSpec(1, 0, 1, 1, 128),
        )
        rows = [
            (
                transfer.source_rank,
                transfer.destination_rank,
                transfer.sample_id,
                row,
            )
            for transfer in plan.window_transfers
            for row in range(transfer.q_begin, transfer.q_end)
        ]
        assert len(rows) == len(set(rows))

    def test_overlap_routes_exist_only_for_ratio4(self):
        fragments = [
            [DsaFragmentSpec(0, 0, 128)],
            [DsaFragmentSpec(0, 128, 256)],
        ]
        ratio4 = build_dsa_dispatch_plan(
            [256], fragments, compress_ratio=4, policy="balanced"
        )
        ratio128 = build_dsa_dispatch_plan(
            [256], fragments, compress_ratio=128, policy="balanced"
        )
        assert ratio4.overlap_transfers == (DsaTransferSpec(0, 1, 0, 124, 128),)
        assert ratio128.overlap_transfers == ()


class TestDsaCompressedBroadcastCost:
    @staticmethod
    def _isolated_cost_model() -> DsaCostModel:
        return DsaCostModel(
            token_weight=0,
            indexer_weight=0,
            fragment_overhead=0,
            window_transfer_weight=0,
            overlap_transfer_weight=0,
            compressed_owner_send_weight=2,
            compressed_remote_receive_weight=3,
            token_memory_bytes=0,
            compressed_block_memory_bytes=5,
            compressed_owner_send_memory_bytes=7,
            compressed_remote_receive_memory_bytes=11,
            compressed_global_memory_bytes=13,
            remote_row_memory_bytes=0,
        )

    @pytest.mark.parametrize("cp_size", [1, 2, 8])
    def test_full_peer_broadcast_cost_and_global_residency(self, cp_size):
        plan = make_sequential_plan(
            [1024],
            cp_size,
            4,
            cost_model=self._isolated_cost_model(),
        )
        total_blocks = 256
        for rank in plan.ranks:
            local_blocks = len(rank.compressed_block_ids)
            owner_send_rows = local_blocks * (cp_size - 1)
            remote_receive_rows = total_blocks - local_blocks
            assert rank.predicted_e2e == (2 * owner_send_rows + 3 * remote_receive_rows)
            assert rank.estimated_memory_bytes == (
                5 * local_blocks
                + 7 * local_blocks * int(cp_size > 1)
                + 11 * remote_receive_rows
                + 13 * total_blocks
            )

    def test_cp8_owner_send_uses_peer_fanout_not_unique_rows(self):
        plan = build_dsa_dispatch_plan(
            [1024],
            [[DsaFragmentSpec(0, 0, 1024)]] + [[] for _ in range(7)],
            compress_ratio=4,
            policy="owner-heavy",
            cost_model=self._isolated_cost_model(),
        )
        owner, receiver = plan.ranks[:2]
        assert owner.predicted_e2e == 2 * 256 * 7
        assert receiver.predicted_e2e == 3 * 256
        assert owner.estimated_memory_bytes == (5 + 7 + 13) * 256
        assert receiver.estimated_memory_bytes == (11 + 13) * 256

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("compressed_owner_send_weight", "weights"),
            ("compressed_remote_receive_memory_bytes", "memory coefficients"),
        ],
    )
    def test_compressed_broadcast_coefficients_must_be_non_negative(
        self, field, message
    ):
        with pytest.raises(ValueError, match=message):
            DsaCostModel(**{field: -1})
