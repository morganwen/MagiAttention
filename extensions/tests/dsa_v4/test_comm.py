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

"""Route planning contract.

Magi-DSA states routes as ranges and lets MagiAttention Core lower them. These
tests pin that boundary: the extension must not re-derive splits, rank routes or
row permutations of its own.
"""

from __future__ import annotations

import inspect

from magi_attn_extensions.DSA import comm as dsa_comm
from magi_attn_extensions.DSA import packing as dsa_packing
from magi_attn_extensions.DSA.config import MagiDSAConfig
from magi_attn_extensions.DSA.solver import build_dsa_execution_plan


def _plan(**kwargs):
    return build_dsa_execution_plan(**kwargs)


def test_every_routed_row_has_exactly_one_producer() -> None:
    plan = _plan(
        config=MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 17, 49, 90),
        source_token_counts=(0, 13, 27, 50),
    )
    for route in plan.routes():
        owned: list[tuple[int, int]] = []
        for ranges in route.owner_ranges_per_rank:
            owned.extend(ranges)
        owned.sort()
        for (_, end), (next_begin, _) in zip(owned, owned[1:]):
            assert end <= next_begin, f"{route.name} owner ranges overlap"


def test_consumer_ranges_are_covered_by_the_owner_ranges() -> None:
    """A consumer may only ask for rows that some rank actually produces."""

    plan = _plan(
        config=MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 17, 49, 90),
        source_token_counts=(0, 13, 27, 50),
    )
    for route in plan.routes():
        produced = set()
        for ranges in route.owner_ranges_per_rank:
            for begin, end in ranges:
                produced.update(range(begin, end))
        for rank, ranges in enumerate(route.consumer_ranges_per_rank):
            for begin, end in ranges:
                missing = set(range(begin, end)) - produced
                assert not missing, f"{route.name} rank {rank} wants unowned rows"


def test_token_layout_route_is_a_global_bijection() -> None:
    plan = _plan(
        config=MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 9, 21),
        source_token_counts=(7, 0, 5, 9),
    )
    consumed: list[int] = []
    for ranges in plan.token_layout_route.consumer_ranges_per_rank:
        for begin, end in ranges:
            consumed.extend(range(begin, end))
    assert sorted(consumed) == list(range(plan.total_tokens))


def test_zero_row_rank_still_participates_in_every_route() -> None:
    plan = _plan(
        config=MagiDSAConfig(ratio=4),
        cu_seqlens=(0, 256),
        source_token_counts=(128, 0, 128),
    )
    empty = plan.rank_plans[1]
    assert empty.source_token_count == 0
    # A rank with no source rows still receives Query rows after TOKEN_LAYOUT.
    assert empty.local_token_count > 0
    assert plan.token_layout_route.owner_ranges_per_rank[1] == ()
    for route in plan.routes():
        assert len(route.owner_ranges_per_rank) == plan.cp_size
        assert len(route.consumer_ranges_per_rank) == plan.cp_size
    assert plan.window_route.consumer_row_count(1) > 0


def test_dsa_routes_are_lowered_by_core_group_collectives() -> None:
    """The extension must not carry its own collective or row-map machinery."""

    comm_source = inspect.getsource(dsa_comm)
    assert "group_cast(" in comm_source
    assert "group_reduce(" in comm_source
    # The hand-rolled All2AllV data plane and its row permutations are gone.
    assert "all2all_v(" not in comm_source
    assert "send_pack" not in comm_source
    assert "consumer_pack" not in comm_source
    assert "owner_reduce" not in comm_source

    packing_source = inspect.getsource(dsa_packing)
    assert "_calc_group_collective_arg_from_ranges" in packing_source
    assert "A2AVBasedGroupCollectiveArg" in packing_source


def test_route_lowering_is_delegated_not_reimplemented() -> None:
    """The solver states ranges only; it must not build splits or rank routes."""

    from magi_attn_extensions.DSA import solver as dsa_solver

    solver_source = inspect.getsource(dsa_solver)
    for forbidden in (
        "input_split_size_list",
        "output_split_size_list",
        "dst_indices_list",
        "src_index_list",
        "send_counts",
        "recv_counts",
    ):
        assert forbidden not in solver_source, (
            f"the planner re-derives {forbidden}, which belongs to Core"
        )
