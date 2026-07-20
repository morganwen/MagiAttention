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
import torch

from magi_attention.dsa_config import DsaRatio, MagiDSAConfig
from magi_attention.functional.dsa_comm import (
    copy_dsa_tensor_with_csr,
    route_dsa_tensor,
)
from magi_attention.functional.dsa_packing import (
    DsaDeviceCopyMap,
    DsaDeviceReduceMap,
    make_dsa_device_rank_plan,
)
from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows, reduce_dsa_rows
from magi_attention.meta.solver.dsa_solver import build_dsa_execution_plan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _small_config(ratio: DsaRatio) -> MagiDSAConfig:
    return MagiDSAConfig(
        ratio=ratio,
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


def test_cute_copy_and_csr_match_torch_with_empty_destination_row() -> None:
    source = (
        torch.arange(128, device="cuda", dtype=torch.float32)
        .reshape(8, 16)
        .to(torch.bfloat16)
    )
    copy_rows = torch.tensor([7, 1, 1, 3], device="cuda", dtype=torch.int32)
    copied = copy_dsa_rows(source, copy_rows)
    assert torch.equal(copied, source[copy_rows.long()])

    offsets = torch.tensor([0, 1, 3, 3, 4], device="cuda", dtype=torch.int32)
    reduce_rows = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
    reduced = reduce_dsa_rows(copied, offsets, reduce_rows)
    expected = torch.stack(
        (copied[0], copied[1] + copied[2], torch.zeros_like(copied[0]), copied[3])
    )
    torch.testing.assert_close(reduced.float(), expected.float(), atol=0.0, rtol=0.0)


def test_duplicate_pack_backward_uses_csr_and_supports_retain_graph() -> None:
    source = torch.randn(5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    copy_map = DsaDeviceCopyMap(
        torch.tensor([4, 1, 1, 3], device="cuda", dtype=torch.int32)
    )
    reduce_map = DsaDeviceReduceMap(
        row_offsets=torch.tensor([0, 0, 2, 2, 3, 4], device="cuda", dtype=torch.int32),
        source_rows=torch.tensor([1, 2, 3, 0], device="cuda", dtype=torch.int32),
    )
    packed = copy_dsa_tensor_with_csr(source, copy_map, reduce_map)
    first = torch.autograd.grad(packed.float().sum(), source, retain_graph=True)[0]
    second = torch.autograd.grad(packed.float().sum(), source)[0]
    expected = (
        torch.tensor([0, 2, 0, 1, 1], device="cuda", dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 16)
    )
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_cp1_device_plan_route_is_identity_and_reentrant() -> None:
    config = _small_config(4)
    plan = build_dsa_execution_plan(config, (0, 17), (17,), policy="indexer_balanced")
    device_plan = make_dsa_device_rank_plan(plan, 0, config, torch.device("cuda"))
    route = device_plan.indexer_qw_route
    assert route is not None
    source = torch.randn(
        17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    first = route_dsa_tensor(source, route, None)
    second = route_dsa_tensor(source, route, None)
    assert torch.equal(first, source)
    assert torch.equal(second, source)
    gradient = torch.autograd.grad(first.float().sum() + second.float().sum(), source)[
        0
    ]
    assert torch.equal(gradient, torch.full_like(source, 2))


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_all_ratio_device_plans_materialize_static_attention_maps(
    ratio: DsaRatio,
) -> None:
    config = _small_config(ratio)
    plan = build_dsa_execution_plan(
        config, (0, 17, 278), (31, 0, 100, 147), policy="indexer_balanced"
    )
    for rank in range(plan.cp_size):
        device_plan = make_dsa_device_rank_plan(
            plan, rank, config, torch.device("cuda")
        )
        assert device_plan.local_q_positions.shape == (
            plan.rank_plans[rank].local_token_count,
        )
        assert device_plan.attention.window_rows.shape == (
            plan.rank_plans[rank].local_token_count,
            config.window_size,
        )
        assert (
            device_plan.attention.window_lengths.numel()
            == plan.rank_plans[rank].local_token_count
        )
        assert (device_plan.compression is None) == (ratio == 0)
        assert (device_plan.indexer is not None) == (ratio == 4)
