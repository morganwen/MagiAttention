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

from dataclasses import fields

import pytest
import torch

from magi_attention.dsa_config import DsaRatio, MagiDSAConfig
from magi_attention.dsa_layer import MagiDSALayer
from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
from magi_attention.dsa_types import MagiDSAInput, MagiDSAPackedMeta


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


def test_formal_input_exposes_only_natural_selection() -> None:
    field_names = {field.name for field in fields(MagiDSAInput)}
    assert "forced_topk_ids" not in field_names
    assert "forced_topk_length" not in field_names


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_runtime_is_parameter_free_and_layer_owns_all_trainable_state(
    ratio: DsaRatio,
) -> None:
    config = _small_config(ratio)
    layer = MagiDSALayer(config)
    runtime = MagiDSARuntimeMgr(config)
    assert list(runtime.parameters()) == []
    assert list(runtime.named_parameters()) == []
    assert runtime.state_dict() == {}
    layer_parameters = dict(layer.named_parameters())
    if ratio == 0:
        assert layer_parameters == {}
    else:
        assert any(name.startswith("compressor.") for name in layer_parameters)
        if ratio == 4:
            assert any(name.startswith("indexer.") for name in layer_parameters)
        else:
            assert not any(name.startswith("indexer.") for name in layer_parameters)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cp1_prepare_is_cold_cached_and_health_checked_once() -> None:
    config = _small_config(4)
    runtime = MagiDSARuntimeMgr(config, policy="indexer_balanced", max_cached_handles=2)
    packed_meta = MagiDSAPackedMeta((0, 17), 17)
    first = runtime.prepare_execution(
        packed_meta,
        torch.device("cuda"),
        local_token_capacity=32,
    )
    first_counters = runtime.counters
    second = runtime.prepare_execution(
        packed_meta,
        torch.device("cuda"),
        local_token_capacity=32,
    )
    assert second is first
    assert runtime.counters == first_counters
    assert first_counters.solver_invocations == 1
    assert first_counters.object_collective_invocations == 0
    assert first_counters.device_materializations == 1
    assert first_counters.health_checks == 1
    assert first_counters.warm_invocations == 0
