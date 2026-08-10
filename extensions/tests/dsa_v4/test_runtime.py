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
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from magi_attn_extensions.DSA.config import DsaRatio, MagiDSAConfig, MagiDSAProModelSpec
from magi_attn_extensions.DSA.modeling import MagiDSALayer, MagiDSAProLayerStack
from magi_attn_extensions.DSA.projection import (
    layout_and_project_dsa_input,
    layout_source_hidden_once,
    project_local_dsa_input,
)
from magi_attn_extensions.DSA.pro_runtime import (
    MagiDSAProExecutionBundle,
    MagiDSAProRuntimeMgr,
)
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr
from magi_attn_extensions.DSA.types import MagiDSAInput, MagiDSAPackedMeta


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


def test_model_adapter_projects_only_after_one_hidden_layout() -> None:
    config = _small_config(4)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    positions = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    backward_calls = 0
    layout_calls = 0

    class CountingLayout(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, source: torch.Tensor) -> torch.Tensor:
            nonlocal layout_calls
            layout_calls += 1
            ctx.save_for_backward(permutation)
            return source.index_select(0, permutation)

        @staticmethod
        def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
            nonlocal backward_calls
            backward_calls += 1
            (saved_permutation,) = ctx.saved_tensors
            restored = torch.empty_like(gradient)
            restored.index_copy_(0, saved_permutation, gradient)
            return restored

    class FakeRuntime:
        def __init__(self) -> None:
            self.config = config

        def layout_hidden(self, source: torch.Tensor, handle: object) -> torch.Tensor:
            del handle
            return CountingLayout.apply(source)

        def get_position_ids(self, handle: object) -> torch.Tensor:
            del handle
            return positions

    observed_positions: list[torch.Tensor] = []

    def projector(
        local_x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_positions.append(position_ids)
        qr = local_x[:, : config.q_lora_rank].contiguous()
        q = (
            local_x[:, :1]
            .view(local_x.shape[0], 1, 1)
            .expand(-1, config.num_query_heads, config.head_dim)
            .contiguous()
        )
        latent_kv = local_x[:, : config.head_dim].contiguous()
        return qr, q, latent_kv

    source_x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16).requires_grad_()
    sink = torch.zeros(config.num_query_heads, dtype=torch.float32)
    meta = MagiDSAPackedMeta((0, 4), (4,))
    handle = SimpleNamespace(
        device=source_x.device,
        device_plan=SimpleNamespace(local_token_count=4),
    )
    runtime = FakeRuntime()
    local_x = layout_source_hidden_once(
        source_x,
        runtime,  # type: ignore[arg-type]
        handle,  # type: ignore[arg-type]
    )
    first_input = project_local_dsa_input(
        local_x,
        sink,
        meta,
        runtime,  # type: ignore[arg-type]
        handle,  # type: ignore[arg-type]
        projector,
    )
    second_input = project_local_dsa_input(
        local_x,
        sink,
        meta,
        runtime,  # type: ignore[arg-type]
        handle,  # type: ignore[arg-type]
        projector,
    )

    assert layout_calls == 1
    assert observed_positions == [positions, positions]
    assert torch.equal(first_input.x, source_x.index_select(0, permutation))
    loss = (
        first_input.x.float().sum()
        + first_input.qr.float().sum()
        + first_input.q.float().sum()
        + first_input.latent_kv.float().sum()
        + second_input.qr.float().sum()
        + second_input.q.float().sum()
        + second_input.latent_kv.float().sum()
    )
    loss.backward()
    assert backward_calls == 1
    assert source_x.grad is not None
    assert torch.all(source_x.grad > 0)

    legacy_input = layout_and_project_dsa_input(
        source_x.detach(),
        sink,
        meta,
        runtime,  # type: ignore[arg-type]
        handle,  # type: ignore[arg-type]
        projector,
    )
    assert legacy_input.x.shape == local_x.shape
    assert layout_calls == 2


@pytest.mark.parametrize("ratio", [4, 128])
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


@pytest.mark.parametrize("ratio", [4, 128])
def test_layer_precision_keeps_linear_weights_bf16_and_explicit_state_fp32(
    ratio: DsaRatio,
) -> None:
    layer = MagiDSALayer(_small_config(ratio))
    linear_parameter_names = {
        f"{module_name}.{parameter_name}" if module_name else parameter_name
        for module_name, module in layer.named_modules()
        if isinstance(module, torch.nn.Linear)
        for parameter_name, _ in module.named_parameters(recurse=False)
    }
    assert linear_parameter_names
    named_parameters = dict(layer.named_parameters())
    assert all(
        named_parameters[name].dtype == torch.bfloat16
        for name in linear_parameter_names
    )

    fp32_parameter_names = set(named_parameters) - linear_parameter_names
    assert fp32_parameter_names
    assert all(
        name.endswith("ape") or name.endswith("norm.weight")
        for name in fp32_parameter_names
    )
    assert all(
        named_parameters[name].dtype == torch.float32 for name in fp32_parameter_names
    )
    assert all(
        buffer.dtype == torch.float32
        for name, buffer in layer.named_buffers()
        if name.endswith("inverse_frequencies")
    )


def test_pro_stack_has_61_independent_parameter_owners_and_parameter_free_runtime() -> (
    None
):
    stack = MagiDSAProLayerStack(device="meta")
    runtime = MagiDSAProRuntimeMgr()
    spec = MagiDSAProModelSpec()

    assert len(stack) == 61
    assert [layer.config.ratio for layer in stack.layers] == list(
        spec.main_compress_ratios
    )
    assert sum(layer.config.ratio == 4 for layer in stack.layers) == 30
    assert sum(layer.config.ratio == 128 for layer in stack.layers) == 31
    assert [layer.layer_id for layer in stack.layers] == list(range(61))
    assert list(runtime.parameters()) == []
    assert list(runtime.named_parameters()) == []
    assert runtime.state_dict() == {}

    parameters = list(stack.parameters())
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert sum(parameter.numel() for parameter in parameters) == 1_171_513_600
    parameter_names = dict(stack.named_parameters())
    assert all(
        any(name.startswith(f"layers.{layer_id}.") for name in parameter_names)
        for layer_id in range(61)
    )

    csa_layers = [layer for layer in stack.layers if layer.indexer is not None]
    assert len(csa_layers) == 30
    first_indexer = csa_layers[0].indexer
    second_indexer = csa_layers[1].indexer
    assert first_indexer is not None and second_indexer is not None
    first_indexer_parameter = next(first_indexer.parameters())
    second_indexer_parameter = next(second_indexer.parameters())
    assert first_indexer_parameter is not second_indexer_parameter

    fake_bundle = SimpleNamespace(csa=object(), hca=object())
    with pytest.raises(IndexError, match="main layer_id"):
        runtime.handle_for_layer(-1, fake_bundle)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="main layer_id"):
        runtime.handle_for_layer(61, fake_bundle)  # type: ignore[arg-type]

    csa_handle = SimpleNamespace(
        runtime_identity=runtime.csa_runtime.runtime_identity,
        plan=SimpleNamespace(ratio=4, query_layout_hash="shared"),
    )
    hca_handle = SimpleNamespace(
        runtime_identity=runtime.hca_runtime.runtime_identity,
        plan=SimpleNamespace(ratio=128, query_layout_hash="shared"),
    )
    bundle = MagiDSAProExecutionBundle(
        csa=csa_handle,  # type: ignore[arg-type]
        hca=hca_handle,  # type: ignore[arg-type]
        pro_runtime_identity=id(runtime),
    )
    for layer_id, ratio in enumerate(spec.main_compress_ratios):
        expected_handle = csa_handle if ratio == 4 else hca_handle
        assert runtime.handle_for_layer(layer_id, bundle) is expected_handle

    with pytest.raises(ValueError, match="bound Pro parameter owner"):
        runtime.calc_layer(
            4,
            stack[2],
            None,  # type: ignore[arg-type]
            bundle,
        )
    other_runtime = MagiDSAProRuntimeMgr()
    with pytest.raises(ValueError, match="different Pro runtime"):
        other_runtime.handle_for_layer(2, bundle)


def test_pro_stack_requires_explicit_materialization_and_all_csa_aux_losses() -> None:
    with pytest.raises(TypeError, match="device"):
        MagiDSAProLayerStack()  # type: ignore[call-arg]

    stack = MagiDSAProLayerStack(device="meta")
    losses = {
        layer_id: torch.tensor(float(layer_id), dtype=torch.float32, requires_grad=True)
        for layer_id in stack.model_spec.csa_layer_ids
    }
    total = stack.aggregate_csa_aux_losses(losses)
    assert total.item() == sum(loss.item() for loss in losses.values())
    total.backward()
    assert all(
        loss.grad is not None and loss.grad.item() == 1.0 for loss in losses.values()
    )

    missing = dict(losses)
    missing.pop(stack.model_spec.csa_layer_ids[-1])
    with pytest.raises(ValueError, match="all 30 layers"):
        stack.aggregate_csa_aux_losses(missing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pro_prepare_shares_one_layout_and_routes_source_hidden_once() -> None:
    runtime = MagiDSAProRuntimeMgr()
    packed_meta = MagiDSAPackedMeta((0, 257), (257,))
    bundle = runtime.prepare_execution(
        packed_meta,
        torch.device("cuda"),
        local_token_capacity=257,
        health_check=False,
    )

    assert bundle.query_layout_hash == bundle.csa.plan.query_layout_hash
    assert bundle.query_layout_hash == bundle.hca.plan.query_layout_hash
    assert bundle.csa.plan.rank_plans[0].query_fragments == (
        bundle.hca.plan.rank_plans[0].query_fragments
    )
    source_x = torch.randn(
        257,
        runtime.csa_runtime.config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    local_x = runtime.layout_source_hidden_once(source_x, bundle)
    local_x.float().sum().backward()
    assert source_x.grad is not None
    torch.testing.assert_close(source_x.grad, torch.ones_like(source_x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cp1_prepare_is_cold_cached_and_health_checked_once() -> None:
    config = _small_config(4)
    runtime = MagiDSARuntimeMgr(config, max_cached_handles=2)
    packed_meta = MagiDSAPackedMeta((0, 17), (17,))
    # The health check is a debugging dry run and is off by default, so it has
    # to be asked for explicitly.
    first = runtime.prepare_execution(
        packed_meta,
        torch.device("cuda"),
        local_token_capacity=32,
        health_check=True,
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
    overlap_streams = (
        first.sparse_backward_stream,
        first.csa_main_stream,
        first.csa_indexer_stream,
        first.csa_route_stream,
    )
    assert all(stream is not None for stream in overlap_streams)
    assert first.hca_main_stream is None
    assert first.hca_route_stream is None
    assert len(
        {stream.cuda_stream for stream in overlap_streams if stream is not None}
    ) == len(overlap_streams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hca_prepare_owns_distinct_overlap_streams() -> None:
    config = _small_config(128)
    runtime = MagiDSARuntimeMgr(config, max_cached_handles=2)
    handle = runtime.prepare_execution(
        MagiDSAPackedMeta((0, 257), (257,)),
        torch.device("cuda"),
        local_token_capacity=257,
    )

    assert handle.sparse_backward_stream is None
    assert handle.csa_main_stream is None
    assert handle.csa_indexer_stream is None
    assert handle.csa_route_stream is None
    assert handle.hca_main_stream is not None
    assert handle.hca_route_stream is not None
    assert handle.hca_main_stream.cuda_stream != handle.hca_route_stream.cuda_stream
    assert handle.hca_main_stream.priority == -1
    assert handle.hca_route_stream.priority == -1
