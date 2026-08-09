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

import copy
from typing import Any, cast

import pytest
import torch
from magi_attn_extensions.DSA.backend import run_grouped_dsa_indexer
from magi_attn_extensions.DSA.config import (
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
)
from magi_attn_extensions.DSA.modeling import MagiDSALayer
from magi_attention.common.range_op import range_gather
from magi_attn_extensions.DSA.reference import (
    _compress_global,
    assert_backend_native_topk_outputs_close,
    dsa_reference,
)
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr
from magi_attn_extensions.DSA.types import MagiDSAInput, MagiDSAPackedMeta

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_cp1_csa_indexer_raw_scores_match_pure_pytorch_reference() -> None:
    torch.manual_seed(20)
    config = MagiDSAConfig(ratio=4)
    tokens = 256
    layer = MagiDSALayer(config).cuda()
    x = torch.randn(
        tokens,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    qr = torch.randn(
        tokens,
        config.q_lora_rank,
        device="cuda",
        dtype=torch.bfloat16,
    )
    meta = MagiDSAPackedMeta((0, tokens), (tokens,))
    runtime = MagiDSARuntimeMgr(
        config,
        structural_layout_config=DsaStructuralLayoutConfig(),
    )
    handle = runtime.prepare_execution(
        meta,
        torch.device("cuda"),
        local_token_capacity=tokens,
    )

    from cudnn import DSA

    captured: dict[str, object] = {}
    original_score = DSA.indexer_forward_wrapper
    original_topk = DSA.indexer_top_k_wrapper

    def assert_explicit_current_stream(kwargs: dict[str, Any]) -> None:
        stream = kwargs["stream"]
        assert int(stream) != 0
        assert int(stream) == torch.cuda.current_stream().cuda_stream

    def capture_score(*args: Any, **kwargs: Any) -> Any:
        assert_explicit_current_stream(kwargs)
        backend_result = original_score(*args, **kwargs)
        captured["scores"] = backend_result["scores"].detach().clone()
        captured["sm_scale"] = kwargs["sm_scale"]
        return backend_result

    def capture_topk(*args: Any, **kwargs: Any) -> Any:
        assert_explicit_current_stream(kwargs)
        assert kwargs.get("return_val") is False
        backend_result = original_topk(*args, **kwargs)
        assert backend_result["values"] is None
        captured["raw_topk_indices"] = backend_result["indices"].detach().clone()
        return backend_result

    assert layer.indexer is not None
    indexer_map = handle.device_plan.indexer
    assert indexer_map is not None
    with torch.no_grad():
        positions = handle.device_plan.local_q_positions
        q_indexer, weights = layer.indexer.project_queries(
            x,
            qr,
            positions,
            detach_trunk=False,
        )
        compressed_ki, _, _ = _compress_global(
            layer.indexer.compressor,
            x,
            meta.cu_seqlens,
        )
        grouped_k = range_gather(
            compressed_ki,
            indexer_map.k_gather_ranges,
            total_size=indexer_map.packed_k_rows,
        )
        score_weights = weights
        DSA.indexer_forward_wrapper = capture_score
        DSA.indexer_top_k_wrapper = capture_topk
        try:
            caller_stream = torch.cuda.current_stream()
            indexer_stream = torch.cuda.Stream()
            indexer_stream.wait_stream(caller_stream)
            with torch.cuda.stream(indexer_stream):
                selection = run_grouped_dsa_indexer(
                    q_indexer,
                    grouped_k,
                    score_weights,
                    indexer_map,
                    config,
                )
            caller_stream.wait_stream(indexer_stream)
        finally:
            DSA.indexer_forward_wrapper = original_score
            DSA.indexer_top_k_wrapper = original_topk
        assert selection.logical_score_calls == 1
        assert selection.logical_topk_calls == 1
        dots = torch.einsum(
            "qhd,kd->qhk",
            q_indexer.float(),
            compressed_ki.float(),
        )
        score_weights = weights.float()
        reference_scores = (
            dots.relu() * config.indexer_head_dim**-0.5 * score_weights.unsqueeze(-1)
        ).sum(dim=1)

    torch.cuda.synchronize()
    if "scores" not in captured:
        raise AssertionError("CP1 did not capture Indexer raw scores")
    if "raw_topk_indices" not in captured:
        raise AssertionError("CP1 did not capture IDs-only Indexer Top-K output")
    backend_scores = captured["scores"]
    if not isinstance(backend_scores, torch.Tensor):
        raise AssertionError("CP1 captured an invalid Indexer score output")
    assert captured["sm_scale"] == config.indexer_head_dim**-0.5
    for row in range(tokens):
        visible = (row + 1) // config.ratio
        torch.testing.assert_close(
            backend_scores[row, :visible],
            reference_scores[row, :visible],
            atol=5e-3,
            rtol=5e-3,
        )
        assert torch.all(torch.isneginf(backend_scores[row, visible:]))


def test_cp1_csa_release_forward_matches_reference() -> None:
    torch.manual_seed(21)
    config = MagiDSAConfig(ratio=4)
    tokens = 1024
    layer = MagiDSALayer(config).cuda()
    x = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(tokens, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
    q = (
        torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    ).contiguous()
    latent_kv = (
        torch.randn(tokens, config.head_dim, device="cuda", dtype=torch.bfloat16) * 0.25
    ).contiguous()
    sink = torch.randn(config.num_query_heads, device="cuda", dtype=torch.float32)
    meta = MagiDSAPackedMeta((0, tokens), (tokens,))
    runtime = MagiDSARuntimeMgr(
        config,
        structural_layout_config=DsaStructuralLayoutConfig(),
    )
    handle = runtime.prepare_execution(
        meta, torch.device("cuda"), local_token_capacity=tokens
    )
    dsa_input = MagiDSAInput(x, qr, q, latent_kv, sink, meta)

    with torch.no_grad():
        expected = dsa_reference(layer, x, qr, q, latent_kv, sink, meta.cu_seqlens)
        actual = runtime.calc_dsa(layer.projections(), dsa_input, handle)
    torch.cuda.synchronize()

    output_diagnostics = assert_backend_native_topk_outputs_close(
        actual.output,
        expected.output,
        actual.topk_ids,
        actual.topk_length,
        expected.topk_ids,
        expected.topk_length,
        label="CP1 forward production vs pure-PyTorch reference",
    )
    assert (
        cast(int, output_diagnostics["output_compared_rows"])
        + cast(int, output_diagnostics["output_tie_exempt_rows"])
        == tokens
    )
    torch.testing.assert_close(
        actual.sparse_lse, expected.sparse_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.indexer_lse, expected.indexer_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(actual.kl, expected.kl, atol=2e-2, rtol=2e-2)
    assert runtime.counters.warm_invocations == 1


def test_cp1_csa_natural_backward_matches_all_input_and_parameter_gradients() -> None:
    torch.manual_seed(22)
    config = MagiDSAConfig(ratio=4)
    tokens = 256
    actual_layer = MagiDSALayer(config).cuda()
    reference_layer = copy.deepcopy(actual_layer)

    def make_input(
        shape: tuple[int, ...], *, scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
        ).contiguous()
        return value.requires_grad_(True), value.detach().clone().requires_grad_(True)

    actual_x, reference_x = make_input((tokens, config.hidden_size))
    actual_qr, reference_qr = make_input((tokens, config.q_lora_rank))
    actual_q, reference_q = make_input(
        (tokens, config.num_query_heads, config.head_dim),
        scale=0.25,
    )
    actual_kv, reference_kv = make_input((tokens, config.head_dim), scale=0.25)
    sink_value = torch.randn(config.num_query_heads, device="cuda", dtype=torch.float32)
    actual_sink = sink_value.clone().requires_grad_(True)
    reference_sink = sink_value.clone().requires_grad_(True)
    meta = MagiDSAPackedMeta((0, tokens), (tokens,))
    runtime = MagiDSARuntimeMgr(
        config,
        structural_layout_config=DsaStructuralLayoutConfig(),
    )
    handle = runtime.prepare_execution(
        meta, torch.device("cuda"), local_token_capacity=tokens
    )

    actual = runtime.calc_dsa(
        actual_layer.projections(),
        MagiDSAInput(
            actual_x,
            actual_qr,
            actual_q,
            actual_kv,
            actual_sink,
            meta,
        ),
        handle,
    )
    expected = dsa_reference(
        reference_layer,
        reference_x,
        reference_qr,
        reference_q,
        reference_kv,
        reference_sink,
        meta.cu_seqlens,
    )
    actual_loss = actual.output.float().square().mean() + actual.kl
    expected_loss = expected.output.float().square().mean() + expected.kl
    actual_loss.backward()
    expected_loss.backward()
    torch.cuda.synchronize()

    output_diagnostics = assert_backend_native_topk_outputs_close(
        actual.output,
        expected.output,
        actual.topk_ids,
        actual.topk_length,
        expected.topk_ids,
        expected.topk_length,
        label="CP1 backward production vs pure-PyTorch reference",
    )
    torch.testing.assert_close(
        actual.indexer_lse, expected.indexer_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.sparse_lse, expected.sparse_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(actual.kl, expected.kl, atol=2e-2, rtol=2e-2)
    assert (
        cast(int, output_diagnostics["output_compared_rows"])
        + cast(int, output_diagnostics["output_tie_exempt_rows"])
        == tokens
    )
    for actual_tensor, expected_tensor in (
        (actual_x, reference_x),
        (actual_qr, reference_qr),
        (actual_q, reference_q),
        (actual_kv, reference_kv),
        (actual_sink, reference_sink),
    ):
        assert actual_tensor.grad is not None and expected_tensor.grad is not None
        torch.testing.assert_close(
            actual_tensor.grad.float(),
            expected_tensor.grad.float(),
            atol=2e-2,
            rtol=2e-2,
        )
    actual_parameters = dict(actual_layer.named_parameters())
    reference_parameters = dict(reference_layer.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    for name in actual_parameters:
        actual_gradient = actual_parameters[name].grad
        expected_gradient = reference_parameters[name].grad
        assert actual_gradient is not None, name
        assert expected_gradient is not None, name
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-2, rtol=2e-2, msg=name
        )


def test_cp1_csa_overlap_backward_supports_retain_graph() -> None:
    torch.manual_seed(23)
    config = MagiDSAConfig(ratio=4)
    tokens = 128
    layer = MagiDSALayer(config).cuda()

    def make_input(shape: tuple[int, ...], *, scale: float = 1.0) -> torch.Tensor:
        return (
            (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale)
            .contiguous()
            .requires_grad_(True)
        )

    x = make_input((tokens, config.hidden_size))
    qr = make_input((tokens, config.q_lora_rank))
    q = make_input(
        (tokens, config.num_query_heads, config.head_dim),
        scale=0.25,
    )
    latent_kv = make_input((tokens, config.head_dim), scale=0.25)
    sink = torch.randn(
        config.num_query_heads,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    meta = MagiDSAPackedMeta((0, tokens), (tokens,))
    runtime = MagiDSARuntimeMgr(
        config,
        structural_layout_config=DsaStructuralLayoutConfig(),
    )
    handle = runtime.prepare_execution(
        meta,
        torch.device("cuda"),
        local_token_capacity=tokens,
    )
    result = runtime.calc_dsa(
        layer.projections(),
        MagiDSAInput(x, qr, q, latent_kv, sink, meta),
        handle,
    )
    targets = (x, qr, q, latent_kv, sink, *tuple(layer.parameters()))
    grad_outputs = (
        torch.randn_like(result.output),
        torch.ones_like(result.kl),
    )
    first = torch.autograd.grad(
        (result.output, result.kl),
        targets,
        grad_outputs,
        retain_graph=True,
    )
    second = torch.autograd.grad(
        (result.output, result.kl),
        targets,
        grad_outputs,
    )
    torch.cuda.synchronize()

    for first_gradient, second_gradient in zip(first, second):
        assert torch.isfinite(first_gradient.float()).all()
        torch.testing.assert_close(
            first_gradient.float(),
            second_gradient.float(),
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("ratio", [128])
def test_cp1_window_and_hca_ragged_forward_backward_match_reference(
    ratio: DsaRatio,
) -> None:
    torch.manual_seed(30 + ratio)
    config = MagiDSAConfig(ratio=ratio)
    cu_seqlens = (0, 129, 257)
    tokens = cu_seqlens[-1]
    actual_layer = MagiDSALayer(config).cuda()
    reference_layer = copy.deepcopy(actual_layer)

    def pair(
        shape: tuple[int, ...], scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
        ).contiguous()
        return value.requires_grad_(True), value.detach().clone().requires_grad_(True)

    actual_x, reference_x = pair((tokens, config.hidden_size))
    actual_qr, reference_qr = pair((tokens, config.q_lora_rank))
    actual_q, reference_q = pair(
        (tokens, config.num_query_heads, config.head_dim), 0.25
    )
    actual_kv, reference_kv = pair((tokens, config.head_dim), 0.25)
    sink_value = torch.randn(config.num_query_heads, device="cuda", dtype=torch.float32)
    actual_sink = sink_value.clone().requires_grad_(True)
    reference_sink = sink_value.clone().requires_grad_(True)
    meta = MagiDSAPackedMeta(cu_seqlens, (tokens,))
    runtime = MagiDSARuntimeMgr(config)
    handle = runtime.prepare_execution(
        meta, torch.device("cuda"), local_token_capacity=tokens
    )
    actual = runtime.calc_dsa(
        actual_layer.projections(),
        MagiDSAInput(actual_x, actual_qr, actual_q, actual_kv, actual_sink, meta),
        handle,
    )
    expected = dsa_reference(
        reference_layer,
        reference_x,
        reference_qr,
        reference_q,
        reference_kv,
        reference_sink,
        cu_seqlens,
    )
    actual.output.float().square().mean().backward()
    expected.output.float().square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.output.float(), expected.output.float(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.sparse_lse, expected.sparse_lse, atol=5e-3, rtol=5e-3
    )
    assert actual.topk_ids.shape == (tokens, 0)
    assert torch.equal(actual.topk_length, torch.zeros_like(actual.topk_length))
    assert actual.kl.item() == 0.0
    for actual_tensor, expected_tensor in (
        (actual_q, reference_q),
        (actual_kv, reference_kv),
        (actual_sink, reference_sink),
    ):
        assert actual_tensor.grad is not None and expected_tensor.grad is not None
        torch.testing.assert_close(
            actual_tensor.grad.float(),
            expected_tensor.grad.float(),
            atol=2e-2,
            rtol=2e-2,
        )
    if ratio == 0:
        assert actual_x.grad is None and reference_x.grad is None
        assert actual_qr.grad is None and reference_qr.grad is None
        assert dict(actual_layer.named_parameters()) == {}
    else:
        assert actual_x.grad is not None and reference_x.grad is not None
        torch.testing.assert_close(
            actual_x.grad.float(), reference_x.grad.float(), atol=2e-2, rtol=2e-2
        )
        for name, parameter in actual_layer.named_parameters():
            reference_parameter = dict(reference_layer.named_parameters())[name]
            assert parameter.grad is not None and reference_parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, atol=2e-2, rtol=2e-2, msg=name
            )
