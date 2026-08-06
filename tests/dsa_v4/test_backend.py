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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.functional import dsa_backend as dsa_backend_module
from magi_attention.functional.dsa_backend import (
    DsaIndexerSelection,
    _finalize_backend_topk,
    _require_current_cudnn_stream,
    _require_explicit_cudnn_stream,
    _unpack_flashmla_sparse_result,
    _validate_release_backend,
    dsa_selected_kl,
    dsa_sparse_attention,
)
from magi_attention.functional.dsa_reference import validate_backend_native_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_release_backend_requires_the_official_pro_shape(monkeypatch) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 3))

    _validate_release_backend(MagiDSAConfig(ratio=4), device)
    with pytest.raises(ValueError, match="DeepSeek-V4-Pro"):
        _validate_release_backend(
            replace(MagiDSAConfig(ratio=4), num_query_heads=64),
            device,
        )


def test_cudnn_caller_stream_contract_rejects_the_legacy_default_stream() -> None:
    with pytest.raises(RuntimeError, match="indexer_backward"):
        _require_explicit_cudnn_stream(0, "indexer_backward")
    _require_explicit_cudnn_stream(17, "indexer_backward")


def test_cudnn_sparse_backward_stream_must_match_the_torch_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_backend_module, "_current_cu_stream", lambda: 23)
    _require_current_cudnn_stream(23, "sparse attention backward")
    with pytest.raises(RuntimeError, match="match the current PyTorch stream"):
        _require_current_cudnn_stream(29, "sparse attention backward")


def test_selected_kl_backward_waits_for_unit_gradient_event(monkeypatch) -> None:
    calls: list[object] = []
    ready_event = object()
    saved = tuple(torch.ones(1, device="cuda") for _ in range(3))

    class _FakeStream:
        def wait_event(self, event: object) -> None:
            calls.append(("wait", event))

    backward_stream = _FakeStream()

    def current_stream(device: torch.device) -> _FakeStream:
        calls.append(("stream", device))
        return backward_stream

    def record_stream(tensor: torch.Tensor, stream: object) -> None:
        calls.append(("record", tensor, stream))

    def scale_gradients(*args):
        calls.append("scale")
        return args[:3]

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    monkeypatch.setattr(
        "magi_attention.kernel.triton.dsa_gradients."
        "fused_dsa_scale_indexer_gradients",
        scale_gradients,
    )
    ctx = SimpleNamespace(
        empty=False,
        saved_tensors=saved,
        unit_gradient_ready_event=ready_event,
    )
    grad_loss = torch.ones((), device="cuda")
    result = dsa_backend_module._DsaSelectedKlFunction.backward(
        ctx,
        grad_loss,
    )
    assert calls == [
        ("stream", torch.device("cuda", 0)),
        ("wait", ready_event),
        ("record", saved[0], backward_stream),
        ("record", saved[1], backward_stream),
        ("record", saved[2], backward_stream),
        ("record", grad_loss, backward_stream),
        "scale",
    ]
    assert len(result) == 12
    assert all(actual is expected for actual, expected in zip(result[:3], saved))


def test_flashmla_result_abi_is_ratio_specific() -> None:
    output = torch.empty(1)
    maximum = torch.empty(1)
    sparse_lse = torch.empty(1)
    compressed_lse = torch.empty(1)
    unpacked = _unpack_flashmla_sparse_result(
        (output, maximum, sparse_lse),
        require_indexer_lse=False,
    )
    assert unpacked[0] is output
    assert unpacked[1] is sparse_lse
    assert unpacked[2] is None
    unpacked = _unpack_flashmla_sparse_result(
        (output, maximum, sparse_lse, compressed_lse),
        require_indexer_lse=True,
    )
    assert unpacked[2] is compressed_lse
    with pytest.raises(RuntimeError, match="incompatible FlashMLA"):
        _unpack_flashmla_sparse_result(
            (output, maximum, sparse_lse),
            require_indexer_lse=True,
        )
    with pytest.raises(RuntimeError, match="incompatible FlashMLA"):
        _unpack_flashmla_sparse_result(
            (output, maximum, sparse_lse, compressed_lse),
            require_indexer_lse=False,
        )


def test_flashmla_forward_cudnn_backward_matches_csa_width_reference() -> None:
    torch.manual_seed(11)
    config = MagiDSAConfig(ratio=4)
    tokens_q, tokens_k, width = 8, 1024, config.window_size + config.indexer_topk
    q = torch.randn(
        tokens_q,
        config.num_query_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv = torch.randn(
        tokens_k,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    sink = torch.randn(
        config.num_query_heads, device="cuda", dtype=torch.float32, requires_grad=True
    )
    compressed_lengths = torch.arange(
        256, 256 + tokens_q, device="cuda", dtype=torch.int32
    )
    window_lengths = torch.arange(
        config.window_size - tokens_q + 1,
        config.window_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    lengths = compressed_lengths + window_lengths
    column = torch.arange(width, device="cuda", dtype=torch.int32).unsqueeze(0)
    compressed_valid = column < compressed_lengths.unsqueeze(1)
    window_column = column - config.indexer_topk
    window_valid = (column >= config.indexer_topk) & (
        window_column < window_lengths.unsqueeze(1)
    )
    valid = compressed_valid | window_valid
    row = torch.arange(tokens_q, device="cuda", dtype=torch.int32).unsqueeze(1)
    indices = (
        (row * 17 + column).remainder(tokens_k).masked_fill(~valid, -1).contiguous()
    )
    dout = torch.randn_like(q)

    output, lse, compressed_lse = dsa_sparse_attention(
        q, kv, sink, indices, lengths, config
    )
    loss = (output.float() * dout.float()).sum()
    actual_gradients = torch.autograd.grad(loss, (q, kv, sink))

    reference_q = q.detach().clone().requires_grad_(True)
    reference_kv = kv.detach().clone().requires_grad_(True)
    reference_sink = sink.detach().clone().requires_grad_(True)
    selected = reference_kv.index_select(0, indices.clamp_min(0).flatten().long()).view(
        tokens_q, width, config.head_dim
    )
    score = torch.einsum("qhd,qkd->qhk", reference_q.float(), selected.float()) * (
        config.head_dim**-0.5
    )
    score = score.masked_fill(~valid.unsqueeze(1), float("-inf"))
    reference_lse = torch.logsumexp(score, dim=-1)
    reference_compressed_lse = torch.logsumexp(
        score[:, :, : config.indexer_topk],
        dim=-1,
    )
    denominator = torch.logaddexp(reference_lse, reference_sink.unsqueeze(0))
    probability = torch.exp(score - denominator.unsqueeze(-1))
    reference_output = torch.einsum("qhk,qkd->qhd", probability, selected.float()).to(
        torch.bfloat16
    )
    reference_loss = (reference_output.float() * dout.float()).sum()
    reference_gradients = torch.autograd.grad(
        reference_loss, (reference_q, reference_kv, reference_sink)
    )

    torch.testing.assert_close(
        output.float(), reference_output.float(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(lse, reference_lse, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        compressed_lse,
        reference_compressed_lse,
        atol=1e-4,
        rtol=1e-4,
    )
    for actual, expected in zip(actual_gradients[:2], reference_gradients[:2]):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    torch.testing.assert_close(
        actual_gradients[2], reference_gradients[2], atol=2e-2, rtol=2e-2
    )


def test_indexer_topk_preserves_backend_native_exact_ties() -> None:
    rows, columns, topk = 5, 8, 3
    scores = torch.full(
        (rows, columns),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    scores[0, :6] = torch.tensor(
        [9.0, 8.0, 7.0, 7.0, 7.0, 2.0],
        device="cuda",
    )
    scores[1, :5] = torch.tensor([9.0, 9.0, 8.0, 7.0, 6.0], device="cuda")
    scores[2, :6] = torch.tensor([9.0, 8.0, 7.001, 7.0, 7.0, 2.0], device="cuda")
    scores[4, :2] = torch.tensor([5.0, 5.0], device="cuda")
    lengths = torch.tensor([6, 5, 6, 0, 2], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([100, 200, 300, 400, 500], device="cuda", dtype=torch.int32)
    first_backend_ids = torch.tensor(
        [[0, 1, 4], [1, 0, 2], [0, 1, 2], [-1, -1, -1], [1, 0, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    second_backend_ids = torch.tensor(
        [[0, 1, 3], [0, 1, 2], [0, 1, 2], [-1, -1, -1], [0, 1, -1]],
        device="cuda",
        dtype=torch.int32,
    )

    first_ids, first_lengths = _finalize_backend_topk(
        lengths,
        first_backend_ids,
        offsets,
    )
    second_ids, second_lengths = _finalize_backend_topk(
        lengths,
        second_backend_ids,
        offsets,
    )
    first_expected = torch.tensor(
        [
            [100, 101, 104],
            [201, 200, 202],
            [300, 301, 302],
            [-1, -1, -1],
            [501, 500, -1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    second_expected = torch.tensor(
        [
            [100, 101, 103],
            [200, 201, 202],
            [300, 301, 302],
            [-1, -1, -1],
            [500, 501, -1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    assert torch.equal(first_ids, first_expected)
    assert torch.equal(second_ids, second_expected)
    assert not torch.equal(first_ids, second_ids)
    assert torch.equal(first_lengths, lengths.clamp(max=topk))
    assert torch.equal(second_lengths, first_lengths)
    for row in (0, 1, 2, 4):
        visible = int(lengths[row].item())
        selected = int(first_lengths[row].item())
        validate_backend_native_topk(
            scores[row, :visible].cpu(),
            first_ids[row, :selected].cpu(),
            selected,
            global_offset=int(offsets[row].item()),
        )


def test_selected_kl_matches_reference_and_does_not_differentiate_teacher() -> None:
    torch.manual_seed(12)
    config = MagiDSAConfig(ratio=4)
    tokens_q, tokens_k, topk = 16, 1024, config.indexer_topk
    q_indexer = torch.randn(
        tokens_q,
        config.indexer_heads,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weights = (
        torch.randn(
            tokens_q,
            config.indexer_heads,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * config.indexer_heads**-0.5
    ).requires_grad_(True)
    k_indexer = torch.randn(
        tokens_k,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    q_attention = torch.randn(
        tokens_q,
        config.num_query_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_attention = torch.randn(
        tokens_k,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(tokens_q, 1)
    lengths = torch.full((tokens_q,), topk, device="cuda", dtype=torch.int32)
    selected_attention = k_attention.index_select(0, indices.flatten().long()).view(
        tokens_q, topk, config.head_dim
    )
    selected_attention_logits = torch.einsum(
        "qhd,qkd->qhk", q_attention.float(), selected_attention.float()
    ) * (config.head_dim**-0.5)
    window_indices = torch.arange(
        tokens_k - config.window_size,
        tokens_k,
        device="cuda",
        dtype=torch.int64,
    )
    window_attention = k_attention.index_select(0, window_indices)
    window_attention_logits = torch.einsum(
        "qhd,kd->qhk", q_attention.float(), window_attention.float()
    ) * (config.head_dim**-0.5)
    compressed_lse = torch.logsumexp(selected_attention_logits, dim=-1).detach()
    full_sparse_lse = torch.logsumexp(
        torch.cat((selected_attention_logits, window_attention_logits), dim=-1),
        dim=-1,
    ).detach()
    assert bool((full_sparse_lse - compressed_lse).abs().gt(1e-4).any().item())

    raw_indexer_scores = (
        torch.einsum("qhd,kd->qhk", q_indexer.float(), k_indexer.float()).relu()
        * config.indexer_head_dim**-0.5
        * weights.float().unsqueeze(-1)
    ).sum(dim=1)
    indexer_lse = torch.logsumexp(raw_indexer_scores, dim=-1).detach()
    selection = DsaIndexerSelection(
        global_ids=indices,
        lengths=lengths,
        lse=indexer_lse,
        logical_score_calls=1,
        logical_topk_calls=1,
    )
    loss_coeff = 0.3

    from cudnn import DSA

    captured: dict[str, object] = {}
    original_recompute = DSA.sparse_indexer_score_recompute_wrapper
    original_attention_recompute = DSA.sparse_attn_score_recompute_wrapper
    original_backward = DSA.indexer_backward_wrapper

    def assert_explicit_current_stream(kwargs) -> None:
        stream = kwargs["stream"]
        assert int(stream) != 0
        assert int(stream) == torch.cuda.current_stream().cuda_stream

    def capture_recompute(*args, **kwargs):
        assert_explicit_current_stream(kwargs)
        captured["recompute_weights"] = args[2].detach().clone()
        return original_recompute(*args, **kwargs)

    def capture_attention_recompute(*args, **kwargs):
        assert_explicit_current_stream(kwargs)
        return original_attention_recompute(*args, **kwargs)

    def capture_backward(*args, **kwargs):
        assert_explicit_current_stream(kwargs)
        captured["backward_weights"] = args[1].detach().clone()
        captured["backward_sm_scale"] = kwargs["sm_scale"]
        return original_backward(*args, **kwargs)

    DSA.sparse_indexer_score_recompute_wrapper = capture_recompute
    DSA.sparse_attn_score_recompute_wrapper = capture_attention_recompute
    DSA.indexer_backward_wrapper = capture_backward
    try:
        caller_stream = torch.cuda.current_stream()
        indexer_stream = torch.cuda.Stream()
        indexer_stream.wait_stream(caller_stream)
        with torch.cuda.stream(indexer_stream):
            loss = dsa_selected_kl(
                q_indexer,
                weights,
                k_indexer,
                q_attention,
                k_attention,
                compressed_lse,
                selection,
                indices,
                indices,
                loss_coeff=loss_coeff,
                config=config,
            )
            actual_gradients = torch.autograd.grad(
                loss,
                (q_indexer, weights, k_indexer, q_attention, k_attention),
                allow_unused=True,
            )
        caller_stream.wait_stream(indexer_stream)
    finally:
        DSA.sparse_indexer_score_recompute_wrapper = original_recompute
        DSA.sparse_attn_score_recompute_wrapper = original_attention_recompute
        DSA.indexer_backward_wrapper = original_backward

    recompute_weights = captured["recompute_weights"]
    backward_weights = captured["backward_weights"]
    assert isinstance(recompute_weights, torch.Tensor)
    assert isinstance(backward_weights, torch.Tensor)
    torch.testing.assert_close(
        recompute_weights,
        (weights.detach().unsqueeze(0).float() * config.indexer_head_dim**-0.5).to(
            weights.dtype
        ),
    )
    torch.testing.assert_close(backward_weights, weights.detach().unsqueeze(0))
    assert captured["backward_sm_scale"] == config.indexer_head_dim**-0.5

    reference_q = q_indexer.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_k = k_indexer.detach().clone().requires_grad_(True)
    reference_scores = (
        torch.einsum("qhd,kd->qhk", reference_q.float(), reference_k.float()).relu()
        * config.indexer_head_dim**-0.5
        * reference_weights.float().unsqueeze(-1)
    ).sum(dim=1)
    selected_indexer_scores = reference_scores.gather(1, indices.long())
    attention_mass = torch.exp(
        selected_attention_logits.detach() - compressed_lse.unsqueeze(-1)
    ).sum(dim=1)
    target = (attention_mass / attention_mass.sum(dim=-1, keepdim=True)).detach()
    log_target = (
        target.clamp_min(torch.exp(torch.tensor(-100.0, device="cuda")))
        .log()
        .clamp(-100.0, 0.0)
    )
    selected_indexer_lse = torch.logsumexp(selected_indexer_scores, dim=-1)
    log_predict = (selected_indexer_scores - selected_indexer_lse.unsqueeze(1)).clamp(
        -100.0, 0.0
    )
    reference_loss = (target * (log_target - log_predict)).sum(
        dim=-1
    ).mean() * loss_coeff
    reference_gradients = torch.autograd.grad(
        reference_loss, (reference_q, reference_weights, reference_k)
    )

    torch.testing.assert_close(loss, reference_loss, atol=2e-5, rtol=2e-5)
    for actual, expected in zip(actual_gradients[:3], reference_gradients):
        assert actual is not None
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    assert actual_gradients[2] is not None
    assert torch.count_nonzero(actual_gradients[2][topk:]).item() == 0
    assert actual_gradients[3] is None
    assert actual_gradients[4] is None


def test_selected_kl_zeroes_empty_rows_in_an_odd_thd_fragment() -> None:
    torch.manual_seed(13)
    config = MagiDSAConfig(ratio=4)
    tokens_q, tokens_k = 13, 3
    q_indexer = torch.randn(
        tokens_q,
        config.indexer_heads,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weights = torch.randn(
        tokens_q,
        config.indexer_heads,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k_indexer = torch.randn(
        tokens_k,
        config.indexer_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    q_attention = torch.randn(
        tokens_q,
        config.num_query_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_attention = torch.randn(
        tokens_k,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    lengths = (
        torch.arange(1, tokens_q + 1, device="cuda", dtype=torch.int32) // config.ratio
    )
    columns = torch.arange(
        config.indexer_topk, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    indices = columns.expand(tokens_q, -1).masked_fill(
        columns >= lengths.unsqueeze(1), -1
    )
    raw_scores = (
        torch.einsum("qhd,kd->qhk", q_indexer.float(), k_indexer.float()).relu()
        * config.indexer_head_dim**-0.5
        * weights.float().unsqueeze(-1)
    ).sum(dim=1)
    key_columns = torch.arange(tokens_k, device="cuda").unsqueeze(0)
    raw_scores = raw_scores.masked_fill(
        key_columns >= lengths.unsqueeze(1), float("-inf")
    )
    indexer_lse = torch.logsumexp(raw_scores, dim=-1).detach()
    attention_logits = torch.einsum(
        "qhd,kd->qhk", q_attention.float(), k_attention.float()
    ) * (config.head_dim**-0.5)
    attention_logits = attention_logits.masked_fill(
        key_columns[:, None, :] >= lengths[:, None, None], float("-inf")
    )
    selected_lse = torch.logsumexp(attention_logits, dim=-1)
    compressed_lse = selected_lse.detach()
    selection = DsaIndexerSelection(
        global_ids=indices,
        lengths=lengths,
        lse=indexer_lse,
        logical_score_calls=1,
        logical_topk_calls=1,
    )

    caller_stream = torch.cuda.current_stream()
    indexer_stream = torch.cuda.Stream()
    indexer_stream.wait_stream(caller_stream)
    with torch.cuda.stream(indexer_stream):
        loss = dsa_selected_kl(
            q_indexer,
            weights,
            k_indexer,
            q_attention,
            k_attention,
            compressed_lse,
            selection,
            indices,
            indices,
            loss_coeff=0.3,
            config=config,
        )
        grad_q, grad_weights, grad_k = torch.autograd.grad(
            loss, (q_indexer, weights, k_indexer)
        )
    caller_stream.wait_stream(indexer_stream)
    empty_rows = lengths == 0
    assert bool(empty_rows.any().item())
    assert torch.count_nonzero(grad_q[empty_rows]).item() == 0
    assert torch.count_nonzero(grad_weights[empty_rows]).item() == 0
    assert bool(torch.isfinite(grad_k.float()).all().item())
