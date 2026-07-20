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

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.functional.dsa_backend import (
    _resolve_deterministic_topk,
    dsa_selected_kl,
    dsa_sparse_attention,
)
from magi_attention.functional.dsa_reference import validate_deterministic_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
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
    row = torch.arange(tokens_q, device="cuda", dtype=torch.int32).unsqueeze(1)
    column = torch.arange(width, device="cuda", dtype=torch.int32).unsqueeze(0)
    indices = (row * 17 + column).remainder(tokens_k).contiguous()
    lengths = torch.arange(
        width - tokens_q + 1, width + 1, device="cuda", dtype=torch.int32
    )
    dout = torch.randn_like(q)

    output, lse = dsa_sparse_attention(q, kv, sink, indices, lengths, config)
    loss = (output.float() * dout.float()).sum()
    actual_gradients = torch.autograd.grad(loss, (q, kv, sink))

    reference_q = q.detach().clone().requires_grad_(True)
    reference_kv = kv.detach().clone().requires_grad_(True)
    reference_sink = sink.detach().clone().requires_grad_(True)
    selected = reference_kv.index_select(0, indices.flatten().long()).view(
        tokens_q, width, config.head_dim
    )
    score = torch.einsum("qhd,qkd->qhk", reference_q.float(), selected.float()) * (
        config.head_dim**-0.5
    )
    valid = torch.arange(width, device="cuda").unsqueeze(0) < lengths.unsqueeze(1)
    score = score.masked_fill(~valid.unsqueeze(1), float("-inf"))
    reference_lse = torch.logsumexp(score, dim=-1)
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
    for actual, expected in zip(actual_gradients[:2], reference_gradients[:2]):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    torch.testing.assert_close(
        actual_gradients[2], reference_gradients[2], atol=2e-2, rtol=2e-2
    )


def test_indexer_topk_cutoff_tie_uses_ascending_global_id() -> None:
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

    first_ids, first_lengths = _resolve_deterministic_topk(
        scores,
        lengths,
        first_backend_ids,
        offsets,
    )
    second_ids, second_lengths = _resolve_deterministic_topk(
        scores,
        lengths,
        second_backend_ids,
        offsets,
    )
    expected = torch.tensor(
        [
            [100, 101, 102],
            [200, 201, 202],
            [300, 301, 302],
            [-1, -1, -1],
            [500, 501, -1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    assert torch.equal(first_ids, expected)
    assert torch.equal(second_ids, expected)
    assert torch.equal(first_lengths, lengths.clamp(max=topk))
    assert torch.equal(second_lengths, first_lengths)
    for row in (0, 1, 2):
        validate_deterministic_topk(
            scores[row, : lengths[row]].cpu(),
            first_ids[row].cpu(),
            topk,
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
        * (config.indexer_head_dim**-0.5 * config.indexer_heads**-0.5)
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
    row = torch.arange(tokens_q, device="cuda", dtype=torch.int32).unsqueeze(1)
    column = torch.arange(topk, device="cuda", dtype=torch.int32).unsqueeze(0)
    indices = (row * 13 + column).remainder(tokens_k).contiguous()
    lengths = torch.full((tokens_q,), topk, device="cuda", dtype=torch.int32)
    selected_attention = k_attention.index_select(0, indices.flatten().long()).view(
        tokens_q, topk, config.head_dim
    )
    attention_logits = torch.einsum(
        "qhd,qkd->qhk", q_attention.float(), selected_attention.float()
    ) * (config.head_dim**-0.5)
    sparse_lse = torch.logsumexp(attention_logits, dim=-1).detach()
    loss_coeff = 0.3

    loss = dsa_selected_kl(
        q_indexer,
        weights,
        k_indexer,
        q_attention,
        k_attention,
        sparse_lse,
        indices,
        indices,
        lengths,
        loss_coeff=loss_coeff,
        config=config,
    )
    actual_gradients = torch.autograd.grad(
        loss,
        (q_indexer, weights, k_indexer, q_attention, k_attention),
        allow_unused=True,
    )

    reference_q = q_indexer.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_k = k_indexer.detach().clone().requires_grad_(True)
    selected_indexer = reference_k.index_select(0, indices.flatten().long()).view(
        tokens_q, topk, config.indexer_head_dim
    )
    indexer_logits = torch.einsum(
        "qhd,qkd->qhk", reference_q.float(), selected_indexer.float()
    )
    indexer_logits = (
        indexer_logits.relu() * reference_weights.float().unsqueeze(-1)
    ).sum(dim=1)
    predict = indexer_logits.softmax(dim=-1)
    attention_mass = torch.exp(
        attention_logits.detach() - sparse_lse.unsqueeze(-1)
    ).sum(dim=1)
    target = (attention_mass / attention_mass.sum(dim=-1, keepdim=True)).detach()
    log_target = (
        target.clamp_min(torch.exp(torch.tensor(-100.0, device="cuda")))
        .log()
        .clamp(-100.0, 0.0)
    )
    log_predict = (
        predict.clamp_min(torch.exp(torch.tensor(-100.0, device="cuda")))
        .log()
        .clamp(-100.0, 0.0)
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
    assert actual_gradients[3] is None
    assert actual_gradients[4] is None
