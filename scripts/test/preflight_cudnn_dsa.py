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

import json
import math
from importlib import metadata

import cudnn
import torch
from cuda.bindings import driver as cuda
from cudnn import DSA
from cutlass import Int32

_PRO_INDEXER_HEADS = 64
_PRO_INDEXER_DIM = 128
_PRO_ATTENTION_HEADS = 128
_PRO_ATTENTION_DIM = 512
_PRO_INDEXER_TOPK = 1024
_PRO_WINDOW_TOPK = 128
_PRO_TOTAL_TOPK = _PRO_INDEXER_TOPK + _PRO_WINDOW_TOPK


def _sync(label: str) -> None:
    torch.cuda.synchronize()
    print(label, flush=True)


def _indexer_forward(stream: cuda.CUstream) -> dict[str, object]:
    torch.manual_seed(0)
    batch, seqlen_q, seqlen_k = 1, 256, 512
    heads, dim = _PRO_INDEXER_HEADS, _PRO_INDEXER_DIM
    q = torch.randn(batch, seqlen_q, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, 1, dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(batch, seqlen_q, heads, device="cuda", dtype=torch.bfloat16)
    q_causal_offsets = torch.tensor([512], device="cuda", dtype=torch.int32)
    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=64,
        q_causal_offsets=q_causal_offsets,
        stream=stream,
    )["scores"]
    _sync("indexer_forward: executed")

    reference = torch.einsum(
        "bqhd,bkhd->bqhk",
        q.float(),
        k.float().expand(-1, -1, heads, -1),
    )
    reference = (reference.relu() * w.float().unsqueeze(-1)).sum(dim=2)
    row = torch.arange(seqlen_q, device="cuda", dtype=torch.int64)
    valid_length = ((q_causal_offsets.long()[0] + row + 1) // 4).clamp(0, seqlen_k)
    column = torch.arange(seqlen_k, device="cuda", dtype=torch.int64)
    valid = column.unsqueeze(0) < valid_length.unsqueeze(1)
    reference = reference.masked_fill(~valid.unsqueeze(0), float("-inf"))
    assert torch.equal(torch.isneginf(scores), torch.isneginf(reference))
    finite = torch.isfinite(reference)
    torch.testing.assert_close(scores[finite], reference[finite], atol=1e-4, rtol=1e-4)
    _sync("indexer_forward: reference matched")
    return {
        "max_valid_length": int(valid_length.max().item()),
        "min_valid_length": int(valid_length.min().item()),
        "shape": list(scores.shape),
    }


def _indexer_grouped_thd(stream: cuda.CUstream) -> dict[str, object]:
    torch.manual_seed(3)
    q_lengths = (128, 64)
    k_lengths = (257, 129)
    offsets = (512, 128)
    logical_max_k = max(k_lengths)
    backend_max_k = (logical_max_k + 7) // 8 * 8
    heads, dim = _PRO_INDEXER_HEADS, _PRO_INDEXER_DIM
    q = torch.randn(sum(q_lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), 1, dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(sum(q_lengths), heads, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor(
        (0, q_lengths[0], sum(q_lengths)), device="cuda", dtype=torch.int32
    )
    cu_k = torch.tensor(
        (0, k_lengths[0], sum(k_lengths)), device="cuda", dtype=torch.int32
    )
    q_causal_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    scores = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=64,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=backend_max_k,
        q_causal_offsets=q_causal_offsets,
        stream=stream,
    )["scores"]
    _sync("indexer_grouped_thd: executed")
    assert scores.shape == (sum(q_lengths), backend_max_k)
    assert scores.is_contiguous()
    for segment, (q_length, k_length, offset) in enumerate(
        zip(q_lengths, k_lengths, offsets)
    ):
        q_begin = int(cu_q[segment].item())
        k_begin = int(cu_k[segment].item())
        reference = torch.einsum(
            "qhd,khd->qhk",
            q[q_begin : q_begin + q_length].float(),
            k[k_begin : k_begin + k_length].float().expand(-1, heads, -1),
        )
        reference = (
            reference.relu() * w[q_begin : q_begin + q_length].float().unsqueeze(-1)
        ).sum(dim=1)
        row = torch.arange(q_length, device="cuda", dtype=torch.int64)
        valid_length = ((offset + row + 1) // 4).clamp(0, k_length)
        column = torch.arange(backend_max_k, device="cuda", dtype=torch.int64)
        valid = column.unsqueeze(0) < valid_length.unsqueeze(1)
        padded = torch.full((q_length, backend_max_k), float("-inf"), device="cuda")
        padded[:, :k_length] = reference
        padded.masked_fill_(~valid, float("-inf"))
        actual = scores[q_begin : q_begin + q_length]
        assert torch.equal(torch.isneginf(actual), torch.isneginf(padded))
        finite = torch.isfinite(padded)
        torch.testing.assert_close(actual[finite], padded[finite], atol=1e-4, rtol=1e-4)
    _sync("indexer_grouped_thd: reference matched")
    return {
        "backend_max_k": backend_max_k,
        "k_lengths": list(k_lengths),
        "logical_max_k": logical_max_k,
        "q_lengths": list(q_lengths),
        "shape": list(scores.shape),
    }


def _indexer_topk(stream: cuda.CUstream) -> dict[str, object]:
    rows, columns, top_k = 32, 2048, _PRO_INDEXER_TOPK
    base = torch.arange(columns, device="cuda", dtype=torch.float32)
    row_offset = (
        torch.arange(rows, device="cuda", dtype=torch.float32).unsqueeze(1) / 4096.0
    )
    values = base.unsqueeze(0) + row_offset
    seq_lens = top_k + torch.arange(rows, device="cuda", dtype=torch.int32) * 32
    result = DSA.indexer_top_k_wrapper(
        values,
        seq_lens,
        top_k,
        next_n=1,
        return_val=False,
        stream=stream,
    )
    indices, selected_values = result["indices"], result["values"]
    _sync("indexer_topk: executed")
    assert selected_values is None
    expected = (
        seq_lens.unsqueeze(1)
        - top_k
        + torch.arange(top_k, device="cuda", dtype=torch.int32).unsqueeze(0)
    )
    assert torch.equal(
        torch.sort(indices, dim=-1).values,
        expected,
    )
    assert bool(torch.all(indices >= 0).item())
    assert bool(torch.all(indices < seq_lens.unsqueeze(1)).item())
    _sync("indexer_topk: exact IDs-only output matched")
    return {
        "max_sequence_length": int(seq_lens.max().item()),
        "min_sequence_length": int(seq_lens.min().item()),
        "return_val": False,
        "shape": list(indices.shape),
        "top_k": top_k,
    }


def _sparse_backward(stream: cuda.CUstream) -> dict[str, object]:
    torch.manual_seed(1)
    tokens_q, tokens_k = 16, 2048
    heads, dim = _PRO_ATTENTION_HEADS, _PRO_ATTENTION_DIM
    indexer_topk, window_topk, top_k = (
        _PRO_INDEXER_TOPK,
        _PRO_WINDOW_TOPK,
        _PRO_TOTAL_TOPK,
    )
    q = torch.randn(tokens_q, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(tokens_k, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    row = torch.arange(tokens_q, device="cuda", dtype=torch.int32).unsqueeze(1)
    column = torch.arange(top_k, device="cuda", dtype=torch.int32).unsqueeze(0)
    compressed_lengths = torch.arange(
        indexer_topk - tokens_q + 1,
        indexer_topk + 1,
        device="cuda",
        dtype=torch.int32,
    )
    window_lengths = torch.arange(
        window_topk - tokens_q + 1,
        window_topk + 1,
        device="cuda",
        dtype=torch.int32,
    )
    compressed_valid = column < compressed_lengths.unsqueeze(1)
    window_column = column - indexer_topk
    window_valid = (column >= indexer_topk) & (
        window_column < window_lengths.unsqueeze(1)
    )
    valid = compressed_valid | window_valid
    topk_indices = (
        (row * 17 + column).remainder(tokens_k).masked_fill(~valid, -1).contiguous()
    )
    effective_lengths = compressed_lengths + window_lengths
    assert topk_indices.shape == (tokens_q, top_k)
    assert bool(torch.all(topk_indices[valid] >= 0).item())
    assert bool(torch.all(topk_indices[~valid] == -1).item())
    scale = 1.0 / math.sqrt(dim)
    selected_kv = kv[topk_indices.clamp_min(0).long()].float()
    score = torch.einsum("thd,tkd->thk", q.float(), selected_kv) * scale
    score = score.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(score, dim=-1)
    denominator = torch.logaddexp(lse, sink.view(1, heads))
    probability = torch.exp(score - denominator.unsqueeze(-1))
    out = torch.einsum("thk,tkd->thd", probability, selected_kv).to(torch.bfloat16)
    dout = torch.randn_like(out)
    result = DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        dout,
        lse,
        sink,
        topk_indices,
        softmax_scale=scale,
        topk_length=None,
        stream=stream,
    )
    dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
    _sync("sparse_attention_backward: executed")
    assert dq.shape == q.shape and torch.isfinite(dq).all() and dq.abs().sum() > 0
    assert dkv.shape == kv.shape and torch.isfinite(dkv).all() and dkv.abs().sum() > 0
    assert d_sink.shape == sink.shape and torch.isfinite(d_sink).all()
    _sync("sparse_attention_backward: finite nonzero gradients")
    return {
        "dkv_shape": list(dkv.shape),
        "dq_shape": list(dq.shape),
        "d_sink_shape": list(d_sink.shape),
        "effective_length_max": int(effective_lengths.max().item()),
        "effective_length_min": int(effective_lengths.min().item()),
        "indexer_topk": indexer_topk,
        "sentinel_count": int((topk_indices < 0).sum().item()),
        "topk_length_mode": "sentinel",
        "total_topk": top_k,
        "window_topk": window_topk,
    }


def _selected_kl(stream: cuda.CUstream) -> dict[str, object]:
    torch.manual_seed(4)
    batch, seqlen_q, seqlen_k = 1, 16, 2048
    indexer_heads, index_dim = _PRO_INDEXER_HEADS, _PRO_INDEXER_DIM
    attention_heads, attn_dim = _PRO_ATTENTION_HEADS, _PRO_ATTENTION_DIM
    top_k = _PRO_INDEXER_TOPK
    index_q = torch.randn(
        batch,
        seqlen_q,
        indexer_heads,
        index_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    index_k = torch.randn(
        batch, seqlen_k, index_dim, device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        batch,
        seqlen_q,
        indexer_heads,
        device="cuda",
        dtype=torch.bfloat16,
    ) * ((indexer_heads * index_dim) ** -0.5)
    attn_q = torch.randn(
        batch,
        seqlen_q,
        attention_heads,
        attn_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attn_k = torch.randn(batch, seqlen_k, attn_dim, device="cuda", dtype=torch.bfloat16)
    base = torch.arange(top_k, device="cuda", dtype=torch.int32).view(1, 1, -1)
    row = torch.arange(seqlen_q, device="cuda", dtype=torch.int32).view(1, -1, 1)
    topk_length = torch.arange(
        top_k - seqlen_q + 1,
        top_k + 1,
        device="cuda",
        dtype=torch.int32,
    ).unsqueeze(0)
    valid = base < topk_length.unsqueeze(-1)
    topk_indices = (
        (base + row * 7).remainder(seqlen_k).masked_fill(~valid, -1).contiguous()
    )
    safe_indices = topk_indices.clamp_min(0)
    selected_attention = (
        attn_k[:, None]
        .expand(-1, seqlen_q, -1, -1)
        .gather(
            2,
            safe_indices.long().unsqueeze(-1).expand(-1, -1, -1, attn_dim),
        )
    )
    selected_indexer = (
        index_k[:, None]
        .expand(-1, seqlen_q, -1, -1)
        .gather(
            2,
            safe_indices.long().unsqueeze(-1).expand(-1, -1, -1, index_dim),
        )
    )
    attention_logits = torch.einsum(
        "bqhd,bqkd->bqhk", attn_q.float(), selected_attention.float()
    ) * (attn_dim**-0.5)
    attention_logits = attention_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(attention_logits, dim=-1)
    attention_mass = torch.exp(attention_logits - lse.unsqueeze(-1)).sum(dim=2)
    reference_target = attention_mass / attention_mass.sum(dim=-1, keepdim=True)
    indexer_logits = (
        torch.einsum(
            "bqhd,bqkd->bqhk", index_q.float(), selected_indexer.float()
        ).relu()
        * weights.float().unsqueeze(-1)
    ).sum(dim=2)
    indexer_logits = indexer_logits.masked_fill(~valid, float("-inf"))
    reference_predict = torch.softmax(indexer_logits, dim=-1)
    predict = DSA.sparse_indexer_score_recompute_wrapper(
        index_q,
        index_k,
        weights,
        topk_indices,
        qhead_per_kv_head=indexer_heads,
        topk_length=topk_length,
        stream=stream,
    )["predict"]
    target = DSA.sparse_attn_score_recompute_wrapper(
        attn_q,
        attn_k,
        lse,
        topk_indices,
        softmax_scale=attn_dim**-0.5,
        qhead_per_kv_head=attention_heads,
        topk_length=topk_length,
        stream=stream,
    )["target"]
    backward = DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        target.clone(),
        predict.clone(),
        topk_indices,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=1.0,
        block_I=128,
        topk_indices_global=False,
        stream=stream,
    )
    _sync("selected_kl: executed")
    for name in ("d_index_q", "d_weights", "d_index_k"):
        gradient = backward[name]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.isfinite(predict).all() and torch.isfinite(target).all()
    torch.testing.assert_close(predict.float(), reference_predict, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(target.float(), reference_target, atol=5e-3, rtol=5e-3)
    assert torch.count_nonzero(predict.masked_select(~valid)).item() == 0
    assert torch.count_nonzero(target.masked_select(~valid)).item() == 0
    torch.testing.assert_close(
        predict.sum(dim=-1), torch.ones_like(predict[..., 0]), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        target.sum(dim=-1), torch.ones_like(target[..., 0]), atol=1e-5, rtol=1e-5
    )
    selected_rows = torch.zeros(seqlen_k, device="cuda", dtype=torch.bool)
    selected_rows[topk_indices[valid].long()] = True
    assert torch.count_nonzero(backward["d_index_k"][0, ~selected_rows]).item() == 0
    _sync("selected_kl: distributions and gradients validated")
    return {
        "attention_heads": attention_heads,
        "effective_length_max": int(topk_length.max().item()),
        "effective_length_min": int(topk_length.min().item()),
        "indexer_heads": indexer_heads,
        "predict_shape": list(predict.shape),
        "sentinel_count": int((topk_indices < 0).sum().item()),
        "target_shape": list(target.shape),
        "top_k": top_k,
        "unselected_key_rows": int((~selected_rows).sum().item()),
    }


def main() -> None:
    assert Int32
    assert torch.cuda.get_device_capability() == (10, 3)
    assert cudnn.__version__ == "1.26.0"
    assert cudnn.backend_version() == 92400
    assert metadata.version("nvidia-cudnn-cu13") == "9.24.0.43"
    assert metadata.version("nvidia-cutlass-dsl") == "4.5.0"
    torch_stream = torch.cuda.Stream()
    if torch_stream.cuda_stream == 0:
        raise RuntimeError("cuDNN preflight requires an explicit nonzero CUDA stream")
    with torch.cuda.stream(torch_stream):
        stream = cuda.CUstream(torch_stream.cuda_stream)
        report = {
            "cudnn_backend": cudnn.backend_version(),
            "cudnn_frontend": cudnn.__version__,
            "cuda_stream_nonzero": int(stream) != 0,
            "device": torch.cuda.get_device_name(),
            "indexer_forward": _indexer_forward(stream),
            "indexer_grouped_thd": _indexer_grouped_thd(stream),
            "indexer_topk": _indexer_topk(stream),
            "selected_kl": _selected_kl(stream),
            "sparse_attention_backward": _sparse_backward(stream),
            "pro_abi": {
                "attention_dim": _PRO_ATTENTION_DIM,
                "attention_heads": _PRO_ATTENTION_HEADS,
                "indexer_dim": _PRO_INDEXER_DIM,
                "indexer_heads": _PRO_INDEXER_HEADS,
                "indexer_topk": _PRO_INDEXER_TOPK,
                "total_topk": _PRO_TOTAL_TOPK,
                "window_topk": _PRO_WINDOW_TOPK,
            },
        }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
