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

import math
from dataclasses import dataclass

import torch

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.functional.dsa_packing import DsaDeviceIndexerMap
from magi_attention.utils import nvtx

from .dsa_phase import dsa_phase


@dataclass(frozen=True)
class DsaIndexerSelection:
    """Grouped Indexer output in global compressed-block ID space."""

    global_ids: torch.Tensor
    lengths: torch.Tensor
    lse: torch.Tensor
    logical_score_calls: int
    logical_topk_calls: int


def _current_cu_stream():
    from cuda.bindings import driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _validate_release_backend(config: MagiDSAConfig, device: torch.device) -> None:
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("Magi-DSA v4 kernels require B300 SM103")
    expected = (
        config.num_query_heads == 64,
        config.head_dim == 512,
        config.indexer_heads == 64,
        config.indexer_head_dim == 128,
    )
    if not all(expected):
        raise ValueError(
            "the fixed FlashMLA/cuDNN backend supports only the release DSA dimensions"
        )


def _resolve_deterministic_topk(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    backend_local_ids: torch.Tensor,
    sample_block_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve exact score ties with ascending canonical global block IDs.

    The fixed cuDNN radix operator supplies the score cutoff and every candidate
    strictly above it. This device-side pass rescans the complete visible row so
    cutoff ties are selected independently of the radix execution layout.
    """

    if scores.ndim != 2 or backend_local_ids.ndim != 2:
        raise ValueError("scores and backend Top-K IDs must be rank-2 tensors")
    rows, columns = scores.shape
    if seq_lens.shape != (rows,) or sample_block_offsets.shape != (rows,):
        raise ValueError("deterministic Top-K metadata has an invalid shape")
    if backend_local_ids.shape[0] != rows:
        raise ValueError("backend Top-K row count does not match scores")
    if backend_local_ids.dtype != torch.int32:
        raise TypeError("backend Top-K IDs must use int32")
    if seq_lens.dtype != torch.int32 or sample_block_offsets.dtype != torch.int32:
        raise TypeError("deterministic Top-K metadata must use int32")

    topk = backend_local_ids.shape[1]
    if topk > columns:
        raise ValueError("backend Top-K width exceeds the score width")
    lengths = seq_lens.clamp(min=0, max=topk)
    if topk == 0:
        return backend_local_ids, lengths

    topk_positions = torch.arange(
        topk, dtype=torch.int32, device=scores.device
    ).unsqueeze(0)
    selected_valid = topk_positions < lengths.unsqueeze(1)
    safe_backend_ids = backend_local_ids.clamp(min=0, max=columns - 1)
    selected_scores = scores.gather(1, safe_backend_ids.to(torch.int64))
    selected_scores = selected_scores.masked_fill(~selected_valid, float("-inf"))

    cutoff = selected_scores.masked_fill(~selected_valid, float("inf")).amin(dim=1)
    strict_valid = selected_valid & (selected_scores > cutoff.unsqueeze(1))
    strict_count = strict_valid.sum(dim=1, dtype=torch.int32)
    needed_ties = lengths - strict_count

    local_columns = torch.arange(
        columns, dtype=torch.int32, device=scores.device
    ).unsqueeze(0)
    visible = local_columns < seq_lens.clamp(min=0, max=columns).unsqueeze(1)
    cutoff_ties = visible & (scores == cutoff.unsqueeze(1))
    tie_candidates = torch.where(cutoff_ties, local_columns, columns)
    tie_local_ids = torch.topk(
        tie_candidates,
        k=topk,
        dim=1,
        largest=False,
        sorted=True,
    ).values
    tie_valid = (topk_positions < needed_ties.unsqueeze(1)) & (tie_local_ids < columns)

    strict_local_ids = backend_local_ids.masked_fill(~strict_valid, columns)
    candidate_local_ids = torch.cat((strict_local_ids, tie_local_ids), dim=1)
    candidate_scores = torch.cat(
        (
            selected_scores.masked_fill(~strict_valid, float("-inf")),
            cutoff.unsqueeze(1).expand(-1, topk).masked_fill(~tie_valid, float("-inf")),
        ),
        dim=1,
    )
    candidate_valid = torch.cat((strict_valid, tie_valid), dim=1)
    candidate_global_ids = candidate_local_ids + sample_block_offsets.unsqueeze(1)
    invalid_id = torch.iinfo(torch.int32).max
    candidate_global_ids = candidate_global_ids.masked_fill(
        ~candidate_valid, invalid_id
    )

    # Stable two-key ordering: global ID ascending first, then score descending.
    id_order = torch.argsort(candidate_global_ids, dim=1, stable=True)
    candidate_global_ids = candidate_global_ids.gather(1, id_order)
    candidate_scores = candidate_scores.gather(1, id_order)
    candidate_valid = candidate_valid.gather(1, id_order)
    score_order = torch.argsort(candidate_scores, dim=1, descending=True, stable=True)
    resolved_ids = candidate_global_ids.gather(1, score_order)[:, :topk]
    resolved_valid = candidate_valid.gather(1, score_order)[:, :topk]
    resolved_ids = resolved_ids.masked_fill(~resolved_valid, -1)
    return resolved_ids, lengths


@torch.no_grad()
def run_grouped_dsa_indexer(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    mapping: DsaDeviceIndexerMap,
    config: MagiDSAConfig,
) -> DsaIndexerSelection:
    """Invoke grouped THD score and top-k exactly once each for this rank."""

    _validate_release_backend(config, q_indexer.device)
    if (
        q_indexer.dtype != torch.bfloat16
        or k_indexer.dtype != torch.bfloat16
        or weights.dtype != torch.bfloat16
    ):
        raise TypeError("DSA Indexer Q/K/weights must use BF16")
    if q_indexer.shape != (
        mapping.seq_lens.numel(),
        config.indexer_heads,
        config.indexer_head_dim,
    ):
        raise ValueError("grouped Indexer Q has an invalid shape")
    if weights.shape != (q_indexer.shape[0], config.indexer_heads):
        raise ValueError("grouped Indexer weights have an invalid shape")
    if k_indexer.shape != (mapping.k_pack.source_rows.numel(), config.indexer_head_dim):
        raise ValueError("grouped Indexer K has an invalid shape")

    total_q = q_indexer.shape[0]
    if total_q == 0 or mapping.max_seqlen_k == 0:
        with nvtx.add_nvtx_event("Magi_DSA/indexer"):
            with nvtx.add_nvtx_event("magi_dsa::indexer_score"):
                lse = torch.full(
                    (total_q,),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q_indexer.device,
                )
            with nvtx.add_nvtx_event("magi_dsa::indexer_topk"):
                global_ids = torch.full(
                    (total_q, config.indexer_topk),
                    -1,
                    dtype=torch.int32,
                    device=q_indexer.device,
                )
                lengths = torch.zeros(
                    (total_q,), dtype=torch.int32, device=q_indexer.device
                )
        return DsaIndexerSelection(global_ids, lengths, lse, 1, 1)

    from cudnn import DSA

    stream = _current_cu_stream()
    with nvtx.add_nvtx_event("Magi_DSA/indexer"):
        with nvtx.add_nvtx_event("magi_dsa::indexer_score"):
            scores = DSA.indexer_forward_wrapper(
                q_indexer,
                k_indexer.unsqueeze(1),
                weights,
                ratio=4,
                qhead_per_kv_head=config.indexer_heads,
                sm_scale=1.0,
                stream=stream,
                cu_seqlens_q=mapping.q_cu_seqlens,
                cu_seqlens_k=mapping.k_cu_seqlens,
                max_seqlen_q=mapping.max_seqlen_q,
                max_seqlen_k=mapping.max_seqlen_k,
                q_causal_offsets=mapping.q_causal_offsets,
            )["scores"]
            lse = torch.logsumexp(scores, dim=-1)

        with nvtx.add_nvtx_event("magi_dsa::indexer_topk"):
            kernel_topk = min(config.indexer_topk, scores.shape[1])
            result = DSA.indexer_top_k_wrapper(
                scores,
                mapping.seq_lens,
                kernel_topk,
                next_n=1,
                return_val=True,
                stream=stream,
            )
            resolved_ids, lengths = _resolve_deterministic_topk(
                scores,
                mapping.seq_lens,
                result["indices"],
                mapping.q_sample_block_offsets,
            )
            global_ids = torch.full(
                (total_q, config.indexer_topk),
                -1,
                dtype=torch.int32,
                device=q_indexer.device,
            )
            global_ids[:, :kernel_topk] = resolved_ids
    return DsaIndexerSelection(global_ids, lengths, lse, 1, 1)


class _DsaSparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[0] == 0:
            output = q.new_empty(q.shape)
            lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        else:
            import flash_mla

            output, _, lse = flash_mla.flash_mla_sparse_fwd(
                q,
                kv.unsqueeze(1),
                indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=q.shape[-1],
                attn_sink=sink,
                topk_length=lengths,
            )
        ctx.save_for_backward(q, kv, output, lse, sink, indices, lengths)
        ctx.softmax_scale = float(softmax_scale)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor | None, dlse: torch.Tensor | None):
        del dlse
        q, kv, output, lse, sink, indices, lengths = ctx.saved_tensors
        if dout is None:
            return (None,) * 6
        if q.shape[0] == 0:
            return (
                torch.zeros_like(q),
                torch.zeros_like(kv),
                torch.zeros_like(sink),
                None,
                None,
                None,
            )
        from cudnn import DSA

        with dsa_phase("sparse_backward"):
            result = DSA.sparse_attention_backward_wrapper(
                q,
                kv,
                output,
                dout.contiguous(),
                lse,
                sink,
                indices,
                softmax_scale=ctx.softmax_scale,
                topk_length=lengths,
                stream=_current_cu_stream(),
            )
        return result["dq"], result["dkv"], result["d_sink"], None, None, None


def dsa_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    config: MagiDSAConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashMLA sparse forward paired with the fixed cuDNN DSA backward."""

    _validate_release_backend(config, q.device)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("DSA attention Q/KV must use BF16")
    if (
        sink.dtype != torch.float32
        or indices.dtype != torch.int32
        or lengths.dtype != torch.int32
    ):
        raise TypeError("DSA sink/indices/lengths have invalid dtypes")
    if q.shape != (indices.shape[0], config.num_query_heads, config.head_dim):
        raise ValueError("DSA attention Q has an invalid shape")
    if kv.ndim != 2 or kv.shape[1] != config.head_dim:
        raise ValueError("DSA attention KV bank has an invalid shape")
    if indices.ndim != 2 or lengths.shape != (q.shape[0],):
        raise ValueError("DSA sparse indices or lengths have an invalid shape")
    return _DsaSparseAttentionFunction.apply(
        q, kv, sink, indices, lengths, config.head_dim**-0.5
    )


class _DsaSelectedKlFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        k_indexer: torch.Tensor,
        q_attention: torch.Tensor,
        k_attention: torch.Tensor,
        sparse_lse: torch.Tensor,
        indexer_indices: torch.Tensor,
        attention_indices: torch.Tensor,
        lengths: torch.Tensor,
        loss_coeff: float,
        attention_scale: float,
    ) -> torch.Tensor:
        total_q = q_indexer.shape[0]
        if total_q == 0 or k_indexer.shape[0] == 0:
            ctx.empty = True
            ctx.shapes = (q_indexer.shape, weights.shape, k_indexer.shape)
            ctx.device = q_indexer.device
            ctx.dtypes = (q_indexer.dtype, weights.dtype, k_indexer.dtype)
            return torch.zeros((), dtype=torch.float32, device=q_indexer.device)

        from cudnn import DSA

        stream = _current_cu_stream()
        topk_length = lengths.unsqueeze(0)
        with dsa_phase("kl_recompute"):
            predict = DSA.sparse_indexer_score_recompute_wrapper(
                q_indexer.unsqueeze(0),
                k_indexer.unsqueeze(0),
                weights.unsqueeze(0),
                indexer_indices.unsqueeze(0),
                qhead_per_kv_head=q_indexer.shape[1],
                topk_length=topk_length,
                stream=stream,
            )["predict"]
            target = DSA.sparse_attn_score_recompute_wrapper(
                q_attention.unsqueeze(0),
                k_attention.unsqueeze(0),
                sparse_lse.unsqueeze(0),
                attention_indices.unsqueeze(0),
                softmax_scale=attention_scale,
                qhead_per_kv_head=q_attention.shape[1],
                topk_length=topk_length,
                stream=stream,
            )["target"]
        columns = torch.arange(
            indexer_indices.shape[1], dtype=torch.int32, device=q_indexer.device
        )
        valid = columns.unsqueeze(0) < lengths.unsqueeze(1)
        target_2d = target[0].masked_fill(~valid, 0.0)
        predict_2d = predict[0].masked_fill(~valid, 0.0)
        minimum = math.exp(-100.0)
        log_target = target_2d.clamp_min(minimum).log().clamp(min=-100.0, max=0.0)
        log_predict = predict_2d.clamp_min(minimum).log().clamp(min=-100.0, max=0.0)
        loss = (target_2d * (log_target - log_predict)).sum(dim=-1).mean() * float(
            loss_coeff
        )

        ctx.empty = False
        ctx.save_for_backward(
            q_indexer,
            weights,
            k_indexer,
            target,
            predict,
            indexer_indices,
        )
        ctx.loss_coeff = float(loss_coeff)
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor | None):
        if grad_loss is None:
            return (None,) * 11
        if ctx.empty:
            q_shape, w_shape, k_shape = ctx.shapes
            q_dtype, w_dtype, k_dtype = ctx.dtypes
            return (
                torch.zeros(q_shape, dtype=q_dtype, device=ctx.device),
                torch.zeros(w_shape, dtype=w_dtype, device=ctx.device),
                torch.zeros(k_shape, dtype=k_dtype, device=ctx.device),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        from cudnn import DSA

        (
            q_indexer,
            weights,
            k_indexer,
            target,
            predict,
            indexer_indices,
        ) = ctx.saved_tensors
        with dsa_phase("indexer_backward"):
            result = DSA.indexer_backward_wrapper(
                q_indexer.unsqueeze(0),
                weights.unsqueeze(0),
                k_indexer.unsqueeze(0),
                target.clone(),
                predict.clone(),
                indexer_indices.unsqueeze(0),
                sm_scale=1.0,
                loss_coeff=ctx.loss_coeff,
                grad_loss=grad_loss,
                stream=_current_cu_stream(),
            )
        return (
            result["d_index_q"][0],
            result["d_weights"][0],
            result["d_index_k"][0],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dsa_selected_kl(
    q_indexer: torch.Tensor,
    weights: torch.Tensor,
    k_indexer: torch.Tensor,
    q_attention: torch.Tensor,
    k_attention: torch.Tensor,
    sparse_lse: torch.Tensor,
    indexer_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    loss_coeff: float,
    config: MagiDSAConfig,
) -> torch.Tensor:
    """Selected-KL on the token owner with gradients only to Indexer tensors."""

    _validate_release_backend(config, q_indexer.device)
    return _DsaSelectedKlFunction.apply(
        q_indexer,
        weights,
        k_indexer,
        q_attention,
        k_attention,
        sparse_lse,
        indexer_indices,
        attention_indices,
        lengths,
        float(loss_coeff),
        config.head_dim**-0.5,
    )


__all__ = [
    "DsaIndexerSelection",
    "dsa_selected_kl",
    "dsa_sparse_attention",
    "run_grouped_dsa_indexer",
]
