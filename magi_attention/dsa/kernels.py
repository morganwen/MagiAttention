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

"""Fused kernel backend for the Magi_DSA V4 unified sparse attention.

Forward runs FlashMLA's ``flash_mla_sparse_fwd`` (nv_dev branch); backward
runs cuDNN Frontend's ``DSA.sparse_attention_backward_wrapper`` — the pair
is co-designed (the cuDNN backward consumes FlashMLA's KV-only FP32 LSE).
The learnable attention sink is passed natively to both kernels and
``d_sink`` comes straight from the cuDNN wrapper.

Both external packages are lazy imports: the reference backend never needs
them. Kernel-path constraints (V1): CUDA bf16 tensors, batch dim 1 (flat
row layout), top-k width padded to the arch alignment (128 on SM90, 64 on
SM100) with ``-1`` sentinels.
"""

from functools import lru_cache

import torch

_flash_mla_sparse_fwd = None
_dsa_namespace = None


def _ensure_flash_mla():
    global _flash_mla_sparse_fwd
    if _flash_mla_sparse_fwd is None:
        try:
            from flash_mla import flash_mla_sparse_fwd as _fwd
        except ImportError as e:
            raise ImportError(
                "Magi_DSA kernel backend needs FlashMLA (nv_dev branch): "
                "`from flash_mla import flash_mla_sparse_fwd` failed."
            ) from e
        _flash_mla_sparse_fwd = _fwd
    return _flash_mla_sparse_fwd


def _ensure_dsa():
    global _dsa_namespace
    if _dsa_namespace is None:
        try:
            from cudnn import DSA as _ns
        except ImportError as e:
            raise ImportError(
                "Magi_DSA kernel backend needs cuDNN Frontend's DSA module: "
                "`from cudnn import DSA` failed."
            ) from e
        _dsa_namespace = _ns
    return _dsa_namespace


@lru_cache(maxsize=1)
def _topk_alignment() -> int:
    """FlashMLA sparse kernel top-k alignment: SM90 dual-warpgroup steps by
    two 64-blocks (128); SM100 single pipeline steps by one (64 for the
    head64 path that D=512 maps to)."""
    sm = torch.cuda.get_device_capability()
    return 64 if sm[0] >= 10 else 128


class _KernelSparseAttn(torch.autograd.Function):
    """FlashMLA forward + cuDNN DSA backward on flat tensors."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # (rows, H, D) bf16
        kv: torch.Tensor,  # (n_kv, D) bf16
        attn_sink: torch.Tensor,  # (H,) f32
        topk_idxs: torch.Tensor,  # (rows, K) int32, -1 invalid
        softmax_scale: float,
    ) -> torch.Tensor:
        fwd = _ensure_flash_mla()

        k_width = topk_idxs.shape[-1]
        align = _topk_alignment()
        k_padded = (k_width + align - 1) // align * align
        if k_padded != k_width:
            topk_idxs = torch.nn.functional.pad(
                topk_idxs, (0, k_padded - k_width), value=-1
            )
        topk_idxs = topk_idxs.contiguous()

        out, _max_logits, lse = fwd(
            q,
            kv.unsqueeze(1),  # (n_kv, h_kv=1, D)
            topk_idxs.unsqueeze(1),  # (rows, h_kv=1, K_padded)
            softmax_scale,
            d_v=q.shape[-1],
            attn_sink=attn_sink,
        )

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out, lse)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        dsa = _ensure_dsa()
        q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors
        result = dsa.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dO.contiguous(),
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=None,
        )
        return result["dq"], result["dkv"], result["d_sink"], None, None


def indexer_select_kernel(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    ratio: int,
    pos_offset: int = 0,
) -> torch.Tensor:
    """Indexer scoring + top-k selection on cuDNN kernels (stop-gradient).

    q_idx (sq, b, H, D) bf16; k_idx (sk, b, D) bf16; weights (sq, b, H)
    bf16 (head scale already applied). ``ratio`` drives the kernel's
    block-causal mask; ``pos_offset`` is the global position of row 0
    (context parallelism), forwarded as ``q_causal_offsets``.
    Returns (b, sq, topk) int32 local block ids, -1 invalid.
    """
    dsa = _ensure_dsa()
    sq, b, h, d = q_idx.shape
    sk = k_idx.shape[0]

    q_b = q_idx.permute(1, 0, 2, 3).contiguous()  # (b, sq, H, D)
    k_b = k_idx.permute(1, 0, 2).unsqueeze(2).contiguous()  # (b, sk, 1, D)
    w_b = weights.permute(1, 0, 2).contiguous()  # (b, sq, H)

    kwargs = {}
    if pos_offset:
        kwargs["q_causal_offsets"] = torch.full(
            (b,), pos_offset, dtype=torch.int32, device=q_idx.device
        )
    scores = dsa.indexer_forward_wrapper(q_b, k_b, w_b, ratio=ratio, **kwargs)[
        "scores"
    ]  # (b, sq, sk) fp32, -inf outside the causal range
    scores_flat = scores.reshape(b * sq, -1)[:, :sk].contiguous()

    rows = torch.arange(sq, device=q_idx.device) + pos_offset
    seq_lens = ((rows + 1) // ratio).clamp(max=sk).to(torch.int32).repeat(b)

    topk_k = min(topk, sk)
    if topk_k == 0:
        return torch.full((b, sq, topk), -1, dtype=torch.int32, device=q_idx.device)
    res = dsa.indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    idx = res["indices"]  # (b*sq, topk_k) int32, -1 invalid
    valid = (idx >= 0) & (idx < seq_lens.unsqueeze(1))
    safe_idx = idx.clamp(min=0, max=max(sk - 1, 0)).long()
    selected_scores = torch.gather(scores_flat, 1, safe_idx)
    selected_scores = selected_scores.masked_fill(~valid, float("-inf"))

    # cuDNN promises the largest set but does not promise output order or the
    # secondary key at the K-boundary. Canonicalize the selected rows first,
    # then replace boundary-score members with the smallest visible block ids.
    invalid_id = torch.iinfo(torch.int32).max
    id_order = torch.argsort(
        torch.where(valid, idx, torch.full_like(idx, invalid_id)),
        dim=-1,
        stable=True,
    )
    idx = torch.gather(idx, 1, id_order)
    selected_scores = torch.gather(selected_scores, 1, id_order)
    score_order = torch.argsort(selected_scores, dim=-1, descending=True, stable=True)
    idx = torch.gather(idx, 1, score_order)
    selected_scores = torch.gather(selected_scores, 1, score_order)

    selected_count = seq_lens.clamp(min=0, max=topk_k).long()
    threshold_position = (selected_count - 1).clamp(min=0).unsqueeze(1)
    threshold = torch.gather(selected_scores, 1, threshold_position).squeeze(1)
    threshold = torch.where(
        selected_count > 0,
        threshold,
        torch.full_like(threshold, float("inf")),
    )
    block_ids = torch.arange(sk, device=q_idx.device).unsqueeze(0)
    visible = block_ids < seq_lens.unsqueeze(1)
    above_count = (visible & (scores_flat > threshold.unsqueeze(1))).sum(dim=1)
    boundary_tie = visible & (scores_flat == threshold.unsqueeze(1))
    tie_keys = torch.where(
        boundary_tie,
        -block_ids.to(scores_flat.dtype),
        torch.full_like(scores_flat, float("-inf")),
    )
    smallest_tie_ids = torch.topk(
        tie_keys,
        k=topk_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices.to(torch.int32)
    positions = torch.arange(topk_k, device=q_idx.device).unsqueeze(0)
    tie_positions = (positions - above_count.unsqueeze(1)).clamp(min=0, max=topk_k - 1)
    idx = torch.where(
        positions < above_count.unsqueeze(1),
        idx,
        torch.gather(smallest_tie_ids, 1, tie_positions),
    )
    idx = torch.where(
        positions < selected_count.unsqueeze(1), idx, torch.full_like(idx, -1)
    )
    if topk_k < topk:
        pad = torch.full(
            (b * sq, topk - topk_k), -1, dtype=torch.int32, device=q_idx.device
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.view(b, sq, -1)


def sparse_attn_with_sink_kernel(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Drop-in kernel replacement for ``reference.sparse_attn_with_sink``.

    Same contract: query (sq, b, np, hn) bf16; kv_full (n_kv, b, hn) bf16;
    attn_sink (np,) f32; topk_indices (b, sq, K) local per-batch ids with
    -1 invalid. V1 kernel path requires b == 1 (flat layout).
    Returns (sq, b, np * hn).
    """
    sq, b, np_, hn = query.shape
    assert b == 1, "kernel backend V1 supports batch 1 (flat rows); use packed"
    assert query.is_cuda and query.dtype == torch.bfloat16, "kernel path is CUDA bf16"

    q_flat = query.squeeze(1).contiguous()
    kv_flat = kv_full.squeeze(1).contiguous()
    idx_flat = topk_indices.squeeze(0).to(torch.int32).contiguous()

    out = _KernelSparseAttn.apply(
        q_flat, kv_flat, attn_sink.float(), idx_flat, softmax_scale
    )  # (sq, np, d_v)
    return out.reshape(sq, 1, -1)


def _indexer_kl_value_only(
    q_idx: torch.Tensor,
    w_idx: torch.Tensor,
    k_global: torch.Tensor,
    query_det: torch.Tensor,
    comp_det: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    indexer_scale: float,
    loss_coeff: float,
    total_global: int,
) -> torch.Tensor:
    """Compute the selected-column KL value without launching its backward.

    The public runtime has its own recompute autograd boundary. Its forward is
    executed under ``no_grad`` and may retain only the frozen minimal state,
    so precomputing cuDNN Indexer gradients here would be discarded. The
    explicit runtime backward invokes :class:`_KernelIndexerKL` once and owns
    the single real ``indexer_backward`` launch.
    """

    dsa = _ensure_dsa()
    sq = q_idx.size(0)
    with torch.no_grad():
        idxs = topk_indices.to(torch.int32)
        k_in = idxs.size(-1)
        k_pad = max(128, (k_in + 127) // 128 * 128)
        if k_pad != k_in:
            idxs = torch.nn.functional.pad(idxs, (0, k_pad - k_in), value=-1)
        idxs = idxs.contiguous()
        valid = idxs >= 0
        safe = idxs.clamp(min=0).long()

        w_scaled = (w_idx.float() * indexer_scale).to(w_idx.dtype)
        predict = dsa.sparse_indexer_score_recompute_wrapper(
            q_idx.permute(1, 0, 2, 3).contiguous(),
            k_global.permute(1, 0, 2).contiguous(),
            w_scaled.permute(1, 0, 2).contiguous(),
            idxs,
            qhead_per_kv_head=q_idx.size(2),
            topk_indices_global=True,
        )["predict"].view(1, sq, idxs.size(-1))

        ckv_sel = comp_det.permute(1, 0, 2)[
            torch.zeros(1, 1, 1, dtype=torch.long, device=idxs.device), safe
        ]
        q_f = query_det.permute(1, 0, 2, 3).float()
        target_logits = (
            torch.einsum("bqnh,bqkh->bqnk", q_f, ckv_sel.float()) * softmax_scale
        )
        row_has = valid.any(dim=-1, keepdim=True)
        target_logits = target_logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
        target_logits = target_logits.masked_fill(~row_has.unsqueeze(2), 0.0)
        target = (
            torch.softmax(target_logits, dim=-1, dtype=torch.float32)
            * row_has.unsqueeze(2).float()
        ).sum(dim=2)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        pred_f = predict.float().clamp(min=0)
        kl = target * (torch.log(target + 1e-10) - torch.log(pred_f + 1e-10))
        kl = torch.where(valid, kl, torch.zeros_like(kl))
        return kl.sum() * (loss_coeff / total_global)


class _KernelIndexerKL(torch.autograd.Function):
    """Selected-columns indexer KL with a fully analytic kernel backward.

    Forward runs entirely under no_grad — the torch target (detached trunk)
    plus cuDNN's sparse_indexer_score_recompute for the predict — so no
    autograd graph (and no gather scatter-add backward, the profiled 60%
    hotspot) is ever built. Backward scales gradients precomputed by
    cuDNN's indexer_backward_wrapper: d_index_q / d_weights / d_index_k in
    one kernel, mirroring the Megatron Path C recipe.
    """

    @staticmethod
    def forward(
        ctx,
        q_idx: torch.Tensor,  # (sq, 1, H, D) bf16, differentiable
        w_idx: torch.Tensor,  # (sq, 1, H) bf16, RAW (unscaled), differentiable
        k_global: torch.Tensor,  # (skc, 1, D) bf16, differentiable
        query_det: torch.Tensor,  # (sq, 1, np, hn) detached
        comp_det: torch.Tensor,  # (skc, 1, hn) detached
        topk_indices: torch.Tensor,  # (1, sq, K) long, -1 invalid
        softmax_scale: float,
        indexer_scale: float,
        loss_coeff: float,
        total_global: int,
    ) -> torch.Tensor:
        dsa = _ensure_dsa()
        sq = q_idx.size(0)
        skc = k_global.size(0)

        with torch.no_grad():
            idxs = topk_indices.to(torch.int32)
            # Pad the selected width to the kernel block (128) with -1.
            k_in = idxs.size(-1)
            k_pad = max(128, (k_in + 127) // 128 * 128)
            if k_pad != k_in:
                idxs = torch.nn.functional.pad(idxs, (0, k_pad - k_in), value=-1)
            idxs = idxs.contiguous()  # (1, sq, K)
            valid = idxs >= 0
            safe = idxs.clamp(min=0).long()

            # predict probabilities at selected columns (cuDNN recompute).
            w_scaled = (w_idx.float() * indexer_scale).to(w_idx.dtype)
            predict = dsa.sparse_indexer_score_recompute_wrapper(
                q_idx.permute(1, 0, 2, 3).contiguous(),  # (1, sq, H, D)
                k_global.permute(1, 0, 2).contiguous(),  # (1, skc, D)
                w_scaled.permute(1, 0, 2).contiguous(),  # (1, sq, H)
                idxs,
                qhead_per_kv_head=q_idx.size(2),
                topk_indices_global=True,
            )["predict"].view(1, sq, idxs.size(-1))

            # target probabilities: per-head softmax over the selected set,
            # head-summed, L1-normalized (identical math to the reference).
            ckv_sel = comp_det.permute(1, 0, 2)[
                torch.zeros(1, 1, 1, dtype=torch.long, device=idxs.device), safe
            ]  # (1, sq, K, hn)
            q_f = query_det.permute(1, 0, 2, 3).float()  # (1, sq, np, hn)
            t = torch.einsum("bqnh,bqkh->bqnk", q_f, ckv_sel.float()) * softmax_scale
            row_has = valid.any(dim=-1, keepdim=True)
            t = t.masked_fill(~valid.unsqueeze(2), float("-inf"))
            t = t.masked_fill(~row_has.unsqueeze(2), 0.0)
            target = (
                torch.softmax(t, dim=-1, dtype=torch.float32)
                * row_has.unsqueeze(2).float()
            ).sum(dim=2)
            target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            pred_f = predict.float().clamp(min=0)
            kl = target * (torch.log(target + 1e-10) - torch.log(pred_f + 1e-10))
            kl = torch.where(valid, kl, torch.zeros_like(kl))
            loss = kl.sum() * (loss_coeff / total_global)

            # Precompute indexer gradients at unit upstream grad; backward
            # only scales. The kernel consumes scores in-place: clone.
            ig = dsa.indexer_backward_wrapper(
                q_idx.permute(1, 0, 2, 3).contiguous(),
                w_idx.permute(1, 0, 2).contiguous(),  # RAW weights
                k_global.permute(1, 0, 2).contiguous(),
                target.clone().contiguous(),
                pred_f.clone().contiguous(),
                idxs,
                grad_loss=torch.ones((), device=q_idx.device, dtype=torch.float32),
                sm_scale=indexer_scale,
                loss_coeff=loss_coeff * sq / total_global,
                topk_indices_global=True,
            )
            dq = ig["d_index_q"].view(1, sq, q_idx.size(2), q_idx.size(3))
            dw = ig["d_weights"].view(1, sq, q_idx.size(2))
            dk = ig["d_index_k"].view(1, skc, k_global.size(2))

        ctx.save_for_backward(dq, dw, dk)
        ctx.shapes = (q_idx.shape, w_idx.shape, k_global.shape)
        return loss

    @staticmethod
    def backward(ctx, d_kl: torch.Tensor):
        dq, dw, dk = ctx.saved_tensors
        qs, ws_, ks = ctx.shapes
        gq = (dq * d_kl).permute(1, 0, 2, 3).reshape(qs).to(torch.bfloat16)
        gw = (dw * d_kl).permute(1, 0, 2).reshape(ws_).to(torch.bfloat16)
        gk = (dk * d_kl).permute(1, 0, 2).reshape(ks).to(torch.bfloat16)
        return gq, gw, gk, None, None, None, None, None, None, None


def indexer_kl_loss_kernel(
    topk_indices: torch.Tensor,
    q_idx: torch.Tensor,
    w_idx: torch.Tensor,
    k_idx: torch.Tensor,
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    softmax_scale: float,
    indexer_scale: float,
    loss_coeff: float,
    total_global: int,
) -> torch.Tensor:
    """Kernel-path drop-in for ``indexer_kl_loss_selected`` (b == 1).

    Returns the KL SUM-over-rows normalized by ``total_global`` (the CP
    global token count; pass local rows for the single-device token-mean).
    """
    assert q_idx.size(1) == 1, "kernel KL path requires batch 1"
    if not torch.is_grad_enabled() or not any(
        tensor.requires_grad for tensor in (q_idx, w_idx, k_idx)
    ):
        return _indexer_kl_value_only(
            q_idx,
            w_idx,
            k_idx,
            query.detach(),
            compressed_kv.detach(),
            topk_indices,
            softmax_scale,
            indexer_scale,
            loss_coeff,
            total_global,
        )
    return _KernelIndexerKL.apply(
        q_idx,
        w_idx,
        k_idx,
        query.detach(),
        compressed_kv.detach(),
        topk_indices,
        softmax_scale,
        indexer_scale,
        loss_coeff,
        total_global,
    )
