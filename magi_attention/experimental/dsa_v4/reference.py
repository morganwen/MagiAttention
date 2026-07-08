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

"""Pure-PyTorch reference math for the Magi_DSA V4 runtime.

Ports of the Megatron dsv4 unfused path (csa.py / dsa.py, commit c6449f0b2):
sliding-window and compressed-prefix index construction, the unified sparse
MQA attention with a learnable per-head sink, and the indexer KL auxiliary
loss. All score math runs in FP32.
"""

from typing import Optional, Tuple

import torch


def get_window_topk_idxs(
    window_size: int, batch: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [b, sq, window]: positions i-window+1..i, -1 padded."""
    rows = torch.arange(seqlen, device=device).unsqueeze(1)
    cols = torch.arange(window_size, device=device).unsqueeze(0)
    idx = rows - (window_size - 1) + cols
    idx = torch.where(idx < 0, torch.full_like(idx, -1), idx)
    return idx.unsqueeze(0).expand(batch, -1, -1)


def get_compress_topk_idxs(
    ratio: int, batch: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All causally-visible compressed ids [b, sq, sq // ratio], offset applied.

    Query position i sees blocks [0, (i + 1) // ratio); invalid slots are -1.
    """
    n_compressed = seqlen // ratio
    cols = torch.arange(n_compressed, device=device).repeat(seqlen, 1)
    visible = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    idx = torch.where(cols >= visible, torch.full_like(cols, -1), cols + offset)
    return idx.unsqueeze(0).expand(batch, -1, -1)


def sparse_attn_with_sink(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse MQA attention with a learnable sink (SBHD).

    query (sq, b, np, hn); kv_full (n_kv, b, hn) single-head, original entries
    first and compressed entries appended; topk_indices (b, sq, topk) local
    per-batch ids with -1 invalid. Returns (sq, b, np * hn).
    """
    sq, b, np_, hn = query.size()
    n_kv = kv_full.size(0)

    q_flat = query.permute(1, 0, 2, 3).reshape(b * sq, np_, hn)
    kv_flat = kv_full.permute(1, 0, 2).reshape(b * n_kv, hn)
    valid = topk_indices >= 0
    batch_ids = torch.arange(b, device=query.device).view(b, 1, 1)
    global_indices = torch.where(
        valid, topk_indices + batch_ids * n_kv, topk_indices
    ).reshape(b * sq, -1)

    safe = global_indices.clamp(min=0).long()
    kv_gathered = torch.gather(
        kv_flat.unsqueeze(0).expand(b * sq, -1, -1),
        dim=1,
        index=safe.unsqueeze(-1).expand(-1, -1, hn),
    )

    q_f = q_flat.float()
    kv_f = kv_gathered.float()
    scores = torch.einsum("inh,ikh->ink", q_f, kv_f) * softmax_scale
    scores = scores.masked_fill((global_indices < 0).unsqueeze(1), float("-inf"))

    sink = attn_sink.view(1, np_, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = torch.max(scores_max, sink)
    exp_scores = torch.exp(scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)

    out = torch.einsum("ink,ikh->inh", attn, kv_f).to(query.dtype)
    return out.reshape(b, sq, np_ * hn).permute(1, 0, 2).contiguous()


def indexer_kl_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    causal_mask: torch.Tensor,
    sparse_loss: bool,
    calculate_per_token_loss: bool = False,
) -> torch.Tensor:
    """KL(target || predict) auxiliary loss for the indexer.

    index_scores (b, sq, sk): differentiable FP32 predict logits (already
    carrying the indexer softmax scale). topk_indices (b, sq, k): detached
    block ids. query (sq, b, np, hn) and compressed_kv (sk, b, hn) must be
    detached by the caller — the target never trains the trunk. causal_mask
    (b, sq, sk): additive block-causal mask.
    """
    sq, b, np_, hn = query.size()
    sk = compressed_kv.size(0)

    q_r = query.permute(1, 2, 0, 3).reshape(b * np_, sq, hn)
    k_r = (
        compressed_kv.unsqueeze(2)
        .expand(-1, -1, np_, -1)
        .permute(1, 2, 3, 0)
        .reshape(b * np_, hn, sk)
    )
    attn_scores = torch.bmm(q_r.float(), k_r.float()) * softmax_scale
    attn_scores = attn_scores.reshape(b, np_, sq, sk)

    causal = causal_mask.float()
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal.device
    ).scatter_(-1, topk_indices, 0.0)

    attn_scores = attn_scores + causal.unsqueeze(1)
    pred_scores = index_scores + causal
    if sparse_loss:
        attn_scores = attn_scores + index_mask.unsqueeze(1)
        pred_scores = pred_scores + index_mask

    row_valid = (causal > float("-inf")).any(dim=-1)  # [b, sq]
    attn_row = row_valid.view(b, 1, sq, 1)
    pred_row = row_valid.view(b, sq, 1)
    attn_scores = attn_scores.masked_fill(~attn_row, 0.0)
    pred_scores = pred_scores.masked_fill(~pred_row, 0.0)

    target = torch.softmax(attn_scores, dim=-1, dtype=torch.float32) * attn_row.float()
    predict = torch.softmax(pred_scores, dim=-1, dtype=torch.float32) * pred_row.float()

    target = target.sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    kl = target * (torch.log(target + 1e-10) - torch.log(predict + 1e-10))
    kl_per_row = kl.sum(dim=-1)
    kl_div = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return kl_div * loss_coeff


def validate_and_offset_topk(
    topk_indices: torch.Tensor, ratio: int, offset: int
) -> torch.Tensor:
    """Reject causally-invalid selections and shift into the flat KV space.

    topk_indices (b, sq, k) block ids; a selection at query position i is
    valid only if id < (i + 1) // ratio. Invalid slots become -1.
    """
    sq = topk_indices.size(1)
    n_valid = (
        torch.arange(1, sq + 1, device=topk_indices.device).unsqueeze(1) // ratio
    )  # [sq, 1]
    valid = topk_indices < n_valid.unsqueeze(0)
    return torch.where(
        valid, topk_indices + offset, torch.full_like(topk_indices, -1)
    )
