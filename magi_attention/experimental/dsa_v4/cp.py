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

"""Context-parallel forward for the Magi_DSA V4 runtime (V1: one sample,
contiguous block-aligned equal split).

Communication structure per the design doc: raw KV and hidden states move
only through a point-to-point left-boundary halo (window reach plus the
CSA overlap block); compressed entries and indexer keys are all-gathered
across the CP group (a ratio-th of the sequence). Both collectives are
autograd-aware so gradients flow back to the owning rank.

Collectives are backend-agnostic: under gloo the payloads are staged
through CPU so the same code runs two ranks on one GPU for correctness
tests and NCCL multi-GPU unchanged.
"""

from typing import Tuple

import torch
import torch.distributed as dist

from .attention import MagiDSAV4
from .indexer import compute_index_scores
from .reference import indexer_kl_loss, sparse_attn_with_sink


def _stage_for_comm(t: torch.Tensor, group) -> torch.Tensor:
    if dist.get_backend(group) == "gloo":
        return t.detach().cpu()
    return t.detach().contiguous()


_HALO_TAG = [0]


def _next_tag() -> int:
    """Monotonic message tag; the call sequence is identical on every rank,
    so tags agree without any coordination."""
    _HALO_TAG[0] += 2
    return _HALO_TAG[0]


class _LeftHaloExchange(torch.autograd.Function):
    """Each rank receives the last ``h`` rows of its left neighbor.

    Rank 0 receives zeros. Backward routes the halo gradient back to the
    owner's tail rows. Blocking tagged send/recv, parity-ordered to avoid
    deadlock; matching by explicit tag is robust under gloo where batched
    P2P ops may interleave across exchanges.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, h: int, group) -> torch.Tensor:
        rank = dist.get_rank(group)
        ws = dist.get_world_size(group)
        tag = _next_tag()
        ctx.h, ctx.group, ctx.rank, ctx.ws, ctx.tag = h, group, rank, ws, tag
        ctx.x_shape, ctx.x_device = x.shape, x.device

        halo = torch.zeros((h, *x.shape[1:]), device=x.device, dtype=x.dtype)

        def do_send():
            if rank + 1 < ws:
                dist.send(
                    _stage_for_comm(x[-h:], group),
                    dist.get_global_rank(group, rank + 1),
                    group=group,
                    tag=tag,
                )

        def do_recv():
            if rank > 0:
                buf = _stage_for_comm(halo, group)
                dist.recv(
                    buf, dist.get_global_rank(group, rank - 1), group=group, tag=tag
                )
                halo.copy_(buf.to(device=x.device, dtype=x.dtype))

        if rank % 2 == 0:
            do_send(), do_recv()
        else:
            do_recv(), do_send()
        return halo

    @staticmethod
    def backward(ctx, g_halo: torch.Tensor):
        h, group, rank, ws = ctx.h, ctx.group, ctx.rank, ctx.ws
        tag = ctx.tag + 1
        grad_x = torch.zeros(ctx.x_shape, device=ctx.x_device, dtype=g_halo.dtype)

        def do_send():
            if rank > 0:
                dist.send(
                    _stage_for_comm(g_halo, group),
                    dist.get_global_rank(group, rank - 1),
                    group=group,
                    tag=tag,
                )

        def do_recv():
            if rank + 1 < ws:
                buf = _stage_for_comm(grad_x[-h:], group)
                dist.recv(
                    buf, dist.get_global_rank(group, rank + 1), group=group, tag=tag
                )
                grad_x[-h:] = buf.to(device=ctx.x_device, dtype=g_halo.dtype)

        if rank % 2 == 0:
            do_send(), do_recv()
        else:
            do_recv(), do_send()
        return grad_x, None, None


class _AllGatherConcat(torch.autograd.Function):
    """All-gather equal-shaped chunks and concatenate on dim 0.

    Backward sums the full gradient across ranks and returns the local
    slice (reduce-scatter semantics, expressed gloo-compatibly).
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor, group) -> torch.Tensor:
        rank = dist.get_rank(group)
        ws = dist.get_world_size(group)
        ctx.group, ctx.rank, ctx.ws = group, rank, ws
        ctx.n_local = t.size(0)
        ctx.t_device, ctx.t_dtype = t.device, t.dtype

        staged = _stage_for_comm(t, group)
        bufs = [torch.empty_like(staged) for _ in range(ws)]
        dist.all_gather(bufs, staged, group=group)
        bufs[rank] = staged
        out = torch.cat([b.to(device=t.device, dtype=t.dtype) for b in bufs], dim=0)
        # Reattach the local chunk so autograd links the graph locally too.
        out[rank * ctx.n_local : (rank + 1) * ctx.n_local] = t
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        # Sum in FP32 per the dKV-merge contract (FP32 accumulate, single
        # cast at the end), then hand back the local slice.
        staged = _stage_for_comm(g.contiguous().float(), ctx.group)
        dist.all_reduce(staged, op=dist.ReduceOp.SUM, group=ctx.group)
        local = staged[ctx.rank * ctx.n_local : (ctx.rank + 1) * ctx.n_local]
        return local.to(device=ctx.t_device, dtype=g.dtype), None


def forward_cp(
    module: MagiDSAV4,
    x: torch.Tensor,
    qr: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context-parallel SBHD forward over one sample.

    Every rank holds an equal contiguous slice of the global sequence;
    the slice length must be a multiple of the compression block grid
    (128-aligned covers both ratios). Inputs are the LOCAL rows of
    x (sq_local, b, hidden), qr, query, kv; RoPE on query/kv is applied
    globally by the caller as usual. Returns the LOCAL output rows and
    this rank's share of the KL scalar, normalized by the GLOBAL token
    count (sum across ranks reproduces the CP=1 scalar).
    """
    cfg = module.config
    rank = dist.get_rank(group)
    ws = dist.get_world_size(group)
    sq_local, b = x.size(0), x.size(1)
    device = x.device
    global_start = rank * sq_local
    sq_global = ws * sq_local
    ratio = cfg.compress_ratio
    if ratio > 1:
        assert sq_local % ratio == 0, "CP slice must be block-aligned"

    halo = max(cfg.window_size, ratio if ratio > 1 else 1)
    kv_halo = _LeftHaloExchange.apply(kv, halo, group)

    kl_loss = torch.zeros((), dtype=torch.float32, device=device)

    def compress_local(compressor, inp, needs_overlap_halo, halo_inp=None):
        block_offset = global_start // ratio
        if needs_overlap_halo and rank > 0:
            comp = compressor(
                torch.cat([halo_inp[-ratio:], inp], dim=0), block_offset=block_offset - 1
            )
            return comp[1:]
        return compressor(inp, block_offset=block_offset)

    comp_global = None
    n_comp_total = 0
    if module.compressor is not None:
        x_halo = _LeftHaloExchange.apply(x, ratio, group) if ratio == 4 else None
        comp_local = compress_local(module.compressor, x, ratio == 4, x_halo)
        if x_halo is not None:
            # Rank 0 never consumes its (zero) halo, so its backward node
            # would be pruned and the halo-grad send/recv pairing across
            # ranks would deadlock. A zero-weight consumption keeps every
            # rank's autograd comm schedule identical without changing
            # any value.
            comp_local = comp_local + x_halo.sum().to(comp_local.dtype) * 0
        comp_global = _AllGatherConcat.apply(comp_local, group)
        n_comp_total = comp_global.size(0)

    # ---- window indices in the flat local KV space -----------------------
    # kv_flat = [halo (halo rows) | local kv (sq_local) | compressed (global)]
    rows = torch.arange(sq_local, device=device).unsqueeze(1) + global_start
    cols = torch.arange(cfg.window_size, device=device).unsqueeze(0)
    win_gid = rows - (cfg.window_size - 1) + cols  # global ids
    win_flat = win_gid - (global_start - halo)
    win_flat = torch.where(win_gid < 0, torch.full_like(win_flat, -1), win_flat)
    window_idxs = win_flat.unsqueeze(0).expand(b, -1, -1)

    if comp_global is not None and n_comp_total > 0:
        comp_offset = halo + sq_local
        visible = (
            torch.arange(1, sq_local + 1, device=device) + global_start
        ).unsqueeze(1) // ratio  # [sq_local, 1] global visible block count

        if module.indexer is not None:
            x_det = x.detach()
            qr_det = qr.detach()
            xh_det = x_halo.detach() if x_halo is not None else None
            q_idx = module.indexer.linear_wq_b(qr_det).reshape(
                sq_local, b, module.indexer.n_heads, module.indexer.head_dim
            )
            from .compressor import rotate_activation
            from .rope import apply_rope_last_dims

            freqs = module.indexer._q_freqs(global_start + sq_local, device)[global_start:]
            q_idx = apply_rope_last_dims(q_idx, freqs, module.indexer.rope_dim)
            q_idx = rotate_activation(q_idx)
            k_local = compress_local(module.indexer.compressor, x_det, ratio == 4, xh_det)
            k_global = _AllGatherConcat.apply(k_local, group)
            w_idx = module.indexer.linear_weights_proj(x_det) * (
                module.indexer.n_heads**-0.5
            )

            mask_cols = torch.arange(n_comp_total, device=device).unsqueeze(0)
            causal_mask = (
                torch.where(mask_cols >= visible, float("-inf"), 0.0)
                .unsqueeze(0)
                .expand(b, -1, -1)
            )

            if module.training and torch.is_grad_enabled():
                scores = compute_index_scores(
                    q_idx, w_idx * module.indexer.softmax_scale, k_global
                )
                scores = scores + causal_mask
                with torch.no_grad():
                    effective_topk = min(cfg.topk, n_comp_total)
                    topk_indices = scores.topk(effective_topk, dim=-1)[1]
                kl_sum = indexer_kl_loss(
                    scores,
                    topk_indices,
                    query.detach(),
                    comp_global.detach(),
                    cfg.softmax_scale,
                    cfg.indexer_loss_coeff,
                    causal_mask,
                    cfg.use_sparse_loss,
                    calculate_per_token_loss=True,
                )
                kl_loss = kl_sum / sq_global
            else:
                scores = compute_index_scores(q_idx, w_idx, k_global) + causal_mask
                effective_topk = min(cfg.topk, n_comp_total)
                topk_indices = scores.topk(effective_topk, dim=-1)[1]

            valid = topk_indices < visible.unsqueeze(0)
            compress_idxs = torch.where(
                valid, topk_indices + comp_offset, torch.full_like(topk_indices, -1)
            )
        else:
            mask_cols = torch.arange(n_comp_total, device=device).repeat(sq_local, 1)
            compress_idxs = torch.where(
                mask_cols >= visible,
                torch.full_like(mask_cols, -1),
                mask_cols + comp_offset,
            )
            compress_idxs = compress_idxs.unsqueeze(0).expand(b, -1, -1)

        kv_flat = torch.cat([kv_halo, kv, comp_global], dim=0)
        topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)
    else:
        kv_flat = torch.cat([kv_halo, kv], dim=0)
        topk_idxs = window_idxs

    output = sparse_attn_with_sink(
        query, kv_flat, attn_sink.float(), topk_idxs.int(), cfg.softmax_scale
    )
    return output, kl_loss
