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


def _wire_tag(group, tag: int) -> int:
    """gloo matches by explicit tag; NCCL does not support tags and its P2P
    is stream-ordered FIFO per pair, so ordering alone is sufficient there."""
    return tag if dist.get_backend(group) == "gloo" else 0


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
                    tag=_wire_tag(group, tag),
                )

        def do_recv():
            if rank > 0:
                buf = _stage_for_comm(halo, group)
                dist.recv(
                    buf, dist.get_global_rank(group, rank - 1), group=group, tag=_wire_tag(group, tag)
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
                    tag=_wire_tag(group, tag),
                )

        def do_recv():
            if rank + 1 < ws:
                buf = _stage_for_comm(grad_x[-h:], group)
                dist.recv(
                    buf, dist.get_global_rank(group, rank + 1), group=group, tag=_wire_tag(group, tag)
                )
                grad_x[-h:] = buf.to(device=ctx.x_device, dtype=g_halo.dtype)

        if rank % 2 == 0:
            do_send(), do_recv()
        else:
            do_recv(), do_send()
        return grad_x, None, None


class _AllGatherVarCat(torch.autograd.Function):
    """All-gather chunks of per-rank row counts and concatenate on dim 0.

    Counts are host-known on every rank (derived from global metadata), so
    payloads are padded to the max count for the collective and sliced back
    after. Backward sums the full gradient across ranks (FP32) and returns
    the local slice.
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor, counts: tuple, group) -> torch.Tensor:
        rank = dist.get_rank(group)
        ws = dist.get_world_size(group)
        ctx.group, ctx.rank, ctx.counts = group, rank, counts
        ctx.t_device = t.device
        assert t.size(0) == counts[rank]

        max_n = max(counts) if counts else 0
        pad = t.new_zeros((max_n - t.size(0), *t.shape[1:]))
        staged = _stage_for_comm(torch.cat([t, pad], dim=0), group)
        bufs = [torch.empty_like(staged) for _ in range(ws)]
        dist.all_gather(bufs, staged, group=group)
        bufs[rank] = staged
        pieces = [
            b.to(device=t.device, dtype=t.dtype)[: counts[r]] for r, b in enumerate(bufs)
        ]
        return torch.cat(pieces, dim=0)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        counts, rank = ctx.counts, ctx.rank
        max_n = max(counts) if counts else 0
        starts = [sum(counts[:r]) for r in range(len(counts))]
        padded = g.new_zeros((len(counts) * max_n, *g.shape[1:]), dtype=torch.float32)
        for r, (s, c) in enumerate(zip(starts, counts)):
            padded[r * max_n : r * max_n + c] = g[s : s + c].float()
        staged = _stage_for_comm(padded, ctx.group)
        dist.all_reduce(staged, op=dist.ReduceOp.SUM, group=ctx.group)
        local = staged[rank * max_n : rank * max_n + counts[rank]]
        return local.to(device=ctx.t_device, dtype=g.dtype), None, None


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

            if cfg.backend == "kernel":
                from .kernels import indexer_select_kernel
                from .reference import indexer_kl_loss_selected

                topk_indices = indexer_select_kernel(
                    q_idx, k_global, w_idx, cfg.topk, ratio, pos_offset=global_start
                ).long()
                if module.training and torch.is_grad_enabled():
                    kl_sum = indexer_kl_loss_selected(
                        topk_indices,
                        q_idx,
                        w_idx,
                        k_global,
                        query.detach(),
                        comp_global.detach(),
                        cfg.softmax_scale,
                        module.indexer.softmax_scale,
                        cfg.indexer_loss_coeff,
                        calculate_per_token_loss=True,
                    )
                    kl_loss = kl_sum / sq_global
            elif module.training and torch.is_grad_enabled():
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

            valid = (topk_indices >= 0) & (topk_indices < visible.unsqueeze(0))
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

    if cfg.backend == "kernel":
        from .kernels import sparse_attn_with_sink_kernel

        output = sparse_attn_with_sink_kernel(
            query, kv_flat, attn_sink.float(), topk_idxs.int(), cfg.softmax_scale
        )
    else:
        output = sparse_attn_with_sink(
            query, kv_flat, attn_sink.float(), topk_idxs.int(), cfg.softmax_scale
        )
    return output, kl_loss


def _plan_packed_cp(bounds, sq_local, ws, ratio):
    """Host-side plan shared by every rank: per-sample block layout and
    per-rank block ownership (a block belongs to the rank holding its last
    token). Returns (sample_block_offset, per-rank counts, per-rank list of
    (sample_id, j0, j1) owned block runs), all pure python."""
    n_blocks = []
    sample_block_offset = []
    acc = 0
    for s in range(len(bounds) - 1):
        sample_block_offset.append(acc)
        nb = (bounds[s + 1] - bounds[s]) // ratio if ratio > 1 else 0
        n_blocks.append(nb)
        acc += nb

    counts = [0] * ws
    runs = [[] for _ in range(ws)]
    for s in range(len(bounds) - 1):
        S = bounds[s]
        for j in range(n_blocks[s]):
            last = S + (j + 1) * ratio - 1
            r = min(ws - 1, last // sq_local)
            counts[r] += 1
            if runs[r] and runs[r][-1][0] == s and runs[r][-1][2] == j - 1:
                runs[r][-1] = (s, runs[r][-1][1], j)
            else:
                runs[r].append((s, j, j))
    return sample_block_offset, counts, runs


def forward_cp_packed(
    module: MagiDSAV4,
    x: torch.Tensor,
    qr: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context-parallel forward over a packed variable-length batch.

    The GLOBAL packed stream (boundaries in ``cu_seqlens``, host tensor) is
    split into equal contiguous row slices; this rank holds rows
    ``[rank * sq_local, (rank + 1) * sq_local)`` of every flat input:
    x (sq_local, hidden), qr (sq_local, q_lora), query (sq_local, nh, hd),
    kv (sq_local, hd). Sample block grids restart at each sample start;
    a compression block belongs to the rank holding its LAST token and is
    computed locally with left-halo rows, then all-gathered (variable
    per-rank counts). Windows, causality and KL never cross sample
    boundaries. Returns local output rows and this rank's KL share
    normalized by the global token count.
    """
    cfg = module.config
    rank = dist.get_rank(group)
    ws = dist.get_world_size(group)
    sq_local = x.size(0)
    device = x.device
    start = rank * sq_local
    bounds = cu_seqlens.tolist()
    total = bounds[-1]
    assert total == ws * sq_local, "global rows must equal ws * sq_local"
    ratio = cfg.compress_ratio

    halo = max(cfg.window_size, 2 * ratio if ratio > 1 else 1)
    kv_halo = _LeftHaloExchange.apply(kv.unsqueeze(1), halo, group).squeeze(1)

    kl_loss = torch.zeros((), dtype=torch.float32, device=device)

    rows_g = torch.arange(sq_local, device=device) + start
    sample_of_row = torch.bucketize(
        rows_g, torch.as_tensor(bounds[1:-1], device=device), right=True
    )
    sample_start_row = torch.as_tensor(bounds[:-1], device=device)[sample_of_row]

    # ---- window indices (never cross a sample boundary) ------------------
    cols = torch.arange(cfg.window_size, device=device).unsqueeze(0)
    win_gid = rows_g.unsqueeze(1) - (cfg.window_size - 1) + cols
    win_ok = win_gid >= sample_start_row.unsqueeze(1)
    win_flat = torch.where(
        win_ok, win_gid - (start - halo), torch.full_like(win_gid, -1)
    )
    window_idxs = win_flat.unsqueeze(0)  # (1, sq_local, w)

    comp_global = None
    n_comp_total = 0
    if module.compressor is not None:
        sample_block_offset, counts, runs = _plan_packed_cp(
            bounds, sq_local, ws, ratio
        )
        n_comp_total = sum(counts)

        x_halo = _LeftHaloExchange.apply(x.unsqueeze(1), halo, group).squeeze(1)
        x_ext = torch.cat([x_halo, x], dim=0)  # rows [start - halo, start + sq_local)

        def run_compressor(compressor, ext_rows, detached):
            src = ext_rows.detach() if detached else ext_rows
            pieces = []
            for s, j0, j1 in runs[rank]:
                S = bounds[s]
                lo = S + j0 * ratio
                prepend = 1 if (ratio == 4 and j0 > 0) else 0
                lo -= prepend * ratio
                hi = S + (j1 + 1) * ratio
                slab = src[lo - (start - halo) : hi - (start - halo)]
                comp = compressor(
                    slab.unsqueeze(1), block_offset=j0 - prepend
                )
                pieces.append(comp[prepend:])
            if pieces:
                return torch.cat(pieces, dim=0)
            return src.new_zeros((0, 1, compressor.head_dim))

        comp_local = run_compressor(module.compressor, x_ext, detached=False)
        comp_local = comp_local + x_halo.sum().to(comp_local.dtype) * 0
        comp_global = _AllGatherVarCat.apply(comp_local, tuple(counts), group)

    if comp_global is not None and n_comp_total > 0:
        comp_base = halo + sq_local
        compress_idxs = torch.full(
            (sq_local, max(cfg.topk, 1)), -1, dtype=torch.long, device=device
        )
        if module.indexer is not None:
            k_local = run_compressor(module.indexer.compressor, x_ext, detached=True)
            k_global = _AllGatherVarCat.apply(k_local, tuple(counts), group)
            kl_sum = torch.zeros((), dtype=torch.float32, device=device)

            frag_bounds = sorted(
                set([start, start + sq_local] + [b for b in bounds if start < b < start + sq_local])
            )
            for fs, fe in zip(frag_bounds[:-1], frag_bounds[1:]):
                s = next(
                    i for i in range(len(bounds) - 1) if bounds[i] <= fs < bounds[i + 1]
                )
                S = bounds[s]
                nb = (bounds[s + 1] - S) // ratio
                if nb == 0:
                    continue
                sl = slice(fs - start, fe - start)
                x_det = x[sl].detach().unsqueeze(1)
                qr_det = qr[sl].detach().unsqueeze(1)
                q_i, _, w_i = module.indexer.forward_before_topk(
                    x_det, qr_det, row_offset=fs - S
                )
                k_s = k_global[sample_block_offset[s] : sample_block_offset[s] + nb]

                if cfg.backend == "kernel":
                    from .kernels import indexer_select_kernel

                    ids = indexer_select_kernel(
                        q_i, k_s, w_i, cfg.topk, ratio, pos_offset=fs - S
                    ).long()[0]
                else:
                    scores = compute_index_scores(q_i, w_i, k_s)[0]
                    vis = (
                        torch.arange(fs - S + 1, fe - S + 1, device=device).unsqueeze(1)
                        // ratio
                    )
                    mask = torch.where(
                        torch.arange(nb, device=device).unsqueeze(0) >= vis,
                        float("-inf"),
                        0.0,
                    )
                    ids = (scores + mask).topk(min(cfg.topk, nb), dim=-1)[1]

                vis = (
                    torch.arange(fs - S + 1, fe - S + 1, device=device).unsqueeze(1)
                    // ratio
                ).clamp(max=nb)
                ok = (ids >= 0) & (ids < vis)
                if module.training and torch.is_grad_enabled():
                    from .reference import indexer_kl_loss_selected

                    kl_sum = kl_sum + indexer_kl_loss_selected(
                        torch.where(ok, ids, torch.full_like(ids, -1)).unsqueeze(0),
                        q_i,
                        w_i,
                        k_s,
                        query[sl].detach().unsqueeze(1),
                        comp_global[
                            sample_block_offset[s] : sample_block_offset[s] + nb
                        ].detach(),
                        cfg.softmax_scale,
                        module.indexer.softmax_scale,
                        cfg.indexer_loss_coeff,
                        calculate_per_token_loss=True,
                    )
                flat = torch.where(
                    ok,
                    ids + sample_block_offset[s] + comp_base,
                    torch.full_like(ids, -1),
                )
                compress_idxs[sl, : flat.size(1)] = flat
            kl_loss = kl_sum / total
        else:
            widths = []
            frag_bounds = sorted(
                set([start, start + sq_local] + [b for b in bounds if start < b < start + sq_local])
            )
            per_frag = []
            for fs, fe in zip(frag_bounds[:-1], frag_bounds[1:]):
                s = next(
                    i for i in range(len(bounds) - 1) if bounds[i] <= fs < bounds[i + 1]
                )
                S = bounds[s]
                nb = (bounds[s + 1] - S) // ratio
                vis = (
                    torch.arange(fs - S + 1, fe - S + 1, device=device).unsqueeze(1)
                    // ratio
                ).clamp(max=nb)
                colsb = torch.arange(max(nb, 1), device=device).repeat(fe - fs, 1)
                idsf = torch.where(
                    colsb < vis,
                    colsb + sample_block_offset[s] + comp_base,
                    torch.full_like(colsb, -1),
                )
                per_frag.append((slice(fs - start, fe - start), idsf))
                widths.append(idsf.size(1))
            wmax = max(widths) if widths else 1
            compress_idxs = torch.full(
                (sq_local, wmax), -1, dtype=torch.long, device=device
            )
            for sl, idsf in per_frag:
                compress_idxs[sl, : idsf.size(1)] = idsf

        kv_flat = torch.cat([kv_halo, kv, comp_global.squeeze(1)], dim=0)
        topk_idxs = torch.cat([window_idxs, compress_idxs.unsqueeze(0)], dim=-1)
    else:
        kv_flat = torch.cat([kv_halo, kv], dim=0)
        topk_idxs = window_idxs

    output = MagiDSAV4._run_attention(
        query.unsqueeze(1),
        kv_flat.unsqueeze(1),
        attn_sink,
        topk_idxs,
        cfg,
    )
    return output.squeeze(1), kl_loss
