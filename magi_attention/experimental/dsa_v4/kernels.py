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
from typing import Optional, Tuple

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
