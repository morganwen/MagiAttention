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

"""Magi_DSA V4 attention runtime (single-GPU SBHD reference path).

One instance serves one layer form, fixed by ``config.compress_ratio``:

- 0: sliding-window only ('W' layers)
- 4: CSA — overlapped 4:1 compression + lightning-indexer top-k + window
- 128: HCA — 128:1 compression, dense over the causal compressed prefix
  + window

All three share the unified sparse attention with a learnable per-head
sink. Forward returns ``(output, kl_loss)`` where ``kl_loss`` is a
differentiable scalar (zero tensor for non-indexer forms so the signature
is uniform). The indexer branch consumes detached inputs: KL trains only
indexer-owned parameters and never the trunk.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .compressor import DSAv4Compressor
from .config import MagiDSAV4Config
from .indexer import DSAv4Indexer, build_block_causal_mask, compute_index_scores
from .reference import (
    get_compress_topk_idxs,
    get_window_topk_idxs,
    indexer_kl_loss,
    sparse_attn_with_sink,
    validate_and_offset_topk,
)


class MagiDSAV4(nn.Module):
    """DeepSeek V4 hybrid sparse attention, reference backend."""

    def __init__(self, config: MagiDSAV4Config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.config = config

        self.compressor: Optional[DSAv4Compressor] = None
        self.indexer: Optional[DSAv4Indexer] = None
        if config.has_compressor:
            self.compressor = DSAv4Compressor(
                config, head_dim=config.kv_dim, rotate=False, dtype=dtype
            )
        if config.has_indexer:
            self.indexer = DSAv4Indexer(config, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        qr: Optional[torch.Tensor],
        query: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SBHD reference forward.

        x (sq, b, hidden): hidden states, feeds the compressors and the
        indexer weights projection. qr (sq, b, q_lora_rank): query latent,
        feeds the indexer; may be None for non-indexer forms. query
        (sq, b, num_heads, kv_dim): RoPE-applied main query. kv (sq, b,
        kv_dim): RoPE-applied single-head latent KV. attn_sink (num_heads,)
        FP32 learnable sink, held by the caller.

        Returns (output (sq, b, num_heads * kv_dim), kl_loss scalar).
        """
        cfg = self.config
        sq, b = x.size(0), x.size(1)
        device = x.device

        kl_loss = torch.zeros((), dtype=torch.float32, device=device)

        window_idxs = get_window_topk_idxs(cfg.window_size, b, sq, device)

        compressed_kv = None
        n_compressed = 0
        if self.compressor is not None:
            compressed_kv = self.compressor(x)
            if compressed_kv is not None:
                n_compressed = compressed_kv.size(0)

        if compressed_kv is not None and n_compressed > 0:
            offset = sq
            kv_full = torch.cat([kv, compressed_kv], dim=0)

            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()
                q_idx, k_idx, w_idx = self.indexer.forward_before_topk(x_det, qr_det)
                causal_mask = build_block_causal_mask(
                    sq, n_compressed, cfg.compress_ratio, b, device
                )
                if self.training and torch.is_grad_enabled():
                    # Predict logits carry the indexer softmax scale in the
                    # loss path only; scale does not change the top-k order.
                    scores = compute_index_scores(
                        q_idx, w_idx * self.indexer.softmax_scale, k_idx
                    )
                    scores = scores + causal_mask
                    topk_indices = self.indexer.select_topk(scores, n_compressed)
                    kl_loss = indexer_kl_loss(
                        scores,
                        topk_indices,
                        query.detach(),
                        compressed_kv.detach(),
                        cfg.softmax_scale,
                        cfg.indexer_loss_coeff,
                        causal_mask,
                        cfg.use_sparse_loss,
                        cfg.calculate_per_token_loss,
                    )
                else:
                    scores = compute_index_scores(q_idx, w_idx, k_idx) + causal_mask
                    topk_indices = self.indexer.select_topk(scores, n_compressed)
                compress_idxs = validate_and_offset_topk(
                    topk_indices, cfg.compress_ratio, offset
                )
            else:
                compress_idxs = get_compress_topk_idxs(
                    cfg.compress_ratio, b, sq, offset, device
                )

            topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)
        else:
            kv_full = kv
            topk_idxs = window_idxs

        output = sparse_attn_with_sink(
            query, kv_full, attn_sink.float(), topk_idxs.int(), cfg.softmax_scale
        )
        return output, kl_loss
