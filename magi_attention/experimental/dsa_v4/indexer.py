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

"""Lightning indexer for the Magi_DSA V4 runtime (CSA layers, ratio 4).

Faithful pure-PyTorch port of the Megatron dsv4 ``CSAIndexer`` (csa.py,
commit c6449f0b2). The indexer is a detached side branch: its inputs are
detached hidden states and query latents, so the KL auxiliary loss trains
only the indexer-owned parameters and never propagates into the trunk.

Scoring: ``score[q, s] = sum_h weights[q, h] * relu(Qi[q, h] . Ki[s])``
computed in FP32; top-k is taken over the compressed-KV axis under the
block-level causal mask (position p sees ``(p + 1) // ratio`` blocks).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .compressor import DSAv4Compressor, rotate_activation
from .config import MagiDSAV4Config
from .rope import apply_rope_last_dims, build_yarn_freqs


def compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """FP32 index scores [b, sq, sk] from q [sq,b,H,D], weights [sq,b,H], k [sk,b,D]."""
    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1).float()
    return scores.sum(dim=2).transpose(0, 1)


def build_block_causal_mask(
    sq: int, n_compressed: int, ratio: int, batch: int, device: torch.device
) -> torch.Tensor:
    """[b, sq, n_compressed] additive mask: -inf where block is not yet fully past.

    Query position p (0-indexed) sees the first ``(p + 1) // ratio`` blocks.
    """
    cols = torch.arange(n_compressed, device=device).unsqueeze(0).expand(sq, -1)
    visible = torch.arange(1, sq + 1, device=device).unsqueeze(1) // ratio
    mask = torch.where(cols >= visible, float("-inf"), 0.0)
    return mask.unsqueeze(0).expand(batch, -1, -1)


class DSAv4Indexer(nn.Module):
    """Learned top-k retrieval over compressed positions."""

    def __init__(
        self, config: MagiDSAV4Config, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        assert config.has_indexer, "indexer requires compress_ratio == 4"
        self.config = config
        self.n_heads = config.indexer_heads
        self.head_dim = config.indexer_dim
        self.topk = config.topk
        self.rope_dim = config.rope_dim
        self.softmax_scale: float = self.head_dim**-0.5

        self.linear_wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.linear_weights_proj = nn.Linear(
            config.hidden_size, self.n_heads, bias=False, dtype=dtype
        )
        self.compressor = DSAv4Compressor(
            config, head_dim=self.head_dim, rotate=True, dtype=dtype
        )

        self._freqs_cache: Optional[torch.Tensor] = None
        self._freqs_cache_len: int = 0

    def _q_freqs(self, sq: int, device: torch.device) -> torch.Tensor:
        if self._freqs_cache is None or self._freqs_cache_len < sq:
            self._freqs_cache = build_yarn_freqs(
                self.rope_dim, sq, self.config.yarn, device
            )
            self._freqs_cache_len = sq
        return self._freqs_cache[:sq]

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        row_offset: int = 0,
        block_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Project detached inputs into indexer Q, compressed K, head weights.

        x: (sq, b, hidden); qr: (sq, b, q_lora_rank). ``row_offset`` and
        ``block_offset`` shift the RoPE positions of Q rows and compressed
        blocks for context-parallel callers.
        Returns q (sq, b, H, D), k (sq // ratio, b, D) or None, weights (sq, b, H).
        """
        q, weights = self.project_queries(x, qr, row_offset=row_offset)
        k = self.compressor(x, block_offset=block_offset)
        return q, k, weights

    def project_queries(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        row_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one sample-relative query fragment without recompressing K.

        Fragment-balanced context parallelism computes compressed Indexer keys
        on their block owners and broadcasts them separately.  This helper
        keeps the Q/weight projection shared with :meth:`forward_before_topk`
        while allowing the distributed runtime to consume those global keys.
        """
        sq, bsz, _ = x.size()
        q = self.linear_wq_b(qr).reshape(sq, bsz, self.n_heads, self.head_dim)
        freqs = self._q_freqs(row_offset + sq, x.device)[row_offset:]
        q = apply_rope_last_dims(q, freqs, self.rope_dim)
        q = rotate_activation(q)

        weights = self.linear_weights_proj(x) * (self.n_heads**-0.5)
        return q, weights

    @torch.no_grad()
    def select_topk(
        self, index_scores: torch.Tensor, n_compressed: int
    ) -> torch.Tensor:
        """Top-k block ids [b, sq, k] from masked FP32 scores [b, sq, n_compressed]."""
        effective_topk = min(self.topk, n_compressed)
        return index_scores.topk(effective_topk, dim=-1)[1]
