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

"""DSA-aware contiguous dispatch solver for the Magi_DSA V4 CP runtime.

The equal contiguous split (SequentialDispatch) is the baseline. This
solver keeps the contiguity and block alignment the compressor and window
require, but places the cut points so every rank carries the same amount
of the FORM-SPECIFIC work instead of the same number of rows:

per row at in-sample position p (0-indexed):
  window attend  : min(p + 1, window_size)
  compressed     : ratio 4  -> min(topk, (p+1)//ratio) attend
                              + (p+1)//ratio indexer scan
                   ratio 128-> (p+1)//ratio attend
                   ratio 0  -> nothing
  compression    : constant per row (each row is pooled once)

Unlike dense causal attention (quadratic prefix), these curves are flat
to gently increasing, so the solver's corrections are modest — its value
shows on packed batches where short samples truncate windows and reset
prefixes. Costs are a relative proxy, not a flops calculator (same
posture as the upstream packing scheduler).
"""

from typing import List, Optional, Sequence

import torch

from .config import MagiDSAV4Config


def per_row_costs(
    cfg: MagiDSAV4Config,
    total_rows: int,
    cu_seqlens: Optional[Sequence[int]] = None,
    compress_weight: float = 0.25,
    indexer_weight: float = 1.0,
) -> torch.Tensor:
    """FP32 host tensor of per-row relative costs, sample-position aware."""
    bounds = list(cu_seqlens) if cu_seqlens is not None else [0, total_rows]
    ratio = cfg.compress_ratio
    cost = torch.zeros(total_rows, dtype=torch.float32)
    for s in range(len(bounds) - 1):
        lo, hi = bounds[s], bounds[s + 1]
        p = torch.arange(hi - lo, dtype=torch.float32)
        c = torch.clamp(p + 1, max=float(cfg.window_size))
        if ratio > 1:
            vis = torch.div(p + 1, ratio, rounding_mode="floor")
            if ratio == 4:
                c = c + torch.clamp(vis, max=float(cfg.topk))
                c = c + indexer_weight * vis
            else:
                c = c + vis
            c = c + compress_weight
        cost[lo:hi] = c
    return cost


def solve_contiguous_cuts(
    cfg: MagiDSAV4Config,
    world_size: int,
    total_rows: int,
    cu_seqlens: Optional[Sequence[int]] = None,
    align: int = 128,
) -> List[int]:
    """Cut points (ws + 1) equalizing cumulative cost, snapped to ``align``.

    Row totals must be divisible by ``align`` so a valid aligned split
    always exists; cuts are strictly monotonic with fixed endpoints.
    """
    assert total_rows % align == 0, "total rows must be align-divisible"
    cost = per_row_costs(cfg, total_rows, cu_seqlens)
    prefix = torch.cumsum(cost, dim=0)
    total = float(prefix[-1])

    cuts = [0]
    for r in range(1, world_size):
        target = total * r / world_size
        idx = int(torch.searchsorted(prefix, torch.tensor(target)).item())
        snapped = round(idx / align) * align
        lo = cuts[-1] + align
        hi = total_rows - (world_size - r) * align
        cuts.append(int(min(max(snapped, lo), hi)))
    cuts.append(total_rows)
    return cuts


def balance_report(
    cfg: MagiDSAV4Config,
    cuts: Sequence[int],
    cu_seqlens: Optional[Sequence[int]] = None,
) -> str:
    """Human-readable predicted per-rank cost shares for a given split."""
    cost = per_row_costs(cfg, cuts[-1], cu_seqlens)
    shares = [float(cost[cuts[r] : cuts[r + 1]].sum()) for r in range(len(cuts) - 1)]
    mean = sum(shares) / len(shares)
    worst = max(shares) / mean - 1 if mean > 0 else 0.0
    rows = ", ".join(f"r{r}={s:.0f}" for r, s in enumerate(shares))
    return f"predicted cost [{rows}] max/mean-1={worst:.3%}"
