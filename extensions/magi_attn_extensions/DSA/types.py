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

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MagiDSAPackedMeta:
    """Global packed metadata and the pre-layout source split of every rank.

    ``source_token_counts`` describes the contiguous source-owner input each
    rank passes to ``MagiDSARuntimeMgr.layout_hidden``. The caller already owns
    that sharding, so stating it here makes the whole execution plan a pure
    function of caller metadata. Every rank then rebuilds a bit-identical plan
    locally, which is why preparing an execution runs no object collective.
    A rank's source count can differ from its final Query-row count after
    ``TOKEN_LAYOUT``.
    """

    cu_seqlens: tuple[int, ...]
    source_token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.cu_seqlens)
        object.__setattr__(self, "cu_seqlens", values)
        counts = tuple(int(value) for value in self.source_token_counts)
        object.__setattr__(self, "source_token_counts", counts)
        if len(values) < 2 or values[0] != 0:
            raise ValueError(
                "cu_seqlens must start at zero and describe at least one sample"
            )
        if any(end < begin for begin, end in zip(values, values[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        if not counts:
            raise ValueError("source_token_counts must contain at least one rank")
        if any(count < 0 for count in counts):
            raise ValueError("source token counts must be non-negative")
        if sum(counts) != values[-1]:
            raise ValueError(
                f"source token counts sum to {sum(counts)}, expected {values[-1]}"
            )

    @property
    def cp_size(self) -> int:
        return len(self.source_token_counts)

    def local_token_count(self, rank: int) -> int:
        """Return the pre-layout source-row count owned by ``rank``."""

        if not 0 <= rank < len(self.source_token_counts):
            raise IndexError("rank is outside the source layout")
        return self.source_token_counts[rank]

    @classmethod
    def from_tensor(
        cls,
        cu_seqlens: torch.Tensor,
        source_token_counts: Sequence[int],
    ) -> MagiDSAPackedMeta:
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
        if cu_seqlens.is_cuda:
            raise ValueError(
                "construct packed metadata from a CPU cu_seqlens tensor on the cold path"
            )
        return cls(
            tuple(int(value) for value in cu_seqlens.tolist()),
            tuple(int(value) for value in source_token_counts),
        )


@dataclass(frozen=True)
class MagiDSAInput:
    """Parameter-free final-Query-local boundary consumed by MagiDSALayer.

    All four token activations are produced from the same hidden state *after*
    the model-boundary ``TOKEN_LAYOUT``. ``packed_meta`` intentionally keeps
    describing the pre-layout source split, since that is what identifies the
    frozen execution plan.
    """

    x: torch.Tensor
    qr: torch.Tensor
    q: torch.Tensor
    latent_kv: torch.Tensor
    sink: torch.Tensor
    packed_meta: MagiDSAPackedMeta
    detach_indexer_trunk: bool = False


@dataclass(frozen=True)
class MagiDSAForwardResult:
    """Internal result with diagnostics required by correctness and profiling."""

    output: torch.Tensor
    kl: torch.Tensor
    sparse_lse: torch.Tensor
    topk_ids: torch.Tensor
    topk_length: torch.Tensor
    indexer_lse: torch.Tensor


__all__ = ["MagiDSAForwardResult", "MagiDSAInput", "MagiDSAPackedMeta"]
