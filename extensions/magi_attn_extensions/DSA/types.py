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

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MagiDSAPackedMeta:
    """Global packed metadata plus this rank's pre-layout source-row count.

    ``local_token_count`` describes the contiguous source-owner input passed to
    ``MagiDSARuntimeMgr.layout_hidden``.  It can differ from the final Query-row
    count carried by an execution handle after ``TOKEN_LAYOUT``.
    """

    cu_seqlens: tuple[int, ...]
    local_token_count: int

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.cu_seqlens)
        object.__setattr__(self, "cu_seqlens", values)
        if len(values) < 2 or values[0] != 0:
            raise ValueError(
                "cu_seqlens must start at zero and describe at least one sample"
            )
        if any(end < begin for begin, end in zip(values, values[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        if self.local_token_count < 0:
            raise ValueError("local_token_count must be non-negative")

    @classmethod
    def from_tensor(
        cls,
        cu_seqlens: torch.Tensor,
        local_token_count: int,
    ) -> MagiDSAPackedMeta:
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
        if cu_seqlens.is_cuda:
            raise ValueError(
                "construct packed metadata from a CPU cu_seqlens tensor on the cold path"
            )
        return cls(
            tuple(int(value) for value in cu_seqlens.tolist()), local_token_count
        )


@dataclass(frozen=True)
class MagiDSAInput:
    """Parameter-free final-Query-local boundary consumed by MagiDSALayer.

    All four token activations are produced from the same hidden state *after*
    the model-boundary ``TOKEN_LAYOUT``.  ``packed_meta.local_token_count``
    intentionally remains the pre-layout source-row count used to identify the
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
