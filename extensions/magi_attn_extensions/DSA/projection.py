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

"""The parameter boundary between Magi-DSA and the model.

Magi-DSA owns no trainable state. Everything with a weight in it -- the Indexer
query and score-weight projections, the Indexer Compressor and the main
Compressor -- is supplied by the model as a callback, exactly as Magi-MSA takes
its projections through ``MsaScheduleOps``. The extension schedules those
callbacks around its collectives and kernels; it never constructs them, never
holds their parameters, and never reduces their gradients.

The callbacks run on tensors that are already in final Query-row order, so a
model implementation needs no knowledge of the CP layout beyond the sample
positions the runtime hands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class DsaIndexerProjection(Protocol):
    """Project local hidden rows into Indexer queries and score weights."""

    def __call__(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        *,
        detach_trunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_indexer, score_weights)`` for the local Query rows."""


class DsaCompression(Protocol):
    """Reduce one padded support group into a single compressed row."""

    def __call__(
        self,
        packed_x: torch.Tensor,
        valid_rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one compressed row per padded support group."""


class DsaOutputTransform(Protocol):
    """Undo the per-head RoPE rotation the attention backend leaves in place."""

    def __call__(
        self, output: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DsaProjections:
    """Model-owned callbacks a DSA layer needs, bundled for one attention call.

    ``indexer_project`` and ``indexer_compress`` are required for CSA and unused
    for HCA, which has no Indexer. ``main_compress`` and ``inverse_output_rope``
    are required for both.
    """

    main_compress: DsaCompression
    inverse_output_rope: DsaOutputTransform
    indexer_project: DsaIndexerProjection | None = None
    indexer_compress: DsaCompression | None = None

    def require_indexer(
        self,
    ) -> tuple[DsaIndexerProjection, DsaCompression]:
        """Return the two CSA-only callbacks, or explain what is missing."""

        if self.indexer_project is None or self.indexer_compress is None:
            raise ValueError(
                "a CSA layer needs both indexer_project and indexer_compress; "
                "HCA layers may omit them"
            )
        return self.indexer_project, self.indexer_compress


__all__ = [
    "DsaCompression",
    "DsaIndexerProjection",
    "DsaOutputTransform",
    "DsaProjections",
]
