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

from typing import TYPE_CHECKING, Protocol

import torch

from .types import MagiDSAInput, MagiDSAPackedMeta

if TYPE_CHECKING:
    from .runtime import DsaExecutionHandle, MagiDSARuntimeMgr


class MagiDSAProjector(Protocol):
    """Model-owned projection from final-layout hidden rows to DSA inputs."""

    def __call__(
        self,
        local_x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(qr, q, latent_kv)`` in final Query-row order."""


def layout_and_project_dsa_input(
    source_x: torch.Tensor,
    sink: torch.Tensor,
    packed_meta: MagiDSAPackedMeta,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    projector: MagiDSAProjector,
    *,
    detach_indexer_trunk: bool = False,
) -> MagiDSAInput:
    """Route hidden rows once, then run all model-owned local projections.

    Keeping every projection downstream of the same ``local_x`` tensor makes
    autograd merge their partial gradients before the inverse ``TOKEN_LAYOUT``
    collective.  The helper owns no parameters and does not alter detach or
    gradient-reduction semantics.
    """

    local_x = layout_source_hidden_once(source_x, runtime, handle)
    return project_local_dsa_input(
        local_x,
        sink,
        packed_meta,
        runtime,
        handle,
        projector,
        detach_indexer_trunk=detach_indexer_trunk,
    )


def layout_source_hidden_once(
    source_x: torch.Tensor,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
) -> torch.Tensor:
    """Move source-owner hidden rows to the final Query layout exactly once."""

    return runtime.layout_hidden(source_x, handle)


def project_local_dsa_input(
    local_x: torch.Tensor,
    sink: torch.Tensor,
    packed_meta: MagiDSAPackedMeta,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    projector: MagiDSAProjector,
    *,
    detach_indexer_trunk: bool = False,
) -> MagiDSAInput:
    """Project hidden rows that are already in the final Query layout.

    This function never invokes ``TOKEN_LAYOUT``.  Pro model integrations call
    it for every layer after one outer ``layout_source_hidden_once`` boundary.
    """

    if (
        local_x.device != handle.device
        or local_x.dtype != torch.bfloat16
        or not local_x.is_contiguous()
        or local_x.shape
        != (handle.device_plan.local_token_count, runtime.config.hidden_size)
    ):
        raise ValueError(
            "local_x must be contiguous final-layout BF16 on the handle device"
        )
    position_ids = runtime.get_position_ids(handle)
    projected = projector(local_x, position_ids)
    if not isinstance(projected, tuple) or len(projected) != 3:
        raise TypeError("projector must return the tuple (qr, q, latent_kv)")
    qr, q, latent_kv = projected
    config = runtime.config
    local_tokens = handle.device_plan.local_token_count
    expected_shapes = (
        (local_tokens, config.q_lora_rank),
        (local_tokens, config.num_query_heads, config.head_dim),
        (local_tokens, config.head_dim),
    )
    for name, tensor, expected_shape in zip(
        ("qr", "q", "latent_kv"), projected, expected_shapes
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"projector {name} output must be a tensor")
        if tensor.shape != expected_shape:
            raise ValueError(
                f"projector {name} output must have shape {expected_shape}"
            )
        if (
            tensor.device != local_x.device
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"projector {name} output must be contiguous BF16 on local_x.device"
            )
    return MagiDSAInput(
        x=local_x,
        qr=qr,
        q=q,
        latent_kv=latent_kv,
        sink=sink,
        packed_meta=packed_meta,
        detach_indexer_trunk=detach_indexer_trunk,
    )


__all__ = [
    "MagiDSAProjector",
    "layout_and_project_dsa_input",
    "layout_source_hidden_once",
    "project_local_dsa_input",
]
