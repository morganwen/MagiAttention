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

"""Autograd-preserving Magi_DSA orchestration."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from magi_attention.api.dsa_attn_interface import MagiDSAInput
    from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr


def dist_dsa_func(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the fragment plan, then execute the current CP=1 compute path."""

    runtime_mgr.validate_input(dsa_input)
    dispatch_plan = runtime_mgr.get_dispatch_plan(dsa_input.packed_meta)
    if runtime_mgr.plan.cp_size != 1:
        raise NotImplementedError(
            "CP=2 DSA fragment plan is ready "
            f"(hash={dispatch_plan.plan_hash[:12]}); tensor data movement starts "
            "with GroupCast/GroupReduce in step 3"
        )

    output_flat, kl_loss = runtime_mgr.dsa_module.forward_packed(
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
        dsa_input.packed_meta.cu_seqlens,
    )
    expected_flat = (
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads * runtime_mgr.config.kv_dim,
    )
    if tuple(output_flat.shape) != expected_flat:
        raise RuntimeError(
            f"legacy DSA path returned shape {tuple(output_flat.shape)}, "
            f"expected {expected_flat}"
        )
    output = output_flat.reshape(
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads,
        runtime_mgr.config.kv_dim,
    )
    if kl_loss.shape != torch.Size([]) or kl_loss.dtype != torch.float32:
        raise RuntimeError(
            "legacy DSA path must return a scalar FP32 KL loss, "
            f"got shape={tuple(kl_loss.shape)}, dtype={kl_loss.dtype}"
        )
    return output, kl_loss


__all__ = ["dist_dsa_func"]
