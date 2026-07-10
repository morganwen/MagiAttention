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

"""Public, structured entry point for DeepSeek V4 hybrid attention."""

from dataclasses import dataclass

import torch

from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr


@dataclass(frozen=True)
class DsaPackedMeta:
    """Packed sample boundaries for one DSA invocation.

    ``cu_seqlens`` is the global logical sample layout. In CP=1 its last
    element must equal the number of packed input rows. CP fragment metadata
    is derived and cached by the runtime's dispatch solver; it is deliberately
    not encoded as positional arguments here.
    """

    cu_seqlens: torch.Tensor

    def validate(self, total_tokens: int | None = None) -> None:
        if not isinstance(self.cu_seqlens, torch.Tensor):
            raise TypeError("packed_meta.cu_seqlens must be a torch.Tensor")
        if self.cu_seqlens.ndim != 1:
            raise ValueError(
                "packed_meta.cu_seqlens must be 1-D, "
                f"got shape {tuple(self.cu_seqlens.shape)}"
            )
        if self.cu_seqlens.dtype != torch.int32:
            raise TypeError(
                "packed_meta.cu_seqlens must have dtype torch.int32, "
                f"got {self.cu_seqlens.dtype}"
            )
        if not self.cu_seqlens.is_contiguous():
            raise ValueError("packed_meta.cu_seqlens must be contiguous")
        if self.cu_seqlens.numel() < 2:
            raise ValueError("packed_meta.cu_seqlens must contain at least [0, T]")

        # Packed metadata is host-inspected when building the static plan.
        # This does not touch device-resident top-k state.
        bounds = self.cu_seqlens.detach().cpu()
        if int(bounds[0]) != 0:
            raise ValueError("packed_meta.cu_seqlens[0] must be 0")
        if bool(torch.any(bounds[1:] < bounds[:-1])):
            raise ValueError("packed_meta.cu_seqlens must be non-decreasing")
        if total_tokens is not None and int(bounds[-1]) != total_tokens:
            raise ValueError(
                "packed_meta.cu_seqlens[-1] must equal the packed token count "
                f"({total_tokens}), got {int(bounds[-1])}"
            )

    @property
    def num_samples(self) -> int:
        return max(self.cu_seqlens.numel() - 1, 0)

    @property
    def total_tokens(self) -> int:
        self.validate()
        return int(self.cu_seqlens.detach().cpu()[-1])


@dataclass(frozen=True)
class MagiDSAInput:
    """All tensors consumed by one packed Magi_DSA forward call.

    Shapes are ``x[T, hidden]``, ``qr[T, q_lora_rank]``,
    ``q[T, 64, 512]``, ``latent_kv[T, 512]`` and ``sink[64]``.  Runtime
    validation enforces CUDA BF16 row tensors and an FP32 sink.
    """

    x: torch.Tensor
    qr: torch.Tensor
    q: torch.Tensor
    latent_kv: torch.Tensor
    sink: torch.Tensor
    packed_meta: DsaPackedMeta


def calc_dsa(
    dsa_input: MagiDSAInput,
    runtime_mgr: MagiDSARuntimeMgr,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed DeepSeek V4 hybrid attention through ``runtime_mgr``.

    Returns an output of shape ``[T, 64, 512]`` and a differentiable FP32 KL
    scalar.  The manager owns compressor and Indexer parameters.
    """

    if not isinstance(runtime_mgr, MagiDSARuntimeMgr):
        raise TypeError("runtime_mgr must be a MagiDSARuntimeMgr")
    return runtime_mgr.calc_dsa(dsa_input)


__all__ = ["DsaPackedMeta", "MagiDSAInput", "MagiDSARuntimeMgr", "calc_dsa"]
