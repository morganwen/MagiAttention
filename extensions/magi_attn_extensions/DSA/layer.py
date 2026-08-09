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

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .config import MagiDSAConfig, MagiDSAProModelSpec
from .nvtx import dsa_nvtx_range
from .types import MagiDSAInput

if TYPE_CHECKING:
    from .pro_runtime import MagiDSAProExecutionBundle, MagiDSAProRuntimeMgr
    from .runtime import DsaExecutionHandle, MagiDSARuntimeMgr


def _apply_fused_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Dispatch the frozen Quack RMSNorm without importing CuTe on CPU paths."""

    from quack.rmsnorm import rmsnorm

    return rmsnorm(tensor, weight=weight, eps=eps)


def _linear_projection(
    tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run production BF16 projection while retaining a CPU reference path."""

    if weight.dtype != torch.bfloat16:
        raise TypeError("DSV4-Pro projection weights must use bfloat16")
    if tensor.is_cuda:
        if tensor.dtype != torch.bfloat16:
            raise TypeError("CUDA DSV4-Pro projection inputs must use bfloat16")
        return F.linear(tensor, weight)
    return F.linear(tensor.float(), weight.float()).to(tensor.dtype)


def _yarn_inverse_frequencies(config: MagiDSAConfig) -> torch.Tensor:
    frequencies = 1.0 / (
        config.compress_rope_theta
        ** (torch.arange(0, config.rope_dim, 2, dtype=torch.float32) / config.rope_dim)
    )
    if config.original_seq_len <= 0:
        return frequencies

    def correction_dim(rotations: int) -> float:
        return (
            config.rope_dim
            * math.log(config.original_seq_len / (rotations * 2 * math.pi))
            / (2 * math.log(config.compress_rope_theta))
        )

    low = max(math.floor(correction_dim(config.rope_beta_fast)), 0)
    high = float(
        min(math.ceil(correction_dim(config.rope_beta_slow)), config.rope_dim - 1)
    )
    if low == high:
        high += 0.001
    ramp = (torch.arange(config.rope_dim // 2, dtype=torch.float32) - low) / (
        high - low
    )
    smooth = 1.0 - ramp.clamp(0.0, 1.0)
    return frequencies / config.rope_factor * (1.0 - smooth) + frequencies * smooth


def apply_dsa_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
    nvtx_scope: str = "rope",
) -> torch.Tensor:
    """Apply sample-relative RoPE to the final dimensions without an in-place view."""

    if tensor.shape[0] != positions.numel():
        raise ValueError("positions must contain one value per tensor row")
    if tensor.shape[-1] < rope_dim or rope_dim % 2:
        raise ValueError("invalid RoPE dimension")
    enabled = tensor.is_cuda
    with dsa_nvtx_range(f"{nvtx_scope}::rope", enabled=enabled):
        prefix, rotary = tensor.split((tensor.shape[-1] - rope_dim, rope_dim), dim=-1)
        with dsa_nvtx_range(f"{nvtx_scope}::rope::angle_generation", enabled=enabled):
            angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequencies.to(
                device=tensor.device
            ).unsqueeze(0)
            if inverse:
                angles = -angles
        with dsa_nvtx_range(f"{nvtx_scope}::rope::sincos", enabled=enabled):
            cos = angles.cos()
            sin = angles.sin()
            for _ in range(tensor.ndim - 2):
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        with dsa_nvtx_range(f"{nvtx_scope}::rope::rotation_hadamard", enabled=enabled):
            even = rotary.float()[..., 0::2]
            odd = rotary.float()[..., 1::2]
            rotated = torch.stack(
                (even * cos - odd * sin, odd * cos + even * sin), dim=-1
            )
        with dsa_nvtx_range(f"{nvtx_scope}::rope::output_assembly", enabled=enabled):
            return torch.cat(
                (prefix, rotated.flatten(-2).to(tensor.dtype)), dim=-1
            ).contiguous()


def apply_normalized_hadamard(tensor: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform on the final dimension."""

    dimension = tensor.shape[-1]
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    dtype = tensor.dtype
    value = tensor.float()
    stride = 1
    while stride < dimension:
        original_shape = value.shape
        grouped = value.reshape(*original_shape[:-1], -1, 2, stride)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        value = torch.cat((left + right, left - right), dim=-1).reshape(original_shape)
        stride *= 2
    return (value * dimension**-0.5).to(dtype)


class DsaRMSNorm(nn.Module):
    """DeepSeek compressor RMSNorm with an FP32 learnable scale."""

    def __init__(
        self, dimension: int, eps: float, *, nvtx_scope: str = "rms_norm"
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.eps = eps
        self.nvtx_scope = nvtx_scope
        self.weight = nn.Parameter(torch.ones(dimension, dtype=torch.float32))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        dtype = tensor.dtype
        enabled = tensor.is_cuda
        scope = f"{self.nvtx_scope}::rms_norm"
        with dsa_nvtx_range(scope, enabled=enabled):
            if tensor.is_cuda:
                with dsa_nvtx_range(f"{scope}::fused_quack", enabled=True):
                    return _apply_fused_rms_norm(tensor, self.weight, self.eps)
            value = tensor.float()
            with dsa_nvtx_range(f"{scope}::variance_reduction", enabled=enabled):
                variance = value.square().mean(dim=-1, keepdim=True)
            with dsa_nvtx_range(f"{scope}::inverse_root", enabled=enabled):
                inverse_rms = torch.rsqrt(variance + self.eps)
            with dsa_nvtx_range(f"{scope}::normalize_hadamard", enabled=enabled):
                value = value * inverse_rms
            with dsa_nvtx_range(f"{scope}::scale_hadamard", enabled=enabled):
                return (value * self.weight).to(dtype)


class DsaCompressor(nn.Module):
    """Trainable CSA overlap or HCA non-overlap compressor."""

    def __init__(
        self,
        config: MagiDSAConfig,
        output_dim: int,
        *,
        overlap: bool,
        rotate_output: bool = False,
        nvtx_scope: str = "compressor",
    ) -> None:
        super().__init__()
        if overlap and config.ratio != 4:
            raise ValueError("only ratio=4 CSA uses overlap compression")
        self.config = config
        self.output_dim = output_dim
        self.overlap = overlap
        self.rotate_output = rotate_output
        self.nvtx_scope = nvtx_scope
        if rotate_output and output_dim & (output_dim - 1):
            raise ValueError("rotated compressor output must have power-of-two width")
        branches = 2 if overlap else 1
        self.wkv = nn.Linear(
            config.hidden_size, branches * output_dim, bias=False, dtype=torch.bfloat16
        )
        self.wgate = nn.Linear(
            config.hidden_size, branches * output_dim, bias=False, dtype=torch.bfloat16
        )
        self.ape = nn.Parameter(
            torch.zeros(config.ratio, branches * output_dim, dtype=torch.float32)
        )
        self.norm = DsaRMSNorm(
            output_dim,
            config.norm_eps,
            nvtx_scope=f"compressor::{nvtx_scope}",
        )
        self.register_buffer(
            "inverse_frequencies", _yarn_inverse_frequencies(config), persistent=False
        )
        self.reset_parameters()

    @property
    def support(self) -> int:
        return self.config.ratio * (2 if self.overlap else 1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wkv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wgate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ape)
        nn.init.ones_(self.norm.weight)

    def forward(
        self,
        packed_x: torch.Tensor,
        valid_rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if packed_x.ndim != 3 or packed_x.shape[1:] != (
            self.support,
            self.config.hidden_size,
        ):
            raise ValueError("packed_x has an invalid compressor shape")
        if valid_rows.shape != packed_x.shape[:2] or valid_rows.dtype != torch.bool:
            raise ValueError("valid_rows must be a bool mask over packed_x rows")
        if positions.shape != (packed_x.shape[0],):
            raise ValueError("positions must contain one compressed-block position")
        dtype = packed_x.dtype
        enabled = packed_x.is_cuda
        scope = f"compressor::{self.nvtx_scope}"
        with dsa_nvtx_range(scope, enabled=enabled):
            with dsa_nvtx_range(f"{scope}::projection_input_cast", enabled=enabled):
                projection_input = packed_x
            with dsa_nvtx_range(f"{scope}::value_projection", enabled=enabled):
                projected_kv = _linear_projection(projection_input, self.wkv.weight)
            with dsa_nvtx_range(f"{scope}::gate_projection", enabled=enabled):
                projected_gate = _linear_projection(projection_input, self.wgate.weight)
            ratio = self.config.ratio
            if self.overlap and packed_x.is_cuda:
                from .kernels.triton.compressor import fused_csa_compressor_reduce

                with dsa_nvtx_range(f"{scope}::post_gemm_fused", enabled=True):
                    compressed = fused_csa_compressor_reduce(
                        projected_kv,
                        projected_gate,
                        self.ape,
                        valid_rows,
                        self.output_dim,
                        dtype,
                    )
            elif self.overlap:
                with dsa_nvtx_range(f"{scope}::overlap_assembly", enabled=enabled):
                    previous_kv = projected_kv[:, :ratio, : self.output_dim]
                    current_kv = projected_kv[:, ratio:, self.output_dim :]
                    previous_gate = (
                        projected_gate[:, :ratio, : self.output_dim]
                        + self.ape[None, :, : self.output_dim]
                    )
                    current_gate = (
                        projected_gate[:, ratio:, self.output_dim :]
                        + self.ape[None, :, self.output_dim :]
                    )
                    values = torch.cat((previous_kv, current_kv), dim=1)
                    logits = torch.cat((previous_gate, current_gate), dim=1)
            else:
                with dsa_nvtx_range(f"{scope}::nonoverlap_assembly", enabled=enabled):
                    values = projected_kv
                    logits = projected_gate + self.ape.unsqueeze(0)
            if not (self.overlap and packed_x.is_cuda):
                with dsa_nvtx_range(f"{scope}::validity_mask", enabled=enabled):
                    logits = logits.masked_fill(
                        ~valid_rows.unsqueeze(-1), float("-inf")
                    )
                with dsa_nvtx_range(f"{scope}::gate_softmax", enabled=enabled):
                    gates = logits.softmax(dim=1)
                with dsa_nvtx_range(f"{scope}::compression_hadamard", enabled=enabled):
                    weighted_values = values * gates
                with dsa_nvtx_range(f"{scope}::support_reduction", enabled=enabled):
                    compressed = weighted_values.sum(dim=1).to(dtype)
            compressed = self.norm(compressed)
            if self.rotate_output:
                with dsa_nvtx_range(f"{scope}::rope_hadamard", enabled=enabled):
                    if compressed.is_cuda:
                        from .kernels.triton.rope import fused_dsa_rope_hadamard

                        with dsa_nvtx_range(
                            f"{scope}::rope_hadamard::fused_triton", enabled=True
                        ):
                            return fused_dsa_rope_hadamard(
                                compressed.unsqueeze(1),
                                positions,
                                self.inverse_frequencies,
                                self.config.rope_dim,
                            )[:, 0]
                    return apply_normalized_hadamard(
                        apply_dsa_rope(
                            compressed,
                            positions,
                            self.inverse_frequencies,
                            self.config.rope_dim,
                            nvtx_scope=scope,
                        )
                    )
            if compressed.is_cuda:
                from .kernels.triton.rope import fused_dsa_rope

                with dsa_nvtx_range(f"{scope}::rope", enabled=True):
                    with dsa_nvtx_range(f"{scope}::rope::fused_triton", enabled=True):
                        return fused_dsa_rope(
                            compressed.unsqueeze(1),
                            positions,
                            self.inverse_frequencies,
                            self.config.rope_dim,
                        )[:, 0]
            return apply_dsa_rope(
                compressed,
                positions,
                self.inverse_frequencies,
                self.config.rope_dim,
                nvtx_scope=scope,
            )


class DsaIndexer(nn.Module):
    """CSA Indexer projections and its independently parameterized compressor."""

    def __init__(self, config: MagiDSAConfig) -> None:
        super().__init__()
        if config.ratio != 4:
            raise ValueError("DsaIndexer is only valid for ratio=4")
        self.config = config
        self.q_proj = nn.Linear(
            config.q_lora_rank,
            config.indexer_heads * config.indexer_head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.weights_proj = nn.Linear(
            config.hidden_size,
            config.indexer_heads,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.compressor = DsaCompressor(
            config,
            config.indexer_head_dim,
            overlap=True,
            rotate_output=True,
            nvtx_scope="indexer",
        )
        self.register_buffer(
            "inverse_frequencies", _yarn_inverse_frequencies(config), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.weights_proj.weight, mean=0.0, std=0.02)

    def project_queries(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        *,
        detach_trunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_x = x.detach() if detach_trunk else x
        branch_qr = qr.detach() if detach_trunk else qr
        enabled = x.is_cuda
        scope = "indexer_projection"
        with dsa_nvtx_range(scope, enabled=enabled):
            with dsa_nvtx_range(f"{scope}::query_projection", enabled=enabled):
                q = _linear_projection(branch_qr, self.q_proj.weight).view(
                    qr.shape[0],
                    self.config.indexer_heads,
                    self.config.indexer_head_dim,
                )
            if q.is_cuda:
                from .kernels.triton.rope import fused_dsa_rope_hadamard

                with dsa_nvtx_range(f"{scope}::query::rope_hadamard", enabled=True):
                    with dsa_nvtx_range(
                        f"{scope}::query::rope_hadamard::fused_triton",
                        enabled=True,
                    ):
                        q = fused_dsa_rope_hadamard(
                            q,
                            positions,
                            self.inverse_frequencies,
                            self.config.rope_dim,
                            output_dtype=qr.dtype,
                        )
            else:
                with dsa_nvtx_range(f"{scope}::query_cast", enabled=enabled):
                    q = q.to(qr.dtype)
                q = apply_normalized_hadamard(
                    apply_dsa_rope(
                        q,
                        positions,
                        self.inverse_frequencies,
                        self.config.rope_dim,
                        nvtx_scope=f"{scope}::query",
                    )
                )
            with dsa_nvtx_range(f"{scope}::weight_projection", enabled=enabled):
                weights = _linear_projection(branch_x, self.weights_proj.weight)
            # Keep the model-side head averaging separate from the Indexer
            # dot-product scale consumed by the cuDNN backend.
            weight_scale = self.config.indexer_heads**-0.5
            with dsa_nvtx_range(f"{scope}::weight_scaling", enabled=enabled):
                if weights.is_cuda:
                    from .kernels.triton.projection import fused_dsa_scale_cast

                    with dsa_nvtx_range(
                        f"{scope}::weight_scaling::fused_triton", enabled=True
                    ):
                        weights = fused_dsa_scale_cast(
                            weights,
                            weight_scale,
                            x.dtype,
                        )
                else:
                    weights = (weights * weight_scale).to(x.dtype)
            return q, weights


class MagiDSALayer(nn.Module):
    """Model-side owner of every trainable Compressor and Indexer parameter."""

    def __init__(
        self,
        config: MagiDSAConfig,
        *,
        layer_id: int | None = None,
    ) -> None:
        super().__init__()
        if layer_id is not None and layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        self.config = config
        self._layer_id = layer_id
        self.compressor = (
            DsaCompressor(
                config,
                config.head_dim,
                overlap=config.ratio == 4,
                nvtx_scope="main",
            )
            if config.has_compressor
            else None
        )
        self.indexer = DsaIndexer(config) if config.has_indexer else None
        self.register_buffer(
            "output_inverse_frequencies",
            _yarn_inverse_frequencies(config),
            persistent=False,
        )

    @property
    def layer_id(self) -> int | None:
        return self._layer_id

    def inverse_output_rope(
        self,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the official per-head output after inverse sample-relative RoPE."""

        if positions.shape != (output.shape[0],):
            raise ValueError(
                "positions must contain one value per attention output row"
            )
        expected_shape = (
            positions.shape[0],
            self.config.num_query_heads,
            self.config.head_dim,
        )
        if output.shape != expected_shape:
            raise ValueError(f"attention output must have shape {expected_shape}")
        scope = "attention::output_inverse_rope"
        if output.is_cuda:
            from .kernels.triton.rope import fused_dsa_rope

            with dsa_nvtx_range(f"{scope}::fused_triton", enabled=True):
                return fused_dsa_rope(
                    output,
                    positions,
                    self.output_inverse_frequencies,
                    self.config.rope_dim,
                    inverse=True,
                )
        return apply_dsa_rope(
            output,
            positions,
            self.output_inverse_frequencies,
            self.config.rope_dim,
            inverse=True,
            nvtx_scope=scope,
        )

    def forward(
        self,
        dsa_input: MagiDSAInput,
        runtime: MagiDSARuntimeMgr,
        handle: DsaExecutionHandle,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = runtime.calc_dsa(self, dsa_input, handle)
        return result.output, result.kl


class MagiDSAProLayerStack(nn.Module):
    """Independent DSA parameter owners for the official 61-layer Pro stack."""

    def __init__(
        self,
        model_spec: MagiDSAProModelSpec | None = None,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.model_spec = model_spec or MagiDSAProModelSpec()
        with torch.device(device):
            self.layers = nn.ModuleList(
                [
                    MagiDSALayer(
                        self.model_spec.make_layer_config(layer_id),
                        layer_id=layer_id,
                    )
                    for layer_id in range(self.model_spec.main_layer_count)
                ]
            )

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_id: int) -> MagiDSALayer:
        return self.layers[layer_id]

    def aggregate_csa_aux_losses(
        self,
        losses: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        """Require and sum exactly one all-Query auxiliary loss per CSA layer."""

        expected_layer_ids = self.model_spec.csa_layer_ids
        if set(losses) != set(expected_layer_ids):
            missing = sorted(set(expected_layer_ids) - set(losses))
            extra = sorted(set(losses) - set(expected_layer_ids))
            raise ValueError(
                "CSA auxiliary losses must cover all 30 layers exactly once; "
                f"missing={missing}, extra={extra}"
            )
        ordered = tuple(losses[layer_id] for layer_id in expected_layer_ids)
        if any(loss.ndim != 0 or loss.dtype != torch.float32 for loss in ordered):
            raise ValueError("every CSA auxiliary loss must be an FP32 scalar")
        first_device = ordered[0].device
        if any(loss.device != first_device for loss in ordered):
            raise ValueError("CSA auxiliary losses must share one device")
        return torch.stack(ordered).sum()

    def forward(
        self,
        layer_id: int,
        dsa_input: MagiDSAInput,
        runtime: MagiDSAProRuntimeMgr,
        bundle: MagiDSAProExecutionBundle,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = runtime.calc_layer(
            layer_id,
            self.layers[layer_id],
            dsa_input,
            bundle,
        )
        return result.output, result.kl


__all__ = [
    "DsaCompressor",
    "DsaIndexer",
    "DsaRMSNorm",
    "MagiDSALayer",
    "MagiDSAProLayerStack",
    "apply_dsa_rope",
    "apply_normalized_hadamard",
]
