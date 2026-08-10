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

"""The owner-local Magi-DSA schedule, forward and backward.

Both directions are ordinary Python. The model callbacks run as detached
subgraphs the schedule drives itself, so when a reverse collective is launched
and when it is waited on is decided by where the call sits in this file, not by
the order the autograd engine happens to reach nodes in. That is the whole
reason the schedule can put independent model-side backward work under a route
that is still in flight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from magi_attention.common.range_op import range_gather

from .backend import dsa_csa_attention_kl, dsa_sparse_attention, run_grouped_dsa_indexer
from .comm import (
    DsaRouteTransfer,
    finish_dsa_reverse_route,
    start_dsa_reverse_route,
    start_dsa_tensor_route,
)
from .config import MagiDSAConfig
from .kernels.triton.indices import build_csa_index_tensors
from .nvtx import dsa_nvtx_range
from .packing import compressor_support_rows, gather_compressor_support
from .projection import DsaProjections
from .schedule import DsaScheduleNode
from .types import MagiDSAForwardResult, MagiDSAInput

if TYPE_CHECKING:
    from .runtime import DsaExecutionHandle


def _validate_inputs(
    config: MagiDSAConfig, dsa_input: MagiDSAInput, handle: DsaExecutionHandle
) -> torch.Tensor:
    local_tokens = handle.device_plan.local_token_count
    expected = {
        "x": (dsa_input.x, (local_tokens, config.hidden_size)),
        "qr": (dsa_input.qr, (local_tokens, config.q_lora_rank)),
        "q": (dsa_input.q, (local_tokens, config.num_query_heads, config.head_dim)),
    }
    for name, (tensor, shape) in expected.items():
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA BF16")
    latent_kv = dsa_input.latent_kv
    if latent_kv.shape == (local_tokens, 1, config.head_dim):
        latent_kv = latent_kv[:, 0]
    if latent_kv.shape != (local_tokens, config.head_dim):
        raise ValueError("latent_kv has an invalid owner-local shape")
    if (
        latent_kv.dtype != torch.bfloat16
        or not latent_kv.is_cuda
        or not latent_kv.is_contiguous()
    ):
        raise ValueError("latent_kv must be contiguous CUDA BF16")
    if (
        dsa_input.sink.shape != (config.num_query_heads,)
        or dsa_input.sink.dtype != torch.float32
    ):
        raise ValueError("sink must be FP32 with one value per query head")
    if not dsa_input.sink.is_cuda or not dsa_input.sink.is_contiguous():
        raise ValueError("sink must be contiguous CUDA FP32")
    return latent_kv


def _build_attention_indices(
    handle: DsaExecutionHandle,
    config: MagiDSAConfig,
    attention_mode: str,
    raw_bank_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the HCA per-query window and compressed runs into flat indices.

    Both regions are contiguous runs in their consumer bank, so the padded
    index matrix is generated from a base and a length instead of being read
    out of a resident per-query table.
    """

    attention = handle.device_plan.attention
    device = attention.window_base.device
    window_length = attention.window_length
    compressed_length = attention.compressed_length
    # Width is a plan-time constant, so building the indices needs no device
    # reduction and therefore no host synchronization on the warm path.
    logical_width = config.window_size + attention.max_compressed_length
    width = ((logical_width + 127) // 128) * 128
    enabled = window_length.is_cuda
    scope = f"attention::{attention_mode}::attention_indices"
    with dsa_nvtx_range(scope, enabled=enabled):
        with dsa_nvtx_range(f"{scope}::column_masks", enabled=enabled):
            columns = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
            window_valid = columns < window_length.unsqueeze(1)
            compressed_columns = columns - window_length.unsqueeze(1)
            compressed_valid = (compressed_columns >= 0) & (
                compressed_columns < compressed_length.unsqueeze(1)
            )
        with dsa_nvtx_range(f"{scope}::window_expand", enabled=enabled):
            window_rows = attention.window_base.unsqueeze(1) + columns
        with dsa_nvtx_range(f"{scope}::compressed_expand", enabled=enabled):
            compressed_rows = (
                raw_bank_rows
                + attention.compressed_base.unsqueeze(1)
                + compressed_columns
            )
        with dsa_nvtx_range(f"{scope}::sentinel_merge", enabled=enabled):
            indices = torch.where(
                window_valid,
                window_rows,
                torch.where(
                    compressed_valid,
                    compressed_rows,
                    torch.full((), -1, dtype=torch.int32, device=device),
                ),
            )
        with dsa_nvtx_range(f"{scope}::length_update", enabled=enabled):
            lengths = window_length + compressed_length
        return indices.contiguous(), lengths.contiguous()


def _build_csa_indices(
    handle: DsaExecutionHandle,
    config: MagiDSAConfig,
    attention_mode: str,
    topk_global_ids: torch.Tensor,
    topk_lengths: torch.Tensor,
    raw_bank_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_plan = handle.device_plan
    indexer = device_plan.indexer
    if indexer is None:
        raise RuntimeError("CSA plan is missing Indexer metadata")
    scope = f"attention::{attention_mode}::attention_indices"
    with dsa_nvtx_range(
        f"{scope}::fused_dual_map",
        enabled=topk_global_ids.is_cuda,
    ):
        return build_csa_index_tensors(
            topk_global_ids,
            topk_lengths,
            device_plan.attention.compressed_global_to_consumer,
            indexer.ki_global_to_consumer,
            device_plan.attention.window_base,
            device_plan.attention.window_length,
            config.window_size,
            raw_bank_rows,
        )


class _DsaScheduler:
    """State shared by the two mode schedules and their backward passes."""

    # CSA trains its Indexer against the attention distribution and returns a
    # real KL scalar. HCA has no Indexer, so its KL is a constant and must not
    # carry a gradient edge at all.
    kl_is_differentiable = True

    def __init__(
        self,
        config: MagiDSAConfig,
        projections: DsaProjections,
        dsa_input: MagiDSAInput,
        handle: DsaExecutionHandle,
        cp_group: dist.ProcessGroup | None,
    ) -> None:
        self.config = config
        self.projections = projections
        self.handle = handle
        self.cp_group = cp_group
        self.attention_mode = {4: "csa", 128: "hca"}[config.ratio]
        self.detach_indexer_trunk = dsa_input.detach_indexer_trunk
        # Read here, outside the autograd node, because grad mode is always
        # off once ``Function.forward`` is running.
        self.schedule_enabled = torch.is_grad_enabled()
        device_plan = handle.device_plan
        self.device_plan = device_plan
        if (
            device_plan.overlap_x_route is None
            or device_plan.compressed_kv_route is None
            or device_plan.compression is None
            or device_plan.window_route is None
        ):
            raise RuntimeError("compressed DSA plan is missing mandatory routes")
        self.overlap_route = device_plan.overlap_x_route
        self.window_route = device_plan.window_route
        self.compressed_kv_route = device_plan.compressed_kv_route
        self.compression = device_plan.compression
        # Kept so the backward can reverse the support gather with the very
        # same index that produced it.
        self.support_rows: torch.Tensor | None = None
        self.support_dtype: torch.dtype | None = None
        self.window_rows = 0
        self.overlap_rows = 0
        self.pending: list[DsaRouteTransfer] = []
        self.rope_node: DsaScheduleNode | None = None
        self.kv_node: DsaScheduleNode | None = None
        self.attention_node: DsaScheduleNode | None = None
        # Non-differentiable results the caller reads after the autograd node.
        self.sparse_lse: torch.Tensor | None = None
        self.topk_global_ids: torch.Tensor | None = None
        self.topk_lengths: torch.Tensor | None = None
        self.indexer_lse: torch.Tensor | None = None

    def _scatter_support_gradient(
        self, grad_packed: torch.Tensor | None
    ) -> torch.Tensor:
        """Adjoint of the support gather: accumulate onto the OVERLAP_X rows.

        A rank that produced no compressed block gets no support gradient back,
        and it still has to launch its OVERLAP_X reverse. Whether a collective
        runs must never depend on whether a gradient happens to exist, so a
        missing gradient becomes zeros here rather than an early return.
        """

        assert self.support_rows is not None
        assert self.support_dtype is not None
        hidden_size = self.config.hidden_size
        with dsa_nvtx_range(
            f"packing::{self.attention_mode}::compression_support::backward_scatter"
        ):
            grad_overlap_x = torch.zeros(
                (self.overlap_rows, hidden_size),
                dtype=self.support_dtype,
                device=self.support_rows.device,
            )
            if grad_packed is not None:
                grad_overlap_x.index_add_(
                    0, self.support_rows, grad_packed.reshape(-1, hidden_size)
                )
        return grad_overlap_x

    def _split_bank_gradient(
        self, grad_kv_bank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with dsa_nvtx_range(
            f"attention::{self.attention_mode}::kv_bank_split",
            enabled=grad_kv_bank.is_cuda,
        ):
            grad_window = grad_kv_bank[: self.window_rows].contiguous()
            grad_compressed = grad_kv_bank[self.window_rows :].contiguous()
        return grad_window, grad_compressed

    def _drain_pending(self) -> None:
        for transfer in self.pending:
            try:
                transfer.wait()
            except BaseException:  # noqa: BLE001 - draining a failed schedule
                pass
        self.pending.clear()


class _CsaScheduler(_DsaScheduler):
    """CSA: Indexer projection and both Compressors around four routes."""

    def __init__(self, config, projections, dsa_input, handle, cp_group) -> None:
        super().__init__(config, projections, dsa_input, handle, cp_group)
        if (
            self.device_plan.compressed_ki_route is None
            or self.device_plan.indexer is None
        ):
            raise RuntimeError("CSA plan is missing Indexer routes or metadata")
        self.compressed_ki_route = self.device_plan.compressed_ki_route
        self.indexer_map = self.device_plan.indexer
        self.indexer_node: DsaScheduleNode | None = None
        self.ki_node: DsaScheduleNode | None = None

    def forward(self, x, qr, q, latent_kv, sink):
        config = self.config
        device_plan = self.device_plan
        handle = self.handle
        cp_group = self.cp_group
        mode = self.attention_mode
        indexer_project, indexer_compress = self.projections.require_indexer()
        caller_stream = torch.cuda.current_stream(x.device)
        route_stream = handle.csa_route_stream
        indexer_stream = handle.csa_indexer_stream
        main_stream = handle.csa_main_stream
        if route_stream is None or indexer_stream is None or main_stream is None:
            raise RuntimeError("CSA execution is missing its overlap streams")
        caller_ready = torch.cuda.Event()
        caller_ready.record(caller_stream)

        try:
            route_stream.wait_event(caller_ready)
            with torch.cuda.stream(route_stream):
                latent_kv.record_stream(route_stream)
                x.record_stream(route_stream)
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::support_routes_launch"
                ):
                    window_transfer = start_dsa_tensor_route(
                        latent_kv, self.window_route, cp_group, attention_mode=mode
                    )
                    self.pending.append(window_transfer)
                    overlap_transfer = start_dsa_tensor_route(
                        x, self.overlap_route, cp_group, attention_mode=mode
                    )
                    self.pending.append(overlap_transfer)

            # The Indexer projection is the compute that hides both routes.
            indexer_stream.wait_event(caller_ready)
            with torch.cuda.stream(indexer_stream):
                x.record_stream(indexer_stream)
                qr.record_stream(indexer_stream)
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::indexer_projection_launch"
                ):
                    self.indexer_node = DsaScheduleNode(
                        lambda hidden, query_rope: indexer_project(
                            hidden,
                            query_rope,
                            device_plan.local_q_positions,
                            detach_trunk=self.detach_indexer_trunk,
                        ),
                        enabled=self.schedule_enabled,
                    )
                    q_indexer, weights = self.indexer_node.forward(x, qr)

            with torch.cuda.stream(route_stream):
                with dsa_nvtx_range("route::attention::csa::support_routes_wait"):
                    overlap_x = overlap_transfer.wait()
                    window_kv = window_transfer.wait()
                routes_ready = torch.cuda.Event()
                routes_ready.record(route_stream)
            self.pending.clear()
            caller_stream.wait_event(routes_ready)
            overlap_x.record_stream(caller_stream)
            window_kv.record_stream(caller_stream)

            self.overlap_rows = overlap_x.shape[0]
            self.window_rows = window_kv.shape[0]
            self.support_rows = compressor_support_rows(
                self.overlap_rows,
                self.compression,
                config.compressor_support,
                overlap_x.device,
            )
            self.support_dtype = overlap_x.dtype
            with dsa_nvtx_range(
                "packing::csa::compression_support::forward_gather",
                enabled=overlap_x.is_cuda,
            ):
                packed = gather_compressor_support(
                    overlap_x, self.compression, config.compressor_support,
                    self.support_rows,
                )
            valid_rows = self.compression.valid_rows
            block_positions = self.compression.block_positions

            # The Indexer Compressor feeds the KI route, so it goes first and
            # the Main Compressor runs while KI is in flight.
            self.ki_node = DsaScheduleNode(
                indexer_compress,
                input_requires_grad=(not self.detach_indexer_trunk, False, False),
                enabled=self.schedule_enabled,
            )
            compressed_ki_local = self.ki_node.forward(
                packed, valid_rows, block_positions
            )
            ki_transfer = start_dsa_tensor_route(
                compressed_ki_local, self.compressed_ki_route, cp_group,
                attention_mode=mode,
            )
            self.pending.append(ki_transfer)

            main_ready_in = torch.cuda.Event()
            main_ready_in.record(caller_stream)
            main_stream.wait_event(main_ready_in)
            with torch.cuda.stream(main_stream):
                packed.record_stream(main_stream)
                valid_rows.record_stream(main_stream)
                block_positions.record_stream(main_stream)
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::main_compressor_launch"
                ):
                    self.kv_node = DsaScheduleNode(
                        self.projections.main_compress,
                        input_requires_grad=(True, False, False),
                        enabled=self.schedule_enabled,
                    )
                    compressed_kv_local = self.kv_node.forward(
                        packed, valid_rows, block_positions
                    )
                    kv_transfer = start_dsa_tensor_route(
                        compressed_kv_local, self.compressed_kv_route, cp_group,
                        attention_mode=mode,
                    )
                    self.pending.append(kv_transfer)
                    compressed_kv = kv_transfer.wait()
                main_ready = torch.cuda.Event()
                main_ready.record(main_stream)

            compressed_ki = ki_transfer.wait()

            grouped_ready_in = torch.cuda.Event()
            grouped_ready_in.record(caller_stream)
            indexer_stream.wait_event(grouped_ready_in)
            # The grouped Indexer waits for the compressed-KV route instead of
            # running under it. Letting the two share the device costs more in
            # SM contention than the route would cost exposed: the Indexer
            # kernel measurably slows down, and by an amount that tracks how
            # much of the route each rank happens to own, which turns a
            # balanced dispatch into an unbalanced profile.
            indexer_stream.wait_event(main_ready)
            with torch.cuda.stream(indexer_stream):
                compressed_ki.record_stream(indexer_stream)
                with (
                    torch.no_grad(),
                    dsa_nvtx_range(
                        "packing::csa::indexer_key_support::forward_copy",
                        enabled=compressed_ki.is_cuda,
                    ),
                ):
                    # Each fragment's grouped-K prefix is a contiguous run of
                    # the unique KI bank, so this is a plain range gather.
                    # Indexer selection is no-grad, so it has no backward.
                    grouped_k = range_gather(
                        compressed_ki,
                        self.indexer_map.k_gather_ranges,
                        total_size=self.indexer_map.packed_k_rows,
                    )
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::grouped_indexer_launch"
                ):
                    selection = run_grouped_dsa_indexer(
                        q_indexer, grouped_k, weights, self.indexer_map, config
                    )
                selection_ready = torch.cuda.Event()
                selection_ready.record(indexer_stream)
            caller_stream.wait_event(selection_ready)
            caller_stream.wait_event(main_ready)
            self.pending.clear()
            for produced in (
                grouped_k,
                selection.global_ids,
                selection.lengths,
                selection.lse,
                compressed_kv,
            ):
                produced.record_stream(caller_stream)
        except BaseException:
            self._drain_pending()
            caller_stream.wait_stream(route_stream)
            caller_stream.wait_stream(indexer_stream)
            caller_stream.wait_stream(main_stream)
            raise

        with dsa_nvtx_range(
            f"attention::{mode}::kv_bank_assembly", enabled=q.is_cuda
        ):
            kv_bank = torch.cat((window_kv, compressed_kv), dim=0).contiguous()
        (
            attention_indices,
            attention_lengths,
            indexer_indices,
            attention_compressed_indices,
        ) = _build_csa_indices(
            handle, config, mode, selection.global_ids, selection.lengths,
            self.window_rows,
        )
        if handle.sparse_backward_stream is None:
            raise RuntimeError("CSA execution is missing its sparse backward stream")
        local_loss_coeff = (
            config.kl_loss_coeff * device_plan.local_token_count
            / handle.plan.total_tokens
        )

        attention_input_ready = torch.cuda.Event()
        attention_input_ready.record(caller_stream)
        indexer_stream.wait_event(attention_input_ready)
        try:
            with torch.cuda.stream(indexer_stream):
                for attention_input in (
                    q, kv_bank, sink, attention_indices, attention_lengths,
                    q_indexer, weights, compressed_ki_local, compressed_ki,
                    compressed_kv, selection.global_ids, selection.lengths,
                    selection.lse, indexer_indices, attention_compressed_indices,
                ):
                    attention_input.record_stream(indexer_stream)
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::combined_attention_kl_launch"
                ):
                    self.attention_node = DsaScheduleNode(
                        lambda q_, kv_, sink_, qi_, w_, ki_: dsa_csa_attention_kl(
                            q_, kv_, sink_, attention_indices, attention_lengths,
                            qi_, w_, ki_, compressed_ki, compressed_kv, selection,
                            indexer_indices, attention_compressed_indices,
                            self.compressed_ki_route, cp_group,
                            handle.sparse_backward_stream,
                            loss_coeff=local_loss_coeff, config=config,
                        ),
                        input_requires_grad=(
                            q.requires_grad, True, sink.requires_grad,
                            q_indexer.requires_grad, weights.requires_grad,
                            compressed_ki_local.requires_grad,
                        ),
                        enabled=self.schedule_enabled,
                    )
                    output, kl, sparse_lse = self.attention_node.forward(
                        q, kv_bank, sink, q_indexer, weights, compressed_ki_local
                    )
                attention_ready = torch.cuda.Event()
                attention_ready.record(indexer_stream)
        except BaseException:
            caller_stream.wait_stream(indexer_stream)
            raise
        caller_stream.wait_event(attention_ready)
        for produced in (output, kl, sparse_lse):
            produced.record_stream(caller_stream)

        with dsa_nvtx_range(f"attention::{mode}::output_inverse_rope", enabled=q.is_cuda):
            self.rope_node = DsaScheduleNode(
                self.projections.inverse_output_rope,
                enabled=self.schedule_enabled,
            )
            output = self.rope_node.forward(output, device_plan.local_q_positions)

        self.sparse_lse = sparse_lse
        self.topk_global_ids = selection.global_ids
        self.topk_lengths = selection.lengths
        self.indexer_lse = selection.lse
        return output, kl

    def backward(self, grad_output, grad_kl):
        """CSA backward, in the order the routes are actually launched.

        The two compressor backwards and the Indexer projection backward are
        the only work independent of a route, so they are placed under the
        three reverse collectives: the Indexer Compressor covers WINDOW_KV and
        COMPRESSED_KV, and the Indexer projection covers OVERLAP_X, which is
        otherwise terminal.
        """

        assert self.rope_node is not None
        assert self.attention_node is not None
        assert self.ki_node is not None
        assert self.kv_node is not None
        assert self.indexer_node is not None
        mode = self.attention_mode
        cp_group = self.cp_group

        with dsa_nvtx_range(f"attention::{mode}::backward_overlap::inverse_rope"):
            (grad_attention_output, _) = self.rope_node.backward((grad_output,))

        # COMPRESSED_KI reverse and the sparse-attention backward overlap
        # inside this node; it hands back gradients for every CSA producer.
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::combined_attention_kl_backward"
        ):
            (
                grad_q,
                grad_kv_bank,
                grad_sink,
                grad_q_indexer,
                grad_weights,
                grad_ki_local,
            ) = self.attention_node.backward((grad_attention_output, grad_kl, None))

        grad_window_kv, grad_compressed_kv = self._split_bank_gradient(grad_kv_bank)
        # COMPRESSED_KV goes out first because it is the one on the critical
        # path: nothing downstream can move until the Main Compressor has its
        # gradient. WINDOW_KV is waited last, so it is launched last and costs
        # the critical route no bandwidth.
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::compressed_kv_reverse_launch"
        ):
            kv_reverse = start_dsa_reverse_route(
                grad_compressed_kv, self.compressed_kv_route, cp_group,
                attention_mode=mode,
            )

        with dsa_nvtx_range(
            "attention::csa::backward_overlap::indexer_compressor_backward"
        ):
            (grad_packed_ki, _, _) = self.ki_node.backward((grad_ki_local,))

        with dsa_nvtx_range(
            "attention::csa::backward_overlap::compressed_kv_reverse_wait"
        ):
            grad_compressed_kv_local = finish_dsa_reverse_route(kv_reverse)
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::main_compressor_backward"
        ):
            (grad_packed, _, _) = self.kv_node.backward((grad_compressed_kv_local,))
        if grad_packed_ki is not None:
            grad_packed = grad_packed + grad_packed_ki

        grad_overlap_x = self._scatter_support_gradient(grad_packed)
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::support_reverse_launch"
        ):
            overlap_reverse = start_dsa_reverse_route(
                grad_overlap_x, self.overlap_route, cp_group, attention_mode=mode
            )
            window_reverse = start_dsa_reverse_route(
                grad_window_kv, self.window_route, cp_group, attention_mode=mode
            )

        # The Indexer projection backward is the last route-independent work,
        # so it is what hides both of the reverses above.
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::indexer_projection_backward"
        ):
            (grad_x_indexer, grad_qr) = self.indexer_node.backward(
                (grad_q_indexer, grad_weights)
            )

        with dsa_nvtx_range(
            "attention::csa::backward_overlap::support_reverse_wait"
        ):
            grad_latent_kv = finish_dsa_reverse_route(window_reverse)
            grad_x = finish_dsa_reverse_route(overlap_reverse)
        if grad_x_indexer is not None:
            grad_x = grad_x + grad_x_indexer
        return grad_x, grad_qr, grad_q, grad_latent_kv, grad_sink


class _HcaScheduler(_DsaScheduler):
    """HCA: one Compressor and three routes, with no Indexer at all."""

    kl_is_differentiable = False

    def forward(self, x, qr, q, latent_kv, sink):
        del qr
        config = self.config
        handle = self.handle
        cp_group = self.cp_group
        mode = self.attention_mode
        caller_stream = torch.cuda.current_stream(x.device)
        route_stream = handle.hca_route_stream
        main_stream = handle.hca_main_stream
        if route_stream is None or main_stream is None:
            raise RuntimeError("HCA execution is missing its overlap streams")
        caller_ready = torch.cuda.Event()
        caller_ready.record(caller_stream)

        try:
            route_stream.wait_event(caller_ready)
            with torch.cuda.stream(route_stream):
                x.record_stream(route_stream)
                latent_kv.record_stream(route_stream)
                with dsa_nvtx_range(
                    "attention::hca::stream_overlap::support_routes_launch"
                ):
                    overlap_transfer = start_dsa_tensor_route(
                        x, self.overlap_route, cp_group, attention_mode=mode
                    )
                    self.pending.append(overlap_transfer)
                    window_transfer = start_dsa_tensor_route(
                        latent_kv, self.window_route, cp_group, attention_mode=mode
                    )
                    self.pending.append(window_transfer)
                    overlap_x = overlap_transfer.wait()
                    overlap_ready = torch.cuda.Event()
                    overlap_ready.record(route_stream)
                    window_kv = window_transfer.wait()
                    window_ready = torch.cuda.Event()
                    window_ready.record(route_stream)
            self.pending.clear()
            caller_stream.wait_event(overlap_ready)
            overlap_x.record_stream(caller_stream)

            self.overlap_rows = overlap_x.shape[0]
            self.support_rows = compressor_support_rows(
                self.overlap_rows, self.compression, config.compressor_support,
                overlap_x.device,
            )
            self.support_dtype = overlap_x.dtype
            with dsa_nvtx_range(
                f"packing::{mode}::compression_support::forward_gather",
                enabled=overlap_x.is_cuda,
            ):
                packed = gather_compressor_support(
                    overlap_x, self.compression, config.compressor_support,
                    self.support_rows,
                )
            valid_rows = self.compression.valid_rows
            block_positions = self.compression.block_positions

            packed_ready = torch.cuda.Event()
            packed_ready.record(caller_stream)
            main_stream.wait_event(packed_ready)
            with torch.cuda.stream(main_stream):
                packed.record_stream(main_stream)
                valid_rows.record_stream(main_stream)
                block_positions.record_stream(main_stream)
                with dsa_nvtx_range(
                    "attention::hca::stream_overlap::main_compressor_launch"
                ):
                    self.kv_node = DsaScheduleNode(
                        self.projections.main_compress,
                        input_requires_grad=(True, False, False),
                        enabled=self.schedule_enabled,
                    )
                    compressed_kv_local = self.kv_node.forward(
                        packed, valid_rows, block_positions
                    )
                    kv_transfer = start_dsa_tensor_route(
                        compressed_kv_local, self.compressed_kv_route, cp_group,
                        attention_mode=mode,
                    )
                    self.pending.append(kv_transfer)
                    compressed_kv = kv_transfer.wait()
                main_ready = torch.cuda.Event()
                main_ready.record(main_stream)
            caller_stream.wait_event(main_ready)
            caller_stream.wait_event(window_ready)
            self.pending.clear()
            compressed_kv.record_stream(caller_stream)
            window_kv.record_stream(caller_stream)
        except BaseException:
            self._drain_pending()
            caller_stream.wait_stream(route_stream)
            caller_stream.wait_stream(main_stream)
            raise

        self.window_rows = window_kv.shape[0]
        with dsa_nvtx_range(
            f"attention::{mode}::kv_bank_assembly", enabled=q.is_cuda
        ):
            kv_bank = torch.cat((window_kv, compressed_kv), dim=0).contiguous()
        attention_indices, attention_lengths = _build_attention_indices(
            handle, config, mode, self.window_rows
        )
        with dsa_nvtx_range(
            "attention::hca::stream_overlap::sparse_attention_launch",
            enabled=q.is_cuda,
        ):
            self.attention_node = DsaScheduleNode(
                lambda q_, kv_, sink_: dsa_sparse_attention(
                    q_, kv_, sink_, attention_indices, attention_lengths, config
                ),
                input_requires_grad=(q.requires_grad, True, sink.requires_grad),
                enabled=self.schedule_enabled,
            )
            output, sparse_lse, _ = self.attention_node.forward(q, kv_bank, sink)

        with dsa_nvtx_range(f"attention::{mode}::output_inverse_rope", enabled=q.is_cuda):
            self.rope_node = DsaScheduleNode(
                self.projections.inverse_output_rope,
                enabled=self.schedule_enabled,
            )
            output = self.rope_node.forward(output, self.device_plan.local_q_positions)

        self.sparse_lse = sparse_lse
        return output, torch.zeros((), dtype=torch.float32, device=x.device)

    def backward(self, grad_output, grad_kl):
        """HCA backward. There is no Indexer, so only the Compressor is free.

        WINDOW_KV is launched with COMPRESSED_KV and waited last, so the Main
        Compressor backward hides it. COMPRESSED_KV feeds that backward and
        OVERLAP_X consumes its result, so both are dependency-bound: HCA owns
        no route-independent work to put under them.
        """

        del grad_kl
        assert self.rope_node is not None
        assert self.attention_node is not None
        assert self.kv_node is not None
        mode = self.attention_mode
        cp_group = self.cp_group

        with dsa_nvtx_range(f"attention::{mode}::backward_overlap::inverse_rope"):
            (grad_attention_output, _) = self.rope_node.backward((grad_output,))
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::sparse_attention_backward"
        ):
            grad_q, grad_kv_bank, grad_sink = self.attention_node.backward(
                (grad_attention_output, None, None)
            )

        grad_window_kv, grad_compressed_kv = self._split_bank_gradient(grad_kv_bank)
        # COMPRESSED_KV first for the same reason as CSA: it gates the Main
        # Compressor backward, which is the only work either mode can put under
        # a route here.
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::support_reverse_launch"
        ):
            kv_reverse = start_dsa_reverse_route(
                grad_compressed_kv, self.compressed_kv_route, cp_group,
                attention_mode=mode,
            )
            window_reverse = start_dsa_reverse_route(
                grad_window_kv, self.window_route, cp_group, attention_mode=mode
            )
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::compressed_kv_reverse_wait"
        ):
            grad_compressed_kv_local = finish_dsa_reverse_route(kv_reverse)
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::main_compressor_backward"
        ):
            (grad_packed, _, _) = self.kv_node.backward((grad_compressed_kv_local,))

        grad_overlap_x = self._scatter_support_gradient(grad_packed)
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::overlap_x_reverse_launch"
        ):
            overlap_reverse = start_dsa_reverse_route(
                grad_overlap_x, self.overlap_route, cp_group, attention_mode=mode
            )
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::support_reverse_wait"
        ):
            grad_latent_kv = finish_dsa_reverse_route(window_reverse)
            grad_x = finish_dsa_reverse_route(overlap_reverse)
        return grad_x, None, grad_q, grad_latent_kv, grad_sink


class _DsaAttentionFunction(torch.autograd.Function):
    """The autograd boundary of one DSA layer.

    Everything between the two halves is explicitly scheduled, so this node
    exists only to hand control to ``scheduler.backward`` at the right moment.
    Gradients are materialized, which keeps the number of collectives a rank
    issues independent of which of its gradients happen to be present.
    """

    @staticmethod
    def forward(ctx, scheduler, x, qr, q, latent_kv, sink):
        ctx.scheduler = scheduler
        output, kl = scheduler.forward(x, qr, q, latent_kv, sink)
        # Detached aliases, because autograd rewrites the ``grad_fn`` of every
        # tensor a Function returns. Handing back the schedule's own tensors
        # would redirect the last node's subgraph at this very node, and
        # driving that subgraph would then re-enter this backward.
        output_alias = output.detach()
        kl_alias = kl.detach()
        if not scheduler.kl_is_differentiable:
            ctx.mark_non_differentiable(kl_alias)
        return output_alias, kl_alias

    @staticmethod
    def backward(ctx, grad_output, grad_kl):
        scheduler = ctx.scheduler
        if scheduler is None:
            raise RuntimeError(
                "a Magi-DSA layer runs its backward once. The schedule drives "
                "the model callbacks itself and frees each subgraph as soon as "
                "it has read its gradients, so there is nothing left to "
                "traverse a second time; retain_graph is not supported."
            )
        ctx.scheduler = None
        return (None, *scheduler.backward(grad_output.contiguous(), grad_kl))


def dist_dsa(
    config: MagiDSAConfig,
    projections: DsaProjections,
    dsa_input: MagiDSAInput,
    handle: DsaExecutionHandle,
    cp_group: dist.ProcessGroup | None,
) -> MagiDSAForwardResult:
    """Execute the owner-local Magi-DSA schedule for one layer."""

    latent_kv = _validate_inputs(config, dsa_input, handle)
    scheduler_type = {4: _CsaScheduler, 128: _HcaScheduler}[config.ratio]
    scheduler = scheduler_type(config, projections, dsa_input, handle, cp_group)
    output, kl = _DsaAttentionFunction.apply(
        scheduler, dsa_input.x, dsa_input.qr, dsa_input.q, latent_kv, dsa_input.sink
    )

    local_tokens = handle.device_plan.local_token_count
    device = dsa_input.x.device
    topk_global_ids = scheduler.topk_global_ids
    if topk_global_ids is None:
        topk_global_ids = torch.empty(
            (local_tokens, 0), dtype=torch.int32, device=device
        )
    topk_lengths = scheduler.topk_lengths
    if topk_lengths is None:
        topk_lengths = torch.zeros((local_tokens,), dtype=torch.int32, device=device)
    indexer_lse = scheduler.indexer_lse
    if indexer_lse is None:
        indexer_lse = torch.full(
            (local_tokens,), float("-inf"), dtype=torch.float32, device=device
        )
    assert scheduler.sparse_lse is not None
    return MagiDSAForwardResult(
        output=output,
        kl=kl,
        sparse_lse=scheduler.sparse_lse,
        topk_ids=topk_global_ids,
        topk_length=topk_lengths,
        indexer_lse=indexer_lse,
    )


__all__ = ["dist_dsa"]
