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

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from magi_attention.dsa_nvtx import dsa_nvtx_range
from magi_attention.dsa_types import MagiDSAForwardResult, MagiDSAInput
from magi_attention.kernel.triton.dsa_indices import build_csa_index_tensors

from .dsa_backend import (
    dsa_csa_attention_kl,
    dsa_sparse_attention,
    run_grouped_dsa_indexer,
)
from .dsa_comm import (
    DsaRouteTransfer,
    copy_dsa_tensor_with_csr,
    finish_dsa_reverse_route,
    finish_dsa_tensor_route,
    start_dsa_reverse_route,
    start_dsa_tensor_route,
)
from .dsa_packing import copy_dsa_device_map

if TYPE_CHECKING:
    from magi_attention.dsa_layer import MagiDSALayer
    from magi_attention.dsa_runtime_mgr import DsaExecutionHandle

    from .dsa_packing import DsaDeviceRoutePlan


class _CsaBackwardProjectionGateFunction(torch.autograd.Function):
    """Release projection and compression gradients from one late join."""

    @staticmethod
    def forward(
        ctx,
        q_indexer: torch.Tensor,
        score_weights: torch.Tensor,
        packed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        return q_indexer, score_weights, packed

    @staticmethod
    def backward(
        ctx,
        grad_q_indexer: torch.Tensor | None,
        grad_score_weights: torch.Tensor | None,
        grad_packed: torch.Tensor | None,
    ):
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::projection_support_release"
        ):
            return grad_q_indexer, grad_score_weights, grad_packed


class _CsaBackwardRouteOrderFunction(torch.autograd.Function):
    """Submit CSA OVERLAP_X reverse before WINDOW_KV reverse."""

    @staticmethod
    def forward(
        ctx,
        overlap_source: torch.Tensor,
        window_source: torch.Tensor,
        overlap_consumer: torch.Tensor,
        window_consumer: torch.Tensor,
        overlap_route: DsaDeviceRoutePlan,
        window_route: DsaDeviceRoutePlan,
        group: dist.ProcessGroup | None,
        route_stream: torch.cuda.Stream,
        source_stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.overlap_route = overlap_route
        ctx.window_route = window_route
        ctx.group = group
        ctx.route_stream = route_stream
        ctx.source_stream = source_stream
        ctx.overlap_source_shape = overlap_source.shape
        ctx.window_source_shape = window_source.shape
        ctx.set_materialize_grads(False)
        return overlap_consumer, window_consumer

    @staticmethod
    def backward(
        ctx,
        grad_overlap_consumer: torch.Tensor | None,
        grad_window_consumer: torch.Tensor | None,
    ):
        if grad_overlap_consumer is None or grad_window_consumer is None:
            raise RuntimeError(
                "CSA backward requires both OVERLAP_X and WINDOW_KV gradients"
            )
        caller_stream = torch.cuda.current_stream(grad_overlap_consumer.device)
        routes_ready = torch.cuda.Event()
        routes_ready.record(caller_stream)
        routes_done = torch.cuda.Event()
        overlap_transfer = None
        window_transfer = None
        with dsa_nvtx_range(
            "attention::csa::backward_overlap::overlap_then_window_reverse"
        ):
            ctx.route_stream.wait_event(routes_ready)
            try:
                with torch.cuda.stream(ctx.route_stream):
                    grad_overlap_consumer.record_stream(ctx.route_stream)
                    grad_window_consumer.record_stream(ctx.route_stream)
                    overlap_transfer = start_dsa_reverse_route(
                        grad_overlap_consumer.contiguous(),
                        ctx.overlap_route,
                        ctx.group,
                        attention_mode="csa",
                    )
                    window_transfer = start_dsa_reverse_route(
                        grad_window_consumer.contiguous(),
                        ctx.window_route,
                        ctx.group,
                        attention_mode="csa",
                    )
                    grad_overlap_source = finish_dsa_reverse_route(overlap_transfer)
                    grad_window_source = finish_dsa_reverse_route(window_transfer)
                    routes_done.record(ctx.route_stream)
            except BaseException:
                if overlap_transfer is not None:
                    overlap_transfer.wait()
                if window_transfer is not None:
                    window_transfer.wait()
                raise
            ctx.source_stream.wait_event(routes_done)
            grad_overlap_source.record_stream(ctx.source_stream)
            grad_window_source.record_stream(ctx.source_stream)
        if grad_overlap_source.shape != ctx.overlap_source_shape:
            raise RuntimeError("CSA OVERLAP_X reverse returned an invalid local shape")
        if grad_window_source.shape != ctx.window_source_shape:
            raise RuntimeError("CSA WINDOW_KV reverse returned an invalid local shape")
        return (
            grad_overlap_source,
            grad_window_source,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _HcaBackwardBranchOrderFunction(torch.autograd.Function):
    """Join HCA KV branches and release Window reverse after CKV reverse."""

    @staticmethod
    def forward(
        ctx,
        compressed_local: torch.Tensor,
        window_source: torch.Tensor,
        window_consumer: torch.Tensor,
        route: DsaDeviceRoutePlan,
        group: dist.ProcessGroup | None,
        route_stream: torch.cuda.Stream,
        source_stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.route = route
        ctx.group = group
        ctx.route_stream = route_stream
        ctx.source_stream = source_stream
        ctx.source_shape = window_source.shape
        ctx.set_materialize_grads(False)
        return compressed_local, window_consumer

    @staticmethod
    def backward(
        ctx,
        grad_compressed_local: torch.Tensor | None,
        grad_window_consumer: torch.Tensor | None,
    ):
        if grad_window_consumer is None:
            raise RuntimeError("HCA backward is missing its Window KV gradient")
        caller_stream = torch.cuda.current_stream(grad_window_consumer.device)
        window_ready = torch.cuda.Event()
        window_ready.record(caller_stream)
        route_done = torch.cuda.Event()
        transfer = None
        with dsa_nvtx_range(
            "attention::hca::backward_overlap::window_reverse_priority"
        ):
            ctx.route_stream.wait_event(window_ready)
            try:
                with torch.cuda.stream(ctx.route_stream):
                    grad_window_consumer.record_stream(ctx.route_stream)
                    transfer = start_dsa_reverse_route(
                        grad_window_consumer.contiguous(),
                        ctx.route,
                        ctx.group,
                        attention_mode="hca",
                    )
                    grad_source = finish_dsa_reverse_route(transfer)
                    route_done.record(ctx.route_stream)
            except BaseException:
                if transfer is not None:
                    transfer.wait()
                raise
            ctx.source_stream.wait_event(route_done)
            grad_source.record_stream(ctx.source_stream)
        if grad_source.shape != ctx.source_shape:
            raise RuntimeError("HCA WINDOW_KV reverse returned an invalid local shape")
        return grad_compressed_local, grad_source, None, None, None, None, None


def _validate_inputs(
    layer: MagiDSALayer, dsa_input: MagiDSAInput, handle: DsaExecutionHandle
) -> torch.Tensor:
    config = layer.config
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
    topk_global_ids: torch.Tensor,
    topk_lengths: torch.Tensor,
    raw_bank_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_plan = handle.device_plan
    attention = device_plan.attention
    ratio = handle.plan.ratio
    local_tokens = device_plan.local_token_count
    enabled = topk_global_ids.is_cuda
    scope = "attention_indices"
    with dsa_nvtx_range(scope, enabled=enabled):
        if ratio == 4:
            raise AssertionError("CSA indices must use the fused dual-map builder")

        if ratio == 128:
            with dsa_nvtx_range(
                f"{scope}::compressed_mask_and_offset", enabled=enabled
            ):
                compressed = attention.compressed_rows
                invalid = compressed < 0
                compressed = (compressed + raw_bank_rows).masked_fill(invalid, -1)
                compressed_lengths = attention.compressed_lengths
        else:
            with dsa_nvtx_range(f"{scope}::window_only_init", enabled=enabled):
                compressed = torch.empty(
                    (local_tokens, 0),
                    dtype=torch.int32,
                    device=attention.window_rows.device,
                )
                compressed_lengths = torch.zeros_like(attention.window_lengths)

        logical_width = attention.window_rows.shape[1] + compressed.shape[1]
        width = ((logical_width + 127) // 128) * 128
        with dsa_nvtx_range(f"{scope}::output_allocation", enabled=enabled):
            indices = torch.full(
                (local_tokens, width),
                -1,
                dtype=torch.int32,
                device=attention.window_rows.device,
            )
        with dsa_nvtx_range(f"{scope}::window_copy", enabled=enabled):
            indices[:, : attention.window_rows.shape[1]] = attention.window_rows
        if compressed.shape[1]:
            with dsa_nvtx_range(f"{scope}::compressed_scatter", enabled=enabled):
                columns = torch.arange(
                    compressed.shape[1], dtype=torch.int32, device=compressed.device
                ).unsqueeze(0)
                destinations = attention.window_lengths.unsqueeze(1) + columns
                indices.scatter_(1, destinations.long(), compressed)
        with dsa_nvtx_range(f"{scope}::length_update", enabled=enabled):
            lengths = attention.window_lengths + compressed_lengths
        return indices, lengths


def _build_csa_indices(
    handle: DsaExecutionHandle,
    topk_global_ids: torch.Tensor,
    topk_lengths: torch.Tensor,
    raw_bank_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_plan = handle.device_plan
    indexer = device_plan.indexer
    if indexer is None:
        raise RuntimeError("CSA plan is missing Indexer metadata")
    with dsa_nvtx_range(
        "attention_indices::fused_csa_dual_map",
        enabled=topk_global_ids.is_cuda,
    ):
        return build_csa_index_tensors(
            topk_global_ids,
            topk_lengths,
            device_plan.attention.compressed_global_to_consumer,
            indexer.ki_global_to_consumer,
            device_plan.attention.window_rows,
            device_plan.attention.window_lengths,
            raw_bank_rows,
        )


def dist_dsa(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    handle: DsaExecutionHandle,
    cp_group: dist.ProcessGroup | None,
) -> MagiDSAForwardResult:
    """Execute the frozen owner-local Magi-DSA forward/autograd DAG."""

    config = layer.config
    device_plan = handle.device_plan
    latent_kv = _validate_inputs(layer, dsa_input, handle)
    local_tokens = device_plan.local_token_count
    attention_mode = {0: "w", 4: "csa", 128: "hca"}[config.ratio]
    caller_stream = torch.cuda.current_stream(dsa_input.x.device)
    csa_main_stream = handle.csa_main_stream
    csa_indexer_stream = handle.csa_indexer_stream
    csa_route_stream = handle.csa_route_stream
    hca_main_stream = handle.hca_main_stream
    hca_route_stream = handle.hca_route_stream
    csa_caller_ready: torch.cuda.Event | None = None
    csa_indexer_ready: torch.cuda.Event | None = None
    csa_overlap_ready: torch.cuda.Event | None = None
    csa_window_ready: torch.cuda.Event | None = None
    csa_main_ready: torch.cuda.Event | None = None
    hca_caller_ready: torch.cuda.Event | None = None
    hca_overlap_ready: torch.cuda.Event | None = None
    hca_window_ready: torch.cuda.Event | None = None
    hca_main_ready: torch.cuda.Event | None = None
    if config.ratio == 4:
        if (
            csa_main_stream is None
            or csa_indexer_stream is None
            or csa_route_stream is None
        ):
            raise RuntimeError("CSA execution is missing its overlap streams")
        csa_caller_ready = torch.cuda.Event()
        csa_caller_ready.record(caller_stream)
    elif config.ratio == 128:
        if hca_main_stream is None or hca_route_stream is None:
            raise RuntimeError("HCA execution is missing its overlap streams")
        hca_caller_ready = torch.cuda.Event()
        hca_caller_ready.record(caller_stream)

    pending_transfers: list[DsaRouteTransfer] = []
    if config.ratio == 4:
        assert csa_route_stream is not None
        assert csa_caller_ready is not None
        csa_route_stream.wait_event(csa_caller_ready)
        with torch.cuda.stream(csa_route_stream):
            latent_kv.record_stream(csa_route_stream)
            window_transfer = start_dsa_tensor_route(
                latent_kv,
                device_plan.window_route,
                cp_group,
                attention_mode=attention_mode,
            )
        pending_transfers.append(window_transfer)
    elif config.ratio == 0:
        window_transfer = start_dsa_tensor_route(
            latent_kv,
            device_plan.window_route,
            cp_group,
            attention_mode=attention_mode,
        )
        pending_transfers.append(window_transfer)
    try:
        q_indexer = dsa_input.x.new_empty((local_tokens, 0, 0))
        weights = dsa_input.x.new_empty((local_tokens, 0))
        score_weights = weights
        compressed_ki_local: torch.Tensor | None = None
        compressed_ki = dsa_input.x.new_empty((0, config.indexer_head_dim))
        grouped_k = dsa_input.x.new_empty((0, config.indexer_head_dim))
        selection = None
        topk_global_ids = torch.empty(
            (local_tokens, 0), dtype=torch.int32, device=dsa_input.x.device
        )
        topk_lengths = torch.zeros(
            (local_tokens,), dtype=torch.int32, device=dsa_input.x.device
        )
        indexer_lse = torch.full(
            (local_tokens,),
            float("-inf"),
            dtype=torch.float32,
            device=dsa_input.x.device,
        )

        if config.ratio:
            overlap_route = device_plan.overlap_x_route
            compressed_kv_route = device_plan.compressed_kv_route
            compression = device_plan.compression
            if (
                overlap_route is None
                or compressed_kv_route is None
                or compression is None
                or layer.compressor is None
            ):
                raise RuntimeError("compressed DSA plan is missing mandatory routes")
            if config.ratio == 4:
                if (
                    layer.indexer is None
                    or csa_indexer_stream is None
                    or csa_route_stream is None
                    or csa_caller_ready is None
                ):
                    raise RuntimeError("CSA execution requires the model-side Indexer")
                with torch.cuda.stream(csa_route_stream):
                    dsa_input.x.record_stream(csa_route_stream)
                    with dsa_nvtx_range(
                        "attention::csa::stream_overlap::" "support_routes_launch"
                    ):
                        overlap_transfer = start_dsa_tensor_route(
                            dsa_input.x,
                            overlap_route,
                            cp_group,
                            attention_mode=attention_mode,
                        )
                        pending_transfers.append(overlap_transfer)
                csa_indexer_stream.wait_event(csa_caller_ready)
                with torch.cuda.stream(csa_indexer_stream):
                    dsa_input.x.record_stream(csa_indexer_stream)
                    dsa_input.qr.record_stream(csa_indexer_stream)
                    with dsa_nvtx_range(
                        "attention::csa::stream_overlap::" "indexer_projection_launch"
                    ):
                        q_indexer, weights = layer.indexer.project_queries(
                            dsa_input.x,
                            dsa_input.qr,
                            device_plan.local_q_positions,
                            detach_trunk=dsa_input.detach_indexer_trunk,
                        )
                        score_weights = weights
                    csa_indexer_ready = torch.cuda.Event()
                    csa_indexer_ready.record(csa_indexer_stream)
                with torch.cuda.stream(csa_route_stream):
                    overlap_x = finish_dsa_tensor_route(overlap_transfer)
                    window_kv = finish_dsa_tensor_route(window_transfer)
                    (
                        overlap_x,
                        window_kv,
                    ) = _CsaBackwardRouteOrderFunction.apply(
                        dsa_input.x,
                        latent_kv,
                        overlap_x,
                        window_kv,
                        overlap_route,
                        device_plan.window_route,
                        cp_group,
                        csa_route_stream,
                        caller_stream,
                    )
                    csa_overlap_ready = torch.cuda.Event()
                    csa_overlap_ready.record(csa_route_stream)
                    csa_window_ready = torch.cuda.Event()
                    csa_window_ready.record(csa_route_stream)
                caller_stream.wait_event(csa_overlap_ready)
                overlap_x.record_stream(caller_stream)
            else:
                if (
                    hca_route_stream is None
                    or hca_caller_ready is None
                    or device_plan.window_route is None
                ):
                    raise RuntimeError("HCA execution is missing its support route")
                hca_route_stream.wait_event(hca_caller_ready)
                with torch.cuda.stream(hca_route_stream):
                    dsa_input.x.record_stream(hca_route_stream)
                    latent_kv.record_stream(hca_route_stream)
                    with dsa_nvtx_range(
                        "attention::hca::stream_overlap::support_routes_launch"
                    ):
                        overlap_transfer = start_dsa_tensor_route(
                            dsa_input.x,
                            overlap_route,
                            cp_group,
                            attention_mode=attention_mode,
                        )
                        pending_transfers.append(overlap_transfer)
                        window_transfer = start_dsa_tensor_route(
                            latent_kv,
                            device_plan.window_route,
                            cp_group,
                            attention_mode=attention_mode,
                        )
                        pending_transfers.append(window_transfer)
                        overlap_x = finish_dsa_tensor_route(overlap_transfer)
                        hca_overlap_ready = torch.cuda.Event()
                        hca_overlap_ready.record(hca_route_stream)
                        window_kv = finish_dsa_tensor_route(window_transfer)
                        hca_window_ready = torch.cuda.Event()
                        hca_window_ready.record(hca_route_stream)
                caller_stream.wait_event(hca_overlap_ready)
                overlap_x.record_stream(caller_stream)

            packed = copy_dsa_tensor_with_csr(
                overlap_x,
                compression.source_pack,
                compression.source_unpack,
                nvtx_scope=f"{attention_mode}::compression_support",
            ).view(-1, config.compressor_support, config.hidden_size)
            valid_rows = compression.valid_rows.view(-1, config.compressor_support)

            compressed_ki_transfer: DsaRouteTransfer | None = None
            compressed_ki_route = device_plan.compressed_ki_route
            indexer_map = device_plan.indexer
            if config.ratio == 4:
                if (
                    layer.indexer is None
                    or compressed_ki_route is None
                    or indexer_map is None
                ):
                    raise RuntimeError("CSA plan is missing Indexer routes or metadata")
                (
                    q_indexer,
                    score_weights,
                    packed,
                ) = _CsaBackwardProjectionGateFunction.apply(
                    q_indexer,
                    score_weights,
                    packed,
                )
                indexer_packed = (
                    packed.detach() if dsa_input.detach_indexer_trunk else packed
                )
                compressed_ki_local = layer.indexer.compressor(
                    indexer_packed,
                    valid_rows,
                    compression.block_positions,
                )
                compressed_ki_transfer = start_dsa_tensor_route(
                    compressed_ki_local,
                    compressed_ki_route,
                    cp_group,
                    attention_mode=attention_mode,
                )
                pending_transfers.append(compressed_ki_transfer)
                if csa_main_stream is None:
                    raise RuntimeError("CSA execution is missing its main stream")
                csa_main_input_ready = torch.cuda.Event()
                csa_main_input_ready.record(caller_stream)
                csa_main_stream.wait_event(csa_main_input_ready)
                with torch.cuda.stream(csa_main_stream):
                    packed.record_stream(csa_main_stream)
                    valid_rows.record_stream(csa_main_stream)
                    compression.block_positions.record_stream(csa_main_stream)
                    with dsa_nvtx_range(
                        "attention::csa::stream_overlap::main_compressor_launch"
                    ):
                        compressed_kv_local = layer.compressor(
                            packed,
                            valid_rows,
                            compression.block_positions,
                        )
                        compressed_kv_transfer = start_dsa_tensor_route(
                            compressed_kv_local,
                            compressed_kv_route,
                            cp_group,
                            attention_mode=attention_mode,
                        )
                        pending_transfers.append(compressed_kv_transfer)
                        compressed_kv = finish_dsa_tensor_route(compressed_kv_transfer)
                    csa_main_ready = torch.cuda.Event()
                    csa_main_ready.record(csa_main_stream)

                assert compressed_ki_transfer is not None
                compressed_ki = finish_dsa_tensor_route(compressed_ki_transfer)
                assert csa_indexer_ready is not None
                assert csa_indexer_stream is not None
                grouped_input_ready = torch.cuda.Event()
                grouped_input_ready.record(caller_stream)
                csa_indexer_stream.wait_event(grouped_input_ready)
                with torch.cuda.stream(csa_indexer_stream):
                    compressed_ki.record_stream(csa_indexer_stream)
                    with (
                        torch.no_grad(),
                        dsa_nvtx_range(
                            "packing::csa::indexer_key_support::forward_copy",
                            enabled=compressed_ki.is_cuda,
                        ),
                    ):
                        grouped_k = copy_dsa_device_map(
                            compressed_ki,
                            indexer_map.k_pack,
                        )
                    with dsa_nvtx_range(
                        "attention::csa::stream_overlap::grouped_indexer_launch"
                    ):
                        selection = run_grouped_dsa_indexer(
                            q_indexer,
                            grouped_k,
                            score_weights,
                            indexer_map,
                            config,
                        )
                    selection_ready = torch.cuda.Event()
                    selection_ready.record(csa_indexer_stream)
                caller_stream.wait_event(selection_ready)
                for indexer_output in (
                    grouped_k,
                    selection.global_ids,
                    selection.lengths,
                    selection.lse,
                ):
                    indexer_output.record_stream(caller_stream)
                topk_global_ids = selection.global_ids
                topk_lengths = selection.lengths
                indexer_lse = selection.lse
                assert csa_main_ready is not None
                caller_stream.wait_event(csa_main_ready)
                compressed_kv.record_stream(caller_stream)
            else:
                if hca_main_stream is None:
                    raise RuntimeError("HCA execution is missing its main stream")
                hca_packed_ready = torch.cuda.Event()
                hca_packed_ready.record(caller_stream)
                hca_main_stream.wait_event(hca_packed_ready)
                with torch.cuda.stream(hca_main_stream):
                    packed.record_stream(hca_main_stream)
                    valid_rows.record_stream(hca_main_stream)
                    compression.block_positions.record_stream(hca_main_stream)
                    with dsa_nvtx_range(
                        "attention::hca::stream_overlap::main_compressor_launch"
                    ):
                        compressed_kv_local = layer.compressor(
                            packed,
                            valid_rows,
                            compression.block_positions,
                        )
                        (
                            compressed_kv_local,
                            window_kv,
                        ) = _HcaBackwardBranchOrderFunction.apply(
                            compressed_kv_local,
                            latent_kv,
                            window_kv,
                            device_plan.window_route,
                            cp_group,
                            hca_route_stream,
                            caller_stream,
                        )
                        compressed_kv_transfer = start_dsa_tensor_route(
                            compressed_kv_local,
                            compressed_kv_route,
                            cp_group,
                            attention_mode=attention_mode,
                        )
                        pending_transfers.append(compressed_kv_transfer)
                        compressed_kv = finish_dsa_tensor_route(compressed_kv_transfer)
                    hca_main_ready = torch.cuda.Event()
                    hca_main_ready.record(hca_main_stream)
                caller_stream.wait_event(hca_main_ready)
                compressed_kv.record_stream(caller_stream)
        else:
            compressed_kv = latent_kv.new_empty((0, config.head_dim))

        if config.ratio == 4:
            assert csa_window_ready is not None
            caller_stream.wait_event(csa_window_ready)
            window_kv.record_stream(caller_stream)
        elif config.ratio == 128:
            assert hca_window_ready is not None
            caller_stream.wait_event(hca_window_ready)
            window_kv.record_stream(caller_stream)
        else:
            window_kv = finish_dsa_tensor_route(window_transfer)
    except BaseException:
        for transfer in pending_transfers:
            transfer.wait()
        if config.ratio == 4:
            assert csa_main_stream is not None
            assert csa_indexer_stream is not None
            assert csa_route_stream is not None
            caller_stream.wait_stream(csa_main_stream)
            caller_stream.wait_stream(csa_indexer_stream)
            caller_stream.wait_stream(csa_route_stream)
        elif config.ratio == 128:
            assert hca_main_stream is not None
            assert hca_route_stream is not None
            caller_stream.wait_stream(hca_main_stream)
            caller_stream.wait_stream(hca_route_stream)
        raise

    with dsa_nvtx_range("attention::kv_bank_assembly", enabled=dsa_input.q.is_cuda):
        kv_bank = torch.cat((window_kv, compressed_kv), dim=0).contiguous()
    if config.ratio == 4:
        (
            attention_indices,
            attention_lengths,
            indexer_indices,
            attention_compressed_indices,
        ) = _build_csa_indices(
            handle,
            topk_global_ids,
            topk_lengths,
            window_kv.shape[0],
        )
    else:
        attention_indices, attention_lengths = _build_attention_indices(
            handle,
            topk_global_ids,
            topk_lengths,
            window_kv.shape[0],
        )
    if config.ratio == 4:
        assert device_plan.indexer is not None
        assert selection is not None
        assert compressed_ki_local is not None
        assert device_plan.compressed_ki_route is not None
        if handle.sparse_backward_stream is None:
            raise RuntimeError("CSA execution is missing its sparse backward stream")
        local_loss_coeff = (
            config.kl_loss_coeff * local_tokens / handle.plan.total_tokens
        )
        if csa_indexer_stream is None:
            raise RuntimeError("CSA execution is missing its Indexer stream")
        attention_input_ready = torch.cuda.Event()
        attention_input_ready.record(caller_stream)
        csa_indexer_stream.wait_event(attention_input_ready)
        try:
            with torch.cuda.stream(csa_indexer_stream):
                for attention_input in (
                    dsa_input.q,
                    kv_bank,
                    dsa_input.sink,
                    attention_indices,
                    attention_lengths,
                    q_indexer,
                    score_weights,
                    compressed_ki_local,
                    compressed_ki,
                    compressed_kv,
                    selection.global_ids,
                    selection.lengths,
                    selection.lse,
                    indexer_indices,
                    attention_compressed_indices,
                ):
                    attention_input.record_stream(csa_indexer_stream)
                with dsa_nvtx_range(
                    "attention::csa::stream_overlap::combined_attention_kl_launch"
                ):
                    output, kl, sparse_lse = dsa_csa_attention_kl(
                        dsa_input.q,
                        kv_bank,
                        dsa_input.sink,
                        attention_indices,
                        attention_lengths,
                        q_indexer,
                        score_weights,
                        compressed_ki_local,
                        compressed_ki,
                        compressed_kv,
                        selection,
                        indexer_indices,
                        attention_compressed_indices,
                        device_plan.compressed_ki_route,
                        cp_group,
                        handle.sparse_backward_stream,
                        loss_coeff=local_loss_coeff,
                        config=config,
                    )
                attention_ready = torch.cuda.Event()
                attention_ready.record(csa_indexer_stream)
        except BaseException:
            caller_stream.wait_stream(csa_indexer_stream)
            raise
        caller_stream.wait_event(attention_ready)
        output.record_stream(caller_stream)
        kl.record_stream(caller_stream)
        sparse_lse.record_stream(caller_stream)
    else:
        output, sparse_lse, _ = dsa_sparse_attention(
            dsa_input.q,
            kv_bank,
            dsa_input.sink,
            attention_indices,
            attention_lengths,
            config,
        )
        kl = torch.zeros((), dtype=torch.float32, device=dsa_input.x.device)

    output = layer.inverse_output_rope(output, device_plan.local_q_positions)

    return MagiDSAForwardResult(
        output=output,
        kl=kl,
        sparse_lse=sparse_lse,
        topk_ids=topk_global_ids,
        topk_length=topk_lengths,
        indexer_lse=indexer_lse,
    )


__all__ = ["dist_dsa"]
