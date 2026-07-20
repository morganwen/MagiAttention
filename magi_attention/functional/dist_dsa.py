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

from magi_attention.dsa_types import MagiDSAForwardResult, MagiDSAInput

from .dsa_backend import dsa_selected_kl, dsa_sparse_attention, run_grouped_dsa_indexer
from .dsa_comm import (
    copy_dsa_tensor_with_csr,
    restore_dsa_bijective_tensor,
    route_dsa_tensor,
)

if TYPE_CHECKING:
    from magi_attention.dsa_layer import MagiDSALayer
    from magi_attention.dsa_runtime_mgr import DsaExecutionHandle


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


def _map_global_ids(
    global_ids: torch.Tensor, global_to_consumer: torch.Tensor
) -> torch.Tensor:
    if global_ids.dtype != torch.int32 or global_to_consumer.dtype != torch.int32:
        raise TypeError("DSA compressed IDs and maps must use int32")
    if global_to_consumer.numel() == 0:
        return torch.full_like(global_ids, -1)
    valid = (global_ids >= 0) & (global_ids < global_to_consumer.numel())
    safe = global_ids.clamp(min=0, max=global_to_consumer.numel() - 1)
    return global_to_consumer[safe.long()].masked_fill(~valid, -1)


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
    if ratio == 4:
        compressed = _map_global_ids(
            topk_global_ids, attention.compressed_global_to_consumer
        )
        columns = torch.arange(
            compressed.shape[1], dtype=torch.int32, device=compressed.device
        ).unsqueeze(0)
        invalid = (compressed < 0) | (columns >= topk_lengths.unsqueeze(1))
        compressed = (compressed + raw_bank_rows).masked_fill(invalid, -1)
        compressed_lengths = topk_lengths
    elif ratio == 128:
        compressed = attention.compressed_rows
        invalid = compressed < 0
        compressed = (compressed + raw_bank_rows).masked_fill(invalid, -1)
        compressed_lengths = attention.compressed_lengths
    else:
        compressed = torch.empty(
            (local_tokens, 0), dtype=torch.int32, device=attention.window_rows.device
        )
        compressed_lengths = torch.zeros_like(attention.window_lengths)

    logical_width = attention.window_rows.shape[1] + compressed.shape[1]
    width = ((logical_width + 127) // 128) * 128
    indices = torch.full(
        (local_tokens, width),
        -1,
        dtype=torch.int32,
        device=attention.window_rows.device,
    )
    indices[:, : attention.window_rows.shape[1]] = attention.window_rows
    if compressed.shape[1]:
        columns = torch.arange(
            compressed.shape[1], dtype=torch.int32, device=compressed.device
        ).unsqueeze(0)
        destinations = attention.window_lengths.unsqueeze(1) + columns
        indices.scatter_(1, destinations.long(), compressed)
    lengths = attention.window_lengths + compressed_lengths
    return indices, lengths


def _pack_indexer_aux(
    global_ids: torch.Tensor, lengths: torch.Tensor, lse: torch.Tensor
) -> torch.Tensor:
    if (
        global_ids.dtype != torch.int32
        or lengths.dtype != torch.int32
        or lse.dtype != torch.float32
    ):
        raise TypeError("Indexer auxiliary tensors have invalid dtypes")
    padding = torch.zeros(
        (global_ids.shape[0], 2), dtype=torch.int32, device=global_ids.device
    )
    return torch.cat(
        (
            global_ids,
            lengths.unsqueeze(1),
            lse.contiguous().view(torch.int32).reshape(-1, 1),
            padding,
        ),
        dim=1,
    ).contiguous()


def _unpack_indexer_aux(
    auxiliary: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if auxiliary.dtype != torch.int32 or auxiliary.shape[1] != topk + 4:
        raise ValueError("restored Indexer auxiliary payload has an invalid schema")
    global_ids = auxiliary[:, :topk].contiguous()
    lengths = auxiliary[:, topk].contiguous()
    lse = auxiliary[:, topk + 1].contiguous().view(torch.float32)
    return global_ids, lengths, lse


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

    window_kv = route_dsa_tensor(latent_kv, device_plan.window_route, cp_group)
    if config.ratio:
        overlap_route = device_plan.overlap_x_route
        compressed_kv_route = device_plan.compressed_kv_route
        if overlap_route is None or compressed_kv_route is None:
            raise RuntimeError("compressed DSA plan is missing mandatory routes")
        overlap_x = route_dsa_tensor(dsa_input.x, overlap_route, cp_group)
        compression = device_plan.compression
        if compression is None or layer.compressor is None:
            raise RuntimeError("compressed DSA plan is missing compression metadata")
        packed = copy_dsa_tensor_with_csr(
            overlap_x,
            compression.source_pack,
            compression.source_unpack,
        ).view(-1, config.compressor_support, config.hidden_size)
        valid_rows = compression.valid_rows.view(-1, config.compressor_support)
        compressed_kv_local = layer.compressor(
            packed, valid_rows, compression.block_positions
        )
        compressed_kv = route_dsa_tensor(
            compressed_kv_local, compressed_kv_route, cp_group
        )
    else:
        packed = None
        valid_rows = None
        compressed_kv = latent_kv.new_empty((0, config.head_dim))

    if config.ratio == 4:
        compression = device_plan.compression
        if (
            layer.indexer is None
            or packed is None
            or valid_rows is None
            or compression is None
        ):
            raise RuntimeError("CSA execution requires the model-side Indexer")
        compressed_ki_route = device_plan.compressed_ki_route
        query_route = device_plan.indexer_qw_route
        indexer_map = device_plan.indexer
        if compressed_ki_route is None or query_route is None or indexer_map is None:
            raise RuntimeError("CSA plan is missing Indexer routes or metadata")
        indexer_packed = packed.detach() if dsa_input.detach_indexer_trunk else packed
        compressed_ki_local = layer.indexer.compressor(
            indexer_packed,
            valid_rows,
            compression.block_positions,
        )
        compressed_ki = route_dsa_tensor(
            compressed_ki_local, compressed_ki_route, cp_group
        )

        q_indexer, weights = layer.indexer.project_queries(
            dsa_input.x,
            dsa_input.qr,
            device_plan.local_q_positions,
            detach_trunk=dsa_input.detach_indexer_trunk,
        )
        packed_qw = torch.cat((q_indexer.flatten(1), weights), dim=1).contiguous()
        worker_qw = route_dsa_tensor(packed_qw, query_route, cp_group)
        q_width = config.indexer_heads * config.indexer_head_dim
        worker_q = worker_qw[:, :q_width].view(
            -1, config.indexer_heads, config.indexer_head_dim
        )
        worker_weights = worker_qw[:, q_width:].view(-1, config.indexer_heads)
        worker_k = copy_dsa_tensor_with_csr(
            compressed_ki,
            indexer_map.k_pack,
            indexer_map.k_unpack,
        )
        selection = run_grouped_dsa_indexer(
            worker_q, worker_k, worker_weights, indexer_map, config
        )
        worker_aux = _pack_indexer_aux(
            selection.global_ids, selection.lengths, selection.lse
        )
        owner_aux = restore_dsa_bijective_tensor(worker_aux, query_route, cp_group)
        topk_global_ids, topk_lengths, indexer_lse = _unpack_indexer_aux(
            owner_aux, config.indexer_topk
        )
    else:
        q_indexer = dsa_input.x.new_empty((local_tokens, 0, 0))
        weights = dsa_input.x.new_empty((local_tokens, 0))
        compressed_ki = dsa_input.x.new_empty((0, config.indexer_head_dim))
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

    kv_bank = torch.cat((window_kv, compressed_kv), dim=0).contiguous()
    attention_indices, attention_lengths = _build_attention_indices(
        handle,
        topk_global_ids,
        topk_lengths,
        window_kv.shape[0],
    )
    output, sparse_lse = dsa_sparse_attention(
        dsa_input.q,
        kv_bank,
        dsa_input.sink,
        attention_indices,
        attention_lengths,
        config,
    )

    if config.ratio == 4:
        assert device_plan.indexer is not None
        indexer_indices = _map_global_ids(
            topk_global_ids,
            device_plan.indexer.ki_global_to_consumer,
        )
        attention_compressed_indices = _map_global_ids(
            topk_global_ids,
            device_plan.attention.compressed_global_to_consumer,
        )
        columns = torch.arange(
            config.indexer_topk, dtype=torch.int32, device=dsa_input.x.device
        ).unsqueeze(0)
        invalid = columns >= topk_lengths.unsqueeze(1)
        indexer_indices = indexer_indices.masked_fill(invalid, -1)
        attention_compressed_indices = attention_compressed_indices.masked_fill(
            invalid, -1
        )
        local_loss_coeff = (
            config.kl_loss_coeff * local_tokens / handle.plan.total_tokens
        )
        kl = dsa_selected_kl(
            q_indexer,
            weights,
            compressed_ki,
            dsa_input.q,
            compressed_kv,
            sparse_lse,
            indexer_indices,
            attention_compressed_indices,
            topk_lengths,
            loss_coeff=local_loss_coeff,
            config=config,
        )
    else:
        kl = torch.zeros((), dtype=torch.float32, device=dsa_input.x.device)

    return MagiDSAForwardResult(
        output=output,
        kl=kl,
        sparse_lse=sparse_lse,
        topk_ids=topk_global_ids,
        topk_length=topk_lengths,
        indexer_lse=indexer_lse,
    )


__all__ = ["dist_dsa"]
