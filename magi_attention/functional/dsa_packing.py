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

from magi_attention.dsa_config import MagiDSAConfig
from magi_attention.dsa_nvtx import dsa_nvtx_range
from magi_attention.meta.collection.dsa_meta import (
    DsaExecutionPlan,
    DsaRankPlan,
    DsaRouteRankPlan,
)


@dataclass(frozen=True)
class DsaHostCopyMap:
    """Validated destination-row to source-row map."""

    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class DsaHostReduceMap:
    """Validated destination-row CSR reduction map."""

    row_offsets: tuple[int, ...]
    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class DsaHostRouteMaps:
    """All row permutations used by a typed unique-row route."""

    send_pack: DsaHostCopyMap
    consumer_pack: DsaHostCopyMap
    received_pack: DsaHostCopyMap
    owner_reduce: DsaHostReduceMap


@dataclass(frozen=True)
class DsaDeviceCopyMap:
    """CUDA-resident destination-row to source-row map."""

    source_rows: torch.Tensor
    is_identity: bool = False


@dataclass(frozen=True)
class DsaDeviceReduceMap:
    """CUDA-resident destination-row CSR reduction map."""

    row_offsets: torch.Tensor
    source_rows: torch.Tensor
    is_identity: bool = False


@dataclass(frozen=True)
class DsaDeviceRoutePlan:
    """One rank's static maps and split sizes for a typed All2AllV route."""

    name: str
    rank: int
    producer_row_count: int
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_pack: DsaDeviceCopyMap
    consumer_pack: DsaDeviceCopyMap
    received_pack: DsaDeviceCopyMap
    owner_reduce: DsaDeviceReduceMap
    owner_restore: DsaDeviceCopyMap | None
    consumer_global_rows: torch.Tensor

    @property
    def send_row_count(self) -> int:
        return sum(self.send_counts)

    @property
    def received_row_count(self) -> int:
        return sum(self.recv_counts)

    @property
    def consumer_row_count(self) -> int:
        return self.consumer_global_rows.numel()


@dataclass(frozen=True)
class DsaDeviceCompressionMap:
    """Padded compression support rows and a matching validity mask."""

    source_pack: DsaDeviceCopyMap
    source_unpack: DsaDeviceReduceMap
    valid_rows: torch.Tensor
    block_positions: torch.Tensor


@dataclass(frozen=True)
class DsaDeviceAttentionMap:
    """Device-generated padded raw-window/HCA rows into unique KV banks."""

    window_rows: torch.Tensor
    window_lengths: torch.Tensor
    compressed_rows: torch.Tensor
    compressed_lengths: torch.Tensor
    compressed_global_to_consumer: torch.Tensor


@dataclass(frozen=True)
class DsaDeviceIndexerMap:
    """Grouped Indexer metadata and its forward-only duplicate-prefix map."""

    q_cu_seqlens: torch.Tensor
    k_cu_seqlens: torch.Tensor
    q_causal_offsets: torch.Tensor
    q_sample_block_offsets: torch.Tensor
    seq_lens: torch.Tensor
    k_pack: DsaDeviceCopyMap
    ki_global_to_consumer: torch.Tensor
    max_seqlen_q: int
    logical_max_seqlen_k: int
    backend_max_seqlen_k: int


@dataclass(frozen=True)
class DsaDeviceRankPlan:
    """All immutable CUDA metadata materialized during cold prepare."""

    rank: int
    source_token_count: int
    local_token_count: int
    local_q_sample_ids: torch.Tensor
    local_q_positions: torch.Tensor
    compression: DsaDeviceCompressionMap | None
    attention: DsaDeviceAttentionMap
    indexer: DsaDeviceIndexerMap | None
    token_layout_route: DsaDeviceRoutePlan | None
    window_route: DsaDeviceRoutePlan
    overlap_x_route: DsaDeviceRoutePlan | None
    compressed_kv_route: DsaDeviceRoutePlan | None
    compressed_ki_route: DsaDeviceRoutePlan | None


def make_dsa_copy_map(
    source_rows: tuple[int, ...],
    source_row_count: int,
) -> DsaHostCopyMap:
    """Validate an arbitrary row gather without imposing uniqueness."""

    if source_row_count < 0:
        raise ValueError("source_row_count must be non-negative")
    normalized = tuple(int(row) for row in source_rows)
    if any(row < 0 or row >= source_row_count for row in normalized):
        raise ValueError("copy map contains a row outside its source")
    return DsaHostCopyMap(source_rows=normalized)


def make_dsa_reduce_map(
    destination_for_source: tuple[int, ...],
    destination_row_count: int,
) -> DsaHostReduceMap:
    """Build a destination-row CSR from a source-row destination table."""

    if destination_row_count < 0:
        raise ValueError("destination_row_count must be non-negative")
    normalized = tuple(int(row) for row in destination_for_source)
    if any(row < 0 or row >= destination_row_count for row in normalized):
        raise ValueError("reduce map contains a row outside its destination")
    occurrences: list[list[int]] = [[] for _ in range(destination_row_count)]
    for source_row, destination_row in enumerate(normalized):
        occurrences[destination_row].append(source_row)
    row_offsets = [0]
    source_rows: list[int] = []
    for rows in occurrences:
        source_rows.extend(rows)
        row_offsets.append(len(source_rows))
    return DsaHostReduceMap(tuple(row_offsets), tuple(source_rows))


def validate_dsa_reduce_map(
    mapping: DsaHostReduceMap,
    source_row_count: int,
    destination_row_count: int,
) -> None:
    """Validate CSR bounds and require every source contribution exactly once."""

    if len(mapping.row_offsets) != destination_row_count + 1:
        raise ValueError("CSR offset count does not match destination rows")
    if not mapping.row_offsets or mapping.row_offsets[0] != 0:
        raise ValueError("CSR offsets must start at zero")
    if any(
        end < begin for begin, end in zip(mapping.row_offsets, mapping.row_offsets[1:])
    ):
        raise ValueError("CSR offsets must be nondecreasing")
    if mapping.row_offsets[-1] != len(mapping.source_rows):
        raise ValueError("CSR terminal offset does not match its item count")
    if sorted(mapping.source_rows) != list(range(source_row_count)):
        raise ValueError("CSR must consume every source row exactly once")


def make_dsa_route_maps(route: DsaRouteRankPlan) -> DsaHostRouteMaps:
    """Convert solver metadata into four independently validated row maps."""

    if len(route.send_counts) != len(route.recv_counts):
        raise ValueError("route split tables have different world sizes")
    if sum(route.send_counts) != len(route.send_source_rows):
        raise ValueError("route send splits do not cover the packed send rows")
    if sum(route.recv_counts) != len(route.received_global_rows):
        raise ValueError("route receive splits do not cover the received rows")
    if len(route.consumer_from_received) != len(route.consumer_global_rows):
        raise ValueError("route consumer permutation has the wrong length")
    if len(route.received_from_consumer) != len(route.received_global_rows):
        raise ValueError("route received permutation has the wrong length")

    send_pack = make_dsa_copy_map(route.send_source_rows, route.producer_row_count)
    consumer_pack = make_dsa_copy_map(
        route.consumer_from_received, len(route.received_global_rows)
    )
    received_pack = make_dsa_copy_map(
        route.received_from_consumer, len(route.consumer_global_rows)
    )
    for consumer_row, received_row in enumerate(route.consumer_from_received):
        if route.received_from_consumer[received_row] != consumer_row:
            raise ValueError("route consumer and received maps are not inverses")

    owner_reduce = DsaHostReduceMap(
        route.reverse_row_offsets, route.reverse_source_rows
    )
    validate_dsa_reduce_map(
        owner_reduce, len(route.send_source_rows), route.producer_row_count
    )
    return DsaHostRouteMaps(send_pack, consumer_pack, received_pack, owner_reduce)


def copy_dsa_rows_reference(
    source: torch.Tensor, mapping: DsaHostCopyMap
) -> torch.Tensor:
    """PyTorch oracle for a destination-to-source row gather."""

    rows = torch.tensor(mapping.source_rows, dtype=torch.int64, device=source.device)
    return source.index_select(0, rows)


def reduce_dsa_rows_reference(
    source: torch.Tensor,
    mapping: DsaHostReduceMap,
) -> torch.Tensor:
    """PyTorch oracle for static CSR sum reduction."""

    output_rows = len(mapping.row_offsets) - 1
    result = torch.zeros(
        (output_rows, *source.shape[1:]), dtype=source.dtype, device=source.device
    )
    if mapping.source_rows:
        source_rows = torch.tensor(
            mapping.source_rows, dtype=torch.int64, device=source.device
        )
        destination_rows = torch.repeat_interleave(
            torch.arange(output_rows, dtype=torch.int64, device=source.device),
            torch.tensor(
                [
                    end - begin
                    for begin, end in zip(mapping.row_offsets, mapping.row_offsets[1:])
                ],
                dtype=torch.int64,
                device=source.device,
            ),
        )
        result.index_add_(0, destination_rows, source.index_select(0, source_rows))
    return result


def _to_int32(
    values: tuple[int, ...] | list[int], device: torch.device
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=device)


def _device_copy_map(
    mapping: DsaHostCopyMap,
    source_row_count: int,
    device: torch.device,
) -> DsaDeviceCopyMap:
    return DsaDeviceCopyMap(
        _to_int32(mapping.source_rows, device),
        len(mapping.source_rows) == source_row_count
        and mapping.source_rows == tuple(range(source_row_count)),
    )


def _device_reduce_map(
    mapping: DsaHostReduceMap, device: torch.device
) -> DsaDeviceReduceMap:
    return DsaDeviceReduceMap(
        row_offsets=_to_int32(mapping.row_offsets, device),
        source_rows=_to_int32(mapping.source_rows, device),
        is_identity=(
            mapping.row_offsets == tuple(range(len(mapping.row_offsets)))
            and mapping.source_rows == tuple(range(len(mapping.source_rows)))
        ),
    )


def make_dsa_device_route_plan(
    name: str,
    route: DsaRouteRankPlan,
    device: torch.device,
) -> DsaDeviceRoutePlan:
    """Upload one typed route once during cold plan materialization."""

    maps = make_dsa_route_maps(route)
    owner_restore = None
    if all(
        end - begin == 1
        for begin, end in zip(
            maps.owner_reduce.row_offsets, maps.owner_reduce.row_offsets[1:]
        )
    ):
        owner_restore = _device_copy_map(
            DsaHostCopyMap(maps.owner_reduce.source_rows),
            len(route.send_source_rows),
            device,
        )
    return DsaDeviceRoutePlan(
        name=name,
        rank=route.rank,
        producer_row_count=route.producer_row_count,
        send_counts=route.send_counts,
        recv_counts=route.recv_counts,
        send_pack=_device_copy_map(maps.send_pack, route.producer_row_count, device),
        consumer_pack=_device_copy_map(
            maps.consumer_pack, len(route.received_global_rows), device
        ),
        received_pack=_device_copy_map(
            maps.received_pack, len(route.consumer_global_rows), device
        ),
        owner_reduce=_device_reduce_map(maps.owner_reduce, device),
        owner_restore=owner_restore,
        consumer_global_rows=_to_int32(route.consumer_global_rows, device),
    )


def _make_compression_map(
    config: MagiDSAConfig,
    rank_plan: DsaRankPlan,
    device: torch.device,
) -> DsaDeviceCompressionMap | None:
    if not config.ratio:
        return None
    support = config.compressor_support
    expected = len(rank_plan.produced_blocks) * support
    if len(rank_plan.compression_source_from_overlap) != expected:
        raise ValueError("compression support map has an invalid row count")
    source_rows = [max(row, 0) for row in rank_plan.compression_source_from_overlap]
    valid_rows = [row >= 0 for row in rank_plan.compression_source_from_overlap]
    overlap_row_count = (
        0
        if rank_plan.overlap_x_route is None
        else len(rank_plan.overlap_x_route.consumer_global_rows)
    )
    source_pack = make_dsa_copy_map(tuple(source_rows), overlap_row_count)
    source_unpack = make_dsa_reduce_map(tuple(source_rows), overlap_row_count)
    validate_dsa_reduce_map(source_unpack, len(source_rows), overlap_row_count)
    return DsaDeviceCompressionMap(
        source_pack=_device_copy_map(source_pack, overlap_row_count, device),
        source_unpack=_device_reduce_map(source_unpack, device),
        valid_rows=torch.tensor(valid_rows, dtype=torch.bool, device=device),
        block_positions=_to_int32(
            [block.position for block in rank_plan.produced_blocks], device
        ),
    )


def _make_attention_map(
    plan: DsaExecutionPlan,
    rank_plan: DsaRankPlan,
    config: MagiDSAConfig,
    device: torch.device,
) -> DsaDeviceAttentionMap:
    window_position = {
        global_row: position
        for position, global_row in enumerate(
            rank_plan.window_route.consumer_global_rows
        )
    }
    compressed_rows = (
        ()
        if rank_plan.compressed_kv_route is None
        else rank_plan.compressed_kv_route.consumer_global_rows
    )
    total_tokens = plan.total_tokens
    token_global_to_consumer = torch.full(
        (total_tokens,), -1, dtype=torch.int32, device=device
    )
    if window_position:
        window_globals = _to_int32(list(window_position), device)
        token_global_to_consumer[window_globals.long()] = torch.arange(
            len(window_position), dtype=torch.int32, device=device
        )
    sample_ids = _to_int32(rank_plan.local_q_sample_ids, device).long()
    positions = _to_int32(rank_plan.local_q_positions, device)
    sample_begins = _to_int32(list(plan.cu_seqlens[:-1]), device)
    window_lengths = (positions + 1).clamp(max=config.window_size)
    window_columns = torch.arange(
        config.window_size, dtype=torch.int32, device=device
    ).unsqueeze(0)
    window_valid = window_columns < window_lengths.unsqueeze(1)
    window_positions = (
        positions.unsqueeze(1) - window_lengths.unsqueeze(1) + 1 + window_columns
    )
    window_globals = sample_begins[sample_ids].unsqueeze(1) + window_positions
    safe_window_globals = window_globals.masked_fill(~window_valid, 0)
    window_rows = token_global_to_consumer[safe_window_globals.long()].masked_fill(
        ~window_valid, -1
    )
    if bool((window_rows[window_valid] < 0).any().item()):
        raise ValueError("WINDOW_KV consumer union does not cover a local raw window")

    global_to_consumer = [-1] * plan.total_compressed_blocks
    for consumer_row, global_block in enumerate(compressed_rows):
        global_to_consumer[global_block] = consumer_row
    compressed_global_to_consumer = _to_int32(global_to_consumer, device)
    if config.ratio == 128:
        compressed_lengths = torch.div(
            positions + 1, config.ratio, rounding_mode="floor"
        )
        max_visible = (
            int(compressed_lengths.max().item()) if compressed_lengths.numel() else 0
        )
        compressed_columns = torch.arange(
            max_visible, dtype=torch.int32, device=device
        ).unsqueeze(0)
        compressed_valid = compressed_columns < compressed_lengths.unsqueeze(1)
        sample_block_offsets = _to_int32(list(rank_plan.sample_block_offsets), device)
        compressed_globals = (
            sample_block_offsets[sample_ids].unsqueeze(1) + compressed_columns
        )
        safe_compressed_globals = compressed_globals.masked_fill(~compressed_valid, 0)
        local_compressed = compressed_global_to_consumer[safe_compressed_globals.long()]
        local_compressed = local_compressed.masked_fill(~compressed_valid, -1)
        if bool((local_compressed[compressed_valid] < 0).any().item()):
            raise ValueError(
                "COMPRESSED_KV consumer union does not cover an HCA causal prefix"
            )
    else:
        compressed_lengths = torch.zeros_like(positions)
        local_compressed = torch.empty(
            (positions.numel(), 0), dtype=torch.int32, device=device
        )
    return DsaDeviceAttentionMap(
        window_rows=window_rows.contiguous(),
        window_lengths=window_lengths.contiguous(),
        compressed_rows=local_compressed.contiguous(),
        compressed_lengths=compressed_lengths.contiguous(),
        compressed_global_to_consumer=compressed_global_to_consumer,
    )


def _make_indexer_map(
    rank_plan: DsaRankPlan,
    device: torch.device,
) -> DsaDeviceIndexerMap | None:
    if rank_plan.compressed_ki_route is None:
        return None
    compressed_position = {
        global_row: position
        for position, global_row in enumerate(
            rank_plan.compressed_ki_route.consumer_global_rows
        )
    }
    k_pack_rows: list[int] = []
    for fragment in rank_plan.query_fragments:
        block_begin = rank_plan.sample_block_offsets[fragment.sample_id]
        k_pack_rows.extend(
            compressed_position[block_begin + block_offset]
            for block_offset in range(fragment.q_end // 4)
        )
    if len(k_pack_rows) != rank_plan.packed_indexer_k_count:
        raise ValueError("Indexer K pack rows do not match grouped cu_seqlens")
    k_pack = make_dsa_copy_map(tuple(k_pack_rows), len(compressed_position))
    total_compressed_blocks = sum(rank_plan.sample_block_counts)
    global_to_consumer = [-1] * total_compressed_blocks
    for global_block, consumer_row in compressed_position.items():
        global_to_consumer[global_block] = consumer_row
    return DsaDeviceIndexerMap(
        q_cu_seqlens=_to_int32(rank_plan.indexer_q_cu_seqlens, device),
        k_cu_seqlens=_to_int32(rank_plan.indexer_k_cu_seqlens, device),
        q_causal_offsets=_to_int32(rank_plan.indexer_q_causal_offsets, device),
        q_sample_block_offsets=_to_int32(
            rank_plan.indexer_q_sample_block_offsets, device
        ),
        seq_lens=_to_int32(rank_plan.indexer_seq_lens, device),
        k_pack=_device_copy_map(k_pack, len(compressed_position), device),
        ki_global_to_consumer=_to_int32(global_to_consumer, device),
        max_seqlen_q=rank_plan.indexer_max_seqlen_q,
        logical_max_seqlen_k=rank_plan.indexer_logical_max_seqlen_k,
        backend_max_seqlen_k=rank_plan.indexer_backend_max_seqlen_k,
    )


def make_dsa_device_rank_plan(
    plan: DsaExecutionPlan,
    rank: int,
    config: MagiDSAConfig,
    device: torch.device,
) -> DsaDeviceRankPlan:
    """Materialize every host layout needed by one warm execution handle."""

    if plan.ratio != config.ratio:
        raise ValueError("execution plan and DSA config ratios do not match")
    if not 0 <= rank < plan.cp_size:
        raise ValueError("rank is outside the execution plan")
    if device.type != "cuda":
        raise ValueError("production DSA device plans require CUDA")
    rank_plan = plan.rank_plans[rank]

    def route(name: str, value: DsaRouteRankPlan | None) -> DsaDeviceRoutePlan | None:
        return (
            None if value is None else make_dsa_device_route_plan(name, value, device)
        )

    window = make_dsa_device_route_plan("WINDOW_KV", rank_plan.window_route, device)
    return DsaDeviceRankPlan(
        rank=rank,
        source_token_count=rank_plan.source_token_count,
        local_token_count=rank_plan.local_token_count,
        local_q_sample_ids=_to_int32(rank_plan.local_q_sample_ids, device),
        local_q_positions=_to_int32(rank_plan.local_q_positions, device),
        compression=_make_compression_map(config, rank_plan, device),
        attention=_make_attention_map(plan, rank_plan, config, device),
        indexer=_make_indexer_map(rank_plan, device),
        token_layout_route=route("TOKEN_LAYOUT", rank_plan.token_layout_route),
        window_route=window,
        overlap_x_route=route("OVERLAP_X", rank_plan.overlap_x_route),
        compressed_kv_route=route("COMPRESSED_KV", rank_plan.compressed_kv_route),
        compressed_ki_route=route("COMPRESSED_KI", rank_plan.compressed_ki_route),
    )


def copy_dsa_device_map(
    source: torch.Tensor, mapping: DsaDeviceCopyMap
) -> torch.Tensor:
    """Run the CuTe row gather using device-resident metadata."""

    if mapping.is_identity:
        return source

    from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows

    with dsa_nvtx_range("packing::cute_row_copy", enabled=source.is_cuda):
        return copy_dsa_rows(source, mapping.source_rows)


def reduce_dsa_device_map(
    source: torch.Tensor, mapping: DsaDeviceReduceMap
) -> torch.Tensor:
    """Run the CuTe FP32-accumulating CSR reduction."""

    if mapping.is_identity:
        return source

    from magi_attention.kernel.cutedsl.dsa_pack import reduce_dsa_rows

    with dsa_nvtx_range("packing::cute_csr_reduce", enabled=source.is_cuda):
        return reduce_dsa_rows(source, mapping.row_offsets, mapping.source_rows)


__all__ = [
    "DsaDeviceAttentionMap",
    "DsaDeviceCompressionMap",
    "DsaDeviceCopyMap",
    "DsaDeviceIndexerMap",
    "DsaDeviceRankPlan",
    "DsaDeviceReduceMap",
    "DsaDeviceRoutePlan",
    "DsaHostCopyMap",
    "DsaHostReduceMap",
    "DsaHostRouteMaps",
    "copy_dsa_device_map",
    "copy_dsa_rows_reference",
    "make_dsa_copy_map",
    "make_dsa_device_rank_plan",
    "make_dsa_device_route_plan",
    "make_dsa_reduce_map",
    "make_dsa_route_maps",
    "reduce_dsa_device_map",
    "reduce_dsa_rows_reference",
    "validate_dsa_reduce_map",
]
