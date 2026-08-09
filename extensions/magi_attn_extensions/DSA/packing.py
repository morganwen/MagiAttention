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

"""Cold materialization of one rank's device metadata.

The host plan is range shaped, so everything here is either a small range table
or a per-query base/length pair. Nothing materializes a padded per-query index
matrix: a raw window and a causal-visible compressed prefix are both contiguous
runs in their consumer bank, so a base and a length describe them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from magi_attention import env
from magi_attention.meta.collection.comm_meta import (
    A2AVBasedGroupCollectiveArg,
    GroupCollectiveArg,
)
from magi_attention.meta.solver.dynamic_attn_solver import DynamicAttnSolver
from magi_attention.utils.general import _make_device_tensor

from .config import MagiDSAConfig
from .meta import DsaExecutionPlan, DsaRankPlan, DsaRoutePlan


@dataclass(frozen=True)
class DsaDeviceRoutePlan:
    """One rank's lowered group-collective plan for a typed payload route."""

    name: str
    rank: int
    cp_size: int
    producer_row_count: int
    consumer_row_count: int
    # ``None`` only for a one-rank CP group, which needs no collective.
    group_collective_arg: GroupCollectiveArg | None
    # One-rank fallback. A one-rank route is not always the identity: a sample
    # tail shorter than the compression ratio produces no block, so OVERLAP_X
    # consumes fewer rows than the Query buffer holds. These are the consumer
    # ranges in producer-local coordinates.
    local_gather_ranges: torch.Tensor | None
    # Maps a global row id to its consumer-local row, or -1 when this rank does
    # not receive that row. Sized by the route's global row domain, not by the
    # number of received rows.
    global_to_consumer: torch.Tensor


@dataclass(frozen=True)
class DsaDeviceCompressionMap:
    """Compressor support rows as one contiguous run per produced block."""

    # Consumer row that support column 0 maps to; leading columns of a
    # sample-initial CSA block are masked instead of read.
    support_offset: torch.Tensor
    valid_rows: torch.Tensor
    block_positions: torch.Tensor


@dataclass(frozen=True)
class DsaDeviceAttentionMap:
    """Per-query base and length of the raw window and compressed prefix."""

    window_base: torch.Tensor
    window_length: torch.Tensor
    compressed_base: torch.Tensor
    compressed_length: torch.Tensor
    compressed_global_to_consumer: torch.Tensor
    # Plan-time bound on the causal-visible compressed prefix, so the warm path
    # can size its index matrix without a device reduction.
    max_compressed_length: int


@dataclass(frozen=True)
class DsaDeviceIndexerMap:
    """Grouped Indexer varlen metadata and its forward-only prefix gather."""

    q_cu_seqlens: torch.Tensor
    k_cu_seqlens: torch.Tensor
    q_causal_offsets: torch.Tensor
    q_sample_block_offsets: torch.Tensor
    seq_lens: torch.Tensor
    # One [begin, end) consumer-row run per fragment; a range gather expands the
    # unique KI bank into the grouped per-fragment prefixes cuDNN expects.
    k_gather_ranges: torch.Tensor
    packed_k_rows: int
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
    compression: DsaDeviceCompressionMap
    attention: DsaDeviceAttentionMap
    indexer: DsaDeviceIndexerMap | None
    token_layout_route: DsaDeviceRoutePlan
    window_route: DsaDeviceRoutePlan
    overlap_x_route: DsaDeviceRoutePlan
    compressed_kv_route: DsaDeviceRoutePlan
    compressed_ki_route: DsaDeviceRoutePlan | None


def _make_global_to_consumer(
    route: DsaRoutePlan,
    rank: int,
    row_domain: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the global-row to consumer-row lookup for one received bank."""

    table = torch.full((row_domain,), -1, dtype=torch.int32, device=device)
    cursor = 0
    for begin, end in route.consumer_ranges_per_rank[rank]:
        length = end - begin
        table[begin:end] = torch.arange(
            cursor, cursor + length, dtype=torch.int32, device=device
        )
        cursor += length
    return table


def make_dsa_device_route_plan(
    route: DsaRoutePlan,
    rank: int,
    row_domain: int,
    device: torch.device,
    *,
    cp_group,
    deterministic: bool,
) -> DsaDeviceRoutePlan:
    """Lower one range-shaped route to a Core group-collective plan."""

    cp_size = route.cp_size
    arg: GroupCollectiveArg | None = None
    if cp_size > 1:
        if env.comm.is_native_grpcoll_enable():
            raise RuntimeError("Magi-DSA routes do not support native grpcoll")
        host_arg = DynamicAttnSolver._calc_group_collective_arg_from_ranges(
            host_ranges=route.owner_attn_ranges(),
            calc_ranges_per_rank=route.consumer_attn_ranges(),
            cp_rank=rank,
            cp_size=cp_size,
            cp_group=cp_group,
            cp_mesh=None,
            deterministic=deterministic,
            split_alignment=1,
            calc_local_range=True,
        )
        if sum(host_arg.input_split_size_list) != route.owner_row_count(rank):
            raise ValueError(f"{route.name}: group-cast input does not match the owner")
        if sum(host_arg.output_split_size_list) != route.consumer_row_count(rank):
            raise ValueError(
                f"{route.name}: group-cast output does not match the consumer"
            )
        # Each route owns an independent device handle so its forward group-cast
        # and backward group-reduce never share buffers with a sibling route.
        arg = A2AVBasedGroupCollectiveArg(
            input_split_size_list=list(host_arg.input_split_size_list),
            output_split_size_list=list(host_arg.output_split_size_list),
            dst_indices_list=[list(indices) for indices in host_arg.dst_indices_list],
            src_index_list=list(host_arg.src_index_list),
            rank=host_arg.rank,
            world_size=host_arg.world_size,
            group=host_arg.group,
            device_mesh=host_arg.device_mesh,
            deterministic=host_arg.deterministic,
            split_alignment=host_arg.split_alignment,
            packed_times=1,
            reduce_op="sum",
            init_group_reduce=True,
        )
    local_gather_ranges = None
    if cp_size == 1:
        local_gather_ranges = _make_device_tensor(
            _local_gather_ranges(route, rank) or [[0, 0]],
            dtype=torch.int32,
            device=device,
        )
    return DsaDeviceRoutePlan(
        name=route.name,
        rank=rank,
        cp_size=cp_size,
        producer_row_count=route.owner_row_count(rank),
        consumer_row_count=route.consumer_row_count(rank),
        group_collective_arg=arg,
        local_gather_ranges=local_gather_ranges,
        global_to_consumer=_make_global_to_consumer(route, rank, row_domain, device),
    )


def _local_gather_ranges(route: DsaRoutePlan, rank: int) -> list[list[int]]:
    """Express the consumer ranges in this rank's producer-local coordinates.

    A consumer range is merged across samples while owner ranges are split at
    sample boundaries, so one consumer range can cover several owner ranges and
    has to be intersected rather than matched whole.
    """

    owner = route.owner_ranges_per_rank[rank]
    bases: list[int] = []
    cursor = 0
    for begin, end in owner:
        bases.append(cursor)
        cursor += end - begin
    ranges: list[list[int]] = []
    for begin, end in route.consumer_ranges_per_rank[rank]:
        covered = 0
        for index, (owner_begin, owner_end) in enumerate(owner):
            lo = max(begin, owner_begin)
            hi = min(end, owner_end)
            if lo >= hi:
                continue
            offset = bases[index] - owner_begin
            ranges.append([lo + offset, hi + offset])
            covered += hi - lo
        if covered != end - begin:
            raise ValueError(
                f"{route.name}: a one-rank consumer range is not owned locally"
            )
    return ranges


def _expand_query_metadata(
    rank_plan: DsaRankPlan,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand fragment bounds into per-query sample ids and positions.

    The host plan only stores fragments, so the two per-token tables are built
    here with one arange and one repeat_interleave instead of being shipped as
    Python tuples.
    """

    fragments = rank_plan.query_fragments
    if not fragments:
        empty = torch.empty((0,), dtype=torch.int32, device=device)
        return empty, empty.clone()
    lengths = _make_device_tensor(
        [fragment.length for fragment in fragments], dtype=torch.int64, device=device
    )
    sample_ids = _make_device_tensor(
        [fragment.sample_id for fragment in fragments],
        dtype=torch.int32,
        device=device,
    )
    q_begins = _make_device_tensor(
        [fragment.q_begin for fragment in fragments], dtype=torch.int32, device=device
    )
    local_sample_ids = torch.repeat_interleave(sample_ids, lengths)
    fragment_begin = torch.repeat_interleave(
        torch.cumsum(lengths, dim=0) - lengths, lengths
    )
    within = (
        torch.arange(int(lengths.sum().item()), device=device, dtype=torch.int64)
        - fragment_begin
    )
    local_positions = (
        torch.repeat_interleave(q_begins, lengths) + within.to(torch.int32)
    ).to(torch.int32)
    return local_sample_ids.contiguous(), local_positions.contiguous()


def _make_compression_map(
    config: MagiDSAConfig,
    plan: DsaExecutionPlan,
    rank_plan: DsaRankPlan,
    overlap_route: DsaDeviceRoutePlan,
    device: torch.device,
) -> DsaDeviceCompressionMap:
    """Locate each produced block's support run inside the OVERLAP_X bank."""

    ratio = config.ratio
    support = config.compressor_support
    overlap = ratio == 4
    # ``anchor`` is the global row that support column ``leading`` maps to, so
    # the run is ``consumer(anchor) - leading + column``. A CSA block that opens
    # a sample has no previous group: its first ``ratio`` columns are masked and
    # the anchor becomes the current group.
    anchors: list[int] = []
    leadings: list[int] = []
    valid: list[list[bool]] = []
    positions: list[int] = []
    cu_seqlens = plan.cu_seqlens
    lookup = overlap_route.global_to_consumer
    for block_begin, block_end in rank_plan.produced_block_ranges:
        # Sound because a produced block range never straddles a sample.
        sample_id = _sample_of_block(rank_plan.sample_block_offsets, block_begin)
        sample_global_begin = cu_seqlens[sample_id]
        sample_block_begin = rank_plan.sample_block_offsets[sample_id]
        for global_block in range(block_begin, block_end):
            sample_block_id = global_block - sample_block_begin
            group_begin = sample_global_begin + sample_block_id * ratio
            positions.append(sample_block_id * ratio)
            if overlap and sample_block_id == 0:
                anchors.append(group_begin)
                leadings.append(ratio)
                valid.append([False] * ratio + [True] * ratio)
            else:
                anchors.append(group_begin - (ratio if overlap else 0))
                leadings.append(0)
                valid.append([True] * support)
    if not anchors:
        return DsaDeviceCompressionMap(
            support_offset=torch.empty((0,), dtype=torch.int32, device=device),
            valid_rows=torch.empty((0, support), dtype=torch.bool, device=device),
            block_positions=torch.empty((0,), dtype=torch.int32, device=device),
        )
    anchor_consumer = lookup[
        _make_device_tensor(anchors, dtype=torch.int64, device=device)
    ].to(torch.int32)
    if bool((anchor_consumer < 0).any().item()):
        raise ValueError("OVERLAP_X consumer bank misses a compressor support row")
    support_offset = anchor_consumer - _make_device_tensor(
        leadings, dtype=torch.int32, device=device
    )
    return DsaDeviceCompressionMap(
        support_offset=support_offset.contiguous(),
        valid_rows=_make_device_tensor(valid, dtype=torch.bool, device=device),
        block_positions=_make_device_tensor(
            positions, dtype=torch.int32, device=device
        ),
    )


def _sample_of_block(sample_block_offsets: tuple[int, ...], global_block: int) -> int:
    import bisect

    return bisect.bisect_right(sample_block_offsets, global_block) - 1


def _make_attention_map(
    plan: DsaExecutionPlan,
    rank_plan: DsaRankPlan,
    config: MagiDSAConfig,
    window_route: DsaDeviceRoutePlan,
    compressed_kv_route: DsaDeviceRoutePlan,
    local_q_sample_ids: torch.Tensor,
    local_q_positions: torch.Tensor,
    device: torch.device,
) -> DsaDeviceAttentionMap:
    """Reduce both attention banks to one base and one length per query."""

    sample_ids = local_q_sample_ids.to(torch.int64)
    positions = local_q_positions
    sample_begins = _make_device_tensor(
        list(plan.cu_seqlens[:-1]), dtype=torch.int32, device=device
    )
    sample_block_offsets = _make_device_tensor(
        list(rank_plan.sample_block_offsets), dtype=torch.int32, device=device
    )

    window_length = (positions + 1).clamp(max=config.window_size)
    window_global_begin = (
        sample_begins[sample_ids] + positions - window_length + 1
    ).to(torch.int64)
    window_base = window_route.global_to_consumer[window_global_begin]
    if positions.numel() and bool((window_base < 0).any().item()):
        raise ValueError("WINDOW_KV consumer bank does not cover a local raw window")

    compressed_length = torch.div(
        positions + 1, config.ratio, rounding_mode="floor"
    ).to(torch.int32)
    sample_block_begin = sample_block_offsets[sample_ids].to(torch.int64)
    compressed_base = compressed_kv_route.global_to_consumer[sample_block_begin]
    # A query whose causal prefix is empty never reads the bank, so a -1 base is
    # only a defect when that query actually has visible compressed rows.
    if positions.numel() and bool(
        ((compressed_base < 0) & (compressed_length > 0)).any().item()
    ):
        raise ValueError("COMPRESSED_KV consumer bank misses a causal prefix")

    max_compressed_length = max(
        (
            fragment.q_end // config.ratio
            for fragment in rank_plan.query_fragments
        ),
        default=0,
    )
    return DsaDeviceAttentionMap(
        window_base=window_base.to(torch.int32).contiguous(),
        window_length=window_length.contiguous(),
        compressed_base=compressed_base.to(torch.int32).contiguous(),
        compressed_length=compressed_length.contiguous(),
        compressed_global_to_consumer=compressed_kv_route.global_to_consumer,
        max_compressed_length=max_compressed_length,
    )


def _make_indexer_map(
    plan: DsaExecutionPlan,
    rank_plan: DsaRankPlan,
    config: MagiDSAConfig,
    compressed_ki_route: DsaDeviceRoutePlan,
    local_q_sample_ids: torch.Tensor,
    local_q_positions: torch.Tensor,
    device: torch.device,
) -> DsaDeviceIndexerMap:
    """Build grouped Indexer metadata and the per-fragment prefix gather."""

    ranges: list[list[int]] = []
    for fragment in rank_plan.query_fragments:
        block_begin = rank_plan.sample_block_offsets[fragment.sample_id]
        length = fragment.q_end // config.ratio
        base = int(compressed_ki_route.global_to_consumer[block_begin].item())
        if length and base < 0:
            raise ValueError("COMPRESSED_KI consumer bank misses a fragment prefix")
        ranges.append([max(base, 0), max(base, 0) + length])
    packed_k_rows = sum(end - begin for begin, end in ranges)
    if packed_k_rows != rank_plan.packed_indexer_k_count:
        raise ValueError("Indexer K gather rows do not match grouped cu_seqlens")

    sample_ids = local_q_sample_ids.to(torch.int64)
    sample_block_offsets = _make_device_tensor(
        list(rank_plan.sample_block_offsets), dtype=torch.int32, device=device
    )
    seq_lens = torch.div(
        local_q_positions + 1, config.ratio, rounding_mode="floor"
    ).to(torch.int32)
    return DsaDeviceIndexerMap(
        q_cu_seqlens=_make_device_tensor(
            list(rank_plan.indexer_q_cu_seqlens), dtype=torch.int32, device=device
        ),
        k_cu_seqlens=_make_device_tensor(
            list(rank_plan.indexer_k_cu_seqlens), dtype=torch.int32, device=device
        ),
        q_causal_offsets=_make_device_tensor(
            list(rank_plan.indexer_q_causal_offsets), dtype=torch.int32, device=device
        ),
        q_sample_block_offsets=sample_block_offsets[sample_ids]
        .to(torch.int32)
        .contiguous(),
        seq_lens=seq_lens.contiguous(),
        k_gather_ranges=_make_device_tensor(
            ranges if ranges else [[0, 0]], dtype=torch.int32, device=device
        ),
        packed_k_rows=packed_k_rows,
        ki_global_to_consumer=compressed_ki_route.global_to_consumer,
        max_seqlen_q=rank_plan.indexer_max_seqlen_q,
        logical_max_seqlen_k=rank_plan.indexer_logical_max_seqlen_k,
        backend_max_seqlen_k=rank_plan.indexer_backend_max_seqlen_k,
    )


def make_dsa_device_rank_plan(
    plan: DsaExecutionPlan,
    rank: int,
    config: MagiDSAConfig,
    device: torch.device,
    *,
    cp_group=None,
    deterministic: bool = False,
) -> DsaDeviceRankPlan:
    """Materialize every host layout needed by one warm execution handle."""

    if plan.ratio != config.ratio:
        raise ValueError("execution plan and DSA config ratios do not match")
    if not 0 <= rank < plan.cp_size:
        raise ValueError("rank is outside the execution plan")
    if device.type != "cuda":
        raise ValueError("production DSA device plans require CUDA")
    rank_plan = plan.rank_plans[rank]

    def lower(route: DsaRoutePlan, row_domain: int) -> DsaDeviceRoutePlan:
        return make_dsa_device_route_plan(
            route,
            rank,
            row_domain,
            device,
            cp_group=cp_group,
            deterministic=deterministic,
        )

    tokens = plan.total_tokens
    blocks = plan.total_compressed_blocks
    token_layout_route = lower(plan.token_layout_route, tokens)
    window_route = lower(plan.window_route, tokens)
    overlap_x_route = lower(plan.overlap_x_route, tokens)
    compressed_kv_route = lower(plan.compressed_kv_route, blocks)
    compressed_ki_route = (
        None
        if plan.compressed_ki_route is None
        else lower(plan.compressed_ki_route, blocks)
    )

    local_q_sample_ids, local_q_positions = _expand_query_metadata(rank_plan, device)
    return DsaDeviceRankPlan(
        rank=rank,
        source_token_count=rank_plan.source_token_count,
        local_token_count=rank_plan.local_token_count,
        local_q_sample_ids=local_q_sample_ids,
        local_q_positions=local_q_positions,
        compression=_make_compression_map(
            config, plan, rank_plan, overlap_x_route, device
        ),
        attention=_make_attention_map(
            plan,
            rank_plan,
            config,
            window_route,
            compressed_kv_route,
            local_q_sample_ids,
            local_q_positions,
            device,
        ),
        indexer=(
            None
            if compressed_ki_route is None
            else _make_indexer_map(
                plan,
                rank_plan,
                config,
                compressed_ki_route,
                local_q_sample_ids,
                local_q_positions,
                device,
            )
        ),
        token_layout_route=token_layout_route,
        window_route=window_route,
        overlap_x_route=overlap_x_route,
        compressed_kv_route=compressed_kv_route,
        compressed_ki_route=compressed_ki_route,
    )


def gather_compressor_support(
    overlap_x: torch.Tensor,
    compression: DsaDeviceCompressionMap,
    support: int,
) -> torch.Tensor:
    """Gather each block's support run, keeping the gradient path differentiable.

    ``index_select`` already accumulates duplicate source rows in backward, so
    the CSA overlap needs no separate CSR reduction.
    """

    if compression.support_offset.numel() == 0:
        return overlap_x.new_empty((0, support, overlap_x.shape[-1]))
    columns = torch.arange(support, device=overlap_x.device, dtype=torch.int32)
    rows = compression.support_offset.unsqueeze(1) + columns.unsqueeze(0)
    rows = rows.clamp_(0, overlap_x.shape[0] - 1).reshape(-1).to(torch.int64)
    return overlap_x.index_select(0, rows).view(-1, support, overlap_x.shape[-1])


__all__ = [
    "DsaDeviceAttentionMap",
    "DsaDeviceCompressionMap",
    "DsaDeviceIndexerMap",
    "DsaDeviceRankPlan",
    "DsaDeviceRoutePlan",
    "gather_compressor_support",
    "make_dsa_device_rank_plan",
    "make_dsa_device_route_plan",
]
