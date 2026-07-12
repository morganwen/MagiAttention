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

"""Build and validate Magi_DSA fragment, restore and transfer plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from magi_attention.meta.collection.dsa_meta import (
    DsaCompressedBlockSpec,
    DsaDispatchPlan,
    DsaFragmentSpec,
    DsaRankPlan,
    DsaRestoreSpec,
    DsaTransferSpec,
    sample_offsets,
)


@dataclass(frozen=True)
class DsaCostModel:
    """Uncalibrated step-2 predictor; B300 coefficients are filled in step 8.

    Compressed communication weights are charged per logical compressed block:
    owner sends count every remote destination, while receives count unique
    remote blocks.  Memory coefficients separately model the owner result,
    packed send buffer, receive buffer and globally reordered resident tensor.
    A ratio=4 runtime configures one block as the combined KV+Ki row size;
    ratio=128 configures KV only.
    """

    token_weight: float = 1.0
    indexer_weight: float = 1.0
    fragment_overhead: float = 64.0
    window_transfer_weight: float = 1.0
    overlap_transfer_weight: float = 1.0
    token_memory_bytes: int = 1
    compressed_block_memory_bytes: int = 1
    remote_row_memory_bytes: int = 1
    # Keep new coefficients after the original positional fields so existing
    # callers that did not use keywords retain their argument mapping.
    compressed_owner_send_weight: float = 1.0
    compressed_remote_receive_weight: float = 1.0
    compressed_owner_send_memory_bytes: int = 1
    compressed_remote_receive_memory_bytes: int = 1
    compressed_global_memory_bytes: int = 1

    def __post_init__(self) -> None:
        numeric = (
            self.token_weight,
            self.indexer_weight,
            self.fragment_overhead,
            self.window_transfer_weight,
            self.overlap_transfer_weight,
            self.compressed_owner_send_weight,
            self.compressed_remote_receive_weight,
        )
        if any(value < 0 for value in numeric):
            raise ValueError("DSA cost weights must be non-negative")
        memory = (
            self.token_memory_bytes,
            self.compressed_block_memory_bytes,
            self.compressed_owner_send_memory_bytes,
            self.compressed_remote_receive_memory_bytes,
            self.compressed_global_memory_bytes,
            self.remote_row_memory_bytes,
        )
        if any(value < 0 for value in memory):
            raise ValueError("DSA memory coefficients must be non-negative")


def indexer_prefix_cost(position_count: int, compress_ratio: int = 4) -> int:
    """Return ``sum(floor((p+1)/ratio), p=0..position_count-1)``."""

    if position_count < 0:
        raise ValueError("position_count must be non-negative")
    if compress_ratio <= 0:
        return 0
    quotient, remainder = divmod(position_count, compress_ratio)
    return compress_ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def fragment_indexer_cost(fragment: DsaFragmentSpec, compress_ratio: int) -> int:
    """Sample-relative Indexer scan work for one fragment.

    Only ratio=4 has a Lightning Indexer. ratio=0 and ratio=128 therefore
    contribute zero even though ratio=128 has compressed attention work.
    """

    if compress_ratio != 4:
        return 0
    return indexer_prefix_cost(fragment.q_end, compress_ratio) - indexer_prefix_cost(
        fragment.q_begin, compress_ratio
    )


def make_atomic_fragments(
    sample_lengths: Sequence[int], alignment: int = 128
) -> tuple[DsaFragmentSpec, ...]:
    """Split samples at legal, sample-relative alignment boundaries."""

    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    atoms: list[DsaFragmentSpec] = []
    for sample_id, length in enumerate(sample_lengths):
        if length < 0:
            raise ValueError(f"sample lengths must be non-negative, got {length}")
        for begin in range(0, int(length), alignment):
            atoms.append(
                DsaFragmentSpec(sample_id, begin, min(begin + alignment, int(length)))
            )
    return tuple(atoms)


def _canonical_fragments(
    fragments: Iterable[DsaFragmentSpec],
) -> tuple[DsaFragmentSpec, ...]:
    ordered = sorted(fragments)
    merged: list[DsaFragmentSpec] = []
    for fragment in ordered:
        if merged and merged[-1].can_merge(fragment):
            merged[-1] = merged[-1].merge(fragment)
        else:
            merged.append(fragment)
    return tuple(merged)


def _owner_segments(
    fragments_per_rank: Sequence[Sequence[DsaFragmentSpec]],
    sample_count: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    per_sample: list[list[tuple[int, int, int]]] = [[] for _ in range(sample_count)]
    for rank, fragments in enumerate(fragments_per_rank):
        for fragment in fragments:
            per_sample[fragment.sample_id].append(
                (fragment.q_begin, fragment.q_end, rank)
            )
    return tuple(tuple(sorted(segments)) for segments in per_sample)


def _rank_holding_position(
    segments: Sequence[tuple[int, int, int]], position: int
) -> int:
    for begin, end, rank in segments:
        if begin <= position < end:
            return rank
    raise ValueError(f"no fragment owns sample-relative position {position}")


def _intersect_owners(
    segments: Sequence[tuple[int, int, int]],
    begin: int,
    end: int,
    destination_rank: int,
    sample_id: int,
) -> list[DsaTransferSpec]:
    transfers: list[DsaTransferSpec] = []
    for owner_begin, owner_end, source_rank in segments:
        intersection_begin = max(begin, owner_begin)
        intersection_end = min(end, owner_end)
        if intersection_begin < intersection_end and source_rank != destination_rank:
            transfers.append(
                DsaTransferSpec(
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    sample_id=sample_id,
                    q_begin=intersection_begin,
                    q_end=intersection_end,
                )
            )
    return transfers


def _merge_transfers(
    transfers: Iterable[DsaTransferSpec],
) -> tuple[DsaTransferSpec, ...]:
    ordered = sorted(transfers)
    merged: list[DsaTransferSpec] = []
    for transfer in ordered:
        if not merged:
            merged.append(transfer)
            continue
        previous = merged[-1]
        same_route = (
            previous.source_rank == transfer.source_rank
            and previous.destination_rank == transfer.destination_rank
            and previous.sample_id == transfer.sample_id
        )
        if same_route and transfer.q_begin <= previous.q_end:
            merged[-1] = DsaTransferSpec(
                source_rank=previous.source_rank,
                destination_rank=previous.destination_rank,
                sample_id=previous.sample_id,
                q_begin=previous.q_begin,
                q_end=max(previous.q_end, transfer.q_end),
            )
        else:
            merged.append(transfer)
    return tuple(merged)


def _make_compressed_blocks(
    sample_lengths: Sequence[int],
    compress_ratio: int,
    owners: Sequence[Sequence[tuple[int, int, int]]],
) -> tuple[DsaCompressedBlockSpec, ...]:
    if compress_ratio == 0:
        return ()
    if compress_ratio not in (4, 128):
        raise ValueError(f"unsupported compress ratio {compress_ratio}")

    blocks: list[DsaCompressedBlockSpec] = []
    logical_block_id = 0
    for sample_id, length in enumerate(sample_lengths):
        for sample_block_id in range(length // compress_ratio):
            last_token = (sample_block_id + 1) * compress_ratio - 1
            blocks.append(
                DsaCompressedBlockSpec(
                    logical_block_id=logical_block_id,
                    sample_id=sample_id,
                    sample_block_id=sample_block_id,
                    owner_rank=_rank_holding_position(owners[sample_id], last_token),
                )
            )
            logical_block_id += 1
    return tuple(blocks)


def _make_restore_map(
    fragments_per_rank: Sequence[Sequence[DsaFragmentSpec]],
    sample_lengths: Sequence[int],
) -> tuple[DsaRestoreSpec, ...]:
    offsets = sample_offsets(sample_lengths)
    restore: list[DsaRestoreSpec] = []
    for rank, fragments in enumerate(fragments_per_rank):
        local_begin = 0
        for fragment in fragments:
            restore.append(
                DsaRestoreSpec(
                    rank=rank,
                    local_begin=local_begin,
                    global_begin=offsets[fragment.sample_id] + fragment.q_begin,
                    length=fragment.token_count,
                )
            )
            local_begin += fragment.token_count
    return tuple(sorted(restore, key=lambda entry: entry.global_begin))


def _make_window_transfers(
    fragments_per_rank: Sequence[Sequence[DsaFragmentSpec]],
    owners: Sequence[Sequence[tuple[int, int, int]]],
    window_size: int,
) -> tuple[DsaTransferSpec, ...]:
    transfers: list[DsaTransferSpec] = []
    for destination_rank, fragments in enumerate(fragments_per_rank):
        for fragment in fragments:
            begin = max(0, fragment.q_begin - (window_size - 1))
            end = fragment.q_begin
            transfers.extend(
                _intersect_owners(
                    owners[fragment.sample_id],
                    begin,
                    end,
                    destination_rank,
                    fragment.sample_id,
                )
            )
    return _merge_transfers(transfers)


def _make_overlap_transfers(
    compressed_blocks: Sequence[DsaCompressedBlockSpec],
    owners: Sequence[Sequence[tuple[int, int, int]]],
    compress_ratio: int,
) -> tuple[DsaTransferSpec, ...]:
    if compress_ratio != 4:
        return ()
    transfers: list[DsaTransferSpec] = []
    for block in compressed_blocks:
        if block.sample_block_id == 0:
            continue
        end = block.sample_block_id * compress_ratio
        begin = end - compress_ratio
        transfers.extend(
            _intersect_owners(
                owners[block.sample_id],
                begin,
                end,
                block.owner_rank,
                block.sample_id,
            )
        )
    return _merge_transfers(transfers)


def build_dsa_dispatch_plan(
    sample_lengths: Sequence[int],
    fragments_per_rank: Sequence[Sequence[DsaFragmentSpec]],
    *,
    compress_ratio: int,
    policy: str,
    alignment: int = 128,
    window_size: int = 128,
    cost_model: DsaCostModel | None = None,
) -> DsaDispatchPlan:
    """Materialize all owner, restore, transfer and predicted-load metadata."""

    if not fragments_per_rank:
        raise ValueError("fragments_per_rank must contain at least one rank")
    if not policy:
        raise ValueError("policy must be non-empty")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    model = cost_model or DsaCostModel()
    lengths = tuple(int(length) for length in sample_lengths)
    if any(length < 0 for length in lengths):
        raise ValueError("sample lengths must be non-negative")
    for fragments in fragments_per_rank:
        for fragment in fragments:
            if fragment.sample_id >= len(lengths):
                raise ValueError(f"invalid sample id {fragment.sample_id}")
            sample_length = lengths[fragment.sample_id]
            if fragment.q_end > sample_length:
                raise ValueError(f"fragment exceeds sample {fragment.sample_id}")
            if fragment.q_begin != 0 and fragment.q_begin % alignment:
                raise ValueError(f"fragment begin is not {alignment}-aligned")
            if fragment.q_end != sample_length and fragment.q_end % alignment:
                raise ValueError(f"fragment end is not {alignment}-aligned")
    canonical = tuple(
        _canonical_fragments(fragments) for fragments in fragments_per_rank
    )
    owners = _owner_segments(canonical, len(lengths))
    blocks = _make_compressed_blocks(lengths, compress_ratio, owners)
    restore_map = _make_restore_map(canonical, lengths)
    window_transfers = _make_window_transfers(canonical, owners, window_size)
    overlap_transfers = _make_overlap_transfers(blocks, owners, compress_ratio)

    blocks_per_rank: list[list[int]] = [[] for _ in canonical]
    for block in blocks:
        blocks_per_rank[block.owner_rank].append(block.logical_block_id)
    window_rows = [0 for _ in canonical]
    overlap_rows = [0 for _ in canonical]
    for transfer in window_transfers:
        window_rows[transfer.destination_rank] += transfer.row_count
    for transfer in overlap_transfers:
        overlap_rows[transfer.destination_rank] += transfer.row_count

    ranks: list[DsaRankPlan] = []
    total_compressed_blocks = len(blocks)
    for rank, fragments in enumerate(canonical):
        token_count = sum(fragment.token_count for fragment in fragments)
        indexer_cost = sum(
            fragment_indexer_cost(fragment, compress_ratio) for fragment in fragments
        )
        local_compressed_blocks = len(blocks_per_rank[rank])
        # Compressed KV (and Ki for ratio=4) are broadcast to every peer.  The
        # transport packs each owner row once, but its network send volume has
        # one copy per remote destination.  Every rank then retains all blocks
        # in logical order for its local attention/Indexer work.
        compressed_owner_send_rows = local_compressed_blocks * (len(canonical) - 1)
        compressed_remote_receive_rows = (
            total_compressed_blocks - local_compressed_blocks
        )
        predicted_e2e = (
            model.token_weight * token_count
            + model.indexer_weight * indexer_cost
            + model.fragment_overhead * len(fragments)
            + model.window_transfer_weight * window_rows[rank]
            + model.overlap_transfer_weight * overlap_rows[rank]
            + model.compressed_owner_send_weight * compressed_owner_send_rows
            + model.compressed_remote_receive_weight * compressed_remote_receive_rows
        )
        estimated_memory_bytes = (
            model.token_memory_bytes * token_count
            + model.compressed_block_memory_bytes * local_compressed_blocks
            + model.compressed_owner_send_memory_bytes
            * local_compressed_blocks
            * int(len(canonical) > 1)
            + model.compressed_remote_receive_memory_bytes
            * compressed_remote_receive_rows
            + model.compressed_global_memory_bytes * total_compressed_blocks
            + model.remote_row_memory_bytes * (window_rows[rank] + overlap_rows[rank])
        )
        ranks.append(
            DsaRankPlan(
                rank=rank,
                fragments=fragments,
                compressed_block_ids=tuple(blocks_per_rank[rank]),
                token_count=token_count,
                indexer_cost=indexer_cost,
                predicted_e2e=predicted_e2e,
                estimated_memory_bytes=estimated_memory_bytes,
            )
        )

    plan = DsaDispatchPlan(
        policy=policy,
        cp_size=len(canonical),
        compress_ratio=compress_ratio,
        alignment=alignment,
        window_size=window_size,
        sample_lengths=lengths,
        ranks=tuple(ranks),
        compressed_blocks=blocks,
        restore_map=restore_map,
        window_transfers=window_transfers,
        overlap_transfers=overlap_transfers,
    )
    validate_dsa_dispatch_plan(plan)
    return plan


def make_sequential_plan(
    sample_lengths: Sequence[int],
    cp_size: int,
    compress_ratio: int,
    *,
    alignment: int = 128,
    window_size: int = 128,
    cost_model: DsaCostModel | None = None,
) -> DsaDispatchPlan:
    """Make the 128-aligned, globally ordered contiguous baseline plan."""

    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    atoms = make_atomic_fragments(sample_lengths, alignment)
    prefix_tokens = [0]
    for atom in atoms:
        prefix_tokens.append(prefix_tokens[-1] + atom.token_count)

    cuts = [0]
    for rank in range(1, cp_size):
        target = prefix_tokens[-1] * rank / cp_size
        cut = min(
            range(cuts[-1], len(atoms) + 1),
            key=lambda index: (abs(prefix_tokens[index] - target), index),
        )
        cuts.append(cut)
    cuts.append(len(atoms))
    per_rank = [atoms[cuts[rank] : cuts[rank + 1]] for rank in range(cp_size)]
    return build_dsa_dispatch_plan(
        sample_lengths,
        per_rank,
        compress_ratio=compress_ratio,
        policy="sequential",
        alignment=alignment,
        window_size=window_size,
        cost_model=cost_model,
    )


def validate_dsa_dispatch_plan(plan: DsaDispatchPlan) -> None:
    """Reject holes, overlap, illegal cuts and inconsistent derived tables."""

    if plan.cp_size <= 0 or len(plan.ranks) != plan.cp_size:
        raise ValueError("plan rank table does not match cp_size")
    if plan.alignment <= 0:
        raise ValueError("plan alignment must be positive")
    if plan.compress_ratio not in (0, 4, 128):
        raise ValueError(f"unsupported compress ratio {plan.compress_ratio}")

    per_sample: list[list[tuple[int, int, int]]] = [[] for _ in plan.sample_lengths]
    for expected_rank, rank_plan in enumerate(plan.ranks):
        if rank_plan.rank != expected_rank:
            raise ValueError("rank plans must be stored in rank order")
        if rank_plan.token_count != sum(
            fragment.token_count for fragment in rank_plan.fragments
        ):
            raise ValueError(f"rank {expected_rank} token_count is inconsistent")
        if tuple(sorted(rank_plan.fragments)) != rank_plan.fragments:
            raise ValueError(f"rank {expected_rank} fragments are not sorted")
        for left, right in zip(rank_plan.fragments, rank_plan.fragments[1:]):
            if left.can_merge(right):
                raise ValueError("adjacent same-rank fragments must be merged")
        for fragment in rank_plan.fragments:
            if fragment.sample_id >= len(plan.sample_lengths):
                raise ValueError(f"invalid sample id {fragment.sample_id}")
            length = plan.sample_lengths[fragment.sample_id]
            if fragment.q_end > length:
                raise ValueError(f"fragment exceeds sample {fragment.sample_id}")
            if fragment.q_begin not in (0, length) and (
                fragment.q_begin % plan.alignment
            ):
                raise ValueError(f"fragment begin is not {plan.alignment}-aligned")
            if fragment.q_end != length and fragment.q_end % plan.alignment:
                raise ValueError(f"fragment end is not {plan.alignment}-aligned")
            per_sample[fragment.sample_id].append(
                (fragment.q_begin, fragment.q_end, expected_rank)
            )

    for sample_id, (length, segments) in enumerate(
        zip(plan.sample_lengths, per_sample)
    ):
        cursor = 0
        for begin, end, _ in sorted(segments):
            if begin != cursor:
                kind = "overlap" if begin < cursor else "hole"
                raise ValueError(
                    f"sample {sample_id} has a fragment {kind} at {cursor}"
                )
            cursor = end
        if cursor != length:
            raise ValueError(f"sample {sample_id} has a fragment hole at {cursor}")

    if len(plan.restore_map) != plan.total_fragments:
        raise ValueError("restore map must contain one entry per fragment")
    fragments_per_rank = tuple(rank.fragments for rank in plan.ranks)
    expected_restore_map = _make_restore_map(fragments_per_rank, plan.sample_lengths)
    if plan.restore_map != expected_restore_map:
        raise ValueError("restore map does not match rank fragment packing order")
    cursor = 0
    local_cursors = [0] * plan.cp_size
    for entry in plan.restore_map:
        if entry.global_begin != cursor or entry.length <= 0:
            raise ValueError("restore map has a hole, overlap or empty entry")
        if not 0 <= entry.rank < plan.cp_size:
            raise ValueError("restore rank is outside the CP group")
        if entry.local_begin != local_cursors[entry.rank]:
            raise ValueError("restore map local offsets are inconsistent")
        cursor = entry.global_end
        local_cursors[entry.rank] = entry.local_end
    if cursor != plan.total_tokens:
        raise ValueError("restore map does not cover all packed tokens")

    owners = _owner_segments(fragments_per_rank, len(plan.sample_lengths))
    expected_blocks_table = _make_compressed_blocks(
        plan.sample_lengths, plan.compress_ratio, owners
    )
    if plan.compressed_blocks != expected_blocks_table:
        raise ValueError("compressed block owner table is inconsistent")
    for logical_id, block in enumerate(plan.compressed_blocks):
        if block.logical_block_id != logical_id:
            raise ValueError("compressed blocks are not in logical-id order")
        if not 0 <= block.owner_rank < plan.cp_size:
            raise ValueError("compressed block owner is outside the CP group")
    expected_blocks = (
        0
        if plan.compress_ratio == 0
        else sum(length // plan.compress_ratio for length in plan.sample_lengths)
    )
    if len(plan.compressed_blocks) != expected_blocks:
        raise ValueError("compressed block table omitted a complete block")

    expected_block_ids: list[list[int]] = [[] for _ in range(plan.cp_size)]
    for block in plan.compressed_blocks:
        expected_block_ids[block.owner_rank].append(block.logical_block_id)
    for rank_plan, block_ids in zip(plan.ranks, expected_block_ids):
        if rank_plan.compressed_block_ids != tuple(block_ids):
            raise ValueError("rank compressed block ids are inconsistent")
        expected_indexer_cost = sum(
            fragment_indexer_cost(fragment, plan.compress_ratio)
            for fragment in rank_plan.fragments
        )
        if rank_plan.indexer_cost != expected_indexer_cost:
            raise ValueError("rank Indexer cost is inconsistent")

    expected_window_transfers = _make_window_transfers(
        fragments_per_rank, owners, plan.window_size
    )
    expected_overlap_transfers = _make_overlap_transfers(
        plan.compressed_blocks, owners, plan.compress_ratio
    )
    if plan.window_transfers != expected_window_transfers:
        raise ValueError("window transfer table is incomplete or inconsistent")
    if plan.overlap_transfers != expected_overlap_transfers:
        raise ValueError("overlap transfer table is incomplete or inconsistent")

    for transfer_table in (plan.window_transfers, plan.overlap_transfers):
        if tuple(sorted(transfer_table)) != transfer_table:
            raise ValueError("transfer table is not in deterministic order")
        for transfer in transfer_table:
            if transfer.source_rank == transfer.destination_rank:
                raise ValueError("transfer table must not contain self routes")
            if not (
                0 <= transfer.source_rank < plan.cp_size
                and 0 <= transfer.destination_rank < plan.cp_size
            ):
                raise ValueError("transfer route rank is outside the CP group")
            if not 0 <= transfer.sample_id < len(plan.sample_lengths):
                raise ValueError("transfer sample is outside the packed batch")
            if not (
                0
                <= transfer.q_begin
                < transfer.q_end
                <= plan.sample_lengths[transfer.sample_id]
            ):
                raise ValueError("transfer interval is outside its sample")


def format_plan_comparison(
    sequential: DsaDispatchPlan, balanced: DsaDispatchPlan
) -> str:
    if (
        sequential.sample_lengths != balanced.sample_lengths
        or sequential.cp_size != balanced.cp_size
        or sequential.compress_ratio != balanced.compress_ratio
    ):
        raise ValueError("plans must describe the same DSA invocation")
    return "\n".join((sequential.balance_report(), balanced.balance_report()))


__all__ = [
    "DsaCostModel",
    "build_dsa_dispatch_plan",
    "format_plan_comparison",
    "fragment_indexer_cost",
    "indexer_prefix_cost",
    "make_atomic_fragments",
    "make_sequential_plan",
    "validate_dsa_dispatch_plan",
]
