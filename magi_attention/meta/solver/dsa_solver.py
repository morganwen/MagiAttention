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

import bisect
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, replace

from magi_attention.dsa_config import DsaPlanPolicy, MagiDSAConfig
from magi_attention.meta.collection.dsa_meta import (
    DsaCompressionBlock,
    DsaExecutionPlan,
    DsaIndexerFragment,
    DsaOwnerFragment,
    DsaRankPlan,
    DsaRouteRankPlan,
    DsaTypedRoutePlan,
)


def _validate_cu_seqlens(cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in cu_seqlens)
    if len(values) < 2 or values[0] != 0:
        raise ValueError(
            "cu_seqlens must start at zero and describe at least one sample"
        )
    if any(end < begin for begin, end in zip(values, values[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    return values


def _validate_local_counts(
    local_counts: Sequence[int], total_tokens: int
) -> tuple[int, ...]:
    values = tuple(int(value) for value in local_counts)
    if not values:
        raise ValueError("local_token_counts must contain at least one rank")
    if any(value < 0 for value in values):
        raise ValueError("local token counts must be non-negative")
    if sum(values) != total_tokens:
        raise ValueError(
            f"local token counts sum to {sum(values)}, expected {total_tokens}"
        )
    return values


def _owner_offsets(local_counts: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for count in local_counts:
        offsets.append(offsets[-1] + count)
    return tuple(offsets)


def _owner_of_global_row(global_row: int, offsets: tuple[int, ...]) -> int:
    if not 0 <= global_row < offsets[-1]:
        raise ValueError(f"global row {global_row} is outside the owner layout")
    rank = bisect.bisect_right(offsets, global_row) - 1
    if rank == len(offsets) - 1:
        rank -= 1
    if not offsets[rank] <= global_row < offsets[rank + 1]:
        raise ValueError(f"global row {global_row} has no non-empty owner")
    return rank


def _build_owner_fragments(
    cu_seqlens: tuple[int, ...],
    local_counts: tuple[int, ...],
) -> tuple[tuple[DsaOwnerFragment, ...], ...]:
    offsets = _owner_offsets(local_counts)
    result: list[tuple[DsaOwnerFragment, ...]] = []
    for rank, (owner_begin, owner_end) in enumerate(zip(offsets, offsets[1:])):
        fragments: list[DsaOwnerFragment] = []
        for sample_id, (sample_begin, sample_end) in enumerate(
            zip(cu_seqlens, cu_seqlens[1:])
        ):
            global_begin = max(owner_begin, sample_begin)
            global_end = min(owner_end, sample_end)
            if global_begin >= global_end:
                continue
            fragments.append(
                DsaOwnerFragment(
                    sample_id=sample_id,
                    owner_rank=rank,
                    q_begin=global_begin - sample_begin,
                    q_end=global_end - sample_begin,
                    global_begin=global_begin,
                    global_end=global_end,
                    owner_local_begin=global_begin - owner_begin,
                    sample_global_begin=sample_begin,
                )
            )
        result.append(tuple(fragments))
    return tuple(result)


def _build_compression_blocks(
    config: MagiDSAConfig,
    cu_seqlens: tuple[int, ...],
    local_counts: tuple[int, ...],
) -> tuple[tuple[DsaCompressionBlock, ...], tuple[int, ...], tuple[int, ...]]:
    sample_offsets: list[int] = []
    sample_counts: list[int] = []
    if config.ratio == 0:
        return (
            (),
            tuple(0 for _ in range(len(cu_seqlens) - 1)),
            tuple(0 for _ in range(len(cu_seqlens) - 1)),
        )

    owner_offsets = _owner_offsets(local_counts)
    owner_block_counts = [0] * len(local_counts)
    blocks: list[DsaCompressionBlock] = []
    ratio = config.ratio
    for sample_id, (sample_begin, sample_end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        sample_offsets.append(len(blocks))
        block_count = (sample_end - sample_begin) // ratio
        sample_counts.append(block_count)
        for sample_block_id in range(block_count):
            global_begin = sample_begin + sample_block_id * ratio
            global_end = global_begin + ratio
            owner_rank = _owner_of_global_row(global_end - 1, owner_offsets)
            owner_local_index = owner_block_counts[owner_rank]
            owner_block_counts[owner_rank] += 1
            if ratio == 4:
                previous = (
                    tuple(range(global_begin - ratio, global_begin))
                    if sample_block_id > 0
                    else (-1,) * ratio
                )
                source_rows = (*previous, *range(global_begin, global_end))
            else:
                source_rows = tuple(range(global_begin, global_end))
            blocks.append(
                DsaCompressionBlock(
                    global_block_id=len(blocks),
                    sample_id=sample_id,
                    sample_block_id=sample_block_id,
                    global_begin=global_begin,
                    global_end=global_end,
                    owner_rank=owner_rank,
                    owner_local_index=owner_local_index,
                    position=sample_block_id * ratio,
                    source_global_rows=tuple(source_rows),
                )
            )
    return tuple(blocks), tuple(sample_offsets), tuple(sample_counts)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _make_indexer_atoms(
    config: MagiDSAConfig,
    owner_fragments: tuple[tuple[DsaOwnerFragment, ...], ...],
) -> list[DsaIndexerFragment]:
    atoms: list[DsaIndexerFragment] = []
    atom_size = config.indexer_atom_size
    for fragments in owner_fragments:
        for fragment in fragments:
            cursor = fragment.q_begin
            while cursor < fragment.q_end:
                if cursor % atom_size:
                    atom_end = min(fragment.q_end, _round_up(cursor, atom_size))
                else:
                    atom_end = min(fragment.q_end, cursor + atom_size)
                visible = tuple(
                    (position + 1) // 4 for position in range(cursor, atom_end)
                )
                score_cost = sum(
                    max(
                        config.indexer_score_block,
                        _round_up(length, config.indexer_score_block),
                    )
                    for length in visible
                )
                topk_cost = sum(
                    max(
                        config.indexer_topk_block,
                        _round_up(length, config.indexer_topk_block),
                    )
                    for length in visible
                )
                global_begin = fragment.sample_global_begin + cursor
                atoms.append(
                    DsaIndexerFragment(
                        sample_id=fragment.sample_id,
                        q_begin=cursor,
                        q_end=atom_end,
                        global_begin=global_begin,
                        global_end=fragment.sample_global_begin + atom_end,
                        owner_rank=fragment.owner_rank,
                        owner_local_begin=fragment.owner_local_begin
                        + cursor
                        - fragment.q_begin,
                        worker_rank=-1,
                        worker_local_begin=-1,
                        score_cost=score_cost,
                        topk_cost=topk_cost,
                    )
                )
                cursor = atom_end
    return atoms


def _assign_indexer_atoms(
    atoms: list[DsaIndexerFragment],
    cp_size: int,
    policy: DsaPlanPolicy,
) -> tuple[tuple[DsaIndexerFragment, ...], ...]:
    if policy not in ("sequential", "indexer_balanced"):
        raise ValueError(f"unsupported DSA policy {policy!r}")
    assigned: list[list[DsaIndexerFragment]] = [[] for _ in range(cp_size)]
    if policy == "sequential":
        for atom in atoms:
            assigned[atom.owner_rank].append(replace(atom, worker_rank=atom.owner_rank))
    else:
        total_score = sum(atom.score_cost for atom in atoms)
        total_topk = sum(atom.topk_cost for atom in atoms)
        score_target = max(total_score / cp_size, 1.0)
        topk_target = max(total_topk / cp_size, 1.0)
        score_load = [0] * cp_size
        topk_load = [0] * cp_size
        atom_count = [0] * cp_size
        ordered = sorted(
            atoms,
            key=lambda atom: (
                -max(atom.score_cost / score_target, atom.topk_cost / topk_target),
                -atom.score_cost,
                -atom.topk_cost,
                -atom.q_begin,
                atom.sample_id,
                atom.owner_rank,
            ),
        )
        for atom in ordered:

            def objective(rank: int) -> tuple[float, float, int, int]:
                next_score = (score_load[rank] + atom.score_cost) / score_target
                next_topk = (topk_load[rank] + atom.topk_cost) / topk_target
                return (
                    max(next_score, next_topk),
                    next_score + next_topk,
                    atom_count[rank],
                    rank,
                )

            worker = min(range(cp_size), key=objective)
            assigned[worker].append(replace(atom, worker_rank=worker))
            score_load[worker] += atom.score_cost
            topk_load[worker] += atom.topk_cost
            atom_count[worker] += 1

    resolved: list[tuple[DsaIndexerFragment, ...]] = []
    for worker, fragments in enumerate(assigned):
        worker_local_begin = 0
        worker_fragments: list[DsaIndexerFragment] = []
        for fragment in sorted(
            fragments,
            key=lambda item: (
                item.sample_id,
                item.q_begin,
                item.q_end,
                item.owner_rank,
            ),
        ):
            worker_fragments.append(
                replace(
                    fragment,
                    worker_rank=worker,
                    worker_local_begin=worker_local_begin,
                )
            )
            worker_local_begin += fragment.length
        resolved.append(tuple(worker_fragments))
    return tuple(resolved)


def _build_route(
    name: str,
    producer_owner: tuple[int, ...],
    producer_local_row: tuple[int, ...],
    producer_row_counts: tuple[int, ...],
    consumer_rows: tuple[tuple[int, ...], ...],
) -> DsaTypedRoutePlan:
    cp_size = len(producer_row_counts)
    if len(consumer_rows) != cp_size:
        raise ValueError(f"{name}: consumer row table has the wrong CP size")
    row_count = len(producer_owner)
    if len(producer_local_row) != row_count:
        raise ValueError(f"{name}: producer maps have different lengths")
    for global_row, (owner, local_row) in enumerate(
        zip(producer_owner, producer_local_row)
    ):
        if not 0 <= owner < cp_size:
            raise ValueError(f"{name}: row {global_row} has invalid owner {owner}")
        if not 0 <= local_row < producer_row_counts[owner]:
            raise ValueError(
                f"{name}: row {global_row} has invalid owner-local row {local_row}"
            )

    normalized_consumers: list[tuple[int, ...]] = []
    for rank, consumer_rank_rows in enumerate(consumer_rows):
        normalized = tuple(int(row) for row in consumer_rank_rows)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name}: rank {rank} consumer rows are not unique")
        if any(row < 0 or row >= row_count for row in normalized):
            raise ValueError(f"{name}: rank {rank} consumes an invalid global row")
        normalized_consumers.append(normalized)

    send_counts = [[0] * cp_size for _ in range(cp_size)]
    recv_counts = [[0] * cp_size for _ in range(cp_size)]
    send_rows: list[list[int]] = [[] for _ in range(cp_size)]
    received_rows: list[list[int]] = [[] for _ in range(cp_size)]
    for source in range(cp_size):
        for destination in range(cp_size):
            routed_rows = [
                row
                for row in normalized_consumers[destination]
                if producer_owner[row] == source
            ]
            count = len(routed_rows)
            send_counts[source][destination] = count
            recv_counts[destination][source] = count
            send_rows[source].extend(producer_local_row[row] for row in routed_rows)
            received_rows[destination].extend(routed_rows)

    rank_plans: list[DsaRouteRankPlan] = []
    for rank in range(cp_size):
        received_position = {
            row: index for index, row in enumerate(received_rows[rank])
        }
        consumer_from_received = tuple(
            received_position[row] for row in normalized_consumers[rank]
        )
        received_from_consumer_list = [-1] * len(received_rows[rank])
        for consumer_index, received_index in enumerate(consumer_from_received):
            if received_from_consumer_list[received_index] != -1:
                raise ValueError(
                    f"{name}: rank {rank} receive permutation is not invertible"
                )
            received_from_consumer_list[received_index] = consumer_index
        if any(index < 0 for index in received_from_consumer_list):
            raise ValueError(f"{name}: rank {rank} receive permutation has a gap")

        occurrences: list[list[int]] = [[] for _ in range(producer_row_counts[rank])]
        for reverse_source, local_row in enumerate(send_rows[rank]):
            occurrences[local_row].append(reverse_source)
        row_offsets = [0]
        reverse_source_rows: list[int] = []
        for row_occurrences in occurrences:
            reverse_source_rows.extend(row_occurrences)
            row_offsets.append(len(reverse_source_rows))

        rank_plans.append(
            DsaRouteRankPlan(
                rank=rank,
                producer_row_count=producer_row_counts[rank],
                send_counts=tuple(send_counts[rank]),
                recv_counts=tuple(recv_counts[rank]),
                send_source_rows=tuple(send_rows[rank]),
                received_global_rows=tuple(received_rows[rank]),
                consumer_global_rows=normalized_consumers[rank],
                consumer_from_received=consumer_from_received,
                received_from_consumer=tuple(received_from_consumer_list),
                reverse_row_offsets=tuple(row_offsets),
                reverse_source_rows=tuple(reverse_source_rows),
            )
        )
    return DsaTypedRoutePlan(name=name, rank_plans=tuple(rank_plans))


def _token_producers(
    local_counts: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = _owner_offsets(local_counts)
    owners: list[int] = []
    local_rows: list[int] = []
    for rank, count in enumerate(local_counts):
        owners.extend([rank] * count)
        local_rows.extend(range(count))
    if len(owners) != offsets[-1]:
        raise AssertionError("token producer map length mismatch")
    return tuple(owners), tuple(local_rows)


def _compressed_producers(
    blocks: tuple[DsaCompressionBlock, ...],
    cp_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    owners = tuple(block.owner_rank for block in blocks)
    local_rows = tuple(block.owner_local_index for block in blocks)
    counts = [0] * cp_size
    for block in blocks:
        counts[block.owner_rank] = max(
            counts[block.owner_rank], block.owner_local_index + 1
        )
    return owners, local_rows, tuple(counts)


def _query_metadata(
    cu_seqlens: tuple[int, ...],
    fragments: tuple[DsaOwnerFragment, ...],
    local_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sample_ids = [-1] * local_count
    positions = [-1] * local_count
    for fragment in fragments:
        for offset in range(fragment.length):
            local_row = fragment.owner_local_begin + offset
            sample_ids[local_row] = fragment.sample_id
            positions[local_row] = fragment.q_begin + offset
    if any(value < 0 for value in sample_ids) or any(value < 0 for value in positions):
        raise ValueError("owner fragments do not cover every local query exactly once")
    return tuple(sample_ids), tuple(positions)


def _sample_blocks(
    blocks: tuple[DsaCompressionBlock, ...],
    sample_count: int,
) -> tuple[tuple[DsaCompressionBlock, ...], ...]:
    grouped: list[list[DsaCompressionBlock]] = [[] for _ in range(sample_count)]
    for block in blocks:
        grouped[block.sample_id].append(block)
    return tuple(tuple(sample_blocks) for sample_blocks in grouped)


def _hash_plan_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dsa_execution_plan(
    config: MagiDSAConfig,
    cu_seqlens: Sequence[int],
    local_token_counts: Sequence[int],
    *,
    policy: DsaPlanPolicy = "indexer_balanced",
) -> DsaExecutionPlan:
    """Build and fully validate one immutable owner-local DSA execution plan."""

    cu = _validate_cu_seqlens(cu_seqlens)
    local_counts = _validate_local_counts(local_token_counts, cu[-1])
    cp_size = len(local_counts)
    owner_offsets = _owner_offsets(local_counts)
    owner_fragments = _build_owner_fragments(cu, local_counts)
    blocks, sample_block_offsets, sample_block_counts = _build_compression_blocks(
        config, cu, local_counts
    )
    blocks_by_sample = _sample_blocks(blocks, len(cu) - 1)

    token_owners, token_local_rows = _token_producers(local_counts)
    (
        compressed_owners,
        compressed_local_rows,
        compressed_local_counts,
    ) = _compressed_producers(blocks, cp_size)

    window_consumers: list[tuple[int, ...]] = []
    overlap_consumers: list[tuple[int, ...]] = []
    compressed_kv_consumers: list[tuple[int, ...]] = []
    compressed_ki_sets: list[set[int]] = [set() for _ in range(cp_size)]
    for rank, rank_owner_fragments in enumerate(owner_fragments):
        window_rows: set[int] = set()
        compressed_rows: set[int] = set()
        for owner_fragment in rank_owner_fragments:
            sample_begin = cu[owner_fragment.sample_id]
            window_begin = sample_begin + max(
                0, owner_fragment.q_begin - config.window_size + 1
            )
            window_rows.update(range(window_begin, owner_fragment.global_end))
            if config.ratio:
                visible = owner_fragment.q_end // config.ratio
                compressed_rows.update(
                    block.global_block_id
                    for block in blocks_by_sample[owner_fragment.sample_id][:visible]
                )
        window_consumers.append(tuple(sorted(window_rows)))
        compressed_kv_consumers.append(tuple(sorted(compressed_rows)))
        compressed_ki_sets[rank].update(compressed_rows)
        overlap_rows = {
            source_row
            for block in blocks
            if block.owner_rank == rank
            for source_row in block.source_global_rows
            if source_row >= 0
        }
        overlap_consumers.append(tuple(sorted(overlap_rows)))

    window_route = _build_route(
        "WINDOW_KV",
        token_owners,
        token_local_rows,
        local_counts,
        tuple(window_consumers),
    )

    if config.ratio:
        overlap_route = _build_route(
            "OVERLAP_X",
            token_owners,
            token_local_rows,
            local_counts,
            tuple(overlap_consumers),
        )
        compressed_kv_route = _build_route(
            "COMPRESSED_KV",
            compressed_owners,
            compressed_local_rows,
            compressed_local_counts,
            tuple(compressed_kv_consumers),
        )
    else:
        overlap_route = None
        compressed_kv_route = None

    if config.ratio == 4:
        atoms = _make_indexer_atoms(config, owner_fragments)
        worker_fragments = _assign_indexer_atoms(atoms, cp_size, policy)
        indexer_q_consumers: list[tuple[int, ...]] = []
        for rank, rank_worker_fragments in enumerate(worker_fragments):
            query_rows = tuple(
                global_row
                for indexer_fragment in rank_worker_fragments
                for global_row in range(
                    indexer_fragment.global_begin, indexer_fragment.global_end
                )
            )
            indexer_q_consumers.append(query_rows)
            for indexer_fragment in rank_worker_fragments:
                compressed_ki_sets[rank].update(
                    block.global_block_id
                    for block in blocks_by_sample[indexer_fragment.sample_id][
                        : indexer_fragment.q_end // 4
                    ]
                )
        compressed_ki_route = _build_route(
            "COMPRESSED_KI",
            compressed_owners,
            compressed_local_rows,
            compressed_local_counts,
            tuple(tuple(sorted(rows)) for rows in compressed_ki_sets),
        )
        indexer_qw_route = _build_route(
            "INDEXER_QW",
            token_owners,
            token_local_rows,
            local_counts,
            tuple(indexer_q_consumers),
        )
    else:
        worker_fragments = tuple(() for _ in range(cp_size))
        compressed_ki_route = None
        indexer_qw_route = None

    rank_plans: list[DsaRankPlan] = []
    for rank in range(cp_size):
        local_blocks = tuple(block for block in blocks if block.owner_rank == rank)
        local_sample_ids, local_positions = _query_metadata(
            cu, owner_fragments[rank], local_counts[rank]
        )
        if overlap_route is not None:
            overlap_position = {
                global_row: index
                for index, global_row in enumerate(
                    overlap_route.rank_plans[rank].consumer_global_rows
                )
            }
            compression_source = tuple(
                -1 if source_row < 0 else overlap_position[source_row]
                for block in local_blocks
                for source_row in block.source_global_rows
            )
        else:
            compression_source = ()

        q_cu = [0]
        k_cu = [0]
        q_offsets: list[int] = []
        q_sample_offsets: list[int] = []
        seq_lens: list[int] = []
        max_q = 0
        max_k = 0
        score_cost = 0
        topk_cost = 0
        for indexer_fragment in worker_fragments[rank]:
            q_cu.append(q_cu[-1] + indexer_fragment.length)
            k_length = indexer_fragment.q_end // 4
            k_cu.append(k_cu[-1] + k_length)
            q_offsets.append(indexer_fragment.q_begin)
            q_sample_offsets.extend(
                [sample_block_offsets[indexer_fragment.sample_id]]
                * indexer_fragment.length
            )
            seq_lens.extend(
                (position + 1) // 4
                for position in range(indexer_fragment.q_begin, indexer_fragment.q_end)
            )
            max_q = max(max_q, indexer_fragment.length)
            max_k = max(max_k, k_length)
            score_cost += indexer_fragment.score_cost
            topk_cost += indexer_fragment.topk_cost

        rank_plans.append(
            DsaRankPlan(
                rank=rank,
                local_token_count=local_counts[rank],
                local_global_begin=owner_offsets[rank],
                local_global_end=owner_offsets[rank + 1],
                owner_fragments=owner_fragments[rank],
                owned_blocks=local_blocks,
                worker_fragments=worker_fragments[rank],
                local_q_sample_ids=local_sample_ids,
                local_q_positions=local_positions,
                sample_block_offsets=sample_block_offsets,
                sample_block_counts=sample_block_counts,
                compression_source_from_overlap=compression_source,
                indexer_q_cu_seqlens=tuple(q_cu),
                indexer_k_cu_seqlens=tuple(k_cu),
                indexer_q_causal_offsets=tuple(q_offsets),
                indexer_q_sample_block_offsets=tuple(q_sample_offsets),
                indexer_seq_lens=tuple(seq_lens),
                indexer_max_seqlen_q=max_q,
                indexer_max_seqlen_k=max_k,
                window_route=window_route.rank_plans[rank],
                overlap_x_route=None
                if overlap_route is None
                else overlap_route.rank_plans[rank],
                compressed_kv_route=(
                    None
                    if compressed_kv_route is None
                    else compressed_kv_route.rank_plans[rank]
                ),
                compressed_ki_route=(
                    None
                    if compressed_ki_route is None
                    else compressed_ki_route.rank_plans[rank]
                ),
                indexer_qw_route=None
                if indexer_qw_route is None
                else indexer_qw_route.rank_plans[rank],
                predicted_score_cost=score_cost,
                predicted_topk_cost=topk_cost,
            )
        )

    collective_order: tuple[str, ...]
    if config.ratio == 0:
        collective_order = ("WINDOW_KV",)
    elif config.ratio == 4:
        collective_order = (
            "WINDOW_KV",
            "OVERLAP_X",
            "COMPRESSED_KV",
            "COMPRESSED_KI",
            "INDEXER_QW",
            "INDEXER_AUX",
        )
    else:
        collective_order = ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KV")

    payload = {
        "cu_seqlens": cu,
        "local_token_counts": local_counts,
        "ratio": config.ratio,
        "policy": policy,
        "compressed_blocks": [asdict(block) for block in blocks],
        "rank_plans": [asdict(rank_plan) for rank_plan in rank_plans],
        "collective_order": collective_order,
    }
    plan = DsaExecutionPlan(
        cu_seqlens=cu,
        local_token_counts=local_counts,
        ratio=config.ratio,
        policy=policy,
        compressed_blocks=blocks,
        rank_plans=tuple(rank_plans),
        collective_order=collective_order,
        plan_hash=_hash_plan_payload(payload),
    )
    validate_dsa_execution_plan(plan)
    return plan


def validate_dsa_execution_plan(plan: DsaExecutionPlan) -> None:
    """Reject coverage, inverse-permutation, route symmetry, and DAG errors."""

    if len(plan.plan_hash) != 64:
        raise ValueError("plan hash must be a SHA-256 hex digest")
    if len(plan.rank_plans) != plan.cp_size:
        raise ValueError("rank plan count does not match CP size")
    covered = [0] * plan.total_tokens
    for rank_plan in plan.rank_plans:
        if rank_plan.rank >= plan.cp_size:
            raise ValueError("rank plan has an invalid rank")
        for owner_fragment in rank_plan.owner_fragments:
            for global_row in range(
                owner_fragment.global_begin, owner_fragment.global_end
            ):
                covered[global_row] += 1
    if any(count != 1 for count in covered):
        raise ValueError("owner fragments must cover every query exactly once")

    if plan.ratio == 4:
        assigned = [0] * plan.total_tokens
        for rank_plan in plan.rank_plans:
            for indexer_fragment in rank_plan.worker_fragments:
                if indexer_fragment.worker_rank != rank_plan.rank:
                    raise ValueError("worker fragment is attached to the wrong rank")
                for global_row in range(
                    indexer_fragment.global_begin, indexer_fragment.global_end
                ):
                    assigned[global_row] += 1
        if any(count != 1 for count in assigned):
            raise ValueError("Indexer fragments must cover every query exactly once")

    route_fields = (
        "window_route",
        "overlap_x_route",
        "compressed_kv_route",
        "compressed_ki_route",
        "indexer_qw_route",
    )
    for field in route_fields:
        routes = [getattr(rank_plan, field) for rank_plan in plan.rank_plans]
        if all(route is None for route in routes):
            continue
        if any(route is None for route in routes):
            raise ValueError(f"{field} must be present or absent on every rank")
        concrete = [route for route in routes if route is not None]
        for source, source_route in enumerate(concrete):
            if source_route.rank != source:
                raise ValueError(f"{field} rank metadata mismatch")
            if source_route.send_row_count != len(source_route.send_source_rows):
                raise ValueError(f"{field} send rows do not match split counts")
            if source_route.received_row_count != len(
                source_route.received_global_rows
            ):
                raise ValueError(f"{field} receive rows do not match split counts")
            if len(source_route.consumer_from_received) != len(
                source_route.consumer_global_rows
            ):
                raise ValueError(f"{field} consumer permutation has an invalid length")
            if len(source_route.received_from_consumer) != len(
                source_route.received_global_rows
            ):
                raise ValueError(f"{field} inverse permutation has an invalid length")
            for destination, send_count in enumerate(source_route.send_counts):
                if send_count != concrete[destination].recv_counts[source]:
                    raise ValueError(f"{field} send/receive counts are asymmetric")
            if (
                len(source_route.reverse_row_offsets)
                != source_route.producer_row_count + 1
            ):
                raise ValueError(f"{field} reverse CSR row count is invalid")
            if source_route.reverse_row_offsets[-1] != len(
                source_route.reverse_source_rows
            ):
                raise ValueError(f"{field} reverse CSR item count is invalid")


__all__ = ["build_dsa_execution_plan", "validate_dsa_execution_plan"]
