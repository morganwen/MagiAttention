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

"""Host-side metadata for a Magi_DSA context-parallel dispatch plan.

All fragment and transfer coordinates are relative to their packed sample.
Keeping sample-relative coordinates is important: compression grids, causal
windows and the Indexer prefix all restart at every packed sample boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True, order=True)
class DsaFragmentSpec:
    """One half-open query fragment in sample-relative coordinates."""

    sample_id: int
    q_begin: int
    q_end: int

    def __post_init__(self) -> None:
        if self.sample_id < 0:
            raise ValueError(f"sample_id must be non-negative, got {self.sample_id}")
        if self.q_begin < 0:
            raise ValueError(f"q_begin must be non-negative, got {self.q_begin}")
        if self.q_end <= self.q_begin:
            raise ValueError(
                "fragment must be non-empty with q_end > q_begin, "
                f"got [{self.q_begin}, {self.q_end})"
            )

    @property
    def token_count(self) -> int:
        return self.q_end - self.q_begin

    def can_merge(self, other: "DsaFragmentSpec") -> bool:
        return self.sample_id == other.sample_id and self.q_end == other.q_begin

    def merge(self, other: "DsaFragmentSpec") -> "DsaFragmentSpec":
        if not self.can_merge(other):
            raise ValueError(f"fragments are not adjacent: {self!r}, {other!r}")
        return DsaFragmentSpec(self.sample_id, self.q_begin, other.q_end)


@dataclass(frozen=True, order=True)
class DsaCompressedBlockSpec:
    """Owner of one complete compressed block.

    ``logical_block_id`` is packed-global across samples. ``sample_block_id``
    restarts at zero for every sample.  A block is owned by the rank holding
    its last source token.
    """

    logical_block_id: int
    sample_id: int
    sample_block_id: int
    owner_rank: int

    def __post_init__(self) -> None:
        if (
            min(
                self.logical_block_id,
                self.sample_id,
                self.sample_block_id,
                self.owner_rank,
            )
            < 0
        ):
            raise ValueError("compressed block ids and owner rank must be non-negative")


@dataclass(frozen=True, order=True)
class DsaRestoreSpec:
    """Copy a contiguous local output segment back to packed-global order."""

    rank: int
    local_begin: int
    global_begin: int
    length: int

    def __post_init__(self) -> None:
        if min(self.rank, self.local_begin, self.global_begin) < 0:
            raise ValueError("restore rank and offsets must be non-negative")
        if self.length <= 0:
            raise ValueError("restore length must be positive")

    @property
    def local_end(self) -> int:
        return self.local_begin + self.length

    @property
    def global_end(self) -> int:
        return self.global_begin + self.length


@dataclass(frozen=True, order=True)
class DsaTransferSpec:
    """Unique sample-relative source rows sent from one rank to another."""

    source_rank: int
    destination_rank: int
    sample_id: int
    q_begin: int
    q_end: int

    def __post_init__(self) -> None:
        if (
            min(self.source_rank, self.destination_rank, self.sample_id, self.q_begin)
            < 0
        ):
            raise ValueError("transfer ranks and coordinates must be non-negative")
        if self.source_rank == self.destination_rank:
            raise ValueError("a transfer cannot route rows to the source rank itself")
        if self.q_end <= self.q_begin:
            raise ValueError("transfer interval must be non-empty")

    @property
    def row_count(self) -> int:
        return self.q_end - self.q_begin


@dataclass(frozen=True)
class DsaRankPlan:
    """Static dispatch and predicted load for one CP rank."""

    rank: int
    fragments: tuple[DsaFragmentSpec, ...]
    compressed_block_ids: tuple[int, ...]
    token_count: int
    indexer_cost: int
    predicted_e2e: float
    estimated_memory_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.rank,
                self.token_count,
                self.indexer_cost,
                self.estimated_memory_bytes,
            )
            < 0
        ):
            raise ValueError("rank plan ids, loads and memory must be non-negative")
        if self.predicted_e2e < 0:
            raise ValueError("predicted_e2e must be non-negative")

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)


@dataclass(frozen=True)
class DsaDispatchPlan:
    """Complete, deterministic host plan shared by all CP ranks."""

    policy: str
    cp_size: int
    compress_ratio: int
    alignment: int
    window_size: int
    sample_lengths: tuple[int, ...]
    ranks: tuple[DsaRankPlan, ...]
    compressed_blocks: tuple[DsaCompressedBlockSpec, ...]
    restore_map: tuple[DsaRestoreSpec, ...]
    window_transfers: tuple[DsaTransferSpec, ...]
    overlap_transfers: tuple[DsaTransferSpec, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.sample_lengths)

    @property
    def max_rank_indexer_cost(self) -> int:
        return max((rank.indexer_cost for rank in self.ranks), default=0)

    @property
    def max_rank_predicted_e2e(self) -> float:
        return max((rank.predicted_e2e for rank in self.ranks), default=0.0)

    @property
    def total_fragments(self) -> int:
        return sum(rank.fragment_count for rank in self.ranks)

    @property
    def plan_hash(self) -> str:
        """Stable SHA256 of logical plan contents (independent of repr/pickle)."""

        payload = {
            "schema": 1,
            "policy": self.policy,
            "cp_size": self.cp_size,
            "compress_ratio": self.compress_ratio,
            "alignment": self.alignment,
            "window_size": self.window_size,
            "sample_lengths": self.sample_lengths,
            "fragments": tuple(
                (
                    rank.rank,
                    tuple(
                        (fragment.sample_id, fragment.q_begin, fragment.q_end)
                        for fragment in rank.fragments
                    ),
                )
                for rank in self.ranks
            ),
            "compressed_blocks": tuple(
                (
                    block.logical_block_id,
                    block.sample_id,
                    block.sample_block_id,
                    block.owner_rank,
                )
                for block in self.compressed_blocks
            ),
            "restore_map": tuple(
                (entry.rank, entry.local_begin, entry.global_begin, entry.length)
                for entry in self.restore_map
            ),
            "window_transfers": tuple(
                (
                    entry.source_rank,
                    entry.destination_rank,
                    entry.sample_id,
                    entry.q_begin,
                    entry.q_end,
                )
                for entry in self.window_transfers
            ),
            "overlap_transfers": tuple(
                (
                    entry.source_rank,
                    entry.destination_rank,
                    entry.sample_id,
                    entry.q_begin,
                    entry.q_end,
                )
                for entry in self.overlap_transfers
            ),
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def rank_for_fragment(self, fragment: DsaFragmentSpec) -> int:
        for rank in self.ranks:
            if fragment in rank.fragments:
                return rank.rank
        raise KeyError(f"fragment is not present in plan: {fragment!r}")

    def iter_fragments(self) -> Iterator[tuple[int, DsaFragmentSpec]]:
        for rank in self.ranks:
            for fragment in rank.fragments:
                yield rank.rank, fragment

    def block_owner(self, logical_block_id: int) -> int:
        if not 0 <= logical_block_id < len(self.compressed_blocks):
            raise IndexError(f"invalid logical block id {logical_block_id}")
        block = self.compressed_blocks[logical_block_id]
        if block.logical_block_id != logical_block_id:
            raise RuntimeError("compressed block table is not in logical-id order")
        return block.owner_rank

    def restore_rows(self) -> tuple[tuple[int, int], ...]:
        """Return packed-global ``(rank, local_row)`` entries for testing/packing."""

        rows: list[tuple[int, int] | None] = [None] * self.total_tokens
        for entry in self.restore_map:
            for offset in range(entry.length):
                rows[entry.global_begin + offset] = (
                    entry.rank,
                    entry.local_begin + offset,
                )
        if any(row is None for row in rows):
            raise RuntimeError("restore map does not cover every packed row")
        return tuple(row for row in rows if row is not None)

    def balance_report(self) -> str:
        tokens = ", ".join(f"r{rank.rank}={rank.token_count}" for rank in self.ranks)
        indexer = ", ".join(f"r{rank.rank}={rank.indexer_cost}" for rank in self.ranks)
        e2e = ", ".join(f"r{rank.rank}={rank.predicted_e2e:.1f}" for rank in self.ranks)
        fragments = ", ".join(
            f"r{rank.rank}={rank.fragment_count}" for rank in self.ranks
        )
        costs = [rank.indexer_cost for rank in self.ranks]
        mean_cost = sum(costs) / len(costs) if costs else 0.0
        imbalance = max(costs) / mean_cost - 1.0 if mean_cost > 0.0 else 0.0
        return (
            f"{self.policy} plan {self.plan_hash[:12]}: tokens [{tokens}], "
            f"indexer [{indexer}] max/mean-1={imbalance:.3%}, "
            f"predicted-e2e [{e2e}], fragments [{fragments}]"
        )


def sample_offsets(sample_lengths: Sequence[int]) -> tuple[int, ...]:
    """Packed-global start offset for every sample plus the final total."""

    offsets = [0]
    for length in sample_lengths:
        if length < 0:
            raise ValueError(f"sample lengths must be non-negative, got {length}")
        offsets.append(offsets[-1] + int(length))
    return tuple(offsets)


__all__ = [
    "DsaCompressedBlockSpec",
    "DsaDispatchPlan",
    "DsaFragmentSpec",
    "DsaRankPlan",
    "DsaRestoreSpec",
    "DsaTransferSpec",
    "sample_offsets",
]
