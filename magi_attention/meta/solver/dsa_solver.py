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

"""Deterministic, fragment-based load balancer for the Magi_DSA Indexer."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence

import torch.distributed as dist

from magi_attention.meta.collection.dsa_meta import (
    DsaDispatchPlan,
    DsaFragmentSpec,
)
from magi_attention.meta.solver.dsa_dispatch import (
    DsaCostModel,
    build_dsa_dispatch_plan,
    fragment_indexer_cost,
    make_atomic_fragments,
    make_sequential_plan,
    validate_dsa_dispatch_plan,
)


@dataclass(frozen=True)
class DsaSolverConstraints:
    """Hard feasibility limits plus the Indexer-optimal selection slack."""

    max_tokens_per_rank: int | None = None
    token_imbalance_tolerance: int = 128
    max_memory_bytes: int | None = None
    max_fragments_per_rank: int | None = 64
    indexer_slack: float = 0.01
    absolute_indexer_slack: int = 0
    local_search_passes: int = 4

    def __post_init__(self) -> None:
        optional_positive = (
            ("max_tokens_per_rank", self.max_tokens_per_rank),
            ("max_memory_bytes", self.max_memory_bytes),
            ("max_fragments_per_rank", self.max_fragments_per_rank),
        )
        for name, value in optional_positive:
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.token_imbalance_tolerance < 0:
            raise ValueError("token_imbalance_tolerance must be non-negative")
        if self.indexer_slack < 0:
            raise ValueError("indexer_slack must be non-negative")
        if self.absolute_indexer_slack < 0:
            raise ValueError("absolute_indexer_slack must be non-negative")
        if self.local_search_passes < 0:
            raise ValueError("local_search_passes must be non-negative")


def _assignment_for_plan(
    atoms: Sequence[DsaFragmentSpec], plan: DsaDispatchPlan
) -> tuple[int, ...]:
    assignment: list[int] = []
    for atom in atoms:
        owner = None
        for rank in plan.ranks:
            if any(
                fragment.sample_id == atom.sample_id
                and fragment.q_begin <= atom.q_begin
                and atom.q_end <= fragment.q_end
                for fragment in rank.fragments
            ):
                owner = rank.rank
                break
        if owner is None:
            raise RuntimeError(f"sequential plan does not own atom {atom!r}")
        assignment.append(owner)
    return tuple(assignment)


def _per_rank_fragments(
    atoms: Sequence[DsaFragmentSpec], assignment: Sequence[int], cp_size: int
) -> tuple[tuple[DsaFragmentSpec, ...], ...]:
    per_rank: list[list[DsaFragmentSpec]] = [[] for _ in range(cp_size)]
    for atom, rank in zip(atoms, assignment):
        per_rank[rank].append(atom)
    return tuple(tuple(fragments) for fragments in per_rank)


def _merged_fragment_counts(
    atoms: Sequence[DsaFragmentSpec], assignment: Sequence[int], cp_size: int
) -> tuple[int, ...]:
    counts = [0] * cp_size
    previous: list[DsaFragmentSpec | None] = [None] * cp_size
    for atom, rank in zip(atoms, assignment):
        if previous[rank] is None or not previous[rank].can_merge(atom):
            counts[rank] += 1
        previous[rank] = atom
    return tuple(counts)


def _assignment_loads(
    atoms: Sequence[DsaFragmentSpec],
    assignment: Sequence[int],
    cp_size: int,
    compress_ratio: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tokens = [0] * cp_size
    indexer = [0] * cp_size
    for atom, rank in zip(atoms, assignment):
        tokens[rank] += atom.token_count
        indexer[rank] += fragment_indexer_cost(atom, compress_ratio)
    return tuple(tokens), tuple(indexer)


def _token_limit(
    total_tokens: int,
    cp_size: int,
    alignment: int,
    constraints: DsaSolverConstraints,
) -> int:
    if constraints.max_tokens_per_rank is not None:
        return constraints.max_tokens_per_rank
    average_ceiling = math.ceil(total_tokens / cp_size) if cp_size else 0
    return average_ceiling + max(constraints.token_imbalance_tolerance, alignment)


def _cheap_feasible(
    atoms: Sequence[DsaFragmentSpec],
    assignment: Sequence[int],
    cp_size: int,
    compress_ratio: int,
    token_limit: int,
    constraints: DsaSolverConstraints,
) -> bool:
    tokens, _ = _assignment_loads(atoms, assignment, cp_size, compress_ratio)
    if any(token_count > token_limit for token_count in tokens):
        return False
    if constraints.max_fragments_per_rank is not None:
        fragments = _merged_fragment_counts(atoms, assignment, cp_size)
        if any(count > constraints.max_fragments_per_rank for count in fragments):
            return False
    return True


def _lpt_assignment(
    atoms: Sequence[DsaFragmentSpec],
    cp_size: int,
    compress_ratio: int,
    token_limit: int,
) -> tuple[int, ...]:
    atom_costs = [fragment_indexer_cost(atom, compress_ratio) for atom in atoms]
    order = sorted(
        range(len(atoms)),
        key=lambda index: (
            -atom_costs[index],
            -atoms[index].token_count,
            atoms[index].sample_id,
            atoms[index].q_begin,
        ),
    )
    assignment = [-1] * len(atoms)
    tokens = [0] * cp_size
    indexer = [0] * cp_size
    for atom_index in order:
        atom = atoms[atom_index]
        feasible_ranks = [
            rank
            for rank in range(cp_size)
            if tokens[rank] + atom.token_count <= token_limit
        ]
        ranks = feasible_ranks or list(range(cp_size))
        rank = min(
            ranks,
            key=lambda candidate: (
                indexer[candidate] if compress_ratio == 4 else tokens[candidate],
                tokens[candidate],
                candidate,
            ),
        )
        assignment[atom_index] = rank
        tokens[rank] += atom.token_count
        indexer[rank] += atom_costs[atom_index]
    return tuple(assignment)


def _outer_middle_assignments(
    atoms: Sequence[DsaFragmentSpec], compress_ratio: int
) -> tuple[tuple[int, ...], ...]:
    """CP=2 candidates with one contiguous middle and two outer fragments.

    For the canonical 1024-token ratio=4 example this contains exactly
    ``[0:256)+[768:1024)`` versus ``[256:768)``.  It is also the low-fragment
    refinement of the cost-sorted LPT solution.
    """

    if not atoms:
        return ((),)
    costs = [fragment_indexer_cost(atom, compress_ratio) for atom in atoms]
    if compress_ratio != 4:
        costs = [atom.token_count for atom in atoms]
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    target = prefix[-1] / 2
    candidates: set[tuple[int, ...]] = set()
    for outer_prefix_end in range(len(atoms) + 1):
        wanted = prefix[outer_prefix_end] + prefix[-1] - target
        middle_end = bisect.bisect_left(prefix, wanted, outer_prefix_end)
        for candidate_end in range(
            max(outer_prefix_end, middle_end - 2),
            min(len(atoms), middle_end + 2) + 1,
        ):
            assignment = tuple(
                0 if index < outer_prefix_end or index >= candidate_end else 1
                for index in range(len(atoms))
            )
            candidates.add(assignment)
            candidates.add(tuple(1 - rank for rank in assignment))
    return tuple(sorted(candidates))


def _surrogate_objective(
    atoms: Sequence[DsaFragmentSpec],
    assignment: Sequence[int],
    cp_size: int,
    compress_ratio: int,
) -> tuple[int, int, int, tuple[int, ...]]:
    tokens, indexer = _assignment_loads(atoms, assignment, cp_size, compress_ratio)
    fragments = _merged_fragment_counts(atoms, assignment, cp_size)
    primary = max(indexer) if compress_ratio == 4 else max(tokens, default=0)
    return primary, max(tokens, default=0), sum(fragments), tuple(assignment)


def _refine_moves_and_swaps(
    atoms: Sequence[DsaFragmentSpec],
    assignment: tuple[int, ...],
    cp_size: int,
    compress_ratio: int,
    token_limit: int,
    constraints: DsaSolverConstraints,
) -> tuple[int, ...]:
    """Deterministic bounded local improvement on aligned atoms."""

    current = assignment
    for _ in range(constraints.local_search_passes):
        current_objective = _surrogate_objective(
            atoms, current, cp_size, compress_ratio
        )
        best = current
        best_objective = current_objective

        for atom_index, source_rank in enumerate(current):
            for destination_rank in range(cp_size):
                if source_rank == destination_rank:
                    continue
                candidate = list(current)
                candidate[atom_index] = destination_rank
                candidate_tuple = tuple(candidate)
                if not _cheap_feasible(
                    atoms,
                    candidate_tuple,
                    cp_size,
                    compress_ratio,
                    token_limit,
                    constraints,
                ):
                    continue
                objective = _surrogate_objective(
                    atoms, candidate_tuple, cp_size, compress_ratio
                )
                if objective < best_objective:
                    best, best_objective = candidate_tuple, objective

        # Swapping nearby cost ranks is enough to repair the greedy LPT tail
        # without an unbounded O(n^2) search on the 49k-token workload.
        order = sorted(
            range(len(atoms)),
            key=lambda index: (
                fragment_indexer_cost(atoms[index], compress_ratio),
                atoms[index].token_count,
                index,
            ),
        )
        for order_index, left in enumerate(order):
            for right in order[order_index + 1 : order_index + 9]:
                if current[left] == current[right]:
                    continue
                candidate = list(current)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_tuple = tuple(candidate)
                if not _cheap_feasible(
                    atoms,
                    candidate_tuple,
                    cp_size,
                    compress_ratio,
                    token_limit,
                    constraints,
                ):
                    continue
                objective = _surrogate_objective(
                    atoms, candidate_tuple, cp_size, compress_ratio
                )
                if objective < best_objective:
                    best, best_objective = candidate_tuple, objective

        if best_objective >= current_objective:
            break
        current = best
    return current


class DsaPlanSolver:
    """Solve and cache sequential or Indexer-balanced immutable plans."""

    def __init__(
        self,
        *,
        alignment: int = 128,
        window_size: int = 128,
        constraints: DsaSolverConstraints | None = None,
        cost_model: DsaCostModel | None = None,
    ) -> None:
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.alignment = alignment
        self.window_size = window_size
        self.constraints = constraints or DsaSolverConstraints()
        self.cost_model = cost_model or DsaCostModel()
        self._cache: dict[tuple[object, ...], DsaDispatchPlan] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_key(
        self,
        sample_lengths: Sequence[int],
        cp_size: int,
        compress_ratio: int,
        policy: str,
    ) -> tuple[object, ...]:
        return (
            tuple(int(length) for length in sample_lengths),
            cp_size,
            compress_ratio,
            policy,
            self.alignment,
            self.window_size,
            self.constraints,
            self.cost_model,
        )

    def solve(
        self,
        sample_lengths: Sequence[int],
        cp_size: int,
        compress_ratio: int,
        *,
        policy: str = "balanced",
    ) -> DsaDispatchPlan:
        if cp_size <= 0:
            raise ValueError("cp_size must be positive")
        if compress_ratio not in (0, 4, 128):
            raise ValueError(f"unsupported compress ratio {compress_ratio}")
        if policy not in ("sequential", "balanced"):
            raise ValueError("policy must be 'sequential' or 'balanced'")
        lengths = tuple(int(length) for length in sample_lengths)
        if any(length < 0 for length in lengths):
            raise ValueError("sample lengths must be non-negative")
        key = self._cache_key(lengths, cp_size, compress_ratio, policy)
        if key in self._cache:
            return self._cache[key]

        sequential = make_sequential_plan(
            lengths,
            cp_size,
            compress_ratio,
            alignment=self.alignment,
            window_size=self.window_size,
            cost_model=self.cost_model,
        )
        if policy == "sequential" or cp_size == 1 or not sequential.total_tokens:
            plan = sequential
            if policy == "balanced" and sequential.policy != policy:
                plan = build_dsa_dispatch_plan(
                    lengths,
                    [rank.fragments for rank in sequential.ranks],
                    compress_ratio=compress_ratio,
                    policy=policy,
                    alignment=self.alignment,
                    window_size=self.window_size,
                    cost_model=self.cost_model,
                )
            self._check_plan_constraints(plan)
            self._cache[key] = plan
            return plan

        atoms = make_atomic_fragments(lengths, self.alignment)
        token_limit = _token_limit(
            sum(lengths), cp_size, self.alignment, self.constraints
        )
        assignments: set[tuple[int, ...]] = {
            _assignment_for_plan(atoms, sequential),
            tuple(index % cp_size for index in range(len(atoms))),
            tuple((len(atoms) - 1 - index) % cp_size for index in range(len(atoms))),
        }
        lpt = _lpt_assignment(atoms, cp_size, compress_ratio, token_limit)
        assignments.add(lpt)
        assignments.add(
            _refine_moves_and_swaps(
                atoms,
                lpt,
                cp_size,
                compress_ratio,
                token_limit,
                self.constraints,
            )
        )
        if cp_size == 2:
            assignments.update(_outer_middle_assignments(atoms, compress_ratio))

        feasible = [
            assignment
            for assignment in sorted(assignments)
            if _cheap_feasible(
                atoms,
                assignment,
                cp_size,
                compress_ratio,
                token_limit,
                self.constraints,
            )
        ]
        if not feasible:
            raise ValueError("no aligned DSA plan satisfies token/fragment constraints")

        max_indexer = [
            max(
                _assignment_loads(atoms, assignment, cp_size, compress_ratio)[1],
                default=0,
            )
            for assignment in feasible
        ]
        best_indexer = min(max_indexer)
        slack = max(
            self.constraints.absolute_indexer_slack,
            math.ceil(best_indexer * self.constraints.indexer_slack),
        )
        candidate_plans: list[DsaDispatchPlan] = []
        for assignment, candidate_indexer in zip(feasible, max_indexer):
            if compress_ratio == 4 and candidate_indexer > best_indexer + slack:
                continue
            candidate = build_dsa_dispatch_plan(
                lengths,
                _per_rank_fragments(atoms, assignment, cp_size),
                compress_ratio=compress_ratio,
                policy="balanced",
                alignment=self.alignment,
                window_size=self.window_size,
                cost_model=self.cost_model,
            )
            try:
                self._check_plan_constraints(candidate)
            except ValueError:
                continue
            candidate_plans.append(candidate)

        if not candidate_plans:
            raise ValueError("no DSA plan satisfies the configured memory constraint")
        sequential_fragments = tuple(rank.fragments for rank in sequential.ranks)
        if any(
            tuple(rank.fragments for rank in candidate.ranks) == sequential_fragments
            for candidate in candidate_plans
        ):
            candidate_plans = [
                candidate
                for candidate in candidate_plans
                if candidate.max_rank_indexer_cost <= sequential.max_rank_indexer_cost
            ]
        plan = min(
            candidate_plans,
            key=lambda candidate: (
                candidate.max_rank_predicted_e2e,
                sum(rank.predicted_e2e for rank in candidate.ranks),
                candidate.total_fragments,
                sum(
                    transfer.row_count
                    for transfer in (
                        candidate.window_transfers + candidate.overlap_transfers
                    )
                ),
                candidate.plan_hash,
            ),
        )
        self._cache[key] = plan
        return plan

    def _check_plan_constraints(self, plan: DsaDispatchPlan) -> None:
        token_limit = _token_limit(
            plan.total_tokens, plan.cp_size, plan.alignment, self.constraints
        )
        if any(rank.token_count > token_limit for rank in plan.ranks):
            raise ValueError(f"plan exceeds max tokens per rank ({token_limit})")
        if self.constraints.max_fragments_per_rank is not None and any(
            rank.fragment_count > self.constraints.max_fragments_per_rank
            for rank in plan.ranks
        ):
            raise ValueError("plan exceeds max fragments per rank")
        if self.constraints.max_memory_bytes is not None and any(
            rank.estimated_memory_bytes > self.constraints.max_memory_bytes
            for rank in plan.ranks
        ):
            raise ValueError("plan exceeds estimated memory per rank")

    def solve_distributed(
        self,
        sample_lengths: Sequence[int],
        compress_ratio: int,
        group: dist.ProcessGroup,
        *,
        policy: str = "balanced",
    ) -> DsaDispatchPlan:
        """Solve on group rank 0, broadcast one immutable plan, then cache it."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        group_rank = dist.get_rank(group)
        cp_size = dist.get_world_size(group)
        if cp_size == 1:
            return self.solve(sample_lengths, cp_size, compress_ratio, policy=policy)

        objects: list[DsaDispatchPlan | None] = [None]
        if group_rank == 0:
            objects[0] = self.solve(
                sample_lengths, cp_size, compress_ratio, policy=policy
            )
        source_rank = (
            dist.get_global_rank(group, 0) if hasattr(dist, "get_global_rank") else 0
        )
        dist.broadcast_object_list(objects, src=source_rank, group=group)
        plan = objects[0]
        if not isinstance(plan, DsaDispatchPlan):
            raise RuntimeError("rank 0 broadcast an invalid DSA plan")
        if (
            plan.cp_size != cp_size
            or plan.compress_ratio != compress_ratio
            or plan.sample_lengths != tuple(int(length) for length in sample_lengths)
            or plan.policy != policy
            or plan.alignment != self.alignment
            or plan.window_size != self.window_size
        ):
            raise RuntimeError("broadcast DSA plan does not match this invocation")
        validate_dsa_dispatch_plan(plan)
        key = self._cache_key(sample_lengths, cp_size, compress_ratio, policy)
        self._cache[key] = plan
        return plan


def solve_dsa_plan(
    sample_lengths: Sequence[int],
    cp_size: int,
    compress_ratio: int,
    *,
    policy: str = "balanced",
    alignment: int = 128,
    window_size: int = 128,
    constraints: DsaSolverConstraints | None = None,
    cost_model: DsaCostModel | None = None,
) -> DsaDispatchPlan:
    """Stateless convenience wrapper used by tests and plan tooling."""

    return DsaPlanSolver(
        alignment=alignment,
        window_size=window_size,
        constraints=constraints,
        cost_model=cost_model,
    ).solve(sample_lengths, cp_size, compress_ratio, policy=policy)


__all__ = ["DsaPlanSolver", "DsaSolverConstraints", "solve_dsa_plan"]
