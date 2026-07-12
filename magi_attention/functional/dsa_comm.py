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

"""Typed Magi_DSA transport over MagiAttention group collectives.

Step 3 introduced torch reference gather/scatter maps.  Step 4 keeps those
references for tests and routes production packing/restore through immutable
device maps plus CuTe DSL kernels, without changing the communication metadata
or the GroupCast/GroupReduce primitives in this module.

The four payload kinds never share metadata, buffer slots, native grpcoll
handles or work objects.  The objects holding tensors and work are per-call;
only :class:`DsaCommMeta` is suitable for caching as static plan state.
"""

from __future__ import annotations

import math
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from magi_attention import env
from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.primitive.grpcoll._buffer import GrpCollBuffer
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.meta.collection.comm_meta import GroupCollectiveArg
from magi_attention.meta.collection.dsa_meta import (
    DsaDispatchPlan,
    DsaTransferSpec,
    sample_offsets,
)

if TYPE_CHECKING:
    from magi_attention.kernel.cutedsl.dsa_pack import (
        DsaDeviceCopyMap,
        DsaDeviceReduceMap,
    )


class DsaPayloadKind(str, Enum):
    """The four independently routed DSA payloads."""

    WINDOW_KV = "dsa_window_kv"
    OVERLAP_X = "dsa_overlap_x"
    COMPRESSED_KV = "dsa_compressed_kv"
    COMPRESSED_KI = "dsa_compressed_ki"


@dataclass(frozen=True)
class DsaTypedPayload:
    """A tensor tagged with the static route it is allowed to use."""

    kind: DsaPayloadKind
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DsaPayloadKind):
            raise TypeError("kind must be a DsaPayloadKind")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if self.tensor.ndim < 1:
            raise ValueError("a DSA payload tensor must have a row dimension")


@dataclass(frozen=True)
class DsaCommMeta:
    """Static GroupCast map and its symmetric GroupReduce map for one payload.

    Row ids are packed-global token ids for ``window_kv``/``overlap_x`` and
    global logical compressed-block ids for ``compressed_kv``/``compressed_ki``.
    ``send_row_indices`` index the owner-local input tensor in fragment/block
    order.  They are unique: one packed source row can carry multiple
    destination ranks through one GroupCast split.
    """

    kind: DsaPayloadKind
    collective_arg: GroupCollectiveArg
    local_row_ids: tuple[int, ...]
    send_row_indices: tuple[int, ...]
    send_row_ids: tuple[int, ...]
    receive_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DsaPayloadKind):
            raise TypeError("kind must be a DsaPayloadKind")
        if len(set(self.local_row_ids)) != len(self.local_row_ids):
            raise ValueError(f"{self.kind.value} local row ids must be unique")
        if len(set(self.send_row_indices)) != len(self.send_row_indices):
            raise ValueError(f"{self.kind.value} send row indices must be unique")
        if any(
            not 0 <= index < len(self.local_row_ids) for index in self.send_row_indices
        ):
            raise ValueError(f"{self.kind.value} send row index is out of range")
        expected_send_ids = tuple(
            self.local_row_ids[index] for index in self.send_row_indices
        )
        if self.send_row_ids != expected_send_ids:
            raise ValueError(f"{self.kind.value} send row ids do not match indices")
        if len(set(self.receive_row_ids)) != len(self.receive_row_ids):
            raise ValueError(f"{self.kind.value} receive rows must be unique")
        if sum(self.collective_arg.input_split_size_list) != len(self.send_row_indices):
            raise ValueError(f"{self.kind.value} GroupCast input size is inconsistent")
        if sum(self.collective_arg.output_split_size_list) != len(self.receive_row_ids):
            raise ValueError(f"{self.kind.value} GroupCast output size is inconsistent")

    @property
    def local_row_count(self) -> int:
        return len(self.local_row_ids)

    @property
    def send_row_count(self) -> int:
        return len(self.send_row_indices)

    @property
    def receive_row_count(self) -> int:
        return len(self.receive_row_ids)


@dataclass(frozen=True)
class DsaCommPlan:
    """All four typed routes for one rank of one dispatch plan."""

    window_kv: DsaCommMeta
    overlap_x: DsaCommMeta
    compressed_kv: DsaCommMeta
    compressed_ki: DsaCommMeta

    def __post_init__(self) -> None:
        expected = (
            DsaPayloadKind.WINDOW_KV,
            DsaPayloadKind.OVERLAP_X,
            DsaPayloadKind.COMPRESSED_KV,
            DsaPayloadKind.COMPRESSED_KI,
        )
        actual = tuple(meta.kind for meta in self)
        if actual != expected:
            raise ValueError(f"DSA communication payload order must be {expected}")
        if len({id(meta.collective_arg) for meta in self}) != len(expected):
            raise ValueError("each DSA payload must own a GroupCollectiveArg")

    def __iter__(self) -> Iterator[DsaCommMeta]:
        yield self.window_kv
        yield self.overlap_x
        yield self.compressed_kv
        yield self.compressed_ki

    def for_kind(self, kind: DsaPayloadKind) -> DsaCommMeta:
        for meta in self:
            if meta.kind is kind:
                return meta
        raise KeyError(kind)


@dataclass(frozen=True)
class DsaDeviceCommMap:
    """Read-only device maps for one typed route of one static plan."""

    meta: DsaCommMeta
    send_copy: "DsaDeviceCopyMap"
    owner_restore: "DsaDeviceReduceMap"

    def __post_init__(self) -> None:
        if self.send_copy.source_row_count != self.meta.local_row_count:
            raise ValueError("DSA send map source row count does not match metadata")
        if self.send_copy.destination_row_count != self.meta.send_row_count:
            raise ValueError(
                "DSA send map destination row count does not match metadata"
            )
        if self.owner_restore.source_row_count != self.meta.send_row_count:
            raise ValueError("DSA restore map source row count does not match metadata")
        if self.owner_restore.destination_row_count != self.meta.local_row_count:
            raise ValueError(
                "DSA restore map destination row count does not match metadata"
            )


@dataclass(frozen=True)
class DsaDeviceCommPlan:
    """All four immutable device mapping sets for one rank plan."""

    window_kv: DsaDeviceCommMap
    overlap_x: DsaDeviceCommMap
    compressed_kv: DsaDeviceCommMap
    compressed_ki: DsaDeviceCommMap

    def __post_init__(self) -> None:
        expected = tuple(DsaPayloadKind)
        actual = tuple(device_map.meta.kind for device_map in self)
        if actual != expected:
            raise ValueError(f"DSA device map payload order must be {expected}")

    def __iter__(self) -> Iterator[DsaDeviceCommMap]:
        yield self.window_kv
        yield self.overlap_x
        yield self.compressed_kv
        yield self.compressed_ki

    def for_kind(self, kind: DsaPayloadKind) -> DsaDeviceCommMap:
        for device_map in self:
            if device_map.meta.kind is kind:
                return device_map
        raise KeyError(kind)


@dataclass
class DsaCommBufferSlot:
    """Per-call buffers for exactly one typed payload."""

    kind: DsaPayloadKind
    send_buffer: torch.Tensor
    receive_buffer: torch.Tensor
    logical_row_shape: tuple[int, ...]
    reduce_output_buffer: torch.Tensor | None = None


@dataclass
class DsaGroupCastWork:
    """Disposable GroupCast work and its payload-private grpcoll handle."""

    meta: DsaCommMeta
    device_map: DsaDeviceCommMap
    buffers: DsaCommBufferSlot
    work: WorkWithPostProcessFn
    native_handle_dict: dict[str, Any]
    _result: DsaTypedPayload | None = field(default=None, init=False, repr=False)
    _reduce_started: bool = field(default=False, init=False, repr=False)
    _wait_error: BaseException | None = field(default=None, init=False, repr=False)
    _collective_failed: bool = field(default=False, init=False, repr=False)

    def wait(self) -> DsaTypedPayload:
        if self._result is not None:
            return self._result
        if self._wait_error is not None:
            raise self._wait_error
        try:
            receive = self.work.wait_post_process(self.buffers.receive_buffer)
            if not isinstance(receive, torch.Tensor):
                raise TypeError("GroupCast must return one tensor for one DSA payload")
            if receive.shape != self.buffers.receive_buffer.shape:
                raise RuntimeError(
                    f"{self.meta.kind.value} GroupCast returned shape "
                    f"{tuple(receive.shape)}, expected "
                    f"{tuple(self.buffers.receive_buffer.shape)}"
                )
            self.buffers.receive_buffer = receive
            logical_receive = _restore_logical_row_shape(
                receive, self.buffers.logical_row_shape
            )
            self._result = DsaTypedPayload(self.meta.kind, logical_receive)
            return self._result
        except BaseException as error:
            # WorkWithPostProcessFn clears ``work`` only after the collective
            # wait succeeds.  Preserve that distinction so the enclosing scope
            # aborts a genuinely failed process group but not a recoverable
            # validation/post-process exception.
            self._collective_failed = getattr(self.work, "work", None) is not None
            self._wait_error = error
            raise


@dataclass
class DsaGroupReduceWork:
    """Disposable symmetric GroupReduce work with FP32 owner accumulation."""

    meta: DsaCommMeta
    device_map: DsaDeviceCommMap
    buffers: DsaCommBufferSlot
    work: WorkWithPostProcessFn
    local_accumulator: torch.Tensor
    output_dtype: torch.dtype
    _result: DsaTypedPayload | None = field(default=None, init=False, repr=False)
    _wait_error: BaseException | None = field(default=None, init=False, repr=False)
    _collective_failed: bool = field(default=False, init=False, repr=False)

    def wait(self) -> DsaTypedPayload:
        if self._result is not None:
            return self._result
        if self._wait_error is not None:
            raise self._wait_error
        try:
            if self.buffers.reduce_output_buffer is None:
                raise RuntimeError("GroupReduce output buffer is missing")
            reduced = self.work.wait_post_process(self.buffers.reduce_output_buffer)
            if not isinstance(reduced, torch.Tensor):
                raise TypeError(
                    "GroupReduce must return one tensor for one DSA payload"
                )
            if reduced.shape != self.buffers.reduce_output_buffer.shape:
                raise RuntimeError(
                    f"{self.meta.kind.value} GroupReduce returned shape "
                    f"{tuple(reduced.shape)}, expected "
                    f"{tuple(self.buffers.reduce_output_buffer.shape)}"
                )

            from magi_attention.kernel.cutedsl.dsa_pack import reduce_dsa_rows_csr

            logical_reduced = _restore_logical_row_shape(
                reduced, self.buffers.logical_row_shape
            )
            result = reduce_dsa_rows_csr(
                logical_reduced,
                self.device_map.owner_restore,
                self.local_accumulator,
                accumulate=False,
            ).to(self.output_dtype)
            self._result = DsaTypedPayload(self.meta.kind, result)
            return self._result
        except BaseException as error:
            self._collective_failed = getattr(self.work, "work", None) is not None
            self._wait_error = error
            raise


DsaAsyncWork = DsaGroupCastWork | DsaGroupReduceWork
_CURRENT_DSA_WORK_TRACKER: ContextVar["DsaWorkTracker | None"] = ContextVar(
    "current_dsa_work_tracker", default=None
)


def _abort_failed_process_group(group: dist.ProcessGroup | None) -> None:
    """Best-effort teardown after an unrecoverable collective wait failure."""

    if group is None or not dist.is_available() or not dist.is_initialized():
        return
    try:
        abort = getattr(group, "abort", None)
        if callable(abort):
            abort()
            return
    except BaseException:
        pass
    try:
        dist.destroy_process_group(group)
    except BaseException:
        pass


@dataclass
class DsaWorkTracker:
    """Per-invocation ownership and deterministic drain for async DSA work.

    A context variable makes nested/reentrant autograd scopes independent.  It
    contains only work launched by the current invocation; the runtime cache
    remains immutable and never owns work, events, or communication tensors.
    """

    group: dist.ProcessGroup | None
    _works: list[DsaAsyncWork] = field(default_factory=list, init=False, repr=False)
    _token: Token | None = field(default=None, init=False, repr=False)
    _group_aborted: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> "DsaWorkTracker":
        if self._token is not None:
            raise RuntimeError("a DSA work tracker cannot be entered twice")
        self._token = _CURRENT_DSA_WORK_TRACKER.set(self)
        return self

    def track(self, work: DsaAsyncWork) -> DsaAsyncWork:
        if not isinstance(work, (DsaGroupCastWork, DsaGroupReduceWork)):
            raise TypeError("only DSA GroupCast/GroupReduce work can be tracked")
        self._works.append(work)
        return work

    def _abort_group_once(self) -> None:
        if not self._group_aborted:
            _abort_failed_process_group(self.group)
            self._group_aborted = True

    def drain(self) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for work in self._works:
            try:
                work.wait()
            except BaseException as error:
                errors.append(error)
                if work._collective_failed:
                    # Further waits on a failed process group can block
                    # forever. Request fail-stop immediately and leave bounded
                    # multi-process teardown to the external launcher.
                    self._abort_group_once()
                    break
        return tuple(errors)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        errors = self.drain()
        if exc is not None:
            # A rank-local compute/kernel exception can make peers enter later
            # collectives that this rank will never launch. After draining the
            # common prefix, request best-effort fail-stop for the group. CUDA,
            # NCCL, or NVSHMEM faults still require an external launcher watchdog.
            self._abort_group_once()
        assert self._token is not None
        _CURRENT_DSA_WORK_TRACKER.reset(self._token)
        self._token = None
        if errors:
            if exc is None:
                raise errors[0]
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"DSA drain observed {len(errors)} additional work error(s): "
                    f"{errors[0]!r}"
                )
        return False


def _track_current_dsa_work(work: DsaAsyncWork) -> DsaAsyncWork:
    tracker = _CURRENT_DSA_WORK_TRACKER.get()
    return work if tracker is None else tracker.track(work)


@dataclass(frozen=True)
class _SendSegment:
    row_indices: tuple[int, ...]
    destinations: tuple[int, ...]


def _local_token_row_ids(plan: DsaDispatchPlan) -> tuple[tuple[int, ...], ...]:
    offsets = sample_offsets(plan.sample_lengths)
    return tuple(
        tuple(
            offsets[fragment.sample_id] + position
            for fragment in rank_plan.fragments
            for position in range(fragment.q_begin, fragment.q_end)
        )
        for rank_plan in plan.ranks
    )


def _local_compressed_row_ids(plan: DsaDispatchPlan) -> tuple[tuple[int, ...], ...]:
    return tuple(rank_plan.compressed_block_ids for rank_plan in plan.ranks)


def _token_destinations(
    plan: DsaDispatchPlan,
    transfers: Sequence[DsaTransferSpec],
    local_row_ids: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    offsets = sample_offsets(plan.sample_lengths)
    local_index = tuple(
        {row_id: index for index, row_id in enumerate(row_ids)}
        for row_ids in local_row_ids
    )
    destinations: list[list[set[int]]] = [
        [set() for _ in row_ids] for row_ids in local_row_ids
    ]
    for transfer in transfers:
        for position in range(transfer.q_begin, transfer.q_end):
            row_id = offsets[transfer.sample_id] + position
            try:
                source_index = local_index[transfer.source_rank][row_id]
            except (
                KeyError
            ) as exc:  # pragma: no cover - dispatch validation guards this
                raise ValueError(
                    f"transfer row {row_id} is not owned by rank "
                    f"{transfer.source_rank}"
                ) from exc
            destinations[transfer.source_rank][source_index].add(
                transfer.destination_rank
            )
    return tuple(
        tuple(tuple(sorted(row_destinations)) for row_destinations in per_rank)
        for per_rank in destinations
    )


def _compressed_destinations(
    plan: DsaDispatchPlan,
    local_row_ids: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(peer for peer in range(plan.cp_size) if peer != rank) for _ in row_ids
        )
        for rank, row_ids in enumerate(local_row_ids)
    )


def _make_send_segments(
    row_destinations: Sequence[Sequence[int]],
    rank: int,
    cp_size: int,
) -> tuple[_SendSegment, ...]:
    segments: list[_SendSegment] = []
    active_indices: list[int] = []
    active_destinations: tuple[int, ...] | None = None
    for index, destinations in enumerate(row_destinations):
        destination_tuple = tuple(destinations)
        if not destination_tuple:
            continue
        if active_indices and destination_tuple != active_destinations:
            assert active_destinations is not None
            segments.append(_SendSegment(tuple(active_indices), active_destinations))
            active_indices = []
        active_indices.append(index)
        active_destinations = destination_tuple
    if active_indices:
        assert active_destinations is not None
        segments.append(_SendSegment(tuple(active_indices), active_destinations))

    peers = tuple(peer for peer in range(cp_size) if peer != rank)
    covered = {
        destination for segment in segments for destination in segment.destinations
    }
    missing = tuple(peer for peer in peers if peer not in covered)
    if missing:
        # Explicit zero-length routes keep collective entry/order symmetric.
        segments.append(_SendSegment((), missing))
    if not segments:
        # CP=1 also keeps a legal zero-length metadata entry.
        segments.append(_SendSegment((), ()))
    return tuple(segments)


def _build_comm_meta(
    kind: DsaPayloadKind,
    plan: DsaDispatchPlan,
    rank: int,
    group: dist.ProcessGroup,
    local_row_ids_per_rank: Sequence[Sequence[int]],
    destinations_per_rank: Sequence[Sequence[Sequence[int]]],
) -> DsaCommMeta:
    segments_per_rank = tuple(
        _make_send_segments(destinations, source_rank, plan.cp_size)
        for source_rank, destinations in enumerate(destinations_per_rank)
    )
    local_segments = segments_per_rank[rank]
    send_row_indices = tuple(
        index for segment in local_segments for index in segment.row_indices
    )
    local_row_ids = tuple(local_row_ids_per_rank[rank])
    send_row_ids = tuple(local_row_ids[index] for index in send_row_indices)

    output_split_sizes: list[int] = []
    src_indices: list[int] = []
    receive_row_ids: list[int] = []
    for source_rank, source_segments in enumerate(segments_per_rank):
        if source_rank == rank:
            continue
        source_row_ids = local_row_ids_per_rank[source_rank]
        for segment in source_segments:
            if rank not in segment.destinations:
                continue
            output_split_sizes.append(len(segment.row_indices))
            src_indices.append(source_rank)
            receive_row_ids.extend(
                source_row_ids[index] for index in segment.row_indices
            )

    collective_arg = GroupCollectiveArg(
        input_split_size_list=[len(segment.row_indices) for segment in local_segments],
        output_split_size_list=output_split_sizes,
        dst_indices_list=[list(segment.destinations) for segment in local_segments],
        src_index_list=src_indices,
        rank=rank,
        world_size=plan.cp_size,
        group=group,
        deterministic=True,
        split_alignment=1,
    )
    return DsaCommMeta(
        kind=kind,
        collective_arg=collective_arg,
        local_row_ids=local_row_ids,
        send_row_indices=send_row_indices,
        send_row_ids=send_row_ids,
        receive_row_ids=tuple(receive_row_ids),
    )


def build_dsa_comm_plan(
    dispatch_plan: DsaDispatchPlan,
    rank: int,
    group: dist.ProcessGroup,
) -> DsaCommPlan:
    """Build four independent collective arguments for one CP rank."""

    if not isinstance(dispatch_plan, DsaDispatchPlan):
        raise TypeError("dispatch_plan must be a DsaDispatchPlan")
    if not 0 <= rank < dispatch_plan.cp_size:
        raise ValueError(f"rank must be in [0, {dispatch_plan.cp_size}), got {rank}")
    if dist.is_available() and dist.is_initialized():
        actual_rank = dist.get_rank(group)
        actual_size = dist.get_world_size(group)
        if (actual_rank, actual_size) != (rank, dispatch_plan.cp_size):
            raise ValueError(
                "group and dispatch plan disagree: "
                f"group=({actual_rank}, {actual_size}), "
                f"plan=({rank}, {dispatch_plan.cp_size})"
            )

    token_rows = _local_token_row_ids(dispatch_plan)
    compressed_rows = _local_compressed_row_ids(dispatch_plan)
    window_destinations = _token_destinations(
        dispatch_plan, dispatch_plan.window_transfers, token_rows
    )
    overlap_destinations = _token_destinations(
        dispatch_plan, dispatch_plan.overlap_transfers, token_rows
    )
    compressed_destinations = _compressed_destinations(dispatch_plan, compressed_rows)
    indexer_rows = (
        compressed_rows
        if dispatch_plan.compress_ratio == 4
        else tuple(() for _ in range(dispatch_plan.cp_size))
    )
    indexer_destinations = _compressed_destinations(dispatch_plan, indexer_rows)

    return DsaCommPlan(
        window_kv=_build_comm_meta(
            DsaPayloadKind.WINDOW_KV,
            dispatch_plan,
            rank,
            group,
            token_rows,
            window_destinations,
        ),
        overlap_x=_build_comm_meta(
            DsaPayloadKind.OVERLAP_X,
            dispatch_plan,
            rank,
            group,
            token_rows,
            overlap_destinations,
        ),
        compressed_kv=_build_comm_meta(
            DsaPayloadKind.COMPRESSED_KV,
            dispatch_plan,
            rank,
            group,
            compressed_rows,
            compressed_destinations,
        ),
        compressed_ki=_build_comm_meta(
            DsaPayloadKind.COMPRESSED_KI,
            dispatch_plan,
            rank,
            group,
            indexer_rows,
            indexer_destinations,
        ),
    )


def materialize_dsa_comm_map(
    meta: DsaCommMeta,
    device: torch.device | str | int,
) -> DsaDeviceCommMap:
    """Upload the validated static copy and inverse-CSR maps for one route."""

    if not isinstance(meta, DsaCommMeta):
        raise TypeError("meta must be a DsaCommMeta")
    from magi_attention.kernel.cutedsl.dsa_pack import (
        make_dsa_device_copy_map,
        make_dsa_device_reduce_map,
    )

    send_copy = make_dsa_device_copy_map(
        meta.send_row_indices,
        meta.local_row_count,
        device,
    )
    restore_rows: list[list[int]] = [[] for _ in range(meta.local_row_count)]
    for packed_row, local_row in enumerate(meta.send_row_indices):
        restore_rows[local_row].append(packed_row)
    owner_restore = make_dsa_device_reduce_map(
        restore_rows,
        meta.send_row_count,
        device,
    )
    return DsaDeviceCommMap(meta, send_copy, owner_restore)


def materialize_dsa_comm_plan(
    comm_plan: DsaCommPlan,
    device: torch.device | str | int,
) -> DsaDeviceCommPlan:
    """Materialize all four read-only maps once for one rank/device plan."""

    if not isinstance(comm_plan, DsaCommPlan):
        raise TypeError("comm_plan must be a DsaCommPlan")
    device_maps = tuple(materialize_dsa_comm_map(meta, device) for meta in comm_plan)
    return DsaDeviceCommPlan(*device_maps)


def _validate_payload(
    payload: DsaTypedPayload,
    meta: DsaCommMeta,
    expected_rows: int,
    role: str,
) -> None:
    if not isinstance(payload, DsaTypedPayload):
        raise TypeError(f"{role} must be a DsaTypedPayload")
    if payload.kind is not meta.kind:
        raise ValueError(
            f"{role} kind {payload.kind.value} cannot use {meta.kind.value} metadata"
        )
    if payload.tensor.size(0) != expected_rows:
        raise ValueError(
            f"{meta.kind.value} {role} must contain {expected_rows} rows, "
            f"got {payload.tensor.size(0)}"
        )
    if not payload.tensor.is_cuda:
        raise ValueError(f"{meta.kind.value} transport requires a CUDA tensor")


def _index_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.int64, device=device)


def _pad_native_hidden_size(tensor: torch.Tensor) -> torch.Tensor:
    """Pad one native-grpcoll row to its dtype-specific vector alignment."""

    if not env.comm.is_native_grpcoll_enable():
        return tensor
    hidden_size = math.prod(tensor.shape[1:])
    alignment = GrpCollBuffer.get_hidden_size_alignment(tensor.dtype)
    padded_hidden_size = (hidden_size + alignment - 1) // alignment * alignment
    flattened = tensor.reshape(tensor.size(0), hidden_size)
    if padded_hidden_size == hidden_size:
        return flattened.contiguous()
    return F.pad(flattened, (0, padded_hidden_size - hidden_size)).contiguous()


def _restore_logical_row_shape(
    tensor: torch.Tensor,
    logical_row_shape: Sequence[int],
) -> torch.Tensor:
    logical_hidden_size = math.prod(logical_row_shape)
    transport_hidden_size = math.prod(tensor.shape[1:])
    flattened = tensor.reshape(tensor.size(0), transport_hidden_size)
    if flattened.size(1) < logical_hidden_size:
        raise RuntimeError(
            f"transport row width {flattened.size(1)} is smaller than logical "
            f"row width {logical_hidden_size}"
        )
    return (
        flattened[:, :logical_hidden_size]
        .reshape(tensor.size(0), *logical_row_shape)
        .contiguous()
    )


def reference_pack_dsa_payload(
    payload: DsaTypedPayload,
    meta: DsaCommMeta,
) -> torch.Tensor:
    """Torch reference gather from owner-local rows into the GroupCast buffer."""

    _validate_payload(payload, meta, meta.local_row_count, "local payload")
    indices = _index_tensor(meta.send_row_indices, payload.tensor.device)
    return payload.tensor.index_select(0, indices).contiguous()


def reference_restore_dsa_gradient(
    reduced_packed_gradient: torch.Tensor,
    local_accumulator: torch.Tensor,
    meta: DsaCommMeta,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Torch reference restore from reduced packed rows to owner-local order."""

    if reduced_packed_gradient.size(0) != meta.send_row_count:
        raise ValueError(
            f"{meta.kind.value} reduced gradient must contain "
            f"{meta.send_row_count} rows, got {reduced_packed_gradient.size(0)}"
        )
    if local_accumulator.size(0) != meta.local_row_count:
        raise ValueError(
            f"{meta.kind.value} local accumulator must contain "
            f"{meta.local_row_count} rows, got {local_accumulator.size(0)}"
        )
    if local_accumulator.dtype != torch.float32:
        raise TypeError("the DSA local gradient accumulator must be FP32")
    if reduced_packed_gradient.shape[1:] != local_accumulator.shape[1:]:
        raise ValueError("packed and local DSA gradients must share their row shape")
    if reduced_packed_gradient.device != local_accumulator.device:
        raise ValueError("packed and local DSA gradients must share a device")
    if meta.send_row_indices:
        indices = _index_tensor(meta.send_row_indices, local_accumulator.device)
        local_accumulator.index_copy_(0, indices, reduced_packed_gradient.float())
    return local_accumulator.to(output_dtype)


def start_dsa_group_cast(
    payload: DsaTypedPayload,
    meta: DsaCommMeta,
    *,
    device_map: DsaDeviceCommMap | None = None,
    async_op: bool = False,
) -> DsaGroupCastWork:
    """Kernel-pack and launch one typed forward GroupCast."""

    _validate_payload(payload, meta, meta.local_row_count, "local payload")
    if device_map is None:
        device_map = materialize_dsa_comm_map(meta, payload.tensor.device)
    elif not isinstance(device_map, DsaDeviceCommMap):
        raise TypeError("device_map must be a DsaDeviceCommMap")
    elif device_map.meta is not meta and device_map.meta != meta:
        raise ValueError("device_map was materialized for different DSA metadata")

    from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows

    logical_send_buffer = copy_dsa_rows(payload.tensor, device_map.send_copy)
    send_buffer = _pad_native_hidden_size(logical_send_buffer)
    receive_buffer = send_buffer.new_empty(
        (meta.receive_row_count, *send_buffer.shape[1:])
    )
    buffers = DsaCommBufferSlot(
        meta.kind,
        send_buffer,
        receive_buffer,
        tuple(payload.tensor.shape[1:]),
    )
    native_handle_dict: dict[str, Any] = {"group_cast": None, "group_reduce": None}
    work = group_cast(
        input=send_buffer.detach(),
        output=receive_buffer,
        group=meta.collective_arg.group,
        async_op=async_op,
        split_alignment=meta.collective_arg.split_alignment,
        buffer_name=meta.kind.value,
        native_grpcoll_handle_dict=native_handle_dict,
        **meta.collective_arg.to_group_cast_args(),
    )
    return _track_current_dsa_work(
        DsaGroupCastWork(
            meta=meta,
            device_map=device_map,
            buffers=buffers,
            work=work,
            native_handle_dict=native_handle_dict,
        )
    )


def start_dsa_group_reduce(
    remote_gradient: DsaTypedPayload,
    local_gradient: DsaTypedPayload,
    forward_work: DsaGroupCastWork,
    *,
    async_op: bool = False,
    output_dtype: torch.dtype | None = None,
) -> DsaGroupReduceWork:
    """Launch the symmetric GroupReduce and restore owner-local row order.

    Both local and remote contributions are converted to FP32 before the
    collective.  ``wait()`` writes the reduced packed rows back into an FP32
    owner-local accumulator, then performs the single final dtype conversion.
    """

    if not isinstance(forward_work, DsaGroupCastWork):
        raise TypeError("forward_work must be a DsaGroupCastWork")
    if forward_work._reduce_started:
        raise RuntimeError("a DSA GroupCast work can start only one symmetric reduce")
    meta = forward_work.meta
    _validate_payload(remote_gradient, meta, meta.receive_row_count, "remote gradient")
    _validate_payload(local_gradient, meta, meta.local_row_count, "local gradient")
    if remote_gradient.tensor.device != local_gradient.tensor.device:
        raise ValueError("local and remote DSA gradients must share a CUDA device")
    if remote_gradient.tensor.shape[1:] != local_gradient.tensor.shape[1:]:
        raise ValueError("local and remote DSA gradients must share their row shape")
    expected_row_shape = forward_work.buffers.logical_row_shape
    if tuple(local_gradient.tensor.shape[1:]) != expected_row_shape:
        raise ValueError(
            f"{meta.kind.value} gradient row shape "
            f"{tuple(local_gradient.tensor.shape[1:])} does not match the forward "
            f"payload row shape {expected_row_shape}"
        )

    # Drain the forward work before reusing its native grpcoll routing handle.
    forward_work.wait()
    forward_work._reduce_started = True
    local_accumulator = local_gradient.tensor.float().clone(
        memory_format=torch.contiguous_format
    )
    from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows

    logical_reduce_output = copy_dsa_rows(
        local_accumulator,
        forward_work.device_map.send_copy,
    )
    reduce_output = _pad_native_hidden_size(logical_reduce_output)
    reduce_input = _pad_native_hidden_size(remote_gradient.tensor.float().contiguous())
    forward_work.buffers.reduce_output_buffer = reduce_output
    work = group_reduce(
        input=reduce_input,
        output=reduce_output,
        group=meta.collective_arg.group,
        async_op=async_op,
        reduce_op="sum",
        acc_reduce=True,
        comm_dtype=torch.float32,
        deterministic=True,
        split_alignment=meta.collective_arg.split_alignment,
        buffer_name=meta.kind.value,
        native_grpcoll_handle_dict=forward_work.native_handle_dict,
        **meta.collective_arg.to_group_reduce_args(),
    )
    return _track_current_dsa_work(
        DsaGroupReduceWork(
            meta=meta,
            device_map=forward_work.device_map,
            buffers=forward_work.buffers,
            work=work,
            local_accumulator=local_accumulator,
            output_dtype=(
                local_gradient.tensor.dtype if output_dtype is None else output_dtype
            ),
        )
    )


def reduce_replicated_dsa_gradient(
    gradient: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    buffer_name: str = "dsa_replicated_gradient",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sum one replicated DSA gradient on every rank with GroupReduce.

    Runtime-owned parameters and the attention sink are replicated rather
    than row-sharded.  Their gradients therefore use a one-row all-to-all
    GroupReduce: every rank sends its local FP32 contribution to every peer
    and accumulates all peer rows into its local contribution.  Keeping this
    helper in the DSA communication layer makes the no-``all_reduce`` contract
    explicit and gives native grpcoll the same padding treatment as the four
    row-routed payloads.
    """

    if not isinstance(gradient, torch.Tensor):
        raise TypeError("gradient must be a torch.Tensor")
    if not gradient.is_cuda:
        raise ValueError("replicated DSA gradient reduction requires CUDA")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        result = gradient.float()
        return result.to(gradient.dtype if output_dtype is None else output_dtype)

    rank = dist.get_rank(group)
    peers = [peer for peer in range(world_size) if peer != rank]
    logical_numel = gradient.numel()
    if logical_numel == 0:
        return gradient.to(gradient.dtype if output_dtype is None else output_dtype)
    flat = gradient.float().reshape(-1)
    if env.comm.is_native_grpcoll_enable():
        # Native scratch grows with the per-row hidden width. Flattening a
        # large parameter into one row can therefore require gigabytes even
        # when the gradient itself is only a few MiB. Bound the width and use
        # multiple rows; GroupReduce still sums the identical logical tensor.
        alignment = GrpCollBuffer.get_hidden_size_alignment(torch.float32)
        row_width = min(4096, max(alignment, logical_numel))
        row_width = (row_width + alignment - 1) // alignment * alignment
        padded_numel = (logical_numel + row_width - 1) // row_width * row_width
        transport = F.pad(flat, (0, padded_numel - logical_numel)).reshape(
            -1, row_width
        )
    else:
        transport = flat.reshape(1, logical_numel).contiguous()
    row_count = transport.size(0)
    send = transport.repeat(len(peers), 1)
    output = transport.clone()
    work = group_reduce(
        input=send,
        output=output,
        input_split_sizes=[row_count] * len(peers),
        output_split_sizes=[row_count],
        dst_index=peers,
        src_indices=[peers],
        group=group,
        async_op=False,
        reduce_op="sum",
        acc_reduce=True,
        comm_dtype=torch.float32,
        deterministic=True,
        split_alignment=1,
        buffer_name=buffer_name,
    )
    reduced = work.wait_post_process(output)
    if not isinstance(reduced, torch.Tensor):
        raise TypeError("replicated DSA GroupReduce must return one tensor")
    result = reduced.reshape(-1)[:logical_numel].reshape_as(gradient)
    return result.to(gradient.dtype if output_dtype is None else output_dtype)


__all__ = [
    "DsaCommBufferSlot",
    "DsaCommMeta",
    "DsaCommPlan",
    "DsaDeviceCommMap",
    "DsaDeviceCommPlan",
    "DsaGroupCastWork",
    "DsaGroupReduceWork",
    "DsaPayloadKind",
    "DsaTypedPayload",
    "DsaWorkTracker",
    "build_dsa_comm_plan",
    "materialize_dsa_comm_map",
    "materialize_dsa_comm_plan",
    "reference_pack_dsa_payload",
    "reference_restore_dsa_gradient",
    "reduce_replicated_dsa_gradient",
    "start_dsa_group_cast",
    "start_dsa_group_reduce",
]
