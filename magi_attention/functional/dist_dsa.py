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

"""Magi_DSA packed forward/backward orchestration.

Distributed inputs are owner-local rows in the immutable fragment order selected by
the runtime solver.  Window KV and ratio-4 overlap X use their static transfer
tables; compressed KV and Indexer K are broadcast to every peer.  All dynamic
Indexer selections stay on the device and one packed sparse-attention call
produces this rank's local output.

The custom autograd boundaries retain only original owner-local inputs plus
O/LSE/top-k state.  Backward reissues every typed GroupCast, recomputes the
compressors, reverses row routes with FP32 CSR + GroupReduce, and reduces
replicated sink/parameter gradients internally without retaining forward work
objects or remote/compressed buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from magi_attention.dsa.telemetry import dsa_phase
from magi_attention.functional.dsa_comm import (
    DsaCommPlan,
    DsaDeviceCommPlan,
    DsaPayloadKind,
    DsaTypedPayload,
    DsaWorkTracker,
    build_dsa_comm_plan,
    materialize_dsa_comm_plan,
    reduce_replicated_dsa_gradient,
    start_dsa_group_cast,
    start_dsa_group_reduce,
)
from magi_attention.meta.collection.dsa_meta import (
    DsaDispatchPlan,
    DsaFragmentSpec,
    sample_offsets,
)

if TYPE_CHECKING:
    from magi_attention.api.dsa_attn_interface import MagiDSAInput
    from magi_attention.dsa.compressor import DSAv4Compressor
    from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
    from magi_attention.kernel.cutedsl.dsa_pack import (
        DsaDeviceCopyMap,
        DsaDeviceReduceMap,
    )


@dataclass(frozen=True)
class DsaCompressorRun:
    """One owner-local compressed block within the gathered X slab."""

    logical_block_id: int
    source_begin: int
    source_end: int
    block_offset: int
    result_index: int


@dataclass(frozen=True)
class DsaIndexerProjection:
    """Invocation-local Indexer Q/weight projection for one fragment."""

    fragment: DsaFragmentSpec
    local_begin: int
    local_end: int
    query: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class DsaForwardPlan:
    """Static host/device metadata cached by :class:`MagiDSARuntimeMgr`."""

    dispatch_plan: DsaDispatchPlan
    comm_plan: DsaCommPlan
    device_comm_plan: DsaDeviceCommPlan
    compressor_source_map: "DsaDeviceCopyMap"
    compressor_source_restore: "DsaDeviceReduceMap"
    compressor_runs: tuple[DsaCompressorRun, ...]
    compressed_global_map: "DsaDeviceCopyMap"
    compressed_available_restore: "DsaDeviceReduceMap"
    window_indices: torch.Tensor
    dense_compressed_indices: torch.Tensor
    sample_block_offsets: tuple[int, ...]

    @property
    def local_token_count(self) -> int:
        return self.comm_plan.window_kv.local_row_count

    @property
    def available_token_count(self) -> int:
        meta = self.comm_plan.window_kv
        return meta.local_row_count + meta.receive_row_count


def _unique_row_lookup(name: str, row_ids: tuple[int, ...]) -> dict[int, int]:
    lookup = {row_id: index for index, row_id in enumerate(row_ids)}
    if len(lookup) != len(row_ids):
        raise RuntimeError(f"{name} contains duplicate logical rows")
    return lookup


def _inverse_reduce_rows(
    destination_to_source: list[int], source_row_count: int
) -> list[list[int]]:
    rows: list[list[int]] = [[] for _ in range(source_row_count)]
    for destination_row, source_row in enumerate(destination_to_source):
        rows[source_row].append(destination_row)
    return rows


def _local_query_rows(
    plan: DsaDispatchPlan, rank: int
) -> tuple[tuple[DsaFragmentSpec, int, int], ...]:
    """Return ``(fragment, sample_position, packed_global_row)`` in input order."""

    offsets = sample_offsets(plan.sample_lengths)
    return tuple(
        (fragment, position, offsets[fragment.sample_id] + position)
        for fragment in plan.ranks[rank].fragments
        for position in range(fragment.q_begin, fragment.q_end)
    )


def _sample_block_offsets(plan: DsaDispatchPlan) -> tuple[int, ...]:
    offsets = [0]
    for length in plan.sample_lengths:
        block_count = 0 if plan.compress_ratio == 0 else length // plan.compress_ratio
        offsets.append(offsets[-1] + block_count)
    return tuple(offsets)


def build_dsa_forward_plan(
    dispatch_plan: DsaDispatchPlan,
    rank: int,
    group: dist.ProcessGroup | None,
    device: torch.device,
) -> DsaForwardPlan:
    """Materialize one rank's immutable communication and attention maps."""

    if group is None:
        raise ValueError("a CP process group is required for a distributed DSA plan")
    if not 0 <= rank < dispatch_plan.cp_size:
        raise ValueError(f"rank must be in [0, {dispatch_plan.cp_size}), got {rank}")

    from magi_attention.kernel.cutedsl.dsa_pack import (
        make_dsa_device_copy_map,
        make_dsa_device_reduce_map,
        make_dsa_device_remap_lut,
        remap_dsa_indices,
    )

    comm_plan = build_dsa_comm_plan(dispatch_plan, rank, group)
    device_comm_plan = materialize_dsa_comm_plan(comm_plan, device)
    local_rows = _local_query_rows(dispatch_plan, rank)
    if len(local_rows) != comm_plan.window_kv.local_row_count:
        raise RuntimeError("fragment rows and window communication rows disagree")

    # The local X tensor and received overlap rows use this stable concatenation
    # order.  Each compressed block gathers either its own source block or, for
    # ratio=4 overlap, the preceding plus current source blocks.
    overlap_ids = (
        comm_plan.overlap_x.local_row_ids + comm_plan.overlap_x.receive_row_ids
    )
    overlap_lookup = _unique_row_lookup("overlap X rows", overlap_ids)
    offsets = sample_offsets(dispatch_plan.sample_lengths)
    compressor_source_rows: list[int] = []
    compressor_runs: list[DsaCompressorRun] = []
    ratio = dispatch_plan.compress_ratio
    for logical_block_id in dispatch_plan.ranks[rank].compressed_block_ids:
        block = dispatch_plan.compressed_blocks[logical_block_id]
        prepend = int(ratio == 4 and block.sample_block_id > 0)
        first_block = block.sample_block_id - prepend
        begin = first_block * ratio
        end = (block.sample_block_id + 1) * ratio
        source_begin = len(compressor_source_rows)
        for position in range(begin, end):
            row_id = offsets[block.sample_id] + position
            try:
                compressor_source_rows.append(overlap_lookup[row_id])
            except KeyError as exc:
                raise RuntimeError(
                    f"compressed block {logical_block_id} is missing overlap X row "
                    f"{row_id} on rank {rank}"
                ) from exc
        compressor_runs.append(
            DsaCompressorRun(
                logical_block_id=logical_block_id,
                source_begin=source_begin,
                source_end=len(compressor_source_rows),
                block_offset=first_block,
                result_index=prepend,
            )
        )
    compressor_source_map = make_dsa_device_copy_map(
        compressor_source_rows,
        len(overlap_ids),
        device,
    )
    compressor_source_restore = make_dsa_device_reduce_map(
        _inverse_reduce_rows(compressor_source_rows, len(overlap_ids)),
        len(compressor_source_rows),
        device,
    )

    # Every rank receives all remote compressed rows. Reorder the stable
    # local+remote concatenation into packed-global logical block order once.
    compressed_ids = (
        comm_plan.compressed_kv.local_row_ids + comm_plan.compressed_kv.receive_row_ids
    )
    compressed_lookup = _unique_row_lookup("compressed KV rows", compressed_ids)
    compressed_global_rows = []
    for logical_block_id in range(len(dispatch_plan.compressed_blocks)):
        try:
            compressed_global_rows.append(compressed_lookup[logical_block_id])
        except KeyError as exc:
            raise RuntimeError(
                f"rank {rank} cannot access compressed block {logical_block_id}"
            ) from exc
    compressed_global_map = make_dsa_device_copy_map(
        compressed_global_rows,
        len(compressed_ids),
        device,
    )
    compressed_available_restore = make_dsa_device_reduce_map(
        _inverse_reduce_rows(compressed_global_rows, len(compressed_ids)),
        len(compressed_global_rows),
        device,
    )

    # Window indices are entirely static for one fragment plan. Build them as
    # logical packed-token ids, upload once, and remap with the step-4 kernel.
    token_ids = comm_plan.window_kv.local_row_ids + comm_plan.window_kv.receive_row_ids
    token_lookup = _unique_row_lookup("window KV rows", token_ids)
    logical_to_local = [-1] * dispatch_plan.total_tokens
    for row_id, local_row in token_lookup.items():
        logical_to_local[row_id] = local_row
    window_logical: list[list[int]] = []
    for fragment, position, _ in local_rows:
        sample_start = offsets[fragment.sample_id]
        begin = position - (dispatch_plan.window_size - 1)
        window_logical.append(
            [
                -1 if source_position < 0 else sample_start + source_position
                for source_position in range(begin, position + 1)
            ]
        )
    window_logical_tensor = torch.tensor(
        window_logical,
        dtype=torch.int32,
        device=device,
    ).reshape(len(local_rows), dispatch_plan.window_size)
    window_lut = make_dsa_device_remap_lut(
        logical_to_local,
        len(token_ids),
        device,
    )
    window_indices = remap_dsa_indices(window_logical_tensor, window_lut)

    block_offsets = _sample_block_offsets(dispatch_plan)
    dense_width = (
        max(
            (length // ratio for length in dispatch_plan.sample_lengths),
            default=0,
        )
        if ratio == 128
        else 0
    )
    dense_rows: list[list[int]] = []
    if dense_width:
        compressed_base = len(token_ids)
        for fragment, position, _ in local_rows:
            visible = (position + 1) // ratio
            sample_block_begin = block_offsets[fragment.sample_id]
            row = [
                compressed_base + sample_block_begin + block_id
                for block_id in range(visible)
            ]
            dense_rows.append(row + [-1] * (dense_width - len(row)))
    dense_compressed_indices = torch.tensor(
        dense_rows,
        dtype=torch.int32,
        device=device,
    ).reshape(len(local_rows), dense_width)

    return DsaForwardPlan(
        dispatch_plan=dispatch_plan,
        comm_plan=comm_plan,
        device_comm_plan=device_comm_plan,
        compressor_source_map=compressor_source_map,
        compressor_source_restore=compressor_source_restore,
        compressor_runs=tuple(compressor_runs),
        compressed_global_map=compressed_global_map,
        compressed_available_restore=compressed_available_restore,
        window_indices=window_indices,
        dense_compressed_indices=dense_compressed_indices,
        sample_block_offsets=block_offsets,
    )


def _run_owner_compressor(
    compressor: "DSAv4Compressor",
    source_rows: torch.Tensor,
    forward_plan: DsaForwardPlan,
) -> torch.Tensor:
    """Compute owner-local compressed rows in logical-block order."""

    pieces: list[torch.Tensor] = []
    for run in forward_plan.compressor_runs:
        slab = source_rows[run.source_begin : run.source_end].unsqueeze(1)
        compressed = compressor(slab, block_offset=run.block_offset)
        if compressed is None or compressed.size(0) <= run.result_index:
            raise RuntimeError(
                f"compressor did not produce logical block {run.logical_block_id}"
            )
        pieces.append(compressed[run.result_index : run.result_index + 1].squeeze(1))
    if pieces:
        return torch.cat(pieces, dim=0)
    return source_rows.new_empty((0, compressor.head_dim))


def _project_ratio4_queries(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    forward_plan: DsaForwardPlan,
) -> tuple[DsaIndexerProjection, ...]:
    """Project all local Indexer queries without waiting for compressed casts."""

    indexer = runtime_mgr.dsa_module.indexer
    if indexer is None:
        return ()
    projections: list[DsaIndexerProjection] = []
    local_begin = 0
    for fragment in forward_plan.dispatch_plan.ranks[
        runtime_mgr.plan.cp_rank
    ].fragments:
        local_end = local_begin + fragment.token_count
        with dsa_phase("indexer_projection"):
            query, weights = indexer.project_queries(
                dsa_input.x[local_begin:local_end].detach().unsqueeze(1),
                dsa_input.qr[local_begin:local_end].detach().unsqueeze(1),
                row_offset=fragment.q_begin,
            )
        projections.append(
            DsaIndexerProjection(
                fragment=fragment,
                local_begin=local_begin,
                local_end=local_end,
                query=query,
                weights=weights,
            )
        )
        local_begin = local_end
    if local_begin != forward_plan.local_token_count:
        raise RuntimeError("Indexer projection did not cover every local query row")
    return tuple(projections)


def _ratio4_indices_and_kl(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    forward_plan: DsaForwardPlan,
    compressed_kv: torch.Tensor,
    compressed_ki: torch.Tensor,
    *,
    force_kl: bool = False,
    projections: tuple[DsaIndexerProjection, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fragment-local Indexer selection against sample-global keys."""

    from magi_attention.dsa.indexer import compute_index_scores

    cfg = runtime_mgr.config
    indexer = runtime_mgr.dsa_module.indexer
    if indexer is None:
        raise RuntimeError("ratio=4 requires an Indexer")
    local_count = forward_plan.local_token_count
    selected_rows = torch.full(
        (local_count, cfg.topk),
        -1,
        dtype=torch.int32,
        device=dsa_input.q.device,
    )
    kl_loss = torch.zeros((), dtype=torch.float32, device=dsa_input.q.device)
    if projections is None:
        projections = _project_ratio4_queries(dsa_input, runtime_mgr, forward_plan)
    fragments = forward_plan.dispatch_plan.ranks[runtime_mgr.plan.cp_rank].fragments
    if len(projections) != len(fragments):
        raise RuntimeError("Indexer projection count does not match local fragments")
    ratio = cfg.compress_ratio
    total_global = max(forward_plan.dispatch_plan.total_tokens, 1)

    for fragment, projection in zip(fragments, projections):
        if projection.fragment != fragment:
            raise RuntimeError("Indexer projection fragment order changed")
        local_begin = projection.local_begin
        local_end = projection.local_end
        sample_block_begin = forward_plan.sample_block_offsets[fragment.sample_id]
        sample_block_end = forward_plan.sample_block_offsets[fragment.sample_id + 1]
        sample_block_count = sample_block_end - sample_block_begin
        if sample_block_count == 0:
            continue

        q_idx, w_idx = projection.query, projection.weights
        k_sample = compressed_ki[sample_block_begin:sample_block_end].unsqueeze(1)
        comp_sample = compressed_kv[sample_block_begin:sample_block_end].unsqueeze(1)

        if cfg.backend == "kernel":
            from magi_attention.dsa.kernels import indexer_select_kernel

            with dsa_phase("indexer_topk"):
                local_ids = indexer_select_kernel(
                    q_idx,
                    k_sample,
                    w_idx,
                    cfg.topk,
                    ratio,
                    pos_offset=fragment.q_begin,
                ).to(torch.int32)
        else:
            with torch.no_grad():
                scores = compute_index_scores(q_idx, w_idx, k_sample)
                visible = (
                    torch.arange(
                        fragment.q_begin + 1,
                        fragment.q_end + 1,
                        device=dsa_input.q.device,
                    ).unsqueeze(1)
                    // ratio
                )
                columns = torch.arange(
                    sample_block_count, device=dsa_input.q.device
                ).unsqueeze(0)
                scores = scores.masked_fill(
                    columns.unsqueeze(0) >= visible.unsqueeze(0),
                    float("-inf"),
                )
                selected_width = min(cfg.topk, sample_block_count)
                local_ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[
                    ..., :selected_width
                ].to(torch.int32)
                if selected_width < cfg.topk:
                    local_ids = torch.nn.functional.pad(
                        local_ids, (0, cfg.topk - selected_width), value=-1
                    )

        visible = (
            torch.arange(
                fragment.q_begin + 1,
                fragment.q_end + 1,
                device=dsa_input.q.device,
                dtype=torch.int32,
            ).view(1, -1, 1)
            // ratio
        ).clamp(max=sample_block_count)
        valid = (local_ids >= 0) & (local_ids < visible)
        local_ids = torch.where(valid, local_ids, torch.full_like(local_ids, -1))

        if runtime_mgr.training and (torch.is_grad_enabled() or force_kl):
            if cfg.backend == "kernel":
                from magi_attention.dsa.kernels import (
                    indexer_kl_loss_kernel,
                )

                with dsa_phase("indexer_score_recompute"):
                    kl_loss = kl_loss + indexer_kl_loss_kernel(
                        local_ids,
                        q_idx,
                        w_idx,
                        k_sample,
                        dsa_input.q[local_begin:local_end].unsqueeze(1),
                        comp_sample,
                        cfg.softmax_scale,
                        indexer.softmax_scale,
                        cfg.indexer_loss_coeff,
                        total_global=total_global,
                    )
            else:
                from magi_attention.dsa.reference import (
                    indexer_kl_loss_selected,
                )

                kl_sum = indexer_kl_loss_selected(
                    local_ids,
                    q_idx,
                    w_idx,
                    k_sample,
                    dsa_input.q[local_begin:local_end].detach().unsqueeze(1),
                    comp_sample.detach(),
                    cfg.softmax_scale,
                    indexer.softmax_scale,
                    cfg.indexer_loss_coeff,
                    calculate_per_token_loss=True,
                )
                kl_loss = kl_loss + kl_sum / total_global

        global_ids = torch.where(
            valid,
            local_ids + sample_block_begin,
            torch.full_like(local_ids, -1),
        )
        selected_rows[local_begin:local_end] = global_ids.squeeze(0)

    return selected_rows, kl_loss


def _run_attention_forward_state(
    q: torch.Tensor,
    kv_full: torch.Tensor,
    sink: torch.Tensor,
    topk_indices: torch.Tensor,
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sparse attention while retaining only its backward ABI state."""

    cfg = runtime_mgr.config
    if q.size(0) == 0:
        return q.new_empty(q.shape), torch.empty(
            (0, cfg.num_heads), dtype=torch.float32, device=q.device
        )
    if cfg.backend == "kernel":
        from magi_attention.dsa.kernels import (
            _ensure_flash_mla,
            _topk_alignment,
        )

        alignment = _topk_alignment()
        padded_width = (topk_indices.size(1) + alignment - 1) // alignment * alignment
        kernel_indices = topk_indices
        if padded_width != topk_indices.size(1):
            kernel_indices = torch.nn.functional.pad(
                topk_indices,
                (0, padded_width - topk_indices.size(1)),
                value=-1,
            )
        output, _max_logits, lse = _ensure_flash_mla()(
            q.contiguous(),
            kv_full.unsqueeze(1),
            kernel_indices.contiguous().unsqueeze(1),
            cfg.softmax_scale,
            d_v=q.size(-1),
            attn_sink=sink.float(),
        )
        return output, lse.float()

    output_flat = runtime_mgr.dsa_module._run_attention(
        q.unsqueeze(1),
        kv_full.unsqueeze(1),
        sink,
        topk_indices.unsqueeze(0),
        cfg,
    ).squeeze(1)
    output = output_flat.reshape(q.size(0), cfg.num_heads, cfg.kv_dim)
    # The reference backward is an explicit testing seam and recomputes the
    # reference attention.  Production kernel backward consumes real FP32 LSE.
    return output, torch.empty(0, dtype=torch.float32, device=q.device)


def _dist_dsa_forward_state(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    forward_plan: DsaForwardPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute distributed forward and return the frozen minimal backward state."""

    from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows

    dispatch_plan = forward_plan.dispatch_plan
    rank_plan = dispatch_plan.ranks[runtime_mgr.plan.cp_rank]
    if dsa_input.q.size(0) != rank_plan.token_count:
        raise ValueError(
            f"rank {runtime_mgr.plan.cp_rank} must provide {rank_plan.token_count} "
            f"fragment-ordered rows, got {dsa_input.q.size(0)}"
        )
    comm_plan = forward_plan.comm_plan
    device_plan = forward_plan.device_comm_plan

    # Every work object below is registered with this invocation-local scope.
    # Normal return and recoverable exceptions both drain all launched work;
    # nested/reentrant calls receive an independent context-variable scope.
    with DsaWorkTracker(runtime_mgr.cp_group):
        window_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.WINDOW_KV, dsa_input.latent_kv),
            comm_plan.window_kv,
            device_map=device_plan.window_kv,
            async_op=True,
        )
        overlap_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.OVERLAP_X, dsa_input.x),
            comm_plan.overlap_x,
            device_map=device_plan.overlap_x,
            async_op=True,
        )

        remote_overlap = overlap_work.wait().tensor
        overlap_available = torch.cat([dsa_input.x, remote_overlap], dim=0).contiguous()
        compressor_source = copy_dsa_rows(
            overlap_available,
            forward_plan.compressor_source_map,
        )

        module = runtime_mgr.dsa_module
        if module.compressor is None:
            compressed_local = dsa_input.x.new_empty((0, runtime_mgr.config.kv_dim))
        else:
            compressed_local = _run_owner_compressor(
                module.compressor,
                compressor_source,
                forward_plan,
            )

        # Launch the main compressed payload as soon as it exists.  The
        # Indexer compressor and (when enabled) local Q/weight projection run
        # while the GroupCast progresses on its communication stream.
        compressed_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KV, compressed_local),
            comm_plan.compressed_kv,
            device_map=device_plan.compressed_kv,
            async_op=True,
        )
        if module.indexer is None:
            compressed_ki_local = dsa_input.x.new_empty(
                (0, runtime_mgr.config.indexer_dim)
            )
        else:
            compressed_ki_local = _run_owner_compressor(
                module.indexer.compressor,
                compressor_source.detach(),
                forward_plan,
            )
        compressed_ki_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KI, compressed_ki_local),
            comm_plan.compressed_ki,
            device_map=device_plan.compressed_ki,
            async_op=True,
        )

        projections = None
        if (
            runtime_mgr.overlap_config.compressed_cast_indexer
            and dispatch_plan.compress_ratio == 4
            and dispatch_plan.compressed_blocks
        ):
            with dsa_phase("overlap_compressed_cast_indexer"):
                projections = _project_ratio4_queries(
                    dsa_input, runtime_mgr, forward_plan
                )

        remote_window = window_work.wait().tensor
        remote_compressed = compressed_work.wait().tensor
        remote_compressed_ki = compressed_ki_work.wait().tensor
        token_available = torch.cat(
            [dsa_input.latent_kv, remote_window], dim=0
        ).contiguous()
        compressed_available = torch.cat(
            [compressed_local, remote_compressed], dim=0
        ).contiguous()
        compressed_global = copy_dsa_rows(
            compressed_available,
            forward_plan.compressed_global_map,
        )

        kl_loss = torch.zeros((), dtype=torch.float32, device=dsa_input.q.device)
        if dispatch_plan.compress_ratio == 4 and compressed_global.size(0):
            compressed_ki_available = torch.cat(
                [compressed_ki_local, remote_compressed_ki], dim=0
            ).contiguous()
            compressed_ki_global = copy_dsa_rows(
                compressed_ki_available,
                forward_plan.compressed_global_map,
            )
            indexer_topk, kl_loss = _ratio4_indices_and_kl(
                dsa_input,
                runtime_mgr,
                forward_plan,
                compressed_global,
                compressed_ki_global,
                force_kl=True,
                projections=projections,
            )
            compressed_indices = torch.where(
                indexer_topk >= 0,
                indexer_topk + forward_plan.available_token_count,
                torch.full_like(indexer_topk, -1),
            )
        elif dispatch_plan.compress_ratio == 128 and compressed_global.size(0):
            compressed_indices = forward_plan.dense_compressed_indices
            indexer_topk = forward_plan.window_indices.new_empty(
                (forward_plan.local_token_count, 0)
            )
        else:
            compressed_indices = forward_plan.window_indices.new_empty(
                (forward_plan.local_token_count, 0)
            )
            indexer_topk = compressed_indices

        kv_full = torch.cat([token_available, compressed_global], dim=0)
        topk_indices = torch.cat(
            [forward_plan.window_indices, compressed_indices], dim=-1
        ).contiguous()
        output, lse = _run_attention_forward_state(
            dsa_input.q,
            kv_full,
            dsa_input.sink,
            topk_indices,
            runtime_mgr,
        )
        return output, kl_loss, lse, indexer_topk.contiguous()


def _attention_backward(
    q: torch.Tensor,
    kv_full: torch.Tensor,
    sink: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    d_output: torch.Tensor,
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the kernel ABI, or the differentiable reference testing seam."""

    cfg = runtime_mgr.config
    if q.size(0) == 0:
        return torch.zeros_like(q), torch.zeros_like(kv_full), torch.zeros_like(sink)
    if cfg.backend == "kernel":
        from magi_attention.dsa.kernels import (
            _ensure_dsa,
            _topk_alignment,
        )

        alignment = _topk_alignment()
        padded_width = (topk_indices.size(1) + alignment - 1) // alignment * alignment
        kernel_indices = topk_indices
        if padded_width != topk_indices.size(1):
            kernel_indices = torch.nn.functional.pad(
                topk_indices,
                (0, padded_width - topk_indices.size(1)),
                value=-1,
            )
        result = _ensure_dsa().sparse_attention_backward_wrapper(
            q,
            kv_full,
            output,
            d_output.contiguous(),
            lse,
            sink,
            kernel_indices.contiguous(),
            softmax_scale=cfg.softmax_scale,
            topk_length=None,
        )
        return result["dq"], result["dkv"], result["d_sink"]

    from magi_attention.dsa.reference import sparse_attn_with_sink

    with torch.enable_grad():
        q_ref = q.detach().requires_grad_(True)
        kv_ref = kv_full.detach().requires_grad_(True)
        sink_ref = sink.detach().requires_grad_(True)
        output_ref = sparse_attn_with_sink(
            q_ref.unsqueeze(1),
            kv_ref.unsqueeze(1),
            sink_ref,
            topk_indices.unsqueeze(0),
            cfg.softmax_scale,
        ).squeeze(1)
        output_ref = output_ref.reshape_as(output)
        dq, dkv, d_sink = torch.autograd.grad(
            output_ref,
            (q_ref, kv_ref, sink_ref),
            d_output,
        )
    return dq, dkv, d_sink


def _saved_topk_kl(
    x: torch.Tensor,
    qr: torch.Tensor,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    compressed_ki: torch.Tensor,
    indexer_topk: torch.Tensor,
    runtime_mgr: "MagiDSARuntimeMgr",
    forward_plan: DsaForwardPlan,
) -> torch.Tensor:
    """Recompute Indexer scores for KL backward without selecting top-k."""

    cfg = runtime_mgr.config
    indexer = runtime_mgr.dsa_module.indexer
    if indexer is None:
        return torch.zeros((), dtype=torch.float32, device=x.device)
    total_global = max(forward_plan.dispatch_plan.total_tokens, 1)
    # Keep the zero-loss short-sample path differentiable. A rank may own only
    # samples shorter than one compression block while another rank makes the
    # global compressed tensor non-empty and therefore enters KL backward.
    kl_loss = compressed_ki.float().sum() * 0.0
    local_begin = 0
    for fragment in forward_plan.dispatch_plan.ranks[
        runtime_mgr.plan.cp_rank
    ].fragments:
        local_end = local_begin + fragment.token_count
        sample_begin = forward_plan.sample_block_offsets[fragment.sample_id]
        sample_end = forward_plan.sample_block_offsets[fragment.sample_id + 1]
        if sample_end == sample_begin:
            local_begin = local_end
            continue
        selected_global = indexer_topk[local_begin:local_end]
        local_ids = torch.where(
            selected_global >= 0,
            selected_global - sample_begin,
            torch.full_like(selected_global, -1),
        ).unsqueeze(0)
        q_idx, w_idx = indexer.project_queries(
            x[local_begin:local_end].detach().unsqueeze(1),
            qr[local_begin:local_end].detach().unsqueeze(1),
            row_offset=fragment.q_begin,
        )
        k_sample = compressed_ki[sample_begin:sample_end].unsqueeze(1)
        comp_sample = compressed_kv[sample_begin:sample_end].detach().unsqueeze(1)
        if cfg.backend == "kernel":
            from magi_attention.dsa.kernels import (
                indexer_kl_loss_kernel,
            )

            kl_fragment = indexer_kl_loss_kernel(
                local_ids,
                q_idx,
                w_idx,
                k_sample,
                q[local_begin:local_end].detach().unsqueeze(1),
                comp_sample,
                cfg.softmax_scale,
                indexer.softmax_scale,
                cfg.indexer_loss_coeff,
                total_global=total_global,
            )
        else:
            from magi_attention.dsa.reference import (
                indexer_kl_loss_selected,
            )

            kl_fragment = indexer_kl_loss_selected(
                local_ids,
                q_idx,
                w_idx,
                k_sample,
                q[local_begin:local_end].detach().unsqueeze(1),
                comp_sample,
                cfg.softmax_scale,
                indexer.softmax_scale,
                cfg.indexer_loss_coeff,
                calculate_per_token_loss=True,
            )
            kl_fragment = kl_fragment / total_global
        kl_loss = kl_loss + kl_fragment
        local_begin = local_end
    if local_begin != forward_plan.local_token_count:
        raise RuntimeError("KL recompute did not cover every local query row")
    return kl_loss


def _add_parameter_gradients(
    accumulators: list[torch.Tensor | None],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        value = gradient.float()
        if accumulators[index] is None:
            accumulators[index] = value
        else:
            accumulators[index] = accumulators[index] + value


def _with_dsa_backward_work_tracker(function):
    """Give each custom backward (including reentrant calls) its own work scope."""

    @wraps(function)
    def wrapped(ctx, *args):
        with DsaWorkTracker(ctx.runtime_mgr.cp_group):
            return function(ctx, *args)

    return wrapped


class _DistDsa(torch.autograd.Function):
    """Distributed autograd boundary with recompute-and-reverse communication."""

    @staticmethod
    def forward(
        ctx,
        runtime_mgr: "MagiDSARuntimeMgr",
        forward_plan: DsaForwardPlan,
        packed_meta,
        x: torch.Tensor,
        qr: torch.Tensor,
        q: torch.Tensor,
        latent_kv: torch.Tensor,
        sink: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from magi_attention.api.dsa_attn_interface import MagiDSAInput

        dsa_input = MagiDSAInput(
            x=x,
            qr=qr,
            q=q,
            latent_kv=latent_kv,
            sink=sink,
            packed_meta=packed_meta,
        )
        output, kl_loss, lse, indexer_topk = _dist_dsa_forward_state(
            dsa_input,
            runtime_mgr,
            forward_plan,
        )
        topk_length = (indexer_topk >= 0).sum(dim=-1, dtype=torch.int32)
        ctx.runtime_mgr = runtime_mgr
        ctx.forward_plan = forward_plan
        ctx.parameter_count = len(parameters)
        ctx.compute_kl = runtime_mgr.training
        ctx.save_for_backward(
            x,
            qr,
            q,
            latent_kv,
            sink,
            output,
            lse,
            indexer_topk,
            topk_length,
        )
        return output, kl_loss

    @staticmethod
    @_with_dsa_backward_work_tracker
    def backward(ctx, d_output: torch.Tensor | None, d_kl: torch.Tensor | None):
        (
            x,
            qr,
            q,
            latent_kv,
            sink,
            output,
            lse,
            indexer_topk,
            topk_length,
        ) = ctx.saved_tensors
        runtime_mgr = ctx.runtime_mgr
        forward_plan = ctx.forward_plan
        module = runtime_mgr.dsa_module
        cfg = runtime_mgr.config
        comm_plan = forward_plan.comm_plan
        device_plan = forward_plan.device_comm_plan
        expected_topk_width = cfg.topk if cfg.compress_ratio == 4 else 0
        if indexer_topk.shape != (forward_plan.local_token_count, expected_topk_width):
            raise RuntimeError(
                "saved Indexer top-k has shape "
                f"{tuple(indexer_topk.shape)}, expected "
                f"{(forward_plan.local_token_count, expected_topk_width)}"
            )
        if topk_length.shape != (forward_plan.local_token_count,):
            raise RuntimeError("saved topk_length must contain one count per query")

        if cfg.compress_ratio == 4:
            compressed_sparse_indices = torch.where(
                indexer_topk >= 0,
                indexer_topk + forward_plan.available_token_count,
                torch.full_like(indexer_topk, -1),
            )
        elif cfg.compress_ratio == 128:
            compressed_sparse_indices = forward_plan.dense_compressed_indices
        else:
            compressed_sparse_indices = forward_plan.window_indices.new_empty(
                (forward_plan.local_token_count, 0)
            )
        sparse_indices = torch.cat(
            [forward_plan.window_indices, compressed_sparse_indices], dim=-1
        ).contiguous()
        parameters = tuple(module.parameters())
        if len(parameters) != ctx.parameter_count:
            raise RuntimeError("DSA parameter set changed between forward and backward")
        parameter_grads: list[torch.Tensor | None] = [None] * len(parameters)

        from magi_attention.kernel.cutedsl.dsa_pack import (
            copy_dsa_rows,
            reduce_dsa_rows_csr,
        )

        # Forward work/buffers are intentionally gone. Reissue all four routes
        # from original owner-local inputs, then recompute both compressors.
        window_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.WINDOW_KV, latent_kv),
            comm_plan.window_kv,
            device_map=device_plan.window_kv,
            async_op=True,
        )
        overlap_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.OVERLAP_X, x),
            comm_plan.overlap_x,
            device_map=device_plan.overlap_x,
            async_op=True,
        )
        remote_overlap = overlap_work.wait().tensor
        overlap_available = torch.cat([x, remote_overlap], dim=0).contiguous()
        compressor_source_data = copy_dsa_rows(
            overlap_available,
            forward_plan.compressor_source_map,
        )

        with torch.enable_grad():
            compressor_source = compressor_source_data.detach().requires_grad_(True)
            if module.compressor is None:
                compressed_local = x.new_empty((0, cfg.kv_dim))
            else:
                compressed_local = _run_owner_compressor(
                    module.compressor,
                    compressor_source,
                    forward_plan,
                )
            if module.indexer is None:
                compressed_ki_local = x.new_empty((0, cfg.indexer_dim))
            else:
                compressed_ki_local = _run_owner_compressor(
                    module.indexer.compressor,
                    compressor_source_data.detach(),
                    forward_plan,
                )

        compressed_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KV, compressed_local),
            comm_plan.compressed_kv,
            device_map=device_plan.compressed_kv,
            async_op=True,
        )
        compressed_ki_work = start_dsa_group_cast(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KI, compressed_ki_local),
            comm_plan.compressed_ki,
            device_map=device_plan.compressed_ki,
            async_op=True,
        )
        remote_window = window_work.wait().tensor
        remote_compressed = compressed_work.wait().tensor
        remote_compressed_ki = compressed_ki_work.wait().tensor
        token_available = torch.cat([latent_kv, remote_window], dim=0).contiguous()
        compressed_available = torch.cat(
            [compressed_local.detach(), remote_compressed], dim=0
        ).contiguous()
        compressed_global = copy_dsa_rows(
            compressed_available,
            forward_plan.compressed_global_map,
        )

        # KL first: score recompute/backward creates dKi, then the symmetric
        # compressed-Ki GroupReduce returns it to the compressor owner.
        if (
            ctx.compute_kl
            and module.indexer is not None
            and compressed_global.size(0)
            and forward_plan.local_token_count
        ):
            compressed_ki_available = torch.cat(
                [compressed_ki_local.detach(), remote_compressed_ki], dim=0
            ).contiguous()
            compressed_ki_global_data = copy_dsa_rows(
                compressed_ki_available,
                forward_plan.compressed_global_map,
            )
            with dsa_phase("indexer_backward"):
                with torch.enable_grad():
                    compressed_ki_global = (
                        compressed_ki_global_data.detach().requires_grad_(True)
                    )
                    kl_recomputed = _saved_topk_kl(
                        x,
                        qr,
                        q,
                        compressed_global,
                        compressed_ki_global,
                        indexer_topk,
                        runtime_mgr,
                        forward_plan,
                    )
                    kl_upstream = (
                        torch.zeros((), dtype=torch.float32, device=x.device)
                        if d_kl is None
                        else d_kl.float()
                    )
                    kl_grads = torch.autograd.grad(
                        kl_recomputed,
                        (compressed_ki_global, *parameters),
                        kl_upstream,
                        allow_unused=True,
                    )
            d_compressed_ki_global = kl_grads[0].float()
            _add_parameter_gradients(parameter_grads, kl_grads[1:])
            d_compressed_ki_available = reduce_dsa_rows_csr(
                d_compressed_ki_global,
                forward_plan.compressed_available_restore,
            )
            ki_local_count = comm_plan.compressed_ki.local_row_count
            d_compressed_ki_local = d_compressed_ki_available[:ki_local_count]
            d_remote_compressed_ki = d_compressed_ki_available[ki_local_count:]
        else:
            d_compressed_ki_local = x.new_zeros(
                (comm_plan.compressed_ki.local_row_count, cfg.indexer_dim),
                dtype=torch.float32,
            )
            d_remote_compressed_ki = x.new_zeros(
                (comm_plan.compressed_ki.receive_row_count, cfg.indexer_dim),
                dtype=torch.float32,
            )
        dki_reduce_work = start_dsa_group_reduce(
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KI, d_remote_compressed_ki),
            DsaTypedPayload(DsaPayloadKind.COMPRESSED_KI, d_compressed_ki_local),
            compressed_ki_work,
            async_op=runtime_mgr.overlap_config.dki_reduce_sparse_backward,
            output_dtype=torch.float32,
        )
        owner_d_compressed_ki = None
        if not runtime_mgr.overlap_config.dki_reduce_sparse_backward:
            owner_d_compressed_ki = dki_reduce_work.wait().tensor

        kv_full = torch.cat([token_available, compressed_global], dim=0)
        if d_output is None:
            d_output = torch.zeros_like(output)
        sparse_phase = (
            "overlap_dki_reduce_sparse_backward"
            if runtime_mgr.overlap_config.dki_reduce_sparse_backward
            else "sparse_backward"
        )
        with dsa_phase(sparse_phase):
            dq, dkv_full, d_sink_local = _attention_backward(
                q,
                kv_full,
                sink,
                output,
                lse,
                sparse_indices,
                d_output,
                runtime_mgr,
            )
            if owner_d_compressed_ki is None:
                owner_d_compressed_ki = dki_reduce_work.wait().tensor
        available_count = forward_plan.available_token_count
        d_token_available = dkv_full[:available_count].float()
        d_compressed_global = dkv_full[available_count:].float()
        d_compressed_available = reduce_dsa_rows_csr(
            d_compressed_global,
            forward_plan.compressed_available_restore,
        )
        compressed_local_count = comm_plan.compressed_kv.local_row_count
        d_compressed_local = d_compressed_available[:compressed_local_count]
        d_remote_compressed = d_compressed_available[compressed_local_count:]
        owner_d_compressed = (
            start_dsa_group_reduce(
                DsaTypedPayload(DsaPayloadKind.COMPRESSED_KV, d_remote_compressed),
                DsaTypedPayload(DsaPayloadKind.COMPRESSED_KV, d_compressed_local),
                compressed_work,
                output_dtype=torch.float32,
            )
            .wait()
            .tensor
        )

        local_token_count = forward_plan.local_token_count
        d_local_kv = d_token_available[:local_token_count]
        d_remote_window = d_token_available[local_token_count:]
        owner_d_kv = (
            start_dsa_group_reduce(
                DsaTypedPayload(DsaPayloadKind.WINDOW_KV, d_remote_window),
                DsaTypedPayload(DsaPayloadKind.WINDOW_KV, d_local_kv),
                window_work,
                output_dtype=latent_kv.dtype,
            )
            .wait()
            .tensor
        )

        # PyTorch owns both compressor backwards. The main compressor returns
        # a source-row gradient; the detached Indexer compressor returns only
        # its parameter gradients.
        d_compressor_source = torch.zeros_like(
            compressor_source_data, dtype=torch.float32
        )
        if module.compressor is not None and compressed_local.numel():
            main_grads = torch.autograd.grad(
                compressed_local,
                (compressor_source, *parameters),
                owner_d_compressed.to(compressed_local.dtype),
                allow_unused=True,
            )
            if main_grads[0] is not None:
                d_compressor_source = main_grads[0].float()
            _add_parameter_gradients(parameter_grads, main_grads[1:])
        if module.indexer is not None and compressed_ki_local.numel():
            indexer_compressor_grads = torch.autograd.grad(
                compressed_ki_local,
                parameters,
                owner_d_compressed_ki.to(compressed_ki_local.dtype),
                allow_unused=True,
            )
            _add_parameter_gradients(parameter_grads, indexer_compressor_grads)

        d_overlap_available = reduce_dsa_rows_csr(
            d_compressor_source,
            forward_plan.compressor_source_restore,
        )
        overlap_local_count = comm_plan.overlap_x.local_row_count
        d_local_x = d_overlap_available[:overlap_local_count]
        d_remote_overlap = d_overlap_available[overlap_local_count:]
        owner_d_x = (
            start_dsa_group_reduce(
                DsaTypedPayload(DsaPayloadKind.OVERLAP_X, d_remote_overlap),
                DsaTypedPayload(DsaPayloadKind.OVERLAP_X, d_local_x),
                overlap_work,
                output_dtype=x.dtype,
            )
            .wait()
            .tensor
        )

        d_sink = reduce_replicated_dsa_gradient(
            d_sink_local,
            runtime_mgr.cp_group,
            output_dtype=sink.dtype,
        )
        setattr(sink, "_magi_dsa_cp_reduced", True)
        reduced_parameter_grads: list[torch.Tensor] = []
        for parameter, local_gradient in zip(parameters, parameter_grads):
            if local_gradient is None:
                local_gradient = torch.zeros_like(parameter, dtype=torch.float32)
            reduced = reduce_replicated_dsa_gradient(
                local_gradient,
                runtime_mgr.cp_group,
                output_dtype=parameter.dtype,
            )
            setattr(parameter, "_magi_dsa_cp_reduced", True)
            reduced_parameter_grads.append(reduced)

        d_x = owner_d_x if module.compressor is not None else None
        return (
            None,
            None,
            None,
            d_x,
            None,
            dq,
            owner_d_kv,
            d_sink,
            *reduced_parameter_grads,
        )


def _dist_dsa_forward(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    dispatch_plan: DsaDispatchPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_plan = runtime_mgr.get_forward_plan(dispatch_plan, dsa_input.q.device)
    parameters = tuple(runtime_mgr.dsa_module.parameters())
    # This invocation owns the CP reduction for these replicated gradients;
    # mark the public tensors before saved-tensor hooks can wrap them.
    setattr(dsa_input.sink, "_magi_dsa_cp_reduced", True)
    for parameter in parameters:
        setattr(parameter, "_magi_dsa_cp_reduced", True)
    return _DistDsa.apply(
        runtime_mgr,
        forward_plan,
        dsa_input.packed_meta,
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
        *parameters,
    )


def _single_forward_state(
    x: torch.Tensor,
    qr: torch.Tensor,
    q: torch.Tensor,
    latent_kv: torch.Tensor,
    sink: torch.Tensor,
    bounds: tuple[int, ...],
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CP=1 packed forward with the same minimal kernel backward ABI."""

    from magi_attention.dsa.indexer import (
        build_block_causal_mask,
        compute_index_scores,
    )
    from magi_attention.dsa.reference import (
        get_compress_topk_idxs,
        get_window_topk_idxs,
        indexer_kl_loss,
        validate_and_offset_topk,
    )

    cfg = runtime_mgr.config
    module = runtime_mgr.dsa_module
    outputs: list[torch.Tensor] = []
    lse_pieces: list[torch.Tensor] = []
    indexer_topk_pieces: list[torch.Tensor] = []
    kl_sum = torch.zeros((), dtype=torch.float32, device=x.device)
    saved_topk_width = cfg.topk if cfg.compress_ratio == 4 else 0
    sample_block_offsets = [0]
    for start, end in zip(bounds[:-1], bounds[1:]):
        block_count = (end - start) // 4 if cfg.compress_ratio == 4 else 0
        sample_block_offsets.append(sample_block_offsets[-1] + block_count)

    for sample_id, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        sq = end - start
        if sq == 0:
            continue
        x_sample = x[start:end].unsqueeze(1)
        qr_sample = qr[start:end].unsqueeze(1)
        q_sample = q[start:end]
        kv_sample = latent_kv[start:end]
        window = get_window_topk_idxs(cfg.window_size, 1, sq, x.device)
        sample_indexer_topk = torch.full(
            (sq, saved_topk_width),
            -1,
            dtype=torch.int32,
            device=x.device,
        )
        compressed = None
        if module.compressor is not None:
            compressed = module.compressor(x_sample)

        if compressed is None or compressed.size(0) == 0:
            kv_full = kv_sample
            if cfg.compress_ratio == 4:
                compressed_indices = torch.full(
                    (1, sq, cfg.topk),
                    -1,
                    dtype=window.dtype,
                    device=window.device,
                )
                indices = torch.cat([window, compressed_indices], dim=-1)
            else:
                indices = window
        else:
            n_compressed = compressed.size(0)
            kv_full = torch.cat([kv_sample, compressed.squeeze(1)], dim=0)
            if module.indexer is not None:
                q_idx, k_idx, w_idx = module.indexer.forward_before_topk(
                    x_sample.detach(),
                    qr_sample.detach(),
                )
                if cfg.backend == "kernel":
                    from magi_attention.dsa.kernels import (
                        indexer_kl_loss_kernel,
                        indexer_select_kernel,
                    )

                    selected = indexer_select_kernel(
                        q_idx,
                        k_idx,
                        w_idx,
                        cfg.topk,
                        cfg.compress_ratio,
                    ).long()
                    if runtime_mgr.training:
                        kl_sum = kl_sum + indexer_kl_loss_kernel(
                            selected,
                            q_idx,
                            w_idx,
                            k_idx,
                            q_sample.unsqueeze(1),
                            compressed,
                            cfg.softmax_scale,
                            module.indexer.softmax_scale,
                            cfg.indexer_loss_coeff,
                            total_global=1,
                        )
                else:
                    causal_mask = build_block_causal_mask(
                        sq,
                        n_compressed,
                        cfg.compress_ratio,
                        1,
                        x.device,
                    )
                    scores = compute_index_scores(
                        q_idx,
                        w_idx * module.indexer.softmax_scale,
                        k_idx,
                    )
                    scores = scores + causal_mask
                    selected = module.indexer.select_topk(scores, n_compressed)
                    if runtime_mgr.training:
                        kl_sum = kl_sum + indexer_kl_loss(
                            scores,
                            selected,
                            q_sample.detach().unsqueeze(1),
                            compressed.detach(),
                            cfg.softmax_scale,
                            cfg.indexer_loss_coeff,
                            causal_mask,
                            cfg.use_sparse_loss,
                            calculate_per_token_loss=True,
                        )
                if selected.size(-1) > cfg.topk:
                    raise RuntimeError("Indexer returned more than configured top-k")
                if selected.size(-1) < cfg.topk:
                    selected = torch.nn.functional.pad(
                        selected, (0, cfg.topk - selected.size(-1)), value=-1
                    )
                local_topk = (
                    validate_and_offset_topk(
                        selected,
                        cfg.compress_ratio,
                        0,
                    )
                    .squeeze(0)
                    .to(torch.int32)
                )
                sample_indexer_topk = torch.where(
                    local_topk >= 0,
                    local_topk + sample_block_offsets[sample_id],
                    torch.full_like(local_topk, -1),
                )
                compressed_indices = torch.where(
                    local_topk >= 0,
                    local_topk + sq,
                    torch.full_like(local_topk, -1),
                ).unsqueeze(0)
            else:
                compressed_indices = get_compress_topk_idxs(
                    cfg.compress_ratio,
                    1,
                    sq,
                    sq,
                    x.device,
                )
            indices = torch.cat([window, compressed_indices], dim=-1)

        indices_flat = indices.squeeze(0).to(torch.int32).contiguous()
        output_sample, lse_sample = _run_attention_forward_state(
            q_sample,
            kv_full,
            sink,
            indices_flat,
            runtime_mgr,
        )
        outputs.append(output_sample)
        indexer_topk_pieces.append(sample_indexer_topk)
        if lse_sample.numel():
            lse_pieces.append(lse_sample)

    output = torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)
    lse = (
        torch.cat(lse_pieces, dim=0)
        if lse_pieces
        else torch.empty(0, dtype=torch.float32, device=x.device)
    )
    indexer_topk = (
        torch.cat(indexer_topk_pieces, dim=0)
        if indexer_topk_pieces
        else torch.empty((0, saved_topk_width), dtype=torch.int32, device=x.device)
    )
    topk_length = (indexer_topk >= 0).sum(dim=-1, dtype=torch.int32)
    return output, kl_sum / max(x.size(0), 1), lse, indexer_topk, topk_length


class _SingleDsa(torch.autograd.Function):
    """CP=1 recompute boundary enforcing the frozen saved-state contract."""

    @staticmethod
    def forward(
        ctx,
        runtime_mgr: "MagiDSARuntimeMgr",
        bounds: tuple[int, ...],
        x: torch.Tensor,
        qr: torch.Tensor,
        q: torch.Tensor,
        latent_kv: torch.Tensor,
        sink: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, kl_loss, lse, indexer_topk, topk_length = _single_forward_state(
            x,
            qr,
            q,
            latent_kv,
            sink,
            bounds,
            runtime_mgr,
        )
        ctx.runtime_mgr = runtime_mgr
        ctx.bounds = bounds
        ctx.parameter_count = len(parameters)
        ctx.compute_kl = runtime_mgr.training
        ctx.save_for_backward(
            x,
            qr,
            q,
            latent_kv,
            sink,
            output,
            lse,
            indexer_topk,
            topk_length,
        )
        return output, kl_loss

    @staticmethod
    def backward(ctx, d_output: torch.Tensor | None, d_kl: torch.Tensor | None):
        (
            x,
            qr,
            q,
            latent_kv,
            sink,
            output,
            lse,
            indexer_topk,
            topk_length,
        ) = ctx.saved_tensors
        runtime_mgr = ctx.runtime_mgr
        module = runtime_mgr.dsa_module
        cfg = runtime_mgr.config
        expected_topk_width = cfg.topk if cfg.compress_ratio == 4 else 0
        if indexer_topk.shape != (x.size(0), expected_topk_width):
            raise RuntimeError(
                "saved Indexer top-k has shape "
                f"{tuple(indexer_topk.shape)}, expected "
                f"{(x.size(0), expected_topk_width)}"
            )
        if topk_length.shape != (x.size(0),):
            raise RuntimeError("saved topk_length must contain one count per query")
        from magi_attention.dsa.reference import (
            get_compress_topk_idxs,
            get_window_topk_idxs,
        )

        sample_block_offsets = [0]
        for start, end in zip(ctx.bounds[:-1], ctx.bounds[1:]):
            block_count = (end - start) // 4 if cfg.compress_ratio == 4 else 0
            sample_block_offsets.append(sample_block_offsets[-1] + block_count)
        parameters = tuple(module.parameters())
        if len(parameters) != ctx.parameter_count:
            raise RuntimeError("DSA parameter set changed between forward and backward")
        parameter_grads: list[torch.Tensor | None] = [None] * len(parameters)
        dq = torch.zeros_like(q)
        dkv = torch.zeros_like(latent_kv)
        dx = torch.zeros_like(x) if module.compressor is not None else None
        d_sink = torch.zeros_like(sink)
        if d_output is None:
            d_output = torch.zeros_like(output)
        kl_upstream = (
            torch.zeros((), dtype=torch.float32, device=x.device)
            if d_kl is None
            else d_kl.float()
        )

        row_begin = 0
        lse_begin = 0
        for sample_id, (start, end) in enumerate(zip(ctx.bounds[:-1], ctx.bounds[1:])):
            sq = end - start
            if sq == 0:
                continue
            sample_indexer_topk = indexer_topk[row_begin : row_begin + sq].contiguous()
            window_indices = get_window_topk_idxs(
                cfg.window_size, 1, sq, x.device
            ).squeeze(0)
            if cfg.compress_ratio == 4:
                local_ids = torch.where(
                    sample_indexer_topk >= 0,
                    sample_indexer_topk - sample_block_offsets[sample_id],
                    torch.full_like(sample_indexer_topk, -1),
                )
                compressed_indices = torch.where(
                    local_ids >= 0,
                    local_ids + sq,
                    torch.full_like(local_ids, -1),
                )
            elif cfg.compress_ratio == 128:
                compressed_indices = get_compress_topk_idxs(
                    cfg.compress_ratio, 1, sq, sq, x.device
                ).squeeze(0)
                local_ids = sample_indexer_topk
            else:
                compressed_indices = window_indices.new_empty((sq, 0))
                local_ids = sample_indexer_topk
            sample_indices = (
                torch.cat([window_indices, compressed_indices], dim=-1)
                .to(torch.int32)
                .contiguous()
            )
            sample_lse = lse[lse_begin : lse_begin + sq] if lse.numel() else lse
            x_sample_data = x[start:end].unsqueeze(1)
            with torch.enable_grad():
                x_sample = x_sample_data.detach().requires_grad_(True)
                compressed = (
                    None if module.compressor is None else module.compressor(x_sample)
                )
            if compressed is None or compressed.size(0) == 0:
                compressed_flat = latent_kv.new_empty((0, cfg.kv_dim))
                kv_full = latent_kv[start:end]
            else:
                compressed_flat = compressed.squeeze(1)
                kv_full = torch.cat(
                    [latent_kv[start:end], compressed_flat.detach()], dim=0
                )

            dq_sample, dkv_full, d_sink_sample = _attention_backward(
                q[start:end],
                kv_full,
                sink,
                output[row_begin : row_begin + sq],
                sample_lse,
                sample_indices,
                d_output[row_begin : row_begin + sq],
                runtime_mgr,
            )
            dq[start:end] = dq_sample
            dkv[start:end] = dkv_full[:sq].to(dkv.dtype)
            d_sink = d_sink + d_sink_sample.float()

            if compressed is not None and compressed.numel():
                main_grads = torch.autograd.grad(
                    compressed,
                    (x_sample, *parameters),
                    dkv_full[sq:].reshape_as(compressed).to(compressed.dtype),
                    allow_unused=True,
                )
                if dx is not None and main_grads[0] is not None:
                    dx[start:end] = main_grads[0].squeeze(1).to(dx.dtype)
                _add_parameter_gradients(parameter_grads, main_grads[1:])

            if ctx.compute_kl and module.indexer is not None and compressed is not None:
                with torch.enable_grad():
                    q_idx, w_idx = module.indexer.project_queries(
                        x_sample_data.detach(),
                        qr[start:end].detach().unsqueeze(1),
                    )
                    compressed_ki = module.indexer.compressor(x_sample_data.detach())
                    local_ids_batched = local_ids.unsqueeze(0)
                    if cfg.backend == "kernel":
                        from magi_attention.dsa.kernels import (
                            indexer_kl_loss_kernel,
                        )

                        kl_sample = indexer_kl_loss_kernel(
                            local_ids_batched,
                            q_idx,
                            w_idx,
                            compressed_ki,
                            q[start:end].detach().unsqueeze(1),
                            compressed.detach(),
                            cfg.softmax_scale,
                            module.indexer.softmax_scale,
                            cfg.indexer_loss_coeff,
                            total_global=max(x.size(0), 1),
                        )
                    else:
                        from magi_attention.dsa.indexer import (
                            build_block_causal_mask,
                            compute_index_scores,
                        )
                        from magi_attention.dsa.reference import (
                            indexer_kl_loss,
                        )

                        causal_mask = build_block_causal_mask(
                            sq,
                            compressed_ki.size(0),
                            cfg.compress_ratio,
                            1,
                            x.device,
                        )
                        scores = compute_index_scores(
                            q_idx,
                            w_idx * module.indexer.softmax_scale,
                            compressed_ki,
                        )
                        scores = scores + causal_mask
                        kl_sample = indexer_kl_loss(
                            scores,
                            local_ids_batched,
                            q[start:end].detach().unsqueeze(1),
                            compressed.detach(),
                            cfg.softmax_scale,
                            cfg.indexer_loss_coeff,
                            causal_mask,
                            cfg.use_sparse_loss,
                            calculate_per_token_loss=True,
                        ) / max(x.size(0), 1)
                    kl_parameter_grads = torch.autograd.grad(
                        kl_sample,
                        parameters,
                        kl_upstream,
                        allow_unused=True,
                    )
                _add_parameter_gradients(parameter_grads, kl_parameter_grads)
            row_begin += sq
            lse_begin += sq

        return (
            None,
            None,
            dx,
            None,
            dq,
            dkv,
            d_sink.to(sink.dtype),
            *[
                None if gradient is None else gradient.to(parameter.dtype)
                for parameter, gradient in zip(parameters, parameter_grads)
            ],
        )


def _single_dsa_forward(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds = tuple(
        int(value) for value in dsa_input.packed_meta.cu_seqlens.detach().cpu()
    )
    return _SingleDsa.apply(
        runtime_mgr,
        bounds,
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
        *tuple(runtime_mgr.dsa_module.parameters()),
    )


def dist_dsa_func(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the static plan and execute local or distributed packed forward."""

    runtime_mgr.validate_input(dsa_input)
    dispatch_plan = runtime_mgr.get_dispatch_plan(dsa_input.packed_meta)
    if runtime_mgr.plan.cp_size > 1:
        return _dist_dsa_forward(dsa_input, runtime_mgr, dispatch_plan)

    output, kl_loss = _single_dsa_forward(dsa_input, runtime_mgr)
    expected_shape = (
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads,
        runtime_mgr.config.kv_dim,
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"single-rank DSA path returned shape {tuple(output.shape)}, "
            f"expected {expected_shape}"
        )
    if kl_loss.shape != torch.Size([]) or kl_loss.dtype != torch.float32:
        raise RuntimeError(
            "legacy DSA path must return a scalar FP32 KL loss, "
            f"got shape={tuple(kl_loss.shape)}, dtype={kl_loss.dtype}"
        )
    return output, kl_loss


__all__ = [
    "DsaCompressorRun",
    "DsaForwardPlan",
    "build_dsa_forward_plan",
    "dist_dsa_func",
]
