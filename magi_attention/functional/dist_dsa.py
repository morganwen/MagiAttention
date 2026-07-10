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

"""Magi_DSA packed context-parallel forward orchestration.

CP=2 inputs are owner-local rows in the immutable fragment order selected by
the runtime solver.  Window KV and ratio-4 overlap X use their static transfer
tables; compressed KV and Indexer K are broadcast to every peer.  All dynamic
Indexer selections stay on the device and one packed sparse-attention call
produces this rank's local output.

Step 5 intentionally owns forward only.  The typed GroupCast work objects are
kept invocation-local so step 6 can pair them with symmetric GroupReduce in a
custom backward without changing the static plan or public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from magi_attention.functional.dsa_comm import (
    DsaCommPlan,
    DsaDeviceCommPlan,
    DsaPayloadKind,
    DsaTypedPayload,
    build_dsa_comm_plan,
    materialize_dsa_comm_plan,
    start_dsa_group_cast,
)
from magi_attention.meta.collection.dsa_meta import (
    DsaDispatchPlan,
    DsaFragmentSpec,
    sample_offsets,
)

if TYPE_CHECKING:
    from magi_attention.api.dsa_attn_interface import MagiDSAInput
    from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
    from magi_attention.experimental.dsa_v4.compressor import DSAv4Compressor
    from magi_attention.kernel.cutedsl.dsa_pack import DsaDeviceCopyMap


@dataclass(frozen=True)
class DsaCompressorRun:
    """One owner-local compressed block within the gathered X slab."""

    logical_block_id: int
    source_begin: int
    source_end: int
    block_offset: int
    result_index: int


@dataclass(frozen=True)
class DsaForwardPlan:
    """Static host/device metadata cached by :class:`MagiDSARuntimeMgr`."""

    dispatch_plan: DsaDispatchPlan
    comm_plan: DsaCommPlan
    device_comm_plan: DsaDeviceCommPlan
    compressor_source_map: "DsaDeviceCopyMap"
    compressor_runs: tuple[DsaCompressorRun, ...]
    compressed_global_map: "DsaDeviceCopyMap"
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
        compressor_runs=tuple(compressor_runs),
        compressed_global_map=compressed_global_map,
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


def _ratio4_indices_and_kl(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    forward_plan: DsaForwardPlan,
    compressed_kv: torch.Tensor,
    compressed_ki: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fragment-local Indexer selection against sample-global keys."""

    from magi_attention.experimental.dsa_v4.indexer import compute_index_scores

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
    local_begin = 0
    ratio = cfg.compress_ratio
    total_global = max(forward_plan.dispatch_plan.total_tokens, 1)

    for fragment in forward_plan.dispatch_plan.ranks[
        runtime_mgr.plan.cp_rank
    ].fragments:
        local_end = local_begin + fragment.token_count
        sample_block_begin = forward_plan.sample_block_offsets[fragment.sample_id]
        sample_block_end = forward_plan.sample_block_offsets[fragment.sample_id + 1]
        sample_block_count = sample_block_end - sample_block_begin
        if sample_block_count == 0:
            local_begin = local_end
            continue

        x_fragment = dsa_input.x[local_begin:local_end].detach().unsqueeze(1)
        qr_fragment = dsa_input.qr[local_begin:local_end].detach().unsqueeze(1)
        q_idx, w_idx = indexer.project_queries(
            x_fragment,
            qr_fragment,
            row_offset=fragment.q_begin,
        )
        k_sample = compressed_ki[sample_block_begin:sample_block_end].unsqueeze(1)
        comp_sample = compressed_kv[sample_block_begin:sample_block_end].unsqueeze(1)

        if cfg.backend == "kernel":
            from magi_attention.experimental.dsa_v4.kernels import indexer_select_kernel

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
                local_ids = scores.topk(selected_width, dim=-1).indices.to(torch.int32)
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

        if runtime_mgr.training and torch.is_grad_enabled():
            if cfg.backend == "kernel":
                from magi_attention.experimental.dsa_v4.kernels import (
                    indexer_kl_loss_kernel,
                )

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
                from magi_attention.experimental.dsa_v4.reference import (
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
            local_ids + sample_block_begin + forward_plan.available_token_count,
            torch.full_like(local_ids, -1),
        )
        selected_rows[local_begin:local_end] = global_ids.squeeze(0)
        local_begin = local_end

    if local_begin != local_count:
        raise RuntimeError("Indexer fragments did not cover every local query row")
    return selected_rows, kl_loss


def _dist_dsa_forward(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
    dispatch_plan: DsaDispatchPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    from magi_attention.kernel.cutedsl.dsa_pack import copy_dsa_rows

    rank_plan = dispatch_plan.ranks[runtime_mgr.plan.cp_rank]
    if dsa_input.q.size(0) != rank_plan.token_count:
        raise ValueError(
            f"rank {runtime_mgr.plan.cp_rank} must provide {rank_plan.token_count} "
            f"fragment-ordered rows, got {dsa_input.q.size(0)}"
        )
    forward_plan = runtime_mgr.get_forward_plan(dispatch_plan, dsa_input.q.device)
    comm_plan = forward_plan.comm_plan
    device_plan = forward_plan.device_comm_plan

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
    if module.indexer is None:
        compressed_ki_local = dsa_input.x.new_empty((0, runtime_mgr.config.indexer_dim))
    else:
        compressed_ki_local = _run_owner_compressor(
            module.indexer.compressor,
            compressor_source.detach(),
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
        compressed_indices, kl_loss = _ratio4_indices_and_kl(
            dsa_input,
            runtime_mgr,
            forward_plan,
            compressed_global,
            compressed_ki_global,
        )
    elif dispatch_plan.compress_ratio == 128 and compressed_global.size(0):
        compressed_indices = forward_plan.dense_compressed_indices
    else:
        compressed_indices = forward_plan.window_indices.new_empty(
            (forward_plan.local_token_count, 0)
        )

    kv_full = torch.cat([token_available, compressed_global], dim=0)
    topk_indices = torch.cat(
        [forward_plan.window_indices, compressed_indices], dim=-1
    ).contiguous()
    output_flat = module._run_attention(
        dsa_input.q.unsqueeze(1),
        kv_full.unsqueeze(1),
        dsa_input.sink,
        topk_indices.unsqueeze(0),
        runtime_mgr.config,
    ).squeeze(1)
    output = output_flat.reshape(
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads,
        runtime_mgr.config.kv_dim,
    )
    return output, kl_loss


def dist_dsa_func(
    dsa_input: "MagiDSAInput",
    runtime_mgr: "MagiDSARuntimeMgr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the static plan and execute CP=1 or CP=2 packed forward."""

    runtime_mgr.validate_input(dsa_input)
    dispatch_plan = runtime_mgr.get_dispatch_plan(dsa_input.packed_meta)
    if runtime_mgr.plan.cp_size == 2:
        return _dist_dsa_forward(dsa_input, runtime_mgr, dispatch_plan)

    output_flat, kl_loss = runtime_mgr.dsa_module.forward_packed(
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
        dsa_input.packed_meta.cu_seqlens,
    )
    expected_flat = (
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads * runtime_mgr.config.kv_dim,
    )
    if tuple(output_flat.shape) != expected_flat:
        raise RuntimeError(
            f"legacy DSA path returned shape {tuple(output_flat.shape)}, "
            f"expected {expected_flat}"
        )
    output = output_flat.reshape(
        dsa_input.q.size(0),
        runtime_mgr.config.num_heads,
        runtime_mgr.config.kv_dim,
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
