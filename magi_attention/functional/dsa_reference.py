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

import math

import torch

from magi_attention.dsa_layer import DsaCompressor, MagiDSALayer
from magi_attention.dsa_types import MagiDSAForwardResult


def dsa_position_ids(
    cu_seqlens: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.arange(end - begin, device=device, dtype=torch.int64)
            for begin, end in zip(cu_seqlens, cu_seqlens[1:])
        ),
        dim=0,
    )


def deterministic_topk_global_ids(
    scores: torch.Tensor,
    k: int,
    *,
    global_offset: int = 0,
) -> torch.Tensor:
    """Select by score descending and exact ties by global ID ascending."""

    if scores.ndim != 1:
        raise ValueError("deterministic Top-K scores must be rank-1")
    if k < 0 or k > scores.numel():
        raise ValueError("deterministic Top-K cardinality is invalid")
    local_ids = torch.arange(scores.numel(), device=scores.device, dtype=torch.int64)
    global_ids = local_ids + int(global_offset)
    id_order = torch.argsort(global_ids, stable=True)
    score_order = torch.argsort(
        scores.index_select(0, id_order), descending=True, stable=True
    )
    return global_ids.index_select(0, id_order).index_select(0, score_order)[:k]


def _compress_global(
    compressor: DsaCompressor,
    x: torch.Tensor,
    cu_seqlens: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    ratio = compressor.config.ratio
    support = compressor.support
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    positions: list[int] = []
    sample_offsets: list[int] = []
    sample_counts: list[int] = []
    for sample_begin, sample_end in zip(cu_seqlens, cu_seqlens[1:]):
        sample_offsets.append(len(rows))
        block_count = (sample_end - sample_begin) // ratio
        sample_counts.append(block_count)
        for block_id in range(block_count):
            current_begin = sample_begin + block_id * ratio
            current = x[current_begin : current_begin + ratio]
            if compressor.overlap:
                if block_id == 0:
                    previous = torch.zeros_like(current)
                    previous_mask = torch.zeros(
                        ratio, device=x.device, dtype=torch.bool
                    )
                else:
                    previous = x[current_begin - ratio : current_begin]
                    previous_mask = torch.ones(ratio, device=x.device, dtype=torch.bool)
                rows.append(torch.cat((previous, current), dim=0))
                masks.append(
                    torch.cat(
                        (
                            previous_mask,
                            torch.ones(ratio, device=x.device, dtype=torch.bool),
                        ),
                        dim=0,
                    )
                )
            else:
                rows.append(current)
                masks.append(torch.ones(support, device=x.device, dtype=torch.bool))
            positions.append(block_id * ratio)
    if not rows:
        return (
            x.new_empty((0, compressor.output_dim)),
            tuple(sample_offsets),
            tuple(sample_counts),
        )
    packed_x = torch.stack(rows)
    valid = torch.stack(masks)
    position_tensor = torch.tensor(positions, device=x.device, dtype=torch.int64)
    return (
        compressor(packed_x, valid, position_tensor),
        tuple(sample_offsets),
        tuple(sample_counts),
    )


def _validate_global_inputs(
    layer: MagiDSALayer,
    x: torch.Tensor,
    qr: torch.Tensor,
    q: torch.Tensor,
    latent_kv: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    config = layer.config
    total_tokens = cu_seqlens[-1]
    if x.shape != (total_tokens, config.hidden_size):
        raise ValueError("x has an invalid global reference shape")
    if qr.shape != (total_tokens, config.q_lora_rank):
        raise ValueError("qr has an invalid global reference shape")
    if q.shape != (total_tokens, config.num_query_heads, config.head_dim):
        raise ValueError("q has an invalid global reference shape")
    if latent_kv.shape == (total_tokens, 1, config.head_dim):
        latent_kv = latent_kv[:, 0]
    if latent_kv.shape != (total_tokens, config.head_dim):
        raise ValueError("latent_kv has an invalid global reference shape")
    if sink.shape != (config.num_query_heads,) or sink.dtype != torch.float32:
        raise ValueError("sink must be a float32 vector with one value per query head")
    devices = {tensor.device for tensor in (x, qr, q, latent_kv, sink)}
    if len(devices) != 1:
        raise ValueError("all reference tensors must share one device")
    if any(end < begin for begin, end in zip(cu_seqlens, cu_seqlens[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    return latent_kv


def dsa_reference(
    layer: MagiDSALayer,
    x: torch.Tensor,
    qr: torch.Tensor,
    q: torch.Tensor,
    latent_kv: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens: tuple[int, ...],
    *,
    detach_indexer_trunk: bool = False,
) -> MagiDSAForwardResult:
    """Pure PyTorch CP1 contract for packed BF16 training prefill."""

    cu = tuple(int(value) for value in cu_seqlens)
    if len(cu) < 2 or cu[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    latent_kv = _validate_global_inputs(layer, x, qr, q, latent_kv, sink, cu)
    config = layer.config
    total_tokens = cu[-1]
    positions = dsa_position_ids(cu, device=x.device)

    if layer.compressor is not None:
        compressed_kv, sample_block_offsets, sample_block_counts = _compress_global(
            layer.compressor, x, cu
        )
    else:
        compressed_kv = x.new_empty((0, config.head_dim))
        sample_block_offsets = tuple(0 for _ in range(len(cu) - 1))
        sample_block_counts = tuple(0 for _ in range(len(cu) - 1))

    if layer.indexer is not None:
        q_index, weights = layer.indexer.project_queries(
            x,
            qr,
            positions,
            detach_trunk=detach_indexer_trunk,
        )
        index_x = x.detach() if detach_indexer_trunk else x
        compressed_ki, index_offsets, index_counts = _compress_global(
            layer.indexer.compressor, index_x, cu
        )
        if index_offsets != sample_block_offsets or index_counts != sample_block_counts:
            raise AssertionError("main and Indexer compression layouts differ")
    else:
        q_index = x.new_empty((total_tokens, 0, 0))
        weights = x.new_empty((total_tokens, 0))
        compressed_ki = x.new_empty((0, config.indexer_head_dim))

    output_rows: list[torch.Tensor] = []
    sparse_lse_rows: list[torch.Tensor] = []
    indexer_lse_rows: list[torch.Tensor] = []
    topk_rows: list[torch.Tensor] = []
    topk_lengths: list[int] = []
    kl_rows: list[torch.Tensor] = []
    scale = config.head_dim**-0.5

    for sample_id, (sample_begin, sample_end) in enumerate(zip(cu, cu[1:])):
        sample_length = sample_end - sample_begin
        block_offset = sample_block_offsets[sample_id]
        block_count = sample_block_counts[sample_id]
        sample_compressed_kv = compressed_kv[block_offset : block_offset + block_count]
        sample_compressed_ki = compressed_ki[block_offset : block_offset + block_count]
        if layer.indexer is not None and block_count:
            dots = torch.einsum(
                "qhd,kd->qhk",
                q_index[sample_begin:sample_end].float(),
                sample_compressed_ki.float(),
            )
            index_scores = (
                dots.relu() * weights[sample_begin:sample_end].float().unsqueeze(-1)
            ).sum(dim=1)
        else:
            index_scores = x.new_empty(
                (sample_length, block_count), dtype=torch.float32
            )

        for position in range(sample_length):
            global_row = sample_begin + position
            raw_begin = max(0, position - config.window_size + 1)
            raw_rows = latent_kv[sample_begin + raw_begin : global_row + 1]
            visible = (position + 1) // config.ratio if config.ratio else 0

            if config.ratio == 4:
                if visible:
                    visible_scores = index_scores[position, :visible]
                    indexer_lse = torch.logsumexp(visible_scores, dim=0)
                else:
                    visible_scores = index_scores.new_empty((0,))
                    indexer_lse = torch.full(
                        (), float("-inf"), device=x.device, dtype=torch.float32
                    )
                selected_length = min(config.indexer_topk, visible)
                if selected_length:
                    selected_global = deterministic_topk_global_ids(
                        visible_scores,
                        selected_length,
                        global_offset=block_offset,
                    )
                    selected_local = selected_global - block_offset
                else:
                    selected_local = torch.empty(0, device=x.device, dtype=torch.int64)
                    selected_global = selected_local
                topk_row = torch.full(
                    (config.indexer_topk,),
                    -1,
                    device=x.device,
                    dtype=torch.int32,
                )
                if selected_length:
                    topk_row[:selected_length] = selected_global.to(torch.int32)
                selected_compressed = sample_compressed_kv.index_select(
                    0, selected_local
                )
            elif config.ratio == 128:
                selected_length = 0
                selected_local = torch.arange(
                    visible, device=x.device, dtype=torch.int64
                )
                selected_compressed = sample_compressed_kv[:visible]
                topk_row = torch.empty(0, device=x.device, dtype=torch.int32)
                indexer_lse = torch.full(
                    (), float("-inf"), device=x.device, dtype=torch.float32
                )
            else:
                selected_length = 0
                selected_local = torch.empty(0, device=x.device, dtype=torch.int64)
                selected_compressed = latent_kv.new_empty((0, config.head_dim))
                topk_row = torch.empty(0, device=x.device, dtype=torch.int32)
                indexer_lse = torch.full(
                    (), float("-inf"), device=x.device, dtype=torch.float32
                )

            key_value = torch.cat((raw_rows, selected_compressed), dim=0)
            logits = (
                torch.einsum("hd,kd->hk", q[global_row].float(), key_value.float())
                * scale
            )
            sparse_lse = torch.logsumexp(logits, dim=-1)
            denominator = torch.logaddexp(sparse_lse, sink)
            probability = torch.exp(logits - denominator.unsqueeze(-1))
            output = torch.einsum("hk,kd->hd", probability, key_value.float()).to(
                q.dtype
            )
            output_rows.append(output)
            sparse_lse_rows.append(sparse_lse)
            indexer_lse_rows.append(indexer_lse)
            topk_rows.append(topk_row)
            topk_lengths.append(selected_length)

            if config.ratio == 4 and selected_length:
                selected_index_scores = index_scores[position].index_select(
                    0, selected_local
                )
                predict = selected_index_scores.softmax(dim=-1)
                compressed_logits = (
                    torch.einsum(
                        "hd,kd->hk",
                        q[global_row].float(),
                        selected_compressed.float(),
                    )
                    * scale
                )
                attention_mass = torch.exp(
                    compressed_logits - sparse_lse.unsqueeze(-1)
                ).sum(dim=0)
                target = (
                    attention_mass / attention_mass.sum().clamp_min(1e-10)
                ).detach()
                log_predict = (
                    predict.clamp_min(math.exp(-100.0)).log().clamp(min=-100.0, max=0.0)
                )
                log_target = (
                    target.clamp_min(math.exp(-100.0)).log().clamp(min=-100.0, max=0.0)
                )
                kl_rows.append((target * (log_target - log_predict)).sum())
            else:
                kl_rows.append(torch.zeros((), device=x.device, dtype=torch.float32))

    if output_rows:
        output_tensor = torch.stack(output_rows)
        sparse_lse_tensor = torch.stack(sparse_lse_rows)
        indexer_lse_tensor = torch.stack(indexer_lse_rows)
        if config.ratio == 4:
            topk_tensor = torch.stack(topk_rows)
        else:
            topk_tensor = torch.empty(
                (total_tokens, 0), device=x.device, dtype=torch.int32
            )
        length_tensor = torch.tensor(topk_lengths, device=x.device, dtype=torch.int32)
        kl = torch.stack(kl_rows).mean() * config.kl_loss_coeff
    else:
        output_tensor = q.new_empty(q.shape)
        sparse_lse_tensor = torch.empty(
            (0, config.num_query_heads), device=x.device, dtype=torch.float32
        )
        indexer_lse_tensor = torch.empty((0,), device=x.device, dtype=torch.float32)
        topk_tensor = torch.empty(
            (0, config.indexer_topk if config.ratio == 4 else 0),
            device=x.device,
            dtype=torch.int32,
        )
        length_tensor = torch.empty((0,), device=x.device, dtype=torch.int32)
        kl = torch.zeros((), device=x.device, dtype=torch.float32)

    return MagiDSAForwardResult(
        output=output_tensor,
        kl=kl,
        sparse_lse=sparse_lse_tensor,
        topk_ids=topk_tensor,
        topk_length=length_tensor,
        indexer_lse=indexer_lse_tensor,
    )


def validate_deterministic_topk(
    scores: torch.Tensor,
    actual_ids: torch.Tensor,
    k: int,
    *,
    global_offset: int = 0,
) -> None:
    """Validate the Q15 global-ID secondary-key contract."""

    if actual_ids.numel() != k:
        raise AssertionError("top-k cardinality mismatch")
    if len(set(int(value) for value in actual_ids.tolist())) != k:
        raise AssertionError("actual top-k IDs are not unique")
    expected = deterministic_topk_global_ids(scores, k, global_offset=global_offset).to(
        device=actual_ids.device, dtype=actual_ids.dtype
    )
    if not torch.equal(actual_ids, expected):
        raise AssertionError("top-k violates deterministic score/global-ID ordering")


def validate_canonical_topk(
    actual_ids: torch.Tensor,
    actual_lengths: torch.Tensor,
    expected_ids: torch.Tensor,
    expected_lengths: torch.Tensor,
    *,
    label: str = "production vs independent reference",
) -> None:
    """Validate Q14 canonical global IDs without comparing score-order position."""

    for name, ids, lengths in (
        ("actual", actual_ids, actual_lengths),
        ("expected", expected_ids, expected_lengths),
    ):
        if ids.ndim != 2 or lengths.ndim != 1:
            raise AssertionError(f"{label}: {name} Top-K tensors have invalid ranks")
        if ids.shape[0] != lengths.shape[0]:
            raise AssertionError(f"{label}: {name} Top-K row and length counts differ")
    if actual_ids.shape[0] != expected_ids.shape[0]:
        raise AssertionError(f"{label}: Top-K row counts differ")
    if actual_lengths.shape != expected_lengths.shape or not torch.equal(
        actual_lengths, expected_lengths
    ):
        raise AssertionError(f"{label}: effective Top-K lengths differ")

    lengths = actual_lengths.detach().to(device="cpu", dtype=torch.int64).tolist()
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length < 0:
            raise AssertionError(f"{label}: row {row} has a negative effective length")
        if length > actual_ids.shape[1] or length > expected_ids.shape[1]:
            raise AssertionError(
                f"{label}: row {row} effective length exceeds Top-K capacity"
            )

        canonical: list[torch.Tensor] = []
        for name, ids in (("actual", actual_ids), ("expected", expected_ids)):
            valid_ids = ids[row, :length]
            padding = ids[row, length:]
            if valid_ids.numel() and bool(torch.any(valid_ids < 0).item()):
                raise AssertionError(
                    f"{label}: row {row} {name} valid prefix contains padding"
                )
            if padding.numel() and bool(torch.any(padding >= 0).item()):
                raise AssertionError(
                    f"{label}: row {row} {name} padding contains a valid global ID"
                )
            if torch.unique(valid_ids).numel() != length:
                raise AssertionError(
                    f"{label}: row {row} {name} valid global IDs are not unique"
                )
            canonical.append(torch.sort(valid_ids).values)
        if not torch.equal(canonical[0], canonical[1]):
            raise AssertionError(
                f"{label}: row {row} canonical valid global IDs differ"
            )


__all__ = [
    "deterministic_topk_global_ids",
    "dsa_position_ids",
    "dsa_reference",
    "validate_canonical_topk",
    "validate_deterministic_topk",
]
