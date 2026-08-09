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

import argparse
import bisect
import hashlib
import importlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from magi_attn_extensions.DSA.comm import unlayout_dsa_query_tensor
from magi_attn_extensions.DSA.config import MagiDSAConfig
from magi_attn_extensions.DSA.modeling import MagiDSALayer
from magi_attn_extensions.DSA.model_adapter import layout_and_project_dsa_input
from magi_attn_extensions.DSA.nvtx import dsa_nvtx_range
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr
from magi_attn_extensions.DSA.types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)

# One planner remains, so a "plan" is now only an artifact-directory label.
_PLAN_LABELS = ("balanced", "sequential")
_STEP_MODES = ("forward", "forward-backward")
_TENSOR_NAMES = ("x", "qr", "q", "latent_kv", "sink")
_INDEXER_BACKWARD_DIAGNOSTICS_ENV = "MAGI_DSA_INDEXER_BACKWARD_DIAGNOSTICS"
_BACKWARD_PIPELINE_DIAGNOSTICS_ENV = "MAGI_DSA_BACKWARD_PIPELINE_DIAGNOSTICS"


@dataclass(frozen=True)
class _ProfileSource:
    """Fixed source-owner tensors reused without mutation across all steps."""

    x: torch.Tensor
    sink: torch.Tensor
    packed_meta: MagiDSAPackedMeta


@dataclass(frozen=True)
class _ProfileDSAInputBoundary:
    """Fixed post-projection DSA input leaves prepared before profiler capture."""

    value: MagiDSAInput
    dout: torch.Tensor
    dkl: torch.Tensor
    token_layout_forward_ms: float = 0.0


@dataclass(frozen=True)
class _BackwardPipelineDiagnosticRecord:
    """One asynchronously queued backward tensor diagnostic."""

    label: str
    shape: tuple[int, ...]
    dtype: str
    counts: torch.Tensor
    max_abs: torch.Tensor


_BACKWARD_PIPELINE_DIAGNOSTIC_RECORDS: list[_BackwardPipelineDiagnosticRecord] = []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Magi-DSA v4 CP8 profile worker")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("diagnostic", "profile", "smoke"), required=True
    )
    parser.add_argument("--plan", choices=_PLAN_LABELS, default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--step-mode", choices=_STEP_MODES, default="forward")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _record(artifact_dir: Path, event: str, rank: int, **fields: object) -> None:
    payload: dict[str, object] = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "rank": rank,
        "record_type": "magi_dsa_profile_control",
        "wall_time_ns": time.time_ns(),
    }
    payload.update(fields)
    path = artifact_dir / f"control_rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"MAGI_DSA_PROFILE {json.dumps(payload, sort_keys=True)}", flush=True)


def _row_location(row: int, cu_seqlens: list[int]) -> dict[str, int]:
    segment = bisect.bisect_right(cu_seqlens, row) - 1
    if segment < 0 or segment + 1 >= len(cu_seqlens):
        raise ValueError(f"row {row} is outside packed segment boundaries")
    return {
        "local_row": row - cu_seqlens[segment],
        "row": row,
        "segment": segment,
    }


def _summarize_nonfinite_rows(
    row_counts: torch.Tensor,
    row_width: int,
    cu_seqlens: list[int],
    *,
    q_causal_offsets: list[int] | None = None,
    ratio: int | None = None,
    k_cu_seqlens: list[int] | None = None,
) -> dict[str, object]:
    counts = row_counts.detach().cpu().to(torch.int64)
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("nonfinite row diagnostics have an invalid schema")
    if not cu_seqlens or cu_seqlens[0] != 0 or cu_seqlens[-1] != counts.shape[0]:
        raise ValueError("nonfinite row diagnostics have invalid segment boundaries")
    nan_counts = counts[:, 0]
    positive_inf_counts = counts[:, 1]
    negative_inf_counts = counts[:, 2]
    nan_rows = torch.nonzero(nan_counts > 0).flatten().tolist()
    positive_inf_rows = torch.nonzero(positive_inf_counts > 0).flatten().tolist()
    negative_inf_rows = torch.nonzero(negative_inf_counts > 0).flatten().tolist()

    def annotate(rows: list[int], *, include_nan_column: bool) -> list[dict[str, int]]:
        records: list[dict[str, int]] = []
        for row in rows[:256]:
            record = _row_location(row, cu_seqlens)
            if include_nan_column:
                record["first_nan_column"] = int(counts[row, 3])
            if (
                q_causal_offsets is not None
                and ratio is not None
                and k_cu_seqlens is not None
            ):
                segment = record["segment"]
                k_length = k_cu_seqlens[segment + 1] - k_cu_seqlens[segment]
                record["causal_columns"] = min(
                    k_length,
                    max(
                        0,
                        (q_causal_offsets[segment] + record["local_row"] + 1) // ratio,
                    ),
                )
            records.append(record)
        return records

    segments: list[dict[str, int]] = []
    for segment, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        segment_counts = counts[begin:end, :3].sum(dim=0)
        segment_nan_rows = int((nan_counts[begin:end] > 0).sum())
        segment_positive_inf_rows = int((positive_inf_counts[begin:end] > 0).sum())
        segment_negative_inf_rows = int((negative_inf_counts[begin:end] > 0).sum())
        if bool(torch.any(segment_counts != 0)):
            segments.append(
                {
                    "begin": begin,
                    "end": end,
                    "nan": int(segment_counts[0]),
                    "nan_rows": segment_nan_rows,
                    "negative_inf": int(segment_counts[2]),
                    "negative_inf_rows": segment_negative_inf_rows,
                    "positive_inf": int(segment_counts[1]),
                    "positive_inf_rows": segment_positive_inf_rows,
                    "segment": segment,
                }
            )
    return {
        "elements": counts.shape[0] * row_width,
        "first_nan_rows": annotate(nan_rows, include_nan_column=True),
        "first_negative_inf_rows": annotate(
            negative_inf_rows, include_nan_column=False
        ),
        "first_positive_inf_rows": annotate(
            positive_inf_rows, include_nan_column=False
        ),
        "nan": int(nan_counts.sum()),
        "nan_row_count": len(nan_rows),
        "negative_inf": int(negative_inf_counts.sum()),
        "negative_inf_row_count": len(negative_inf_rows),
        "positive_inf": int(positive_inf_counts.sum()),
        "positive_inf_row_count": len(positive_inf_rows),
        "row_count": counts.shape[0],
        "row_width": row_width,
        "segments_with_nonfinite": segments,
    }


def _install_indexer_backward_diagnostics(
    args: argparse.Namespace,
    rank: int,
) -> None:
    setting = os.environ.get(_INDEXER_BACKWARD_DIAGNOSTICS_ENV)
    if setting is None:
        return
    if setting != "1":
        raise ValueError(f"{_INDEXER_BACKWARD_DIAGNOSTICS_ENV} must be exactly 1")
    if args.mode != "diagnostic":
        raise ValueError("Indexer backward diagnostics require --mode diagnostic")

    from cudnn import DSA
    from magi_attn_extensions.DSA.kernels.triton.diagnostics import (
        dsa_nonfinite_row_counts,
    )

    original = DSA.indexer_backward_wrapper
    call_index = 0

    def diagnostic_wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
        nonlocal call_index
        if len(wrapper_args) < 6:
            raise ValueError("sparse Indexer backward diagnostic requires six inputs")
        index_q = cast(torch.Tensor, wrapper_args[0])
        index_k = cast(torch.Tensor, wrapper_args[2])
        target = cast(torch.Tensor, wrapper_args[3])
        predict = cast(torch.Tensor, wrapper_args[4])
        topk_indices = cast(torch.Tensor, wrapper_args[5])
        if index_q.ndim != 4 or index_q.shape[0] != 1:
            raise ValueError("sparse Indexer backward diagnostic expects fake-BSHD Q")
        if index_k.ndim != 3 or index_k.shape[0] != 1:
            raise ValueError("sparse Indexer backward diagnostic expects fake-BSD K")
        query_rows = index_q.shape[1]
        key_rows = index_k.shape[1]
        q_cu = [0, query_rows]
        k_cu = [0, key_rows]

        before_counts = {
            "predict": dsa_nonfinite_row_counts(
                predict.contiguous().view(query_rows, -1)
            ),
            "target": dsa_nonfinite_row_counts(
                target.contiguous().view(query_rows, -1)
            ),
        }
        torch.cuda.synchronize(index_q.device)
        before = {
            name: _summarize_nonfinite_rows(
                counts,
                tensor.numel() // query_rows,
                q_cu,
            )
            for (name, counts), tensor in zip(
                before_counts.items(),
                (predict, target),
            )
        }

        result = original(*wrapper_args, **wrapper_kwargs)
        torch.cuda.synchronize(index_q.device)
        output_tensors = {
            "d_index_k": result["d_index_k"].view(key_rows, -1),
            "d_index_q": result["d_index_q"].view(query_rows, -1),
            "d_weights": result["d_weights"].view(query_rows, -1),
            "predict_sum_grad": predict.view(query_rows, -1),
            "target_grad_signal": target.view(query_rows, -1),
        }
        output_counts = {
            name: dsa_nonfinite_row_counts(tensor)
            for name, tensor in output_tensors.items()
        }
        torch.cuda.synchronize(index_q.device)
        outputs: dict[str, object] = {}
        for name, tensor in output_tensors.items():
            row_boundaries = k_cu if name == "d_index_k" else q_cu
            outputs[name] = _summarize_nonfinite_rows(
                output_counts[name],
                tensor.numel() // tensor.shape[0],
                row_boundaries,
            )

        payload: dict[str, object] = {
            "before": before,
            "call": call_index,
            "geometry": {
                "index_k_shape": list(index_k.shape),
                "index_q_shape": list(index_q.shape),
                "topk_indices_shape": list(topk_indices.shape),
            },
            "outputs": outputs,
            "plan": args.plan,
            "rank": rank,
        }
        _atomic_json(
            args.artifact_dir
            / f"indexer_backward_diagnostic_call{call_index:03d}_rank{rank}.json",
            payload,
        )
        _record(
            args.artifact_dir,
            "indexer_backward_diagnostic",
            rank,
            call=call_index,
            d_index_k_nan=cast(dict[str, object], outputs["d_index_k"])["nan"],
            grad_signal_nan=cast(dict[str, object], outputs["target_grad_signal"])[
                "nan"
            ],
            plan=args.plan,
        )
        call_index += 1
        return result

    DSA.indexer_backward_wrapper = diagnostic_wrapper
    _record(args.artifact_dir, "indexer_backward_diagnostic_installed", rank)


def _queue_backward_pipeline_diagnostic(
    label: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    from magi_attn_extensions.DSA.kernels.triton.diagnostics import (
        dsa_nonfinite_block_stats,
    )

    observed = tensor if tensor.is_contiguous() else tensor.contiguous()
    counts, block_max_abs = dsa_nonfinite_block_stats(observed)
    count_totals = counts.sum(dim=0, dtype=torch.int64)
    max_abs = (
        block_max_abs.max()
        if block_max_abs.numel()
        else torch.zeros((), dtype=torch.float32, device=tensor.device)
    )
    _BACKWARD_PIPELINE_DIAGNOSTIC_RECORDS.append(
        _BackwardPipelineDiagnosticRecord(
            label=label,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            counts=count_totals,
            max_abs=max_abs,
        )
    )
    return tensor


def _install_backward_pipeline_diagnostics(
    args: argparse.Namespace,
    rank: int,
    layer: MagiDSALayer,
) -> None:
    setting = os.environ.get(_BACKWARD_PIPELINE_DIAGNOSTICS_ENV)
    if setting is None:
        return
    if setting != "1":
        raise ValueError(f"{_BACKWARD_PIPELINE_DIAGNOSTICS_ENV} must be exactly 1")
    if args.mode != "diagnostic" or args.step_mode != "forward-backward":
        raise ValueError(
            "backward pipeline diagnostics require diagnostic forward-backward mode"
        )
    if os.environ.get(_INDEXER_BACKWARD_DIAGNOSTICS_ENV) is not None:
        raise ValueError(
            "Indexer and asynchronous backward diagnostics cannot run together"
        )

    dist_dsa_module = importlib.import_module("magi_attn_extensions.DSA.dist")
    dsa_comm_module = importlib.import_module("magi_attn_extensions.DSA.comm")
    dsa_layer_module = importlib.import_module("magi_attn_extensions.DSA.modeling")
    dsa_compressor_module = importlib.import_module(
        "magi_attn_extensions.DSA.kernels.triton.compressor"
    )
    dsa_rope_module = importlib.import_module(
        "magi_attn_extensions.DSA.kernels.triton.rope"
    )
    original_support_gather = dist_dsa_module.gather_compressor_support
    original_finish_reverse_route = dsa_comm_module.finish_dsa_reverse_route
    original_rms_norm = dsa_layer_module._apply_fused_rms_norm
    original_compressor_reduce = dsa_compressor_module.fused_csa_compressor_reduce
    original_rope_hadamard = dsa_rope_module.fused_dsa_rope_hadamard

    def attach_gradient(label: str, tensor: torch.Tensor) -> None:
        if tensor.requires_grad:
            tensor.register_hook(
                lambda grad: _queue_backward_pipeline_diagnostic(label, grad)
            )

    def diagnostic_support_gather(
        overlap_x: torch.Tensor,
        compression: Any,
        support: int,
    ) -> torch.Tensor:
        output = original_support_gather(overlap_x, compression, support)
        if torch.is_grad_enabled():
            attach_gradient("overlap_x_grad_after_support_gather", overlap_x)
            attach_gradient("packed_support_grad_before_gather", output)
        return output

    def diagnostic_finish_reverse_route(transfer: Any) -> torch.Tensor:
        route = transfer.route
        if route.name == "COMPRESSED_KI":
            _queue_backward_pipeline_diagnostic(
                "compressed_ki_route_grad_before_reverse_exchange",
                transfer.output,
            )
        output = original_finish_reverse_route(transfer)
        if route.name == "COMPRESSED_KI":
            _queue_backward_pipeline_diagnostic(
                "compressed_ki_local_grad_after_reverse_owner_reduce",
                output,
            )
        return output

    def diagnostic_rms_norm(
        tensor: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output = original_rms_norm(tensor, weight, eps)
        if tensor.shape[-1] == 128 and torch.is_grad_enabled():
            attach_gradient("indexer_rms_norm_output_grad", output)
            attach_gradient("indexer_rms_norm_input_grad", tensor)
        return output

    def diagnostic_compressor_reduce(
        projected_kv: torch.Tensor,
        projected_gate: torch.Tensor,
        ape: torch.Tensor,
        valid_rows: torch.Tensor,
        output_dim: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = original_compressor_reduce(
            projected_kv,
            projected_gate,
            ape,
            valid_rows,
            output_dim,
            output_dtype,
        )
        if output_dim == 128 and torch.is_grad_enabled():
            attach_gradient("indexer_post_gemm_output_grad", output)
            attach_gradient("indexer_projected_kv_grad", projected_kv)
            attach_gradient("indexer_projected_gate_grad", projected_gate)
        return output

    def diagnostic_rope_hadamard(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        inverse_frequencies: torch.Tensor,
        rope_dim: int,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        output = original_rope_hadamard(
            tensor,
            positions,
            inverse_frequencies,
            rope_dim,
            output_dtype=output_dtype,
        )
        if tensor.shape[-2:] == (1, 128) and torch.is_grad_enabled():
            attach_gradient("indexer_rope_hadamard_output_grad", output)
            attach_gradient("indexer_rope_hadamard_input_grad", tensor)
        return output

    def indexer_compressor_forward_hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Indexer Compressor diagnostic requires tensor input")
        if not isinstance(output, torch.Tensor):
            raise TypeError("Indexer Compressor diagnostic requires tensor output")
        attach_gradient("indexer_compressor_output_grad", output)
        attach_gradient("indexer_compressor_packed_input_grad", inputs[0])

    setattr(dist_dsa_module, "gather_compressor_support", diagnostic_support_gather)
    setattr(
        dsa_comm_module,
        "finish_dsa_reverse_route",
        diagnostic_finish_reverse_route,
    )
    setattr(dsa_layer_module, "_apply_fused_rms_norm", diagnostic_rms_norm)
    setattr(
        dsa_compressor_module,
        "fused_csa_compressor_reduce",
        diagnostic_compressor_reduce,
    )
    setattr(dsa_rope_module, "fused_dsa_rope_hadamard", diagnostic_rope_hadamard)
    if layer.indexer is None:
        raise RuntimeError("backward pipeline diagnostics require the CSA Indexer")
    layer.indexer.compressor.register_forward_hook(indexer_compressor_forward_hook)
    for name, parameter in layer.indexer.compressor.named_parameters():
        attach_gradient(f"indexer_compressor_parameter_grad::{name}", parameter)
    _record(args.artifact_dir, "backward_pipeline_diagnostic_installed", rank)


def _drain_backward_pipeline_diagnostics(
    artifact_dir: Path,
    rank: int,
    plan: str,
    iteration: int,
) -> None:
    if not _BACKWARD_PIPELINE_DIAGNOSTIC_RECORDS:
        return
    records: list[dict[str, object]] = []
    nonfinite = 0
    for record in _BACKWARD_PIPELINE_DIAGNOSTIC_RECORDS:
        counts = record.counts.detach().cpu().tolist()
        nan_count = int(counts[0])
        positive_inf_count = int(counts[1])
        negative_inf_count = int(counts[2])
        nonfinite += nan_count + positive_inf_count + negative_inf_count
        records.append(
            {
                "dtype": record.dtype,
                "label": record.label,
                "max_abs_finite": float(record.max_abs.detach().cpu()),
                "nan": nan_count,
                "negative_inf": negative_inf_count,
                "positive_inf": positive_inf_count,
                "shape": list(record.shape),
            }
        )
    _BACKWARD_PIPELINE_DIAGNOSTIC_RECORDS.clear()
    payload: dict[str, object] = {
        "iteration": iteration,
        "plan": plan,
        "rank": rank,
        "records": records,
    }
    _atomic_json(
        artifact_dir
        / f"backward_pipeline_{plan}_iteration{iteration:02d}_rank{rank}.json",
        payload,
    )
    _record(
        artifact_dir,
        "backward_pipeline_diagnostic",
        rank,
        iteration=iteration,
        nonfinite=nonfinite,
        plan=plan,
    )


def _derived_seed(base_seed: int, tensor_name: str, rank: int) -> int:
    encoded = f"magi-dsa-v4-profile:{base_seed}:{tensor_name}:{rank}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    byte_tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(memoryview(byte_tensor.numpy())).hexdigest()


def _input_digest(
    source: _ProfileSource,
    artifact_dir: Path,
    rank: int,
) -> tuple[str, dict[str, str]]:
    tensor_digests: dict[str, str] = {}
    overall = hashlib.sha256()
    for name, tensor in (("x", source.x), ("sink", source.sink)):
        _record(artifact_dir, "input_hash_begin", rank, tensor=name)
        digest = _tensor_sha256(tensor)
        tensor_digests[name] = digest
        overall.update(name.encode("utf-8"))
        overall.update(digest.encode("ascii"))
        _record(artifact_dir, "input_hash_end", rank, tensor=name, sha256=digest)
    return overall.hexdigest(), tensor_digests


def _parameter_digest(layer: MagiDSALayer, artifact_dir: Path, rank: int) -> str:
    overall = hashlib.sha256()
    for name, parameter in layer.state_dict().items():
        _record(artifact_dir, "parameter_hash_begin", rank, parameter=name)
        digest = _tensor_sha256(parameter)
        overall.update(name.encode("utf-8"))
        overall.update(digest.encode("ascii"))
        _record(artifact_dir, "parameter_hash_end", rank, parameter=name, sha256=digest)
    return overall.hexdigest()


def _make_inputs(
    config: MagiDSAConfig,
    tokens: int,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
    *,
    requires_grad: bool = False,
) -> tuple[_ProfileSource, dict[str, int], dict[str, tuple[int, int]]]:
    if tokens <= 0 or tokens % world_size:
        raise ValueError(
            "the profile token count must be positive and divisible by world size"
        )
    local_tokens = tokens // world_size
    # The profile shards the source evenly, and the planner is told that split
    # explicitly so no rank has to gather it.
    source_counts = tuple(local_tokens for _ in range(world_size))
    shapes: dict[str, tuple[int, ...]] = {
        "x": (local_tokens, config.hidden_size),
        "sink": (config.num_query_heads,),
    }
    tensors: dict[str, torch.Tensor] = {}
    tensor_seeds: dict[str, int] = {}
    for name, shape in shapes.items():
        seed_rank = -1 if name == "sink" else rank
        tensor_seed = _derived_seed(seed, name, seed_rank)
        tensor_seeds[name] = tensor_seed
        generator = torch.Generator(device=device).manual_seed(tensor_seed)
        dtype = torch.float32 if name == "sink" else torch.bfloat16
        value = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        tensors[name] = value.contiguous().requires_grad_(requires_grad)
    tensor_seeds["dout"] = _derived_seed(seed, "dout", -1)
    source = _ProfileSource(
        x=tensors["x"],
        sink=tensors["sink"],
        packed_meta=MagiDSAPackedMeta((0, tokens), tuple(source_counts)),
    )
    identities = {
        name: (tensor.data_ptr(), tensor._version)
        for name, tensor in (("x", source.x), ("sink", source.sink))
    }
    return source, tensor_seeds, identities


def _assert_input_identity(
    source: _ProfileSource,
    identities: dict[str, tuple[int, int]],
) -> None:
    for name, tensor in (("x", source.x), ("sink", source.sink)):
        expected_pointer, expected_version = identities[name]
        if tensor.data_ptr() != expected_pointer or tensor._version != expected_version:
            raise AssertionError(
                f"profile input tensor was mutated or replaced: {name}"
            )


def _repeat_feature(source: torch.Tensor, width: int) -> torch.Tensor:
    if width <= source.shape[1]:
        return source[:, :width].clone(memory_format=torch.contiguous_format)
    repeats = (width + source.shape[1] - 1) // source.shape[1]
    return source.repeat(1, repeats)[:, :width].contiguous()


def _profile_projector(
    local_x: torch.Tensor,
    position_ids: torch.Tensor,
    config: MagiDSAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed stateless projection recipe for communication/backend profiling."""

    if position_ids.shape != (local_x.shape[0],):
        raise ValueError("profile projector received invalid local position IDs")
    with dsa_nvtx_range("model_projection::profile_qr", enabled=local_x.is_cuda):
        qr = _repeat_feature(local_x, config.q_lora_rank)
    with dsa_nvtx_range("model_projection::profile_q", enabled=local_x.is_cuda):
        q_base = _repeat_feature(local_x, config.head_dim)
        q = (
            q_base.unsqueeze(1)
            .expand(-1, config.num_query_heads, -1)
            .contiguous()
            .mul_(0.25)
        )
    with dsa_nvtx_range("model_projection::profile_kv", enabled=local_x.is_cuda):
        latent_kv = _repeat_feature(local_x, config.head_dim).mul_(0.25)
    return qr, q, latent_kv


def _make_plan_input(
    source: _ProfileSource,
    runtime: MagiDSARuntimeMgr,
    handle: Any,
    *,
    retain_input_gradients: bool,
    input_boundary: _ProfileDSAInputBoundary | None = None,
) -> MagiDSAInput:
    config = runtime.config

    def projector(
        local_x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _profile_projector(local_x, position_ids, config)

    if input_boundary is None:
        dsa_input = layout_and_project_dsa_input(
            source.x,
            source.sink,
            source.packed_meta,
            runtime,
            handle,
            projector,
        )
    else:
        dsa_input = input_boundary.value
        expected_shapes = {
            "x": (handle.device_plan.local_token_count, config.hidden_size),
            "qr": (handle.device_plan.local_token_count, config.q_lora_rank),
            "q": (
                handle.device_plan.local_token_count,
                config.num_query_heads,
                config.head_dim,
            ),
            "latent_kv": (handle.device_plan.local_token_count, config.head_dim),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = cast(torch.Tensor, getattr(dsa_input, name))
            if (
                not tensor.is_leaf
                or not tensor.requires_grad
                or tensor.shape != expected_shape
                or tensor.device != source.x.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "profile DSA input boundary requires contiguous BF16 gradient "
                    f"leaves: {name}"
                )
        if dsa_input.sink is not source.sink:
            raise ValueError("profile DSA input boundary must reuse the model sink")
        if dsa_input.packed_meta is not source.packed_meta:
            raise ValueError("profile DSA input boundary changed packed metadata")
        if (
            input_boundary.dout.shape != expected_shapes["q"]
            or input_boundary.dout.device != source.x.device
            or input_boundary.dout.dtype != torch.bfloat16
            or input_boundary.dout.requires_grad
            or not input_boundary.dout.is_contiguous()
        ):
            raise ValueError("profile dout must be a fixed contiguous BF16 tensor")
        if (
            input_boundary.dkl.shape
            or input_boundary.dkl.device != source.x.device
            or input_boundary.dkl.dtype != torch.float32
            or input_boundary.dkl.requires_grad
        ):
            raise ValueError("profile dkl must be a fixed FP32 scalar one")
    if retain_input_gradients:
        for tensor in (dsa_input.qr, dsa_input.q, dsa_input.latent_kv):
            tensor.retain_grad()
    return dsa_input


def _prepare_profile_dsa_input_boundary(
    source: _ProfileSource,
    runtime: MagiDSARuntimeMgr,
    handle: Any,
    global_dout: torch.Tensor,
) -> _ProfileDSAInputBoundary:
    """Run layout/projection once and expose fixed inputs/backward seeds."""

    layout_start = torch.cuda.Event(enable_timing=True)
    layout_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        layout_start.record()
        local_x = runtime.layout_hidden(source.x.detach(), handle)
        layout_end.record()
        qr, q, latent_kv = _profile_projector(
            local_x,
            runtime.get_position_ids(handle),
            runtime.config,
        )
        rank_plan = handle.plan.rank_plans[handle.rank]
        query_rows = torch.tensor(
            rank_plan.local_query_global_rows,
            dtype=torch.int64,
            device=global_dout.device,
        )
        dout = global_dout.index_select(0, query_rows).contiguous()
    torch.cuda.synchronize(local_x.device)
    dsa_input = MagiDSAInput(
        x=local_x.detach().requires_grad_(True),
        qr=qr.detach().requires_grad_(True),
        q=q.detach().requires_grad_(True),
        latent_kv=latent_kv.detach().requires_grad_(True),
        sink=source.sink,
        packed_meta=source.packed_meta,
    )
    return _ProfileDSAInputBoundary(
        value=dsa_input,
        dout=dout,
        dkl=torch.ones((), dtype=torch.float32, device=local_x.device),
        token_layout_forward_ms=float(layout_start.elapsed_time(layout_end)),
    )


def _cudnn_cache_inventory() -> dict[str, int]:
    attributes = {
        "indexer_backward_objects": (
            "cudnn.deepseek_sparse_attention.indexer_backward.api",
            "_cache_of_IndexerBackwardObjects",
        ),
        "indexer_forward_kernels": (
            "cudnn.deepseek_sparse_attention.indexer_forward._interface",
            "_compile_cache",
        ),
        "indexer_topk_objects": (
            "cudnn.deepseek_sparse_attention.indexer_top_k.api",
            "_cache_of_IndexerTopKObjects",
        ),
        "sparse_attn_recompute_objects": (
            "cudnn.deepseek_sparse_attention.score_recompute.api",
            "_cache_of_SparseAttnScoreRecomputeObjects",
        ),
        "sparse_indexer_recompute_objects": (
            "cudnn.deepseek_sparse_attention.score_recompute.api",
            "_cache_of_SparseIndexerScoreRecomputeObjects",
        ),
        "sparse_attention_backward_objects": (
            "cudnn.deepseek_sparse_attention.sparse_attention_backward.api",
            "_cache_of_SparseAttentionBackwardObjects",
        ),
    }
    inventory: dict[str, int] = {}
    for name, (module_name, attribute_name) in attributes.items():
        module = importlib.import_module(module_name)
        cache = getattr(module, attribute_name)
        if not isinstance(cache, dict):
            raise TypeError(
                f"cuDNN cache is not a dictionary: {module_name}.{attribute_name}"
            )
        inventory[name] = len(cache)
    return inventory


def _counter_dict(runtime: MagiDSARuntimeMgr) -> dict[str, int]:
    return {name: int(value) for name, value in asdict(runtime.counters).items()}


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before[name] for name in before}


def _prepare_runtimes(
    layer: MagiDSALayer,
    source: _ProfileSource,
    artifact_dir: Path,
    rank: int,
) -> dict[str, tuple[MagiDSARuntimeMgr, Any]]:
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]] = {}
    for plan in ("sequential", "balanced"):
        _record(artifact_dir, "prepare_begin", rank, plan=plan)
        runtime = MagiDSARuntimeMgr(
            layer.config,
            dist.group.WORLD,
        )
        handle = runtime.prepare_execution(
            source.packed_meta,
            source.x.device,
            local_token_capacity=max(
                source.packed_meta.local_token_count(rank),
                (source.packed_meta.cu_seqlens[-1] + dist.get_world_size() - 1)
                // dist.get_world_size(),
            ),
            health_check=True,
        )
        prepared[plan] = (runtime, handle)
        _record(
            artifact_dir,
            "prepare_end",
            rank,
            counters=_counter_dict(runtime),
            plan=plan,
            plan_hash=handle.plan_hash,
        )
    return prepared


def _clear_training_gradients(
    layer: MagiDSALayer,
    source: _ProfileSource,
    input_boundary: _ProfileDSAInputBoundary | None = None,
) -> None:
    layer.zero_grad(set_to_none=True)
    source.x.grad = None
    source.sink.grad = None
    if input_boundary is not None:
        dsa_input = input_boundary.value
        for tensor in (dsa_input.x, dsa_input.qr, dsa_input.q, dsa_input.latent_kv):
            tensor.grad = None


def _all_reduce_training_gradients(
    layer: MagiDSALayer,
    source: _ProfileSource,
) -> None:
    sink_gradient = source.sink.grad
    missing = [
        name for name, parameter in layer.named_parameters() if parameter.grad is None
    ]
    if sink_gradient is None:
        missing.append("sink")
    if missing:
        raise AssertionError(
            f"forward-backward profile is missing model gradients: {missing}"
        )
    assert sink_gradient is not None
    named_gradients = [("sink", source.sink, sink_gradient)]
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None
        named_gradients.append((name, parameter, parameter.grad))

    devices = {gradient.device for _, _, gradient in named_gradients}
    if len(devices) != 1:
        raise AssertionError(
            "model gradients must share one device before the FP32 main-grad reducer"
        )
    device = next(iter(devices))
    enabled = device.type == "cuda"
    device_name = str(device).replace(":", "_")
    scope = f"gradient_allreduce::bucket::0::{device_name}::fp32_main_grad"
    with dsa_nvtx_range(f"{scope}::pack", enabled=enabled):
        flat_gradient = torch.cat(
            [
                gradient.detach().reshape(-1).float()
                for _, _, gradient in named_gradients
            ]
        )
    with dsa_nvtx_range(f"{scope}::collective", enabled=enabled):
        dist.all_reduce(flat_gradient)
    with dsa_nvtx_range(f"{scope}::bind_views", enabled=enabled):
        offset = 0
        for _, owner, gradient in named_gradients:
            next_offset = offset + gradient.numel()
            reduced = flat_gradient[offset:next_offset].view_as(gradient)
            owner.grad = reduced.to(dtype=gradient.dtype)
            offset = next_offset
        if offset != flat_gradient.numel():
            raise AssertionError("FP32 main-grad bucket reconstruction is incomplete")


def _run_forward_backward_step(
    layer: MagiDSALayer,
    source: _ProfileSource,
    runtime: MagiDSARuntimeMgr,
    handle: Any,
    plan: str,
    rank: int,
    input_boundary: _ProfileDSAInputBoundary | None = None,
) -> tuple[MagiDSAForwardResult, MagiDSAInput]:
    if input_boundary is None:
        raise ValueError("forward-backward profile requires a fixed DSA input boundary")
    dsa_input = _make_plan_input(
        source,
        runtime,
        handle,
        retain_input_gradients=True,
        input_boundary=input_boundary,
    )
    with torch.cuda.nvtx.range("magi_dsa::forward"):
        with torch.cuda.nvtx.range(f"{plan}/rank_{rank}/O"):
            result = runtime.calc_dsa(layer, dsa_input, handle)
    if result.output.shape != input_boundary.dout.shape:
        raise ValueError("profile dout shape does not match the DSA output")
    if result.kl.ndim != 0 or result.kl.dtype != input_boundary.dkl.dtype:
        raise ValueError("profile dkl does not match the DSA KL scalar")
    with torch.cuda.nvtx.range("magi_dsa::backward"):
        torch.autograd.backward(
            (result.output, result.kl),
            (input_boundary.dout, input_boundary.dkl),
        )
    with torch.cuda.nvtx.range("magi_dsa::parameter_gradient_allreduce"):
        _all_reduce_training_gradients(layer, source)
    return result, dsa_input


def _prewarm(
    layer: MagiDSALayer,
    source: _ProfileSource,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    warmup: int,
    artifact_dir: Path,
    rank: int,
    step_mode: str,
    seed: int,
) -> dict[str, _ProfileDSAInputBoundary]:
    if warmup <= 0:
        raise ValueError("profile warmup count must be positive")
    input_boundaries: dict[str, _ProfileDSAInputBoundary] = {}
    if step_mode == "forward-backward":
        total_tokens = source.packed_meta.cu_seqlens[-1]
        dout_seed = _derived_seed(seed, "dout", -1)
        global_output_elements = (
            total_tokens * layer.config.num_query_heads * layer.config.head_dim
        )
        if global_output_elements <= 0:
            raise ValueError("profile dout requires a non-empty global output")
        dout_scale = 1.0 / global_output_elements
        _record(
            artifact_dir,
            "profile_global_dout_begin",
            rank,
            global_output_elements=global_output_elements,
            scale=dout_scale,
            seed=dout_seed,
            shape=(
                total_tokens,
                layer.config.num_query_heads,
                layer.config.head_dim,
            ),
        )
        with torch.no_grad():
            generator = torch.Generator(device=source.x.device).manual_seed(dout_seed)
            global_dout = torch.randn(
                (
                    total_tokens,
                    layer.config.num_query_heads,
                    layer.config.head_dim,
                ),
                dtype=torch.bfloat16,
                device=source.x.device,
                generator=generator,
            )
            global_dout.mul_(dout_scale)
        for plan in ("sequential", "balanced"):
            runtime, handle = prepared[plan]
            input_boundaries[plan] = _prepare_profile_dsa_input_boundary(
                source,
                runtime,
                handle,
                global_dout,
            )
            _record(
                artifact_dir,
                "profile_dsa_input_boundary_ready",
                rank,
                gradient_boundary="post_projection_magi_dsa_input",
                backward_seed=("precomputed_global_mean_scaled_dout_and_unit_dkl"),
                dout_scale=dout_scale,
                plan=plan,
                projection_capture="pre_capture_once",
                token_layout_capture="pre_capture_once",
            )
        del global_dout
        torch.cuda.empty_cache()
        torch.cuda.synchronize(source.x.device)
        _record(
            artifact_dir,
            "profile_global_dout_end",
            rank,
            seed=dout_seed,
        )
    for plan in ("sequential", "balanced"):
        runtime, handle = prepared[plan]
        input_boundary = input_boundaries.get(plan)
        for iteration in range(warmup):
            _record(
                artifact_dir,
                "prewarm_begin",
                rank,
                iteration=iteration,
                plan=plan,
                step_mode=step_mode,
            )
            if step_mode == "forward":
                with torch.no_grad():
                    dsa_input = _make_plan_input(
                        source,
                        runtime,
                        handle,
                        retain_input_gradients=False,
                    )
                    result = runtime.calc_dsa(layer, dsa_input, handle)
            elif step_mode == "forward-backward":
                _clear_training_gradients(layer, source, input_boundary)
                result, dsa_input = _run_forward_backward_step(
                    layer,
                    source,
                    runtime,
                    handle,
                    plan,
                    rank,
                    input_boundary,
                )
            else:
                raise ValueError(f"unsupported profile step mode: {step_mode}")
            torch.cuda.synchronize()
            _drain_backward_pipeline_diagnostics(
                artifact_dir,
                rank,
                plan,
                iteration,
            )
            del result, dsa_input
            _record(
                artifact_dir,
                "prewarm_end",
                rank,
                iteration=iteration,
                plan=plan,
                step_mode=step_mode,
            )
    if step_mode == "forward-backward":
        for input_boundary in input_boundaries.values():
            _clear_training_gradients(layer, source, input_boundary)
    return input_boundaries


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    if not bool(torch.any(finite).item()):
        return 0.0
    return float((actual_float[finite] - expected_float[finite]).abs().max().item())


def _source_order_tensor(value: torch.Tensor, handle: Any) -> torch.Tensor:
    route = handle.device_plan.token_layout_route
    if route is None:
        if (
            handle.device_plan.local_token_count
            != handle.device_plan.source_token_count
        ):
            raise RuntimeError("a changed Query layout has no TOKEN_LAYOUT route")
        return value
    if value.ndim == 1:
        vector_width = 8 if value.dtype == torch.bfloat16 else 4
        padded = torch.zeros(
            (value.shape[0], vector_width), dtype=value.dtype, device=value.device
        )
        padded[:, 0] = value
        return unlayout_dsa_query_tensor(padded, route, dist.group.WORLD)[:, 0]
    return unlayout_dsa_query_tensor(value, route, dist.group.WORLD)


def _source_order_result(
    result: MagiDSAForwardResult,
    handle: Any,
) -> MagiDSAForwardResult:
    """Canonicalize one post-capture result at the TOKEN_LAYOUT boundary."""

    global_kl = result.kl.detach().clone()
    with dsa_nvtx_range("diagnostic::global_kl", enabled=global_kl.is_cuda):
        dist.all_reduce(global_kl)
    return MagiDSAForwardResult(
        output=_source_order_tensor(result.output, handle),
        kl=global_kl,
        sparse_lse=_source_order_tensor(result.sparse_lse, handle),
        topk_ids=_source_order_tensor(result.topk_ids, handle),
        topk_length=_source_order_tensor(result.topk_length, handle),
        indexer_lse=_source_order_tensor(result.indexer_lse, handle),
    )


def _assert_unique_topk(result: MagiDSAForwardResult, label: str) -> None:
    ids = result.topk_ids
    lengths = result.topk_length
    if ids.ndim != 2 or lengths.ndim != 1 or ids.shape[0] != lengths.shape[0]:
        raise AssertionError(f"{label} has invalid Top-K tensor shapes")
    if bool(torch.any((lengths < 0) | (lengths > ids.shape[1])).item()):
        raise AssertionError(f"{label} has an invalid effective Top-K length")
    columns = torch.arange(
        ids.shape[1], dtype=torch.int32, device=ids.device
    ).unsqueeze(0)
    valid = columns < lengths.unsqueeze(1)
    if bool(torch.any(ids[valid] < 0).item()):
        raise AssertionError(f"{label} has negative IDs in the effective Top-K prefix")
    if bool(torch.any(ids[~valid] >= 0).item()):
        raise AssertionError(
            f"{label} has non-padding IDs after the effective Top-K prefix"
        )
    sentinel = torch.iinfo(torch.int32).max
    sorted_ids = torch.sort(ids.masked_fill(~valid, sentinel), dim=1).values
    adjacent_valid = columns[:, 1:] < lengths.unsqueeze(1)
    duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & adjacent_valid
    if bool(torch.any(duplicate).item()):
        raise AssertionError(f"{label} has duplicate IDs in an effective Top-K row")


def _topk_diagnostics(
    target: MagiDSAForwardResult,
    shadow: MagiDSAForwardResult,
) -> dict[str, object]:
    target_ids = target.topk_ids
    shadow_ids = shadow.topk_ids
    target_lengths = target.topk_length
    shadow_lengths = shadow.topk_length
    if (
        target_ids.shape != shadow_ids.shape
        or target_lengths.shape != shadow_lengths.shape
    ):
        return {
            "canonical_exact": False,
            "length_exact": False,
            "ordered_exact": False,
            "shape_mismatch": {
                "shadow_ids": list(shadow_ids.shape),
                "shadow_lengths": list(shadow_lengths.shape),
                "target_ids": list(target_ids.shape),
                "target_lengths": list(target_lengths.shape),
            },
        }
    columns = torch.arange(
        target_ids.shape[1], dtype=torch.int32, device=target_ids.device
    ).unsqueeze(0)
    target_valid = columns < target_lengths.unsqueeze(1)
    shadow_valid = columns < shadow_lengths.unsqueeze(1)
    sentinel = torch.iinfo(torch.int32).max
    target_sorted = torch.sort(
        target_ids.masked_fill(~target_valid, sentinel), dim=1
    ).values
    shadow_sorted = torch.sort(
        shadow_ids.masked_fill(~shadow_valid, sentinel), dim=1
    ).values
    length_row_mismatch = target_lengths != shadow_lengths
    ordered_row_mismatch = torch.any(target_ids != shadow_ids, dim=1)
    canonical_row_mismatch = length_row_mismatch | torch.any(
        target_sorted != shadow_sorted, dim=1
    )
    ordered_rows = torch.nonzero(ordered_row_mismatch, as_tuple=False).flatten()
    canonical_rows = torch.nonzero(canonical_row_mismatch, as_tuple=False).flatten()
    first: dict[str, object] | None = None
    if ordered_rows.numel():
        row = int(ordered_rows[0].item())
        target_length = int(target_lengths[row].item())
        shadow_length = int(shadow_lengths[row].item())
        target_values = target_ids[row, :target_length].detach().cpu().tolist()
        shadow_values = shadow_ids[row, :shadow_length].detach().cpu().tolist()
        first = {
            "local_row": row,
            "shadow_ids": shadow_values,
            "shadow_length": shadow_length,
            "shadow_sorted_ids": sorted(shadow_values),
            "target_ids": target_values,
            "target_length": target_length,
            "target_sorted_ids": sorted(target_values),
        }
    return {
        "canonical_exact": not bool(torch.any(canonical_row_mismatch).item()),
        "canonical_mismatch_local_rows": canonical_rows.detach().cpu().tolist(),
        "canonical_mismatch_rows": int(canonical_row_mismatch.sum().item()),
        "first_ordered_mismatch": first,
        "length_exact": not bool(torch.any(length_row_mismatch).item()),
        "length_mismatch_rows": int(length_row_mismatch.sum().item()),
        "ordered_exact": not bool(torch.any(ordered_row_mismatch).item()),
        "ordered_mismatch_local_rows": ordered_rows.detach().cpu().tolist(),
        "ordered_mismatch_elements": int((target_ids != shadow_ids).sum().item()),
        "ordered_mismatch_rows": int(ordered_row_mismatch.sum().item()),
        "order_only_mismatch_rows": int(
            (ordered_row_mismatch & ~canonical_row_mismatch).sum().item()
        ),
        "shadow_ids_sha256": _tensor_sha256(shadow_ids),
        "target_ids_sha256": _tensor_sha256(target_ids),
    }


def _compare_results(
    target: MagiDSAForwardResult,
    shadow: MagiDSAForwardResult,
    *,
    diagnostic_path: Path | None = None,
) -> dict[str, object]:
    topk_diagnostics = _topk_diagnostics(target, shadow)
    if diagnostic_path is not None:
        _atomic_json(diagnostic_path, topk_diagnostics)
    if not bool(topk_diagnostics["length_exact"]):
        raise AssertionError("sequential and balanced effective Top-K lengths differ")
    _assert_unique_topk(target, "target")
    _assert_unique_topk(shadow, "shadow")
    if target.output.shape != shadow.output.shape:
        raise AssertionError("sequential and balanced output shapes differ")
    if target.output.shape[0] != target.topk_ids.shape[0]:
        raise AssertionError("profile output and Top-K row counts differ")
    output_finite = bool(torch.all(torch.isfinite(target.output)).item()) and bool(
        torch.all(torch.isfinite(shadow.output)).item()
    )
    if not output_finite:
        raise AssertionError("profile output contains non-finite values")
    canonical_mismatch_rows = cast(
        list[int], topk_diagnostics["canonical_mismatch_local_rows"]
    )
    output_compare_mask = torch.ones(
        target.output.shape[0], dtype=torch.bool, device=target.output.device
    )
    if canonical_mismatch_rows:
        output_compare_mask[
            torch.tensor(
                canonical_mismatch_rows,
                dtype=torch.int64,
                device=target.output.device,
            )
        ] = False
    compared_output = target.output[output_compare_mask]
    compared_shadow_output = shadow.output[output_compare_mask]
    torch.testing.assert_close(
        compared_output.float(),
        compared_shadow_output.float(),
        atol=5e-3,
        rtol=5e-3,
    )
    for name, actual, expected, atol, rtol in (
        ("sparse_lse", target.sparse_lse, shadow.sparse_lse, 5e-3, 5e-3),
        ("indexer_lse", target.indexer_lse, shadow.indexer_lse, 5e-3, 5e-3),
        ("kl", target.kl, shadow.kl, 2e-2, 2e-2),
    ):
        if not torch.equal(torch.isfinite(actual), torch.isfinite(expected)):
            raise AssertionError(f"sequential and balanced {name} finite masks differ")
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
    return {
        "canonical_topk_exact": bool(topk_diagnostics["canonical_exact"]),
        "indexer_lse_max_abs": _max_abs(target.indexer_lse, shadow.indexer_lse),
        "kl_abs": _max_abs(target.kl, shadow.kl),
        "ordered_topk_exact": bool(topk_diagnostics["ordered_exact"]),
        "output_compared_rows": int(output_compare_mask.sum().item()),
        "output_max_abs": _max_abs(target.output, shadow.output),
        "output_non_tie_max_abs": _max_abs(compared_output, compared_shadow_output),
        "output_finite": True,
        "output_tie_exempt_rows": len(canonical_mismatch_rows),
        "sparse_lse_max_abs": _max_abs(target.sparse_lse, shadow.sparse_lse),
        "topk_backend_native_valid": True,
        "topk_length_exact": True,
        "topk_unique": True,
    }


def _wait_for_file(
    path: Path, timeout_seconds: float, artifact_dir: Path, rank: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.05)
    _record(artifact_dir, "control_wait_timeout", rank, path=str(path))
    raise TimeoutError(f"timed out waiting for profile control file: {path}")


def _rank_metadata(
    plan: str,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    rank: int,
) -> dict[str, object]:
    runtime, handle = prepared[plan]
    rank_plan = handle.plan.rank_plans[rank]
    indexer_packing: dict[str, object] | None = None
    if runtime.config.ratio == 4:
        packed_rows = rank_plan.packed_indexer_k_count
        route = rank_plan.compressed_ki_route
        if route is None:
            raise AssertionError("CSA profile plan is missing COMPRESSED_KI metadata")
        unique_rows = len(route.consumer_global_rows)
        duplicate_rows = packed_rows - unique_rows
        if duplicate_rows < 0:
            raise AssertionError("CSA packed Indexer rows are smaller than unique rows")
        row_bytes = runtime.config.indexer_head_dim * 2
        indexer_packing = {
            "duplicate_indexer_k_bytes": duplicate_rows * row_bytes,
            "duplicate_indexer_k_rows": duplicate_rows,
            "indexer_k_row_bytes": row_bytes,
            "packed_indexer_k_bytes": packed_rows * row_bytes,
            "packed_indexer_k_rows": packed_rows,
            "packing_amplification": (
                packed_rows / unique_rows if unique_rows else 1.0
            ),
            "unique_indexer_k_bytes": unique_rows * row_bytes,
            "unique_indexer_k_rows": unique_rows,
        }
    return {
        "counter_snapshot": _counter_dict(runtime),
        "final_query_tokens": rank_plan.local_token_count,
        "backend_max_seqlen_k": rank_plan.indexer_backend_max_seqlen_k,
        "logical_max_seqlen_k": rank_plan.indexer_logical_max_seqlen_k,
        "max_seqlen_q": rank_plan.indexer_max_seqlen_q,
        "packed_indexer_k_rows": rank_plan.packed_indexer_k_count,
        "indexer_k_packing": indexer_packing,
        "plan": plan,
        "plan_hash": handle.plan_hash,
        "policy": "structural_balanced",
        "predicted_score_cost": rank_plan.predicted_score_cost,
        "predicted_topk_cost": rank_plan.predicted_topk_cost,
        "query_fragments": len(rank_plan.query_fragments),
        "source_tokens": rank_plan.source_token_count,
    }


def _run_smoke(
    args: argparse.Namespace,
    layer: MagiDSALayer,
    source: _ProfileSource,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    identities: dict[str, tuple[int, int]],
    control_group: dist.ProcessGroup,
    rank: int,
) -> dict[str, object]:
    dist.barrier(group=control_group)
    _record(args.artifact_dir, "smoke_execute_begin", rank)
    start = time.monotonic()
    with torch.no_grad():
        sequential_runtime, sequential_handle = prepared["sequential"]
        balanced_runtime, balanced_handle = prepared["balanced"]
        sequential_input = _make_plan_input(
            source,
            sequential_runtime,
            sequential_handle,
            retain_input_gradients=False,
        )
        sequential_local = sequential_runtime.calc_dsa(
            layer, sequential_input, sequential_handle
        )
        balanced_input = _make_plan_input(
            source,
            balanced_runtime,
            balanced_handle,
            retain_input_gradients=False,
        )
        balanced_local = balanced_runtime.calc_dsa(
            layer, balanced_input, balanced_handle
        )
        sequential = _source_order_result(sequential_local, sequential_handle)
        balanced = _source_order_result(balanced_local, balanced_handle)
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    _record(args.artifact_dir, "smoke_execute_end", rank, elapsed_seconds=elapsed)
    if elapsed >= 60.0:
        raise TimeoutError(f"post-prewarm CP8 smoke took {elapsed:.6f}s, limit is 60s")
    metrics = _compare_results(
        sequential,
        balanced,
        diagnostic_path=args.artifact_dir / f"topk_diagnostic_rank{rank}.json",
    )
    _assert_input_identity(source, identities)
    dist.barrier(group=control_group)
    return {
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "rank": rank,
        "result": "PASS",
    }


def _run_with_raw_scores(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    runtime: MagiDSARuntimeMgr,
    handle: Any,
) -> tuple[MagiDSAForwardResult, torch.Tensor]:
    from cudnn import DSA

    captured: list[torch.Tensor] = []
    original = DSA.indexer_forward_wrapper

    def capture(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
        result = original(*wrapper_args, **wrapper_kwargs)
        captured.append(result["scores"].detach().clone())
        return result

    DSA.indexer_forward_wrapper = capture
    try:
        result = runtime.calc_dsa(layer, dsa_input, handle)
    finally:
        DSA.indexer_forward_wrapper = original
    if len(captured) != 1:
        raise AssertionError(
            f"diagnostic expected one grouped score invocation, got {len(captured)}"
        )
    return result, captured[0]


def _local_score_rows(
    scores: torch.Tensor,
    handle: Any,
    requested_global_rows: set[int],
    rank: int,
    plan: str,
) -> list[dict[str, object]]:
    rank_plan = handle.plan.rank_plans[rank]
    indexer_map = handle.device_plan.indexer
    if indexer_map is None:
        raise AssertionError("CSA diagnostic is missing Indexer metadata")
    if len(rank_plan.local_query_global_rows) != scores.shape[0]:
        raise AssertionError("captured score rows do not match local Query rows")
    lengths = indexer_map.seq_lens.detach().cpu().tolist()
    block_offsets = indexer_map.q_sample_block_offsets.detach().cpu().tolist()
    rows: list[dict[str, object]] = []
    for local_row, global_row_value in enumerate(rank_plan.local_query_global_rows):
        global_row = int(global_row_value)
        if global_row not in requested_global_rows:
            continue
        length = int(lengths[local_row])
        values = scores[local_row, :length].float().detach().cpu()
        rows.append(
            {
                "block_offset": int(block_offsets[local_row]),
                "dtype": str(scores.dtype),
                "global_row": global_row,
                "length": length,
                "plan": plan,
                "score_sha256": hashlib.sha256(memoryview(values.numpy())).hexdigest(),
                "values": values.tolist(),
                "query_local_row": local_row,
                "query_rank": rank,
            }
        )
    return rows


def _owner_topk_rows(
    target: MagiDSAForwardResult,
    shadow: MagiDSAForwardResult,
    handle: Any,
    requested_global_rows: set[int],
    rank: int,
) -> list[dict[str, object]]:
    rank_plan = handle.plan.rank_plans[rank]
    rows: list[dict[str, object]] = []
    for global_row in sorted(requested_global_rows):
        if not (
            rank_plan.source_global_begin <= global_row < rank_plan.source_global_end
        ):
            continue
        local_row = global_row - rank_plan.source_global_begin
        target_length = int(target.topk_length[local_row].item())
        shadow_length = int(shadow.topk_length[local_row].item())
        rows.append(
            {
                "global_row": global_row,
                "owner_rank": rank,
                "shadow_ids": shadow.topk_ids[local_row, :shadow_length]
                .detach()
                .cpu()
                .tolist(),
                "shadow_length": shadow_length,
                "target_ids": target.topk_ids[local_row, :target_length]
                .detach()
                .cpu()
                .tolist(),
                "target_length": target_length,
            }
        )
    return rows


def _close_diagnostics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    actual_float = actual.float()
    expected_float = expected.float()
    finite_equal = torch.equal(
        torch.isfinite(actual_float), torch.isfinite(expected_float)
    )
    close = torch.isclose(actual_float, expected_float, atol=atol, rtol=rtol)
    mismatch_count = int((~close).sum().item())
    return {
        "atol": atol,
        "finite_masks_exact": finite_equal,
        "max_abs": _max_abs(actual, expected),
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / close.numel() if close.numel() else 0.0,
        "rtol": rtol,
    }


def _training_gradient_snapshot(
    layer: MagiDSALayer,
    source: _ProfileSource,
    dsa_input: MagiDSAInput,
    handle: Any,
    input_boundary: _ProfileDSAInputBoundary | None = None,
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    input_x = source.x if input_boundary is None else input_boundary.value.x
    tensors = (
        input_x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        source.sink,
    )
    for name, tensor in zip(_TENSOR_NAMES, tensors):
        if tensor.grad is None:
            raise AssertionError(
                f"forward-backward profile is missing input grad: {name}"
            )
        gradient = tensor.grad.detach()
        if name in ("qr", "q", "latent_kv") or (
            name == "x" and input_boundary is not None
        ):
            gradient = _source_order_tensor(gradient, handle)
        else:
            gradient = gradient.clone()
        snapshot[f"input::{name}"] = gradient
    for name, parameter in layer.named_parameters():
        if parameter.grad is None:
            raise AssertionError(
                f"forward-backward profile is missing parameter grad: {name}"
            )
        snapshot[f"parameter::{name}"] = parameter.grad.detach().clone()
    return snapshot


def _gradient_nonfinite_counts(value: torch.Tensor) -> dict[str, int]:
    value_float = value.float()
    finite_count = int(torch.isfinite(value_float).sum().item())
    return {
        "elements": value_float.numel(),
        "finite": finite_count,
        "nan": int(torch.isnan(value_float).sum().item()),
        "negative_inf": int(torch.isneginf(value_float).sum().item()),
        "nonfinite": value_float.numel() - finite_count,
        "positive_inf": int(torch.isposinf(value_float).sum().item()),
    }


def _training_gradient_finite_diagnostics(
    target: dict[str, torch.Tensor],
    shadow: dict[str, torch.Tensor],
) -> dict[str, object]:
    if target.keys() != shadow.keys():
        raise AssertionError("forward-backward gradient schemas differ")
    tensors: dict[str, object] = {}
    for name in sorted(target):
        target_value = target[name]
        shadow_value = shadow[name]
        if target_value.shape != shadow_value.shape:
            raise AssertionError(
                f"forward-backward gradient shape differs for {name}: "
                f"{tuple(target_value.shape)} != {tuple(shadow_value.shape)}"
            )
        target_finite = torch.isfinite(target_value.float())
        shadow_finite = torch.isfinite(shadow_value.float())
        tensors[name] = {
            "balanced_target": _gradient_nonfinite_counts(target_value),
            "finite_mask_mismatch_count": int(
                (target_finite != shadow_finite).sum().item()
            ),
            "sequential_shadow": _gradient_nonfinite_counts(shadow_value),
        }
    return {
        "labels": {
            "target": "balanced",
            "shadow": "sequential",
        },
        "tensors": tensors,
    }


def _compare_training_gradients(
    target: dict[str, torch.Tensor],
    shadow: dict[str, torch.Tensor],
    *,
    diagnostic_path: Path | None = None,
) -> dict[str, object]:
    if target.keys() != shadow.keys():
        raise AssertionError("forward-backward gradient schemas differ")
    max_abs = 0.0
    latent_kv_mismatch_ratio = 0.0
    failures: list[tuple[str, dict[str, object]]] = []
    tensor_diagnostics: dict[str, object] = {}
    for name in sorted(target):
        actual = target[name]
        expected = shadow[name]
        if actual.shape != expected.shape:
            raise AssertionError(
                f"forward-backward gradient shape differs for {name}: "
                f"{tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        actual_float = actual.float()
        expected_float = expected.float()
        actual_finite = torch.isfinite(actual_float)
        expected_finite = torch.isfinite(expected_float)
        finite_masks_exact = torch.equal(actual_finite, expected_finite)
        finite = actual_finite & expected_finite
        tensor_max_abs = 0.0
        if bool(torch.any(finite).item()):
            tensor_max_abs = float(
                (actual_float[finite] - expected_float[finite]).abs().max().item()
            )
            max_abs = max(max_abs, tensor_max_abs)
        if name == "input::latent_kv":
            atol = 1e-8
            rtol = 5e-2
            close = torch.isclose(
                actual_float,
                expected_float,
                atol=atol,
                rtol=rtol,
            )
            latent_kv_mismatch_ratio = (
                float((~close).float().mean().item()) if close.numel() else 0.0
            )
            mismatch_ratio = latent_kv_mismatch_ratio
            gate_pass = finite_masks_exact and mismatch_ratio <= 0.08
            threshold = 0.08
        else:
            atol = 2e-2
            rtol = 2e-2
            close = torch.isclose(
                actual_float,
                expected_float,
                atol=atol,
                rtol=rtol,
            )
            mismatch_ratio = (
                float((~close).float().mean().item()) if close.numel() else 0.0
            )
            gate_pass = finite_masks_exact and mismatch_ratio == 0.0
            threshold = 0.0
        diagnostic = {
            "atol": atol,
            "elements": actual.numel(),
            "finite_masks_exact": finite_masks_exact,
            "gate_pass": gate_pass,
            "max_abs": tensor_max_abs,
            "mismatch_ratio": mismatch_ratio,
            "mismatch_ratio_threshold": threshold,
            "rtol": rtol,
        }
        tensor_diagnostics[name] = diagnostic
        if not gate_pass:
            failures.append((name, diagnostic))
    if diagnostic_path is not None:
        _atomic_json(
            diagnostic_path,
            {
                "all_close": not failures,
                "max_abs": max_abs,
                "tensors": tensor_diagnostics,
            },
        )
    if failures:
        name, diagnostic = failures[0]
        if name == "input::latent_kv":
            raise AssertionError(
                "forward-backward latent_kv gradient mismatch ratio "
                f"{diagnostic['mismatch_ratio']:.9f} exceeds 0.08; "
                f"max_abs={diagnostic['max_abs']:.9g}"
            )
        raise AssertionError(
            f"forward-backward gradient differs for {name} beyond "
            f"atol=rtol=2e-2; max_abs={diagnostic['max_abs']:.9g}, "
            f"mismatch_ratio={diagnostic['mismatch_ratio']:.9g}"
        )
    return {
        "all_close": True,
        "latent_kv_mismatch_ratio": latent_kv_mismatch_ratio,
        "max_abs": max_abs,
        "tensor_count": len(target),
    }


def _merge_raw_score_diagnostics(
    requested_global_rows: list[int],
    gathered_score_rows: list[list[dict[str, object]] | None],
    gathered_topk_rows: list[list[dict[str, object]] | None],
) -> dict[str, object]:
    score_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for rank_rows in gathered_score_rows:
        if rank_rows is None:
            continue
        for record in rank_rows:
            key = (
                cast(str, record["plan"]),
                cast(int, record["global_row"]),
            )
            if key in score_by_key:
                raise AssertionError(f"duplicate diagnostic score row: {key}")
            score_by_key[key] = record
    topk_by_row: dict[int, dict[str, object]] = {}
    for rank_rows in gathered_topk_rows:
        if rank_rows is None:
            continue
        for record in rank_rows:
            global_row = cast(int, record["global_row"])
            if global_row in topk_by_row:
                raise AssertionError(
                    f"duplicate diagnostic owner Top-K row: {global_row}"
                )
            topk_by_row[global_row] = record

    summaries: list[dict[str, object]] = []
    for global_row in requested_global_rows:
        target_record = score_by_key.get(("sequential", global_row))
        shadow_record = score_by_key.get(("balanced", global_row))
        topk_record = topk_by_row.get(global_row)
        if target_record is None or shadow_record is None or topk_record is None:
            raise AssertionError(
                f"incomplete raw-score diagnostic for global row {global_row}"
            )
        target_scores = torch.tensor(
            cast(list[float], target_record["values"]), dtype=torch.float32
        )
        shadow_scores = torch.tensor(
            cast(list[float], shadow_record["values"]), dtype=torch.float32
        )
        if target_scores.shape != shadow_scores.shape:
            raise AssertionError(
                f"score vector shape differs for global row {global_row}"
            )
        finite_equal = torch.equal(
            torch.isfinite(target_scores), torch.isfinite(shadow_scores)
        )
        finite = torch.isfinite(target_scores) & torch.isfinite(shadow_scores)
        exact_mismatch = int((target_scores != shadow_scores).sum().item())
        close = torch.isclose(target_scores, shadow_scores, atol=5e-3, rtol=5e-3)
        max_abs = (
            float((target_scores[finite] - shadow_scores[finite]).abs().max().item())
            if bool(torch.any(finite).item())
            else 0.0
        )
        target_ids = cast(list[int], topk_record["target_ids"])
        shadow_ids = cast(list[int], topk_record["shadow_ids"])
        first_order_difference = next(
            (
                position
                for position, (target_id, shadow_id) in enumerate(
                    zip(target_ids, shadow_ids)
                )
                if target_id != shadow_id
            ),
            None,
        )
        target_only = sorted(set(target_ids) - set(shadow_ids))
        shadow_only = sorted(set(shadow_ids) - set(target_ids))
        candidate_ids: list[int] = []
        if first_order_difference is not None:
            candidate_ids.extend(
                (target_ids[first_order_difference], shadow_ids[first_order_difference])
            )
        candidate_ids.extend(target_only[:4])
        candidate_ids.extend(shadow_only[:4])
        candidate_ids.extend(target_ids[-2:])
        candidate_ids.extend(shadow_ids[-2:])
        target_offset = cast(int, target_record["block_offset"])
        shadow_offset = cast(int, shadow_record["block_offset"])
        candidate_scores: dict[str, dict[str, float | None]] = {}
        for candidate_id in dict.fromkeys(candidate_ids):
            target_column = candidate_id - target_offset
            shadow_column = candidate_id - shadow_offset
            candidate_scores[str(candidate_id)] = {
                "balanced": (
                    float(shadow_scores[shadow_column].item())
                    if 0 <= shadow_column < shadow_scores.numel()
                    else None
                ),
                "sequential": (
                    float(target_scores[target_column].item())
                    if 0 <= target_column < target_scores.numel()
                    else None
                ),
            }
        summaries.append(
            {
                "balanced_block_offset": shadow_offset,
                "balanced_score_sha256": shadow_record["score_sha256"],
                "candidate_scores": candidate_scores,
                "canonical_topk_exact": not target_only and not shadow_only,
                "close_mismatch_count_5e3": int((~close).sum().item()),
                "exact_score_mismatch_count": exact_mismatch,
                "finite_masks_exact": finite_equal,
                "first_order_difference": first_order_difference,
                "global_row": global_row,
                "max_abs_score_difference": max_abs,
                "score_length": target_scores.numel(),
                "sequential_block_offset": target_offset,
                "sequential_score_sha256": target_record["score_sha256"],
                "shadow_only_ids": shadow_only,
                "target_only_ids": target_only,
            }
        )
    return {
        "requested_global_rows": requested_global_rows,
        "rows": summaries,
    }


def _run_diagnostic(
    args: argparse.Namespace,
    layer: MagiDSALayer,
    source: _ProfileSource,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    identities: dict[str, tuple[int, int]],
    control_group: dist.ProcessGroup,
    rank: int,
) -> dict[str, object]:
    dist.barrier(group=control_group)
    _record(args.artifact_dir, "diagnostic_execute_begin", rank)
    start = time.monotonic()
    with torch.no_grad():
        sequential_runtime, sequential_handle = prepared["sequential"]
        balanced_runtime, balanced_handle = prepared["balanced"]
        sequential_input = _make_plan_input(
            source,
            sequential_runtime,
            sequential_handle,
            retain_input_gradients=False,
        )
        sequential_local, sequential_scores = _run_with_raw_scores(
            layer, sequential_input, sequential_runtime, sequential_handle
        )
        balanced_input = _make_plan_input(
            source,
            balanced_runtime,
            balanced_handle,
            retain_input_gradients=False,
        )
        balanced_local, balanced_scores = _run_with_raw_scores(
            layer, balanced_input, balanced_runtime, balanced_handle
        )
        sequential = _source_order_result(sequential_local, sequential_handle)
        balanced = _source_order_result(balanced_local, balanced_handle)
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    _record(args.artifact_dir, "diagnostic_execute_end", rank, elapsed_seconds=elapsed)
    if elapsed >= 60.0:
        raise TimeoutError(
            f"post-prewarm CP8 diagnostic took {elapsed:.6f}s, limit is 60s"
        )
    topk_diagnostics = _topk_diagnostics(sequential, balanced)
    _atomic_json(
        args.artifact_dir / f"topk_diagnostic_rank{rank}.json", topk_diagnostics
    )
    rank_plan = sequential_handle.plan.rank_plans[rank]
    local_requested = [
        rank_plan.source_global_begin + int(local_row)
        for local_row in cast(
            list[int], topk_diagnostics["canonical_mismatch_local_rows"]
        )
    ]
    ordered_rows = cast(list[int], topk_diagnostics["ordered_mismatch_local_rows"])
    if ordered_rows:
        local_requested.append(rank_plan.source_global_begin + int(ordered_rows[0]))
    requested_by_rank: list[list[int] | None] = [None] * dist.get_world_size(
        control_group
    )
    dist.all_gather_object(
        requested_by_rank, sorted(set(local_requested)), group=control_group
    )
    requested_global_rows = sorted(
        {
            global_row
            for rank_rows in requested_by_rank
            if rank_rows is not None
            for global_row in rank_rows
        }
    )
    requested_set = set(requested_global_rows)
    local_score_rows = _local_score_rows(
        sequential_scores, sequential_handle, requested_set, rank, "sequential"
    )
    local_score_rows.extend(
        _local_score_rows(
            balanced_scores, balanced_handle, requested_set, rank, "balanced"
        )
    )
    local_topk_rows = _owner_topk_rows(
        sequential, balanced, sequential_handle, requested_set, rank
    )
    gathered_score_rows: list[list[dict[str, object]] | None] = [
        None
    ] * dist.get_world_size(control_group)
    gathered_topk_rows: list[list[dict[str, object]] | None] = [
        None
    ] * dist.get_world_size(control_group)
    dist.all_gather_object(gathered_score_rows, local_score_rows, group=control_group)
    dist.all_gather_object(gathered_topk_rows, local_topk_rows, group=control_group)
    if rank == 0:
        merged = _merge_raw_score_diagnostics(
            requested_global_rows,
            gathered_score_rows,
            gathered_topk_rows,
        )
        _atomic_json(args.artifact_dir / "RAW_SCORE_DIAGNOSTIC.json", merged)
    _assert_input_identity(source, identities)
    canonical_mismatch_rows = cast(
        list[int], topk_diagnostics["canonical_mismatch_local_rows"]
    )
    output_compare_mask = torch.ones(
        sequential.output.shape[0],
        dtype=torch.bool,
        device=sequential.output.device,
    )
    if canonical_mismatch_rows:
        output_compare_mask[
            torch.tensor(
                canonical_mismatch_rows,
                dtype=torch.int64,
                device=sequential.output.device,
            )
        ] = False
    output_diagnostics = {
        "indexer_lse": _close_diagnostics(
            sequential.indexer_lse, balanced.indexer_lse, atol=5e-3, rtol=5e-3
        ),
        "kl": _close_diagnostics(sequential.kl, balanced.kl, atol=2e-2, rtol=2e-2),
        "output": _close_diagnostics(
            sequential.output[output_compare_mask],
            balanced.output[output_compare_mask],
            atol=5e-3,
            rtol=5e-3,
        ),
        "sparse_lse": _close_diagnostics(
            sequential.sparse_lse, balanced.sparse_lse, atol=5e-3, rtol=5e-3
        ),
    }
    _assert_unique_topk(sequential, "sequential diagnostic")
    _assert_unique_topk(balanced, "balanced diagnostic")
    mismatch_counts = [
        diagnostic["mismatch_count"] for diagnostic in output_diagnostics.values()
    ]
    if not all(isinstance(count, int) for count in mismatch_counts):
        raise TypeError("output diagnostic mismatch count is not an integer")
    numerical_pass = (
        all(
            bool(diagnostic["finite_masks_exact"]) and diagnostic["mismatch_count"] == 0
            for diagnostic in output_diagnostics.values()
        )
        and bool(torch.all(torch.isfinite(sequential.output)).item())
        and bool(torch.all(torch.isfinite(balanced.output)).item())
    )
    topk_backend_native_valid = bool(topk_diagnostics["length_exact"])
    dist.barrier(group=control_group)
    return {
        "canonical_topk_exact": bool(topk_diagnostics["canonical_exact"]),
        "elapsed_seconds": elapsed,
        "ordered_topk_exact": bool(topk_diagnostics["ordered_exact"]),
        "output_diagnostics": output_diagnostics,
        "output_tie_exempt_rows": len(canonical_mismatch_rows),
        "rank": rank,
        "result": ("PASS" if topk_backend_native_valid and numerical_pass else "FAIL"),
        "topk_backend_native_valid": topk_backend_native_valid,
    }


def _run_profile(
    args: argparse.Namespace,
    layer: MagiDSALayer,
    source: _ProfileSource,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    input_boundaries: dict[str, _ProfileDSAInputBoundary],
    identities: dict[str, tuple[int, int]],
    control_group: dist.ProcessGroup,
    rank: int,
) -> dict[str, object]:
    plan = args.plan
    shadow_plan = "balanced" if plan == "sequential" else "sequential"
    runtime, handle = prepared[plan]
    shadow_runtime, shadow_handle = prepared[shadow_plan]
    input_boundary = input_boundaries.get(plan)
    shadow_input_boundary = input_boundaries.get(shadow_plan)
    if args.step_mode == "forward-backward" and (
        input_boundary is None or shadow_input_boundary is None
    ):
        raise AssertionError("forward-backward profile is missing DSA input boundaries")
    before = _counter_dict(runtime)
    cache_before = _cudnn_cache_inventory()
    _atomic_json(
        args.artifact_dir / f"ready_rank{rank}.json",
        {
            "cache": cache_before,
            "counters": before,
            "plan": plan,
            "plan_hash": handle.plan_hash,
            "rank": rank,
            "step_mode": args.step_mode,
        },
    )
    _record(
        args.artifact_dir,
        "profile_ready",
        rank,
        plan=plan,
        step_mode=args.step_mode,
    )
    _wait_for_file(
        args.artifact_dir / "control" / "start", 900.0, args.artifact_dir, rank
    )
    dist.barrier(group=control_group)
    torch.cuda.reset_peak_memory_stats()

    outer_name = (
        "$Magi_DSA/capture_five_forward_backward_steps"
        if args.step_mode == "forward-backward"
        else "$Magi_DSA/capture_five_training_steps"
    )
    torch.cuda.nvtx.range_push(outer_name)
    last_result: MagiDSAForwardResult | None = None
    last_input: MagiDSAInput | None = None
    submitted_steps: list[int] = []
    try:
        if args.step_mode == "forward":
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()
        with grad_context:
            for step in range(args.steps):
                step_name = f"{plan}/rank_{rank}/training_step_{step}"
                torch.cuda.nvtx.range_push(step_name)
                try:
                    if args.step_mode == "forward-backward":
                        with dsa_nvtx_range("gradient_clear", enabled=source.x.is_cuda):
                            _clear_training_gradients(layer, source, input_boundary)
                        last_result, last_input = _run_forward_backward_step(
                            layer,
                            source,
                            runtime,
                            handle,
                            plan,
                            rank,
                            input_boundary,
                        )
                    else:
                        last_input = _make_plan_input(
                            source,
                            runtime,
                            handle,
                            retain_input_gradients=False,
                        )
                        torch.cuda.nvtx.range_push(f"{plan}/rank_{rank}/O")
                        try:
                            last_result = runtime.calc_dsa(layer, last_input, handle)
                        finally:
                            torch.cuda.nvtx.range_pop()
                finally:
                    torch.cuda.nvtx.range_pop()
                submitted_steps.append(step)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    for step in submitted_steps:
        _record(
            args.artifact_dir,
            "profile_step_submitted",
            rank,
            deferred_until_capture_sync=True,
            plan=plan,
            step=step,
            step_mode=args.step_mode,
        )
    if last_result is None or last_input is None:
        raise AssertionError("profile did not execute a forward step")
    after = _counter_dict(runtime)
    delta = _counter_delta(before, after)
    expected_delta = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": args.steps,
    }
    if delta != expected_delta:
        raise AssertionError(f"warm profile counter delta mismatch: {delta}")
    dist.barrier(group=control_group)
    _atomic_json(
        args.artifact_dir / f"capture_done_rank{rank}.json",
        {
            "counter_delta": delta,
            "plan": plan,
            "rank": rank,
            "step_mode": args.step_mode,
        },
    )
    _record(
        args.artifact_dir,
        "profile_capture_done",
        rank,
        plan=plan,
        step_mode=args.step_mode,
    )

    _wait_for_file(
        args.artifact_dir / "control" / "capture_stopped",
        600.0,
        args.artifact_dir,
        rank,
    )
    target_gradients: dict[str, torch.Tensor] | None = None
    if args.step_mode == "forward-backward":
        target_gradients = _training_gradient_snapshot(
            layer,
            source,
            last_input,
            handle,
            input_boundary,
        )
        _clear_training_gradients(layer, source, input_boundary)
    _record(
        args.artifact_dir,
        "shadow_begin",
        rank,
        plan=shadow_plan,
        step_mode=args.step_mode,
    )
    if args.step_mode == "forward-backward":
        _clear_training_gradients(layer, source, shadow_input_boundary)
        shadow_result_local, shadow_input = _run_forward_backward_step(
            layer,
            source,
            shadow_runtime,
            shadow_handle,
            shadow_plan,
            rank,
            shadow_input_boundary,
        )
        torch.cuda.synchronize()
        shadow_gradients = _training_gradient_snapshot(
            layer,
            source,
            shadow_input,
            shadow_handle,
            shadow_input_boundary,
        )
        assert target_gradients is not None
        gradient_finite_diagnostics = _training_gradient_finite_diagnostics(
            target_gradients,
            shadow_gradients,
        )
        gradient_finite_diagnostics.update(
            {
                "rank": rank,
                "shadow_plan": shadow_plan,
                "target_plan": plan,
            }
        )
        _atomic_json(
            args.artifact_dir / f"gradient_finite_diagnostic_rank{rank}.json",
            gradient_finite_diagnostics,
        )
        gradient_metrics = _compare_training_gradients(
            target_gradients,
            shadow_gradients,
            diagnostic_path=(
                args.artifact_dir / f"gradient_comparison_rank{rank}.json"
            ),
        )
    else:
        with torch.no_grad():
            shadow_input = _make_plan_input(
                source,
                shadow_runtime,
                shadow_handle,
                retain_input_gradients=False,
            )
            shadow_result_local = shadow_runtime.calc_dsa(
                layer, shadow_input, shadow_handle
            )
            torch.cuda.synchronize()
        gradient_metrics = None
    target_result = _source_order_result(last_result, handle)
    shadow_result = _source_order_result(shadow_result_local, shadow_handle)
    metrics = _compare_results(
        target_result,
        shadow_result,
        diagnostic_path=args.artifact_dir / f"topk_diagnostic_rank{rank}.json",
    )
    _assert_input_identity(source, identities)
    cache_after = _cudnn_cache_inventory()
    if cache_after != cache_before:
        raise AssertionError(
            f"cuDNN cache changed after capture/prewarm: before={cache_before}, after={cache_after}"
        )
    result = {
        "cache_after": cache_after,
        "cache_before": cache_before,
        "capture_counter_delta": delta,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "metrics": metrics,
        "plan": plan,
        "rank": rank,
        "result": "PASS",
        "shadow_plan": shadow_plan,
        "step_mode": args.step_mode,
        "backward_seed": (
            "precomputed_global_mean_scaled_dout_and_unit_dkl"
            if args.step_mode == "forward-backward"
            else "none"
        ),
        "dout_scale": (
            1.0 / (args.tokens * layer.config.num_query_heads * layer.config.head_dim)
            if args.step_mode == "forward-backward"
            else None
        ),
        "loss_capture": "none",
        "projection_capture": (
            "pre_capture_once" if args.step_mode == "forward-backward" else "per_step"
        ),
        "token_layout_capture": (
            "pre_capture_once" if args.step_mode == "forward-backward" else "per_step"
        ),
    }
    if gradient_metrics is not None:
        result["gradient_metrics"] = gradient_metrics
    _record(
        args.artifact_dir,
        "shadow_end",
        rank,
        plan=shadow_plan,
        step_mode=args.step_mode,
    )
    dist.barrier(group=control_group)
    return result


def main() -> None:
    args = _parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "control").mkdir(exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError("the Magi-DSA v4 CP8 profile requires exactly eight ranks")
    if args.seed != 0:
        raise ValueError("the release profile seed is frozen to zero")
    if args.mode == "profile" and (args.tokens != 131072 or args.steps != 5):
        raise ValueError(
            "the release profile is frozen to 131072 tokens and five steps"
        )
    if (
        args.mode == "profile"
        and args.step_mode == "forward-backward"
        and args.plan != "balanced"
    ):
        raise ValueError("the forward-backward profile captures only the balanced plan")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(minutes=10),
    )
    try:
        device = torch.device("cuda", local_rank)
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise RuntimeError("the Magi-DSA v4 profile requires B300 SM103")
        config = MagiDSAConfig(ratio=4)
        config.validate_release_contract()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        layer = MagiDSALayer(config).to(device)
        source, tensor_seeds, identities = _make_inputs(
            config,
            args.tokens,
            rank,
            world_size,
            args.seed,
            device,
            requires_grad=args.step_mode == "forward-backward",
        )
        _record(args.artifact_dir, "setup_tensors_ready", rank, tokens=args.tokens)
        input_sha256, input_tensor_sha256 = _input_digest(
            source, args.artifact_dir, rank
        )
        parameter_sha256 = _parameter_digest(layer, args.artifact_dir, rank)
        prepared = _prepare_runtimes(layer, source, args.artifact_dir, rank)
        _install_indexer_backward_diagnostics(args, rank)
        _install_backward_pipeline_diagnostics(args, rank, layer)
        input_boundaries = _prewarm(
            layer,
            source,
            prepared,
            args.warmup,
            args.artifact_dir,
            rank,
            args.step_mode,
            args.seed,
        )
        dist.barrier(group=control_group)
        metadata = {
            "config": asdict(config),
            "config_sha256": hashlib.sha256(
                json.dumps(
                    asdict(config), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "dtype": "torch.bfloat16",
            "dout_global_output_elements": (
                args.tokens * config.num_query_heads * config.head_dim
                if args.step_mode == "forward-backward"
                else None
            ),
            "dout_global_shape": (
                [args.tokens, config.num_query_heads, config.head_dim]
                if args.step_mode == "forward-backward"
                else None
            ),
            "dout_recipe": (
                "global torch.randn BF16 with shared seed, scale by "
                "1/global_output_elements, then index_select by plan "
                "local_query_global_rows"
                if args.step_mode == "forward-backward"
                else None
            ),
            "input_sha256": input_sha256,
            "input_tensor_sha256": input_tensor_sha256,
            "local_source_tokens": source.packed_meta.local_token_count(rank),
            "mode": args.mode,
            "backward_seed": (
                "precomputed_global_mean_scaled_dout_and_unit_dkl"
                if args.step_mode == "forward-backward"
                else "none"
            ),
            "dout_scale": (
                1.0 / (args.tokens * config.num_query_heads * config.head_dim)
                if args.step_mode == "forward-backward"
                else None
            ),
            "loss_capture": "none",
            "parameter_sha256": parameter_sha256,
            "profile_gradient_boundary": (
                "post_projection_magi_dsa_input"
                if args.step_mode == "forward-backward"
                else "source_owner_x"
            ),
            "projection_capture": (
                "pre_capture_once"
                if args.step_mode == "forward-backward"
                else "per_step"
            ),
            "projection_recipe": (
                "TOKEN_LAYOUT(x) then deterministic repeat/slice profile projector; "
                "qr=x[:q_lora_rank], q=repeat(x[:head_dim], heads)*0.25, "
                "latent_kv=x[:head_dim]*0.25"
            ),
            "plans": {
                name: _rank_metadata(name, prepared, rank)
                for name in ("sequential", "balanced")
            },
            "rank": rank,
            "seed": args.seed,
            "seed_recipe": "sha256('magi-dsa-v4-profile:{seed}:{tensor}:{rank-or--1}')[:8]",
            "step_mode": args.step_mode,
            "tensor_seeds": tensor_seeds,
            "token_layout_capture": (
                "pre_capture_once"
                if args.step_mode == "forward-backward"
                else "per_step"
            ),
            "tokens": args.tokens,
            "warmup": args.warmup,
            "world_size": world_size,
        }
        _atomic_json(args.artifact_dir / f"metadata_rank{rank}.json", metadata)
        if args.mode == "smoke":
            report = _run_smoke(
                args,
                layer,
                source,
                prepared,
                identities,
                control_group,
                rank,
            )
        elif args.mode == "diagnostic":
            report = _run_diagnostic(
                args,
                layer,
                source,
                prepared,
                identities,
                control_group,
                rank,
            )
        else:
            report = _run_profile(
                args,
                layer,
                source,
                prepared,
                input_boundaries,
                identities,
                control_group,
                rank,
            )
        _atomic_json(args.artifact_dir / f"result_rank{rank}.json", report)
        if args.mode == "diagnostic" and report["result"] != "PASS":
            raise AssertionError(
                "128K diagnostic failed backend-native Top-K structure or numerical gates"
            )
        _record(args.artifact_dir, "worker_complete", rank, mode=args.mode)
    except BaseException as error:
        failure = {
            "error": str(error),
            "error_type": type(error).__name__,
            "rank": rank,
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_json(args.artifact_dir / f"failure_rank{rank}.json", failure)
            _record(args.artifact_dir, "worker_failed", rank, error=repr(error))
        finally:
            raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
