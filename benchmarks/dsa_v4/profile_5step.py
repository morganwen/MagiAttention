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
import hashlib
import importlib
import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist

from magi_attention.dsa_config import DsaPlanPolicy, MagiDSAConfig
from magi_attention.dsa_layer import MagiDSALayer
from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr
from magi_attention.dsa_types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)

_POLICY_BY_PLAN: dict[str, DsaPlanPolicy] = {
    "balanced": "indexer_balanced",
    "sequential": "sequential",
}
_TENSOR_NAMES = ("x", "qr", "q", "latent_kv", "sink")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Magi-DSA v4 CP8 profile worker")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("diagnostic", "profile", "smoke"), required=True
    )
    parser.add_argument("--plan", choices=tuple(_POLICY_BY_PLAN), default="sequential")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
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


def _derived_seed(base_seed: int, tensor_name: str, rank: int) -> int:
    encoded = f"magi-dsa-v4-profile:{base_seed}:{tensor_name}:{rank}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    byte_tensor = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(memoryview(byte_tensor.numpy())).hexdigest()


def _input_digest(
    dsa_input: MagiDSAInput,
    artifact_dir: Path,
    rank: int,
) -> tuple[str, dict[str, str]]:
    tensor_digests: dict[str, str] = {}
    overall = hashlib.sha256()
    tensors = (
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
    )
    for name, tensor in zip(_TENSOR_NAMES, tensors):
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
) -> tuple[MagiDSAInput, dict[str, int], dict[str, tuple[int, int]]]:
    if tokens <= 0 or tokens % world_size:
        raise ValueError(
            "the profile token count must be positive and divisible by world size"
        )
    local_tokens = tokens // world_size
    shapes: dict[str, tuple[int, ...]] = {
        "x": (local_tokens, config.hidden_size),
        "qr": (local_tokens, config.q_lora_rank),
        "q": (local_tokens, config.num_query_heads, config.head_dim),
        "latent_kv": (local_tokens, config.head_dim),
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
        if name in ("q", "latent_kv"):
            value = value.mul_(0.25)
        tensors[name] = value.contiguous()
    dsa_input = MagiDSAInput(
        x=tensors["x"],
        qr=tensors["qr"],
        q=tensors["q"],
        latent_kv=tensors["latent_kv"],
        sink=tensors["sink"],
        packed_meta=MagiDSAPackedMeta((0, tokens), local_tokens),
    )
    identities = {
        name: (tensor.data_ptr(), tensor._version)
        for name, tensor in zip(
            _TENSOR_NAMES,
            (
                dsa_input.x,
                dsa_input.qr,
                dsa_input.q,
                dsa_input.latent_kv,
                dsa_input.sink,
            ),
        )
    }
    return dsa_input, tensor_seeds, identities


def _assert_input_identity(
    dsa_input: MagiDSAInput,
    identities: dict[str, tuple[int, int]],
) -> None:
    tensors = (
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
    )
    for name, tensor in zip(_TENSOR_NAMES, tensors):
        expected_pointer, expected_version = identities[name]
        if tensor.data_ptr() != expected_pointer or tensor._version != expected_version:
            raise AssertionError(
                f"profile input tensor was mutated or replaced: {name}"
            )


def _cudnn_cache_inventory() -> dict[str, int]:
    attributes = {
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
    dsa_input: MagiDSAInput,
    artifact_dir: Path,
    rank: int,
) -> dict[str, tuple[MagiDSARuntimeMgr, Any]]:
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]] = {}
    for plan in ("sequential", "balanced"):
        _record(artifact_dir, "prepare_begin", rank, plan=plan)
        runtime = MagiDSARuntimeMgr(
            layer.config,
            dist.group.WORLD,
            policy=_POLICY_BY_PLAN[plan],
        )
        handle = runtime.prepare_execution(
            dsa_input.packed_meta,
            dsa_input.x.device,
            local_token_capacity=dsa_input.packed_meta.local_token_count,
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


def _prewarm(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    warmup: int,
    artifact_dir: Path,
    rank: int,
) -> None:
    if warmup <= 0:
        raise ValueError("profile warmup count must be positive")
    with torch.no_grad():
        for plan in ("sequential", "balanced"):
            runtime, handle = prepared[plan]
            for iteration in range(warmup):
                _record(
                    artifact_dir,
                    "prewarm_begin",
                    rank,
                    iteration=iteration,
                    plan=plan,
                )
                result = runtime.calc_dsa(layer, dsa_input, handle)
                torch.cuda.synchronize()
                del result
                _record(
                    artifact_dir,
                    "prewarm_end",
                    rank,
                    iteration=iteration,
                    plan=plan,
                )


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    if not bool(torch.any(finite).item()):
        return 0.0
    return float((actual_float[finite] - expected_float[finite]).abs().max().item())


def _assert_unique_topk(result: MagiDSAForwardResult, label: str) -> None:
    ids = result.topk_ids
    lengths = result.topk_length
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
    if not bool(topk_diagnostics["ordered_exact"]):
        raise AssertionError("sequential and balanced ordered Top-K IDs differ")
    if not bool(topk_diagnostics["length_exact"]):
        raise AssertionError("sequential and balanced effective Top-K lengths differ")
    _assert_unique_topk(target, "target")
    _assert_unique_topk(shadow, "shadow")
    for name, actual, expected, atol, rtol in (
        ("output", target.output, shadow.output, 5e-3, 5e-3),
        ("sparse_lse", target.sparse_lse, shadow.sparse_lse, 5e-3, 5e-3),
        ("indexer_lse", target.indexer_lse, shadow.indexer_lse, 5e-3, 5e-3),
        ("kl", target.kl, shadow.kl, 2e-2, 2e-2),
    ):
        if not torch.equal(torch.isfinite(actual), torch.isfinite(expected)):
            raise AssertionError(f"sequential and balanced {name} finite masks differ")
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
    if not bool(torch.all(torch.isfinite(target.output)).item()):
        raise AssertionError("profile output contains non-finite values")
    return {
        "indexer_lse_max_abs": _max_abs(target.indexer_lse, shadow.indexer_lse),
        "kl_abs": _max_abs(target.kl, shadow.kl),
        "ordered_topk_exact": True,
        "output_max_abs": _max_abs(target.output, shadow.output),
        "output_finite": True,
        "sparse_lse_max_abs": _max_abs(target.sparse_lse, shadow.sparse_lse),
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
    return {
        "counter_snapshot": _counter_dict(runtime),
        "local_tokens": rank_plan.local_token_count,
        "max_seqlen_k": rank_plan.indexer_max_seqlen_k,
        "max_seqlen_q": rank_plan.indexer_max_seqlen_q,
        "packed_indexer_k_rows": rank_plan.packed_indexer_k_count,
        "plan": plan,
        "plan_hash": handle.plan_hash,
        "policy": handle.plan.policy,
        "predicted_score_cost": rank_plan.predicted_score_cost,
        "predicted_topk_cost": rank_plan.predicted_topk_cost,
        "worker_fragments": len(rank_plan.worker_fragments),
        "worker_tokens": rank_plan.worker_token_count,
    }


def _run_smoke(
    args: argparse.Namespace,
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
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
        sequential = sequential_runtime.calc_dsa(layer, dsa_input, sequential_handle)
        balanced = balanced_runtime.calc_dsa(layer, dsa_input, balanced_handle)
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
    _assert_input_identity(dsa_input, identities)
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
        captured.append(result["scores"].detach())
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
    query_route = rank_plan.indexer_qw_route
    indexer_map = handle.device_plan.indexer
    if query_route is None or indexer_map is None:
        raise AssertionError("CSA diagnostic is missing Indexer routing metadata")
    if len(query_route.consumer_global_rows) != scores.shape[0]:
        raise AssertionError("captured score rows do not match worker query routing")
    lengths = indexer_map.seq_lens.detach().cpu().tolist()
    block_offsets = indexer_map.q_sample_block_offsets.detach().cpu().tolist()
    rows: list[dict[str, object]] = []
    for worker_row, global_row_value in enumerate(query_route.consumer_global_rows):
        global_row = int(global_row_value)
        if global_row not in requested_global_rows:
            continue
        length = int(lengths[worker_row])
        values = scores[worker_row, :length].float().detach().cpu()
        rows.append(
            {
                "block_offset": int(block_offsets[worker_row]),
                "dtype": str(scores.dtype),
                "global_row": global_row,
                "length": length,
                "plan": plan,
                "score_sha256": hashlib.sha256(memoryview(values.numpy())).hexdigest(),
                "values": values.tolist(),
                "worker_rank": rank,
                "worker_row": worker_row,
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
            rank_plan.local_global_begin <= global_row < rank_plan.local_global_end
        ):
            continue
        local_row = global_row - rank_plan.local_global_begin
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
    dsa_input: MagiDSAInput,
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
        sequential, sequential_scores = _run_with_raw_scores(
            layer, dsa_input, sequential_runtime, sequential_handle
        )
        balanced, balanced_scores = _run_with_raw_scores(
            layer, dsa_input, balanced_runtime, balanced_handle
        )
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
        rank_plan.local_global_begin + int(local_row)
        for local_row in cast(
            list[int], topk_diagnostics["canonical_mismatch_local_rows"]
        )
    ]
    ordered_rows = cast(list[int], topk_diagnostics["ordered_mismatch_local_rows"])
    if ordered_rows:
        local_requested.append(rank_plan.local_global_begin + int(ordered_rows[0]))
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
    _assert_input_identity(dsa_input, identities)
    output_diagnostics = {
        "indexer_lse": _close_diagnostics(
            sequential.indexer_lse, balanced.indexer_lse, atol=5e-3, rtol=5e-3
        ),
        "kl": _close_diagnostics(sequential.kl, balanced.kl, atol=2e-2, rtol=2e-2),
        "output": _close_diagnostics(
            sequential.output, balanced.output, atol=5e-3, rtol=5e-3
        ),
        "sparse_lse": _close_diagnostics(
            sequential.sparse_lse, balanced.sparse_lse, atol=5e-3, rtol=5e-3
        ),
    }
    dist.barrier(group=control_group)
    return {
        "elapsed_seconds": elapsed,
        "ordered_topk_exact": bool(topk_diagnostics["ordered_exact"]),
        "output_diagnostics": output_diagnostics,
        "rank": rank,
        "result": "PASS" if bool(topk_diagnostics["ordered_exact"]) else "FAIL",
    }


def _run_profile(
    args: argparse.Namespace,
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    prepared: dict[str, tuple[MagiDSARuntimeMgr, Any]],
    identities: dict[str, tuple[int, int]],
    control_group: dist.ProcessGroup,
    rank: int,
) -> dict[str, object]:
    plan = args.plan
    shadow_plan = "balanced" if plan == "sequential" else "sequential"
    runtime, handle = prepared[plan]
    shadow_runtime, shadow_handle = prepared[shadow_plan]
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
        },
    )
    _record(args.artifact_dir, "profile_ready", rank, plan=plan)
    _wait_for_file(
        args.artifact_dir / "control" / "start", 900.0, args.artifact_dir, rank
    )
    dist.barrier(group=control_group)
    torch.cuda.reset_peak_memory_stats()

    outer_name = "$Magi_DSA/capture_five_training_steps"
    torch.cuda.nvtx.range_push(outer_name)
    last_result: MagiDSAForwardResult | None = None
    try:
        with torch.no_grad():
            for step in range(args.steps):
                step_name = f"{plan}/rank_{rank}/training_step_{step}"
                torch.cuda.nvtx.range_push(step_name)
                try:
                    torch.cuda.nvtx.range_push(f"{plan}/rank_{rank}/O")
                    try:
                        last_result = runtime.calc_dsa(layer, dsa_input, handle)
                    finally:
                        torch.cuda.nvtx.range_pop()
                finally:
                    torch.cuda.nvtx.range_pop()
                _record(
                    args.artifact_dir,
                    "profile_step_submitted",
                    rank,
                    plan=plan,
                    step=step,
                )
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if last_result is None:
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
        {"counter_delta": delta, "plan": plan, "rank": rank},
    )
    _record(args.artifact_dir, "profile_capture_done", rank, plan=plan)

    _wait_for_file(
        args.artifact_dir / "control" / "capture_stopped",
        600.0,
        args.artifact_dir,
        rank,
    )
    _record(args.artifact_dir, "shadow_begin", rank, plan=shadow_plan)
    with torch.no_grad():
        shadow_result = shadow_runtime.calc_dsa(layer, dsa_input, shadow_handle)
        torch.cuda.synchronize()
    metrics = _compare_results(
        last_result,
        shadow_result,
        diagnostic_path=args.artifact_dir / f"topk_diagnostic_rank{rank}.json",
    )
    _assert_input_identity(dsa_input, identities)
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
    }
    _record(args.artifact_dir, "shadow_end", rank, plan=shadow_plan)
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
        dsa_input, tensor_seeds, identities = _make_inputs(
            config,
            args.tokens,
            rank,
            world_size,
            args.seed,
            device,
        )
        _record(args.artifact_dir, "setup_tensors_ready", rank, tokens=args.tokens)
        input_sha256, input_tensor_sha256 = _input_digest(
            dsa_input, args.artifact_dir, rank
        )
        parameter_sha256 = _parameter_digest(layer, args.artifact_dir, rank)
        prepared = _prepare_runtimes(layer, dsa_input, args.artifact_dir, rank)
        _prewarm(layer, dsa_input, prepared, args.warmup, args.artifact_dir, rank)
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
            "input_sha256": input_sha256,
            "input_tensor_sha256": input_tensor_sha256,
            "local_tokens": dsa_input.packed_meta.local_token_count,
            "mode": args.mode,
            "parameter_sha256": parameter_sha256,
            "plans": {
                name: _rank_metadata(name, prepared, rank)
                for name in ("sequential", "balanced")
            },
            "rank": rank,
            "seed": args.seed,
            "seed_recipe": "sha256('magi-dsa-v4-profile:{seed}:{tensor}:{rank-or--1}')[:8]",
            "tensor_seeds": tensor_seeds,
            "tokens": args.tokens,
            "warmup": args.warmup,
            "world_size": world_size,
        }
        _atomic_json(args.artifact_dir / f"metadata_rank{rank}.json", metadata)
        if args.mode == "smoke":
            report = _run_smoke(
                args,
                layer,
                dsa_input,
                prepared,
                identities,
                control_group,
                rank,
            )
        elif args.mode == "diagnostic":
            report = _run_diagnostic(
                args,
                layer,
                dsa_input,
                prepared,
                identities,
                control_group,
                rank,
            )
        else:
            report = _run_profile(
                args,
                layer,
                dsa_input,
                prepared,
                identities,
                control_group,
                rank,
            )
        _atomic_json(args.artifact_dir / f"result_rank{rank}.json", report)
        if args.mode == "diagnostic" and report["result"] != "PASS":
            raise AssertionError("128K diagnostic confirmed ordered Top-K mismatch")
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
