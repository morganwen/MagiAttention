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
import copy
import importlib
import json
import os
import time
import traceback
from datetime import timedelta
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

from magi_attention.dsa_config import DsaPlanPolicy, DsaRatio, MagiDSAConfig
from magi_attention.dsa_layer import MagiDSALayer
from magi_attention.dsa_runtime_mgr import DsaExecutionHandle, MagiDSARuntimeMgr
from magi_attention.dsa_types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)
from magi_attention.functional.dsa_comm import (
    restore_dsa_bijective_tensor,
    route_dsa_tensor,
    route_dsa_tensor_no_grad,
)
from magi_attention.functional.dsa_phase import dsa_phase
from magi_attention.functional.dsa_reference import (
    dsa_reference,
    validate_canonical_topk,
)

_CP2_CU_SEQLENS = (0, 13, 32)
_CP2_LOCAL_COUNTS = (15, 17)
_CP8_CU_SEQLENS = (0, 256)
_CP8_LOCAL_COUNTS = (32,) * 8
_ALL_GRADIENT_NAMES = ("grad_x", "grad_qr", "grad_q", "grad_kv", "grad_sink")


def _internal_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _installed_wheel_metadata() -> dict[str, str] | None:
    if os.environ.get("MAGI_DSA_REQUIRE_INSTALLED_WHEEL") != "1":
        return None
    import magi_attention

    expected_revision = os.environ.get("MAGI_DSA_EXPECTED_REVISION", "")
    if len(expected_revision) != 40:
        raise RuntimeError("installed-wheel validation requires an exact revision")
    package_path = str(Path(magi_attention.__file__).resolve())
    package_version = package_metadata.version("magi-attention")
    expected_version = f"1.1.1+g{expected_revision}"
    installation_parts = set(Path(package_path).parts)
    if not installation_parts.intersection({"site-packages", "dist-packages"}):
        raise RuntimeError(
            "Magi-DSA was not imported from a Python installation directory: "
            f"{package_path}"
        )
    if package_version != expected_version:
        raise RuntimeError(
            f"installed Magi-DSA version is {package_version}, expected {expected_version}"
        )
    return {
        "package_path": package_path,
        "package_version": package_version,
        "source_revision": expected_revision,
    }


def _cp2_record(event: str, **fields: object) -> None:
    payload: dict[str, object] = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "-1")),
        "record_type": "cp2_control",
        "wall_time_ns": time.time_ns(),
    }
    payload.update(fields)
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if artifact_dir:
        rank = _internal_int(payload["rank"], "rank")
        path = Path(artifact_dir) / f"control_rank{rank}.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"MAGI_DSA_CP2 {json.dumps(payload, sort_keys=True)}", flush=True)


def _save_rank_report(report: dict[str, object]) -> None:
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if not artifact_dir:
        return
    rank = _internal_int(report["rank"], "rank")
    destination = Path(artifact_dir) / f"result_rank{rank}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _save_rank_failure(error: BaseException) -> None:
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if not artifact_dir:
        return
    rank = int(os.environ.get("RANK", "-1"))
    destination = Path(artifact_dir) / f"failure_rank{rank}.json"
    temporary = destination.with_suffix(".json.tmp")
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "rank": rank,
        "traceback": traceback.format_exc(),
    }
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception as artifact_error:
        print(
            f"failed to persist rank {rank} failure: {artifact_error}",
            flush=True,
        )


def _wait_for_rank_reports(world_size: int, timeout_seconds: float = 600.0) -> None:
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if not artifact_dir:
        return
    directory = Path(artifact_dir)
    deadline = time.monotonic() + timeout_seconds
    _cp2_record("rank_report_ready")
    while time.monotonic() < deadline:
        if (
            sum(
                (directory / f"result_rank{rank}.json").is_file()
                for rank in range(world_size)
            )
            == world_size
        ):
            _cp2_record("all_rank_reports_visible")
            return
        time.sleep(0.05)
    raise TimeoutError("not all per-rank result files became visible")


def _cudnn_cache_inventory() -> dict[str, object]:
    cache_attributes = {
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
        "sparse_attention_backward_objects": (
            "cudnn.deepseek_sparse_attention.sparse_attention_backward.api",
            "_cache_of_SparseAttentionBackwardObjects",
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
    entries: dict[str, int] = {}
    for name, (module_name, attribute_name) in cache_attributes.items():
        module = importlib.import_module(module_name)
        cache = getattr(module, attribute_name)
        if not isinstance(cache, dict):
            raise TypeError(
                f"cuDNN DSA cache {module_name}.{attribute_name} is not a dictionary"
            )
        entries[name] = len(cache)
    return {
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        "cute_dsl_cache_dir": os.environ.get("CUTE_DSL_CACHE_DIR"),
        "entries": entries,
    }


def _save_cudnn_cache_inventory(stage: str, inventory: dict[str, object]) -> None:
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if not artifact_dir:
        return
    rank = int(os.environ["RANK"])
    path = Path(artifact_dir) / f"cudnn_cache_rank{rank}.jsonl"
    record = {
        "inventory": inventory,
        "rank": rank,
        "stage": stage,
        "wall_time_ns": time.time_ns(),
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _small_config() -> MagiDSAConfig:
    return MagiDSAConfig(
        ratio=4,
        hidden_size=32,
        q_lora_rank=16,
        num_query_heads=4,
        head_dim=16,
        rope_dim=4,
        indexer_heads=4,
        indexer_head_dim=8,
        indexer_topk=4,
        window_size=4,
        indexer_atom_size=4,
    )


def _cp2_backend_config() -> MagiDSAConfig:
    """Keep fixed backend dimensions while reducing only the CP2 model trunk."""

    return MagiDSAConfig(
        ratio=4,
        hidden_size=128,
        q_lora_rank=128,
    )


def _run_route_smoke(rank: int, world_size: int) -> dict[str, object]:
    if world_size != 2:
        raise ValueError("the CP2 smoke case requires exactly two ranks")
    local_count = 0 if rank == 0 else 31
    meta = MagiDSAPackedMeta((0, 7, 31), local_count)
    runtime = MagiDSARuntimeMgr(
        _small_config(), dist.group.WORLD, policy="indexer_balanced"
    )
    handle = runtime.prepare_execution(
        meta,
        torch.device("cuda", rank),
        local_token_capacity=max(local_count, 1),
    )
    route = handle.device_plan.indexer_qw_route
    if route is None:
        raise RuntimeError("CSA smoke plan has no INDEXER_QW route")
    rank_plan = handle.plan.rank_plans[rank]
    global_ids = torch.arange(
        rank_plan.local_global_begin,
        rank_plan.local_global_end,
        dtype=torch.int32,
        device="cuda",
    )
    int_payload = global_ids.unsqueeze(1).expand(-1, 4).contiguous()
    worker_payload = route_dsa_tensor_no_grad(int_payload, route, dist.group.WORLD)
    expected_worker = route.consumer_global_rows.unsqueeze(1).expand(-1, 4)
    if not torch.equal(worker_payload, expected_worker):
        raise AssertionError("forward INDEXER_QW permutation does not match global IDs")
    restored = restore_dsa_bijective_tensor(worker_payload, route, dist.group.WORLD)
    if not torch.equal(restored, int_payload):
        raise AssertionError("INDEXER_AUX restore is not the inverse query permutation")

    differentiable = (
        global_ids.to(torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 8)
        .contiguous()
        .requires_grad_(True)
    )
    worker_float = route_dsa_tensor(differentiable, route, dist.group.WORLD)
    gradient = torch.autograd.grad(worker_float.float().sum(), differentiable)[0]
    if not torch.equal(gradient, torch.ones_like(differentiable)):
        raise AssertionError("query permutation backward is not a global bijection")
    torch.cuda.synchronize()
    counters = runtime.counters
    if counters.health_checks != 1 or counters.device_materializations != 1:
        raise AssertionError("cold route health check/materialization count mismatch")
    return {
        "consumer_rows": route.consumer_row_count,
        "local_rows": local_count,
        "object_collectives": counters.object_collective_invocations,
        "plan_hash": handle.plan_hash,
        "rank": rank,
        "solver_invocations": counters.solver_invocations,
    }


def _release_layers(
    config: MagiDSAConfig,
) -> tuple[MagiDSALayer, MagiDSALayer, MagiDSALayer]:
    torch.manual_seed(410)
    reference = MagiDSALayer(config).cuda()
    return reference, copy.deepcopy(reference), copy.deepcopy(reference)


def _owner_bounds(
    rank: int,
    world_size: int,
    local_counts: tuple[int, ...] = _CP2_LOCAL_COUNTS,
) -> tuple[int, int]:
    if world_size != len(local_counts):
        raise ValueError(
            "distributed correctness world size does not match owner counts"
        )
    begin = sum(local_counts[:rank])
    return begin, begin + local_counts[rank]


def _global_inputs(
    config: MagiDSAConfig,
    *,
    requires_grad: bool,
    total_tokens: int = _CP2_CU_SEQLENS[-1],
    seed: int = 411,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    global_x = torch.randn(
        total_tokens,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_qr = torch.randn(
        total_tokens,
        config.q_lora_rank,
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_q = (
        torch.randn(
            total_tokens,
            config.num_query_heads,
            config.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    global_kv = (
        torch.randn(
            total_tokens,
            config.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    sink = torch.randn(
        config.num_query_heads,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    tensors = (global_x, global_qr, global_q, global_kv)
    if requires_grad:
        tensors = tuple(value.requires_grad_(True) for value in tensors)
        sink = sink.requires_grad_(True)
    return (*tensors, sink)


def _owner_input(
    config: MagiDSAConfig,
    rank: int,
    world_size: int,
    *,
    requires_grad: bool,
    cu_seqlens: tuple[int, ...] = _CP2_CU_SEQLENS,
    local_counts: tuple[int, ...] = _CP2_LOCAL_COUNTS,
    seed: int = 411,
) -> tuple[MagiDSAInput, tuple[torch.Tensor, ...]]:
    begin, end = _owner_bounds(rank, world_size, local_counts)
    global_tensors = _global_inputs(
        config,
        requires_grad=False,
        total_tokens=cu_seqlens[-1],
        seed=seed,
    )
    tensors = tuple(
        value[begin:end].clone().contiguous().requires_grad_(requires_grad)
        for value in global_tensors[:-1]
    )
    sink = global_tensors[-1].clone().contiguous().requires_grad_(requires_grad)

    meta = MagiDSAPackedMeta(cu_seqlens, end - begin)
    dsa_input = MagiDSAInput(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        sink,
        meta,
    )
    return dsa_input, (*tensors, sink)


def _runtime_result(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    policy: DsaPlanPolicy,
) -> tuple[MagiDSAForwardResult, MagiDSARuntimeMgr]:
    runtime, handle = _prepare_runtime(layer, dsa_input, policy)
    return runtime.calc_dsa(layer, dsa_input, handle), runtime


def _prepare_runtime(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    policy: DsaPlanPolicy,
    *,
    health_check: bool = False,
) -> tuple[MagiDSARuntimeMgr, DsaExecutionHandle]:
    runtime = MagiDSARuntimeMgr(layer.config, dist.group.WORLD, policy=policy)
    handle = runtime.prepare_execution(
        dsa_input.packed_meta,
        dsa_input.x.device,
        local_token_capacity=dsa_input.packed_meta.local_token_count,
        health_check=health_check,
    )
    return runtime, handle


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    if not torch.any(finite):
        return 0.0
    return float((actual_float[finite] - expected_float[finite]).abs().max().item())


def _assert_forward_equal(
    actual: MagiDSAForwardResult, expected: MagiDSAForwardResult
) -> None:
    if not torch.equal(actual.topk_ids, expected.topk_ids):
        raise AssertionError("sequential and balanced natural top-k IDs differ")
    if not torch.equal(actual.topk_length, expected.topk_length):
        raise AssertionError("sequential and balanced top-k lengths differ")
    torch.testing.assert_close(
        actual.output.float(), expected.output.float(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.sparse_lse, expected.sparse_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.indexer_lse, expected.indexer_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(actual.kl, expected.kl, atol=2e-2, rtol=2e-2)


def _run_csa_natural(rank: int, world_size: int) -> dict[str, object]:
    config = MagiDSAConfig(ratio=4)
    _, sequential_layer, balanced_layer = _release_layers(config)
    dsa_input, _ = _owner_input(
        config,
        rank,
        world_size,
        requires_grad=False,
    )
    with torch.no_grad():
        sequential, sequential_runtime = _runtime_result(
            sequential_layer, dsa_input, "sequential"
        )
        balanced, balanced_runtime = _runtime_result(
            balanced_layer, dsa_input, "indexer_balanced"
        )
    torch.cuda.synchronize()
    _assert_forward_equal(balanced, sequential)
    return {
        "case": "csa-natural",
        "indexer_lse_max_abs": _max_abs(balanced.indexer_lse, sequential.indexer_lse),
        "kl_abs": _max_abs(balanced.kl, sequential.kl),
        "local_rows": dsa_input.packed_meta.local_token_count,
        "output_max_abs": _max_abs(balanced.output, sequential.output),
        "rank": rank,
        "sequential_warm_calls": sequential_runtime.counters.warm_invocations,
        "balanced_warm_calls": balanced_runtime.counters.warm_invocations,
    }


def _all_reduce_model_gradients(layer: MagiDSALayer, sink: torch.Tensor) -> None:
    missing = [
        name for name, parameter in layer.named_parameters() if parameter.grad is None
    ]
    if sink.grad is None:
        missing.append("sink")
    missing_flag = torch.tensor(
        [bool(missing)],
        dtype=torch.int32,
        device=sink.device,
    )
    with dsa_phase("collective_gradient_presence"):
        dist.all_reduce(missing_flag, op=dist.ReduceOp.MAX)
    if missing_flag.item():
        missing_by_rank: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(missing_by_rank, missing)
        raise AssertionError(f"missing model parameter gradients: {missing_by_rank}")
    assert sink.grad is not None
    with dsa_phase("gradient_allreduce"):
        dist.all_reduce(sink.grad)
        for parameter in layer.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad)


def _backward_snapshot(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    input_tensors: tuple[torch.Tensor, ...],
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    gradient_names: tuple[str, ...] = _ALL_GRADIENT_NAMES,
) -> dict[str, torch.Tensor]:
    captured_raw_scores: dict[str, torch.Tensor] = {}
    original_score = None
    if layer.indexer is not None:
        from cudnn import DSA

        original_score = DSA.indexer_forward_wrapper

        def capture_score(*args: Any, **kwargs: Any) -> Any:
            backend_result = original_score(*args, **kwargs)
            captured_raw_scores["indexer_raw_scores"] = backend_result[
                "scores"
            ].detach()
            return backend_result

        DSA.indexer_forward_wrapper = capture_score
    try:
        result = runtime.calc_dsa(layer, dsa_input, handle)
    finally:
        if original_score is not None:
            DSA.indexer_forward_wrapper = original_score
    global_output_elements = (
        handle.plan.total_tokens * result.output.shape[1] * result.output.shape[2]
    )
    loss = result.output.float().square().sum() / global_output_elements + result.kl
    loss.backward()
    torch.cuda.synchronize()
    _all_reduce_model_gradients(layer, input_tensors[-1])
    global_kl = result.kl.detach().clone()
    with dsa_phase("collective_kl_value"):
        dist.all_reduce(global_kl)
    snapshots: dict[str, torch.Tensor] = {
        "indexer_lse": result.indexer_lse.detach().clone(),
        "kl": result.kl.detach().clone(),
        "kl_global": global_kl,
        "output": result.output.detach().clone(),
        "sparse_lse": result.sparse_lse.detach().clone(),
        "topk_ids": result.topk_ids.detach().clone(),
        "topk_length": result.topk_length.detach().clone(),
    }
    if layer.indexer is not None:
        indexer_map = handle.device_plan.indexer
        query_route = handle.plan.rank_plans[dist.get_rank()].indexer_qw_route
        if indexer_map is None or query_route is None:
            raise AssertionError("CSA snapshot is missing Indexer plan metadata")
        if "indexer_raw_scores" not in captured_raw_scores:
            if indexer_map.seq_lens.numel():
                raise AssertionError("CSA snapshot did not capture Indexer raw scores")
            captured_raw_scores["indexer_raw_scores"] = torch.empty(
                (0, indexer_map.max_seqlen_k),
                dtype=torch.float32,
                device=dsa_input.x.device,
            )
        raw_scores = captured_raw_scores["indexer_raw_scores"]
        snapshots.update(
            {
                "indexer_raw_scores": raw_scores,
                "indexer_score_block_offsets": (
                    indexer_map.q_sample_block_offsets.detach().clone()
                ),
                "indexer_score_global_rows": torch.tensor(
                    query_route.consumer_global_rows,
                    dtype=torch.int64,
                    device=raw_scores.device,
                ),
                "indexer_score_lengths": indexer_map.seq_lens.detach().clone(),
            }
        )
    for name, tensor in zip(("x", "qr", "q", "kv", "sink"), input_tensors):
        gradient_name = f"grad_{name}"
        if gradient_name in gradient_names and tensor.grad is None:
            raise AssertionError(f"{name} gradient is missing")
        if gradient_name not in gradient_names and tensor.grad is not None:
            raise AssertionError(f"{name} unexpectedly received a gradient")
        if tensor.grad is not None:
            snapshots[gradient_name] = tensor.grad.detach().clone()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None
        snapshots[f"parameter::{name}"] = parameter.grad.detach().clone()
    return snapshots


def _reference_snapshot(
    layer: MagiDSALayer,
    input_tensors: tuple[torch.Tensor, ...],
    cu_seqlens: tuple[int, ...] = _CP2_CU_SEQLENS,
    gradient_names: tuple[str, ...] = _ALL_GRADIENT_NAMES,
) -> dict[str, torch.Tensor]:
    x, qr, q, latent_kv, sink = input_tensors
    result = dsa_reference(
        layer,
        x,
        qr,
        q,
        latent_kv,
        sink,
        cu_seqlens,
    )
    reference_raw_scores = None
    if layer.indexer is not None:
        reference_raw_scores = _reference_csa_index_scores(
            layer,
            input_tensors,
            cu_seqlens,
        )
    loss = result.output.float().square().mean() + result.kl
    loss.backward()
    torch.cuda.synchronize()
    snapshots: dict[str, torch.Tensor] = {
        "indexer_lse": result.indexer_lse.detach().clone(),
        "kl": result.kl.detach().clone(),
        "kl_global": result.kl.detach().clone(),
        "output": result.output.detach().clone(),
        "sparse_lse": result.sparse_lse.detach().clone(),
        "topk_ids": result.topk_ids.detach().clone(),
        "topk_length": result.topk_length.detach().clone(),
    }
    if reference_raw_scores is not None:
        snapshots["indexer_raw_scores"] = reference_raw_scores.detach().clone()
    for name, tensor in zip(("x", "qr", "q", "kv", "sink"), input_tensors):
        gradient_name = f"grad_{name}"
        if gradient_name in gradient_names and tensor.grad is None:
            raise AssertionError(f"reference {name} gradient is missing")
        if gradient_name not in gradient_names and tensor.grad is not None:
            raise AssertionError(f"reference {name} unexpectedly received a gradient")
        if tensor.grad is not None:
            snapshots[gradient_name] = tensor.grad.detach().clone()
    for name, parameter in layer.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"reference parameter gradient is missing: {name}")
        snapshots[f"parameter::{name}"] = parameter.grad.detach().clone()
    return snapshots


def _assert_bf16_reduction_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> float:
    close = torch.isclose(actual.float(), expected.float(), atol=1e-8, rtol=5e-2)
    mismatch_ratio = float((~close).float().mean().item()) if close.numel() else 0.0
    if mismatch_ratio > 0.08:
        raise AssertionError(f"{name} mismatch ratio {mismatch_ratio:.6f} exceeds 0.08")
    return mismatch_ratio


def _assert_regular_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape {tuple(actual.shape)} differs from {tuple(expected.shape)}"
        )
    actual_float = actual.float()
    expected_float = expected.float()
    close = torch.isclose(
        actual_float,
        expected_float,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )
    if bool(torch.all(close).item()):
        return
    mismatch_ratio = float((~close).float().mean().item()) if close.numel() else 0.0
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    max_abs = (
        float((actual_float[finite] - expected_float[finite]).abs().max().item())
        if bool(torch.any(finite).item())
        else float("inf")
    )
    raise AssertionError(
        f"{name}: mismatch_ratio={mismatch_ratio:.9f}, max_abs={max_abs:.9g}, "
        f"atol={atol}, rtol={rtol}"
    )


def _assert_raw_indexer_scores(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    label: str,
) -> float:
    required = {
        "indexer_raw_scores",
        "indexer_score_block_offsets",
        "indexer_score_global_rows",
        "indexer_score_lengths",
    }
    missing = required - actual.keys()
    if missing:
        raise AssertionError(f"{label}: missing raw-score fields {sorted(missing)}")
    if "indexer_raw_scores" not in expected:
        raise AssertionError(f"{label}: reference raw scores are missing")

    raw_scores = actual["indexer_raw_scores"]
    block_offsets = actual["indexer_score_block_offsets"]
    global_rows = actual["indexer_score_global_rows"]
    lengths = actual["indexer_score_lengths"]
    reference_scores = expected["indexer_raw_scores"]
    row_count = raw_scores.shape[0]
    if raw_scores.ndim != 2 or any(
        value.ndim != 1 or value.numel() != row_count
        for value in (block_offsets, global_rows, lengths)
    ):
        raise AssertionError(f"{label}: raw-score metadata has an invalid shape")

    metadata = zip(
        global_rows.detach().cpu().tolist(),
        block_offsets.detach().cpu().tolist(),
        lengths.detach().cpu().tolist(),
    )
    max_abs = 0.0
    for worker_row, (global_row_value, block_offset_value, length_value) in enumerate(
        metadata
    ):
        global_row = int(global_row_value)
        block_offset = int(block_offset_value)
        length = int(length_value)
        if (
            global_row < 0
            or global_row >= reference_scores.shape[0]
            or block_offset < 0
            or length < 0
            or length > raw_scores.shape[1]
            or block_offset + length > reference_scores.shape[1]
        ):
            raise AssertionError(f"{label}: raw-score metadata is out of bounds")
        backend_valid = raw_scores[worker_row, :length]
        reference_valid = reference_scores[
            global_row,
            block_offset : block_offset + length,
        ]
        if not torch.equal(
            torch.isfinite(backend_valid), torch.isfinite(reference_valid)
        ):
            raise AssertionError(f"{label}: raw-score finite masks differ")
        _assert_regular_close(
            backend_valid,
            reference_valid,
            atol=5e-3,
            rtol=5e-3,
            name=f"{label}: Indexer raw score row {global_row}",
        )
        backend_padding = raw_scores[worker_row, length:]
        if backend_padding.numel() and not bool(
            torch.all(torch.isneginf(backend_padding)).item()
        ):
            raise AssertionError(
                f"{label}: Indexer raw-score causal padding is not -inf"
            )
        max_abs = max(max_abs, _max_abs(backend_valid, reference_valid))
    return max_abs


_LOCAL_REFERENCE_FIELDS = frozenset(
    {
        "grad_kv",
        "grad_q",
        "grad_qr",
        "grad_x",
        "indexer_lse",
        "output",
        "sparse_lse",
        "topk_ids",
        "topk_length",
    }
)


def _expected_tensor(
    snapshots: dict[str, torch.Tensor],
    name: str,
    reference_bounds: tuple[int, int] | None,
) -> torch.Tensor:
    value = snapshots[name]
    if reference_bounds is not None and name in _LOCAL_REFERENCE_FIELDS:
        begin, end = reference_bounds
        return value[begin:end]
    return value


def _compare_natural_snapshots(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    label: str,
    reference_bounds: tuple[int, int] | None = None,
    gradient_names: tuple[str, ...] = _ALL_GRADIENT_NAMES,
    compare_parameter_values: bool = True,
    topk_comparison: Literal["ordered_exact", "canonical_reference"] = "ordered_exact",
) -> tuple[dict[str, float], list[str]]:
    expected_topk_ids = _expected_tensor(
        expected,
        "topk_ids",
        reference_bounds,
    )
    expected_topk_length = _expected_tensor(
        expected,
        "topk_length",
        reference_bounds,
    )
    if topk_comparison == "ordered_exact":
        if not torch.equal(actual["topk_ids"], expected_topk_ids):
            raise AssertionError(f"{label}: ordered natural Top-K IDs differ")
        if not torch.equal(actual["topk_length"], expected_topk_length):
            raise AssertionError(f"{label}: natural Top-K lengths differ")
    elif topk_comparison == "canonical_reference":
        validate_canonical_topk(
            actual["topk_ids"],
            actual["topk_length"],
            expected_topk_ids,
            expected_topk_length,
            label=label,
        )
    else:
        raise ValueError(f"unsupported Top-K comparison mode: {topk_comparison}")
    raw_score_max_abs = 0.0
    if topk_comparison == "canonical_reference" and (
        "indexer_raw_scores" in actual or "indexer_raw_scores" in expected
    ):
        raw_score_max_abs = _assert_raw_indexer_scores(
            actual,
            expected,
            label=label,
        )
    for name in ("output", "sparse_lse", "indexer_lse"):
        expected_value = _expected_tensor(expected, name, reference_bounds)
        _assert_regular_close(
            actual[name],
            expected_value,
            atol=5e-3,
            rtol=5e-3,
            name=f"{label}: {name}",
        )
    _assert_regular_close(
        actual["kl_global"],
        expected["kl_global"],
        atol=2e-2,
        rtol=2e-2,
        name=f"{label}: KL",
    )
    if reference_bounds is None:
        _assert_regular_close(
            actual["kl"],
            expected["kl"],
            atol=2e-2,
            rtol=2e-2,
            name=f"{label}: local KL",
        )
    for name in gradient_names:
        if name == "grad_kv":
            continue
        expected_value = _expected_tensor(expected, name, reference_bounds)
        _assert_regular_close(
            actual[name],
            expected_value,
            atol=2e-2,
            rtol=2e-2,
            name=f"{label}: {name}",
        )
    kv_mismatch = 0.0
    if "grad_kv" in gradient_names:
        kv_mismatch = _assert_bf16_reduction_close(
            actual["grad_kv"],
            _expected_tensor(expected, "grad_kv", reference_bounds),
            name=f"{label}: latent_kv gradient",
        )
    parameter_names = sorted(name for name in actual if name.startswith("parameter::"))
    expected_parameter_names = sorted(
        name for name in expected if name.startswith("parameter::")
    )
    if parameter_names != expected_parameter_names:
        raise AssertionError(f"{label}: parameter gradient schemas differ")
    if compare_parameter_values:
        for name in parameter_names:
            _assert_regular_close(
                actual[name],
                expected[name],
                atol=2e-2,
                rtol=2e-2,
                name=f"{label}: {name}",
            )
    expected_output = _expected_tensor(expected, "output", reference_bounds)
    expected_indexer_lse = _expected_tensor(
        expected,
        "indexer_lse",
        reference_bounds,
    )
    return (
        {
            "indexer_lse_max_abs": _max_abs(
                actual["indexer_lse"],
                expected_indexer_lse,
            ),
            "indexer_raw_score_max_abs": raw_score_max_abs,
            "kl_abs": _max_abs(actual["kl_global"], expected["kl_global"]),
            "kv_gradient_mismatch_ratio": kv_mismatch,
            "output_max_abs": _max_abs(actual["output"], expected_output),
        },
        parameter_names,
    )


def _prewarm_natural_plan(
    layer: MagiDSALayer,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    rank: int,
    world_size: int,
    policy: DsaPlanPolicy,
) -> tuple[float, dict[str, object]]:
    warm_input, warm_tensors = _owner_input(
        layer.config,
        rank,
        world_size,
        requires_grad=True,
    )
    layer.zero_grad(set_to_none=True)
    _cp2_record("prewarm_begin", policy=policy)
    started = time.monotonic()
    with dsa_phase(f"cudnn_dsa_prewarm_{policy}"):
        snapshots = _backward_snapshot(
            layer,
            warm_input,
            warm_tensors,
            runtime,
            handle,
        )
    elapsed = time.monotonic() - started
    inventory = _cudnn_cache_inventory()
    entries = inventory["entries"]
    if not isinstance(entries, dict) or any(
        int(count) < 1 for count in entries.values()
    ):
        raise AssertionError(
            f"cuDNN DSA prewarm left an empty required cache: {inventory}"
        )
    _save_cudnn_cache_inventory(f"after_prewarm_{policy}", inventory)
    _cp2_record(
        "prewarm_end",
        elapsed_seconds=elapsed,
        inventory=inventory,
        policy=policy,
    )
    layer.zero_grad(set_to_none=True)
    for tensor in warm_tensors:
        tensor.grad = None
    del snapshots, warm_input, warm_tensors
    with dsa_phase("collective_prewarm_ready"):
        dist.barrier()
    torch.cuda.synchronize()
    return elapsed, inventory


def _run_csa_natural_backward(
    rank: int,
    world_size: int,
) -> dict[str, object]:
    config = _cp2_backend_config()
    reference_layer, sequential_layer, balanced_layer = _release_layers(config)
    reference_inputs = _global_inputs(config, requires_grad=True)
    _cp2_record("reference_begin")
    reference_started = time.monotonic()
    reference = _reference_snapshot(reference_layer, reference_inputs)
    reference_seconds = time.monotonic() - reference_started
    _cp2_record("reference_end", elapsed_seconds=reference_seconds)

    sequential_input, sequential_tensors = _owner_input(
        config,
        rank,
        world_size,
        requires_grad=True,
    )
    balanced_input, balanced_tensors = _owner_input(
        config,
        rank,
        world_size,
        requires_grad=True,
    )
    sequential_runtime, sequential_handle = _prepare_runtime(
        sequential_layer,
        sequential_input,
        "sequential",
    )
    balanced_runtime, balanced_handle = _prepare_runtime(
        balanced_layer,
        balanced_input,
        "indexer_balanced",
    )
    sequential_prewarm_seconds, _ = _prewarm_natural_plan(
        sequential_layer,
        sequential_runtime,
        sequential_handle,
        rank,
        world_size,
        "sequential",
    )
    balanced_prewarm_seconds, prewarm_inventory = _prewarm_natural_plan(
        balanced_layer,
        balanced_runtime,
        balanced_handle,
        rank,
        world_size,
        "indexer_balanced",
    )
    sequential_layer.zero_grad(set_to_none=True)
    balanced_layer.zero_grad(set_to_none=True)
    with dsa_phase("collective_actual_ready"):
        dist.barrier()
    torch.cuda.synchronize()

    _cp2_record(
        "execute_begin",
        deadline_seconds=60,
        plans=("sequential", "indexer_balanced"),
    )
    execution_started = time.monotonic()
    sequential = _backward_snapshot(
        sequential_layer,
        sequential_input,
        sequential_tensors,
        sequential_runtime,
        sequential_handle,
    )
    balanced = _backward_snapshot(
        balanced_layer,
        balanced_input,
        balanced_tensors,
        balanced_runtime,
        balanced_handle,
    )
    torch.cuda.synchronize()
    execution_seconds = time.monotonic() - execution_started
    _cp2_record(
        "execute_end",
        elapsed_seconds=execution_seconds,
        plans=("sequential", "indexer_balanced"),
    )
    if execution_seconds >= 60.0:
        raise TimeoutError(
            f"CP2 post-compile natural execution took {execution_seconds:.6f}s"
        )

    final_inventory = _cudnn_cache_inventory()
    prewarm_entries = prewarm_inventory["entries"]
    final_entries = final_inventory["entries"]
    if not isinstance(prewarm_entries, dict) or not isinstance(final_entries, dict):
        raise TypeError("cuDNN DSA cache inventory has an invalid schema")
    if final_entries != prewarm_entries:
        raise AssertionError(
            "CP2 actual execution compiled a new cuDNN DSA specialization after prewarm"
        )
    _save_cudnn_cache_inventory("after_execute", final_inventory)

    _cp2_record("verification_begin", case="csa-natural-backward")
    plan_metrics, parameter_names = _compare_natural_snapshots(
        balanced,
        sequential,
        label="balanced vs sequential",
    )
    reference_bounds = _owner_bounds(rank, world_size)
    sequential_reference_metrics, _ = _compare_natural_snapshots(
        sequential,
        reference,
        label="sequential vs CP1 reference",
        reference_bounds=reference_bounds,
        topk_comparison="canonical_reference",
    )
    balanced_reference_metrics, _ = _compare_natural_snapshots(
        balanced,
        reference,
        label="balanced vs CP1 reference",
        reference_bounds=reference_bounds,
        topk_comparison="canonical_reference",
    )
    _cp2_record("verification_end", case="csa-natural-backward")
    return {
        "balanced_prewarm_seconds": balanced_prewarm_seconds,
        "balanced_reference": balanced_reference_metrics,
        "balanced_warm_calls": balanced_runtime.counters.warm_invocations,
        "case": "csa-natural-backward",
        "execution_seconds": execution_seconds,
        "local_rows": sequential_input.packed_meta.local_token_count,
        "plan_alignment": plan_metrics,
        "parameter_gradients": len(parameter_names),
        "rank": rank,
        "reference_seconds": reference_seconds,
        "sequential_prewarm_seconds": sequential_prewarm_seconds,
        "sequential_reference": sequential_reference_metrics,
        "sequential_warm_calls": sequential_runtime.counters.warm_invocations,
    }


def _release_gradient_names(ratio: DsaRatio) -> tuple[str, ...]:
    if ratio == 0:
        return ("grad_q", "grad_kv", "grad_sink")
    if ratio == 4:
        return _ALL_GRADIENT_NAMES
    if ratio == 128:
        return ("grad_x", "grad_q", "grad_kv", "grad_sink")
    raise ValueError(f"unsupported release ratio: {ratio}")


def _release_layer_copies(
    config: MagiDSAConfig,
    *,
    seed: int,
    count: int,
) -> tuple[MagiDSALayer, ...]:
    torch.manual_seed(seed)
    reference = MagiDSALayer(config).cuda()
    return (reference, *(copy.deepcopy(reference) for _ in range(count - 1)))


@torch.no_grad()
def _reference_csa_index_scores(
    layer: MagiDSALayer,
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    """Return packed CP1 raw scores in canonical global compressed-ID space."""

    if len(cu_seqlens) < 2 or layer.indexer is None:
        raise ValueError("the raw-score reference requires a packed CSA input")
    from magi_attention.functional.dsa_reference import (
        _compress_global,
        dsa_position_ids,
    )

    x, qr = inputs[:2]
    positions = dsa_position_ids(cu_seqlens, device=x.device)
    q_indexer, weights = layer.indexer.project_queries(
        x,
        qr,
        positions,
        detach_trunk=False,
    )
    compressed_ki, block_offsets, block_counts = _compress_global(
        layer.indexer.compressor,
        x,
        cu_seqlens,
    )
    scores = torch.full(
        (cu_seqlens[-1], compressed_ki.shape[0]),
        float("-inf"),
        dtype=torch.float32,
        device=x.device,
    )
    for sample_id, (q_begin, q_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        block_begin = block_offsets[sample_id]
        block_end = block_begin + block_counts[sample_id]
        if q_end == q_begin or block_end == block_begin:
            continue
        dots = torch.einsum(
            "qhd,kd->qhk",
            q_indexer[q_begin:q_end].float(),
            compressed_ki[block_begin:block_end].float(),
        )
        scores[q_begin:q_end, block_begin:block_end] = (
            dots.relu() * weights[q_begin:q_end].float().unsqueeze(-1)
        ).sum(dim=1)
    return scores


def _capture_indexer_forward(
    layer: MagiDSALayer,
    dsa_input: MagiDSAInput,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
) -> tuple[MagiDSAForwardResult, dict[str, torch.Tensor]]:
    """Capture fixed cuDNN wrapper outputs without changing production code."""

    from cudnn import DSA

    captured: dict[str, torch.Tensor] = {}
    original_score = DSA.indexer_forward_wrapper
    original_topk = DSA.indexer_top_k_wrapper

    def capture_score(*args: Any, **kwargs: Any) -> Any:
        result = original_score(*args, **kwargs)
        captured["scores"] = result["scores"].detach()
        return result

    def capture_topk(*args: Any, **kwargs: Any) -> Any:
        result = original_topk(*args, **kwargs)
        captured["raw_topk_indices"] = result["indices"].detach()
        captured["raw_topk_values"] = result["values"].detach()
        return result

    DSA.indexer_forward_wrapper = capture_score
    DSA.indexer_top_k_wrapper = capture_topk
    try:
        result = runtime.calc_dsa(layer, dsa_input, handle)
    finally:
        DSA.indexer_forward_wrapper = original_score
        DSA.indexer_top_k_wrapper = original_topk
    missing = {"scores", "raw_topk_indices", "raw_topk_values"} - captured.keys()
    if missing:
        raise AssertionError(f"cuDNN diagnostic capture is missing {sorted(missing)}")
    return result, captured


def _save_topk_backend_capture(
    rank: int,
    captured: dict[str, torch.Tensor],
    handle: DsaExecutionHandle,
) -> None:
    artifact_dir = os.environ.get("MAGI_DSA_CP2_ARTIFACT_DIR")
    if not artifact_dir:
        return
    rank_plan = handle.plan.rank_plans[rank]
    query_route = rank_plan.indexer_qw_route
    indexer_map = handle.device_plan.indexer
    if query_route is None or indexer_map is None:
        raise RuntimeError("CP8 diagnostic plan is missing Indexer metadata")
    payload: dict[str, object] = {
        name: value.detach().cpu() for name, value in captured.items()
    }
    payload.update(
        {
            "q_sample_block_offsets": indexer_map.q_sample_block_offsets.cpu(),
            "rank": rank,
            "seq_lens": indexer_map.seq_lens.cpu(),
            "worker_global_rows": tuple(query_route.consumer_global_rows),
        }
    )
    destination = Path(artifact_dir) / f"topk_backend_rank{rank}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _topk_diagnostic_report(
    actual: MagiDSAForwardResult,
    expected: MagiDSAForwardResult,
    reference_scores: torch.Tensor,
    captured: dict[str, torch.Tensor],
    handle: DsaExecutionHandle,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    begin, end = _owner_bounds(rank, world_size, _CP8_LOCAL_COUNTS)
    expected_ids = expected.topk_ids[begin:end]
    expected_lengths = expected.topk_length[begin:end]
    actual_ids = actual.topk_ids.cpu()
    actual_lengths = actual.topk_length.cpu()
    expected_ids = expected_ids.cpu()
    expected_lengths = expected_lengths.cpu()
    validate_canonical_topk(
        actual_ids,
        actual_lengths,
        expected_ids,
        expected_lengths,
        label="CP8 diagnostic production vs pure-PyTorch reference",
    )
    reference_scores = reference_scores.cpu()
    backend_scores = captured["scores"].float().cpu()

    rank_plan = handle.plan.rank_plans[rank]
    query_route = rank_plan.indexer_qw_route
    indexer_map = handle.device_plan.indexer
    if query_route is None or indexer_map is None:
        raise RuntimeError("CP8 diagnostic plan is missing Indexer metadata")
    worker_rows = tuple(int(value) for value in query_route.consumer_global_rows)
    worker_lookup = {global_row: row for row, global_row in enumerate(worker_rows)}
    offsets = indexer_map.q_sample_block_offsets.cpu()
    backend_lengths = indexer_map.seq_lens.cpu()

    mismatch_mask = torch.any(actual_ids != expected_ids, dim=1)
    mismatch_mask |= actual_lengths != expected_lengths
    mismatch_rows = torch.nonzero(mismatch_mask, as_tuple=False).flatten().tolist()
    details: list[dict[str, object]] = []
    score_max_abs = 0.0
    all_sets_equal = True
    all_actual_match_backend_order = True
    for local_row in range(end - begin):
        global_row = begin + local_row
        worker_row = worker_lookup.get(global_row)
        if worker_row is None:
            if local_row in mismatch_rows:
                details.append(
                    {
                        "global_row": global_row,
                        "local_row": local_row,
                        "worker_row": None,
                    }
                )
            all_actual_match_backend_order = False
            continue
        backend_length = int(backend_lengths[worker_row].item())
        block_offset = int(offsets[worker_row].item())
        if backend_length:
            backend_visible = backend_scores[worker_row, :backend_length]
            reference_visible = reference_scores[
                global_row,
                block_offset : block_offset + backend_length,
            ].float()
            if not torch.equal(
                torch.isfinite(backend_visible), torch.isfinite(reference_visible)
            ):
                raise AssertionError(
                    "CP8 diagnostic Indexer raw-score finite masks differ"
                )
            _assert_regular_close(
                backend_visible,
                reference_visible,
                atol=5e-3,
                rtol=5e-3,
                name=f"CP8 diagnostic Indexer raw score row {global_row}",
            )
            score_max_abs = max(
                score_max_abs,
                float((backend_visible - reference_visible).abs().max().item()),
            )
        backend_padding = backend_scores[worker_row, backend_length:]
        if backend_padding.numel() and not bool(
            torch.all(torch.isneginf(backend_padding)).item()
        ):
            raise AssertionError(
                "CP8 diagnostic Indexer raw-score causal padding is not -inf"
            )
        if local_row not in mismatch_rows:
            continue

        actual_length = int(actual_lengths[local_row].item())
        expected_length = int(expected_lengths[local_row].item())
        actual_valid = [
            int(value) for value in actual_ids[local_row, :actual_length].tolist()
        ]
        expected_valid = [
            int(value) for value in expected_ids[local_row, :expected_length].tolist()
        ]
        sets_equal = torch.equal(
            torch.sort(torch.tensor(actual_valid, dtype=torch.int64)).values,
            torch.sort(torch.tensor(expected_valid, dtype=torch.int64)).values,
        )
        all_sets_equal &= sets_equal
        first_difference = next(
            (
                position
                for position, (actual_id, expected_id) in enumerate(
                    zip(actual_valid, expected_valid)
                )
                if actual_id != expected_id
            ),
            min(actual_length, expected_length),
        )
        backend_visible = backend_scores[worker_row, :backend_length]
        backend_order = (
            torch.argsort(
                backend_visible,
                descending=True,
                stable=True,
            )
            + block_offset
        ).to(torch.int32)
        actual_matches_backend_order = torch.equal(
            actual_ids[local_row, :actual_length],
            backend_order[:actual_length],
        )
        all_actual_match_backend_order &= actual_matches_backend_order
        detail: dict[str, object] = {
            "actual_ids": actual_valid,
            "actual_length": actual_length,
            "actual_matches_backend_score_order": actual_matches_backend_order,
            "backend_length": backend_length,
            "expected_ids": expected_valid,
            "expected_length": expected_length,
            "first_difference": first_difference,
            "global_row": global_row,
            "local_row": local_row,
            "sets_equal": sets_equal,
            "worker_row": worker_row,
        }
        if first_difference < min(actual_length, expected_length):
            actual_id = actual_valid[first_difference]
            expected_id = expected_valid[first_difference]
            actual_column = actual_id - block_offset
            expected_column = expected_id - block_offset
            detail["first_pair"] = {
                "actual_backend_score": float(
                    backend_scores[worker_row, actual_column].item()
                ),
                "actual_id": actual_id,
                "actual_reference_score": float(
                    reference_scores[global_row, actual_id].item()
                ),
                "backend_gap_actual_minus_expected": float(
                    (
                        backend_scores[worker_row, actual_column]
                        - backend_scores[worker_row, expected_column]
                    ).item()
                ),
                "expected_backend_score": float(
                    backend_scores[worker_row, expected_column].item()
                ),
                "expected_id": expected_id,
                "expected_reference_score": float(
                    reference_scores[global_row, expected_id].item()
                ),
                "reference_gap_expected_minus_actual": float(
                    (
                        reference_scores[global_row, expected_id]
                        - reference_scores[global_row, actual_id]
                    ).item()
                ),
            }
        details.append(detail)

    return {
        "all_actual_match_backend_score_order": all_actual_match_backend_order,
        "all_mismatch_sets_equal": all_sets_equal,
        "backend_reference_score_max_abs": score_max_abs,
        "backend_reference_score_tolerance": {"atol": 5e-3, "rtol": 5e-3},
        "backend_reference_score_tolerance_passed": True,
        "lengths_exact": torch.equal(actual_lengths, expected_lengths),
        "mismatch_row_count": len(mismatch_rows),
        "mismatch_rows": details,
        "owner_global_bounds": [begin, end],
        "selected_all_visible": bool(
            torch.all(expected_lengths <= actual.topk_ids.shape[1]).item()
        ),
        "topk_ids_exact": torch.equal(actual_ids, expected_ids),
    }


def _run_cp8_topk_diagnostic(rank: int, world_size: int) -> dict[str, object]:
    if world_size != 8:
        raise ValueError("CP8 Top-K diagnostic requires exactly eight ranks")
    config = MagiDSAConfig(ratio=4)
    config.validate_release_contract()
    reference_layer, sequential_layer = _release_layer_copies(
        config,
        seed=440,
        count=2,
    )
    global_inputs = _global_inputs(
        config,
        requires_grad=False,
        total_tokens=_CP8_CU_SEQLENS[-1],
        seed=450,
    )
    with torch.no_grad():
        x, qr, q, latent_kv, sink = global_inputs
        expected = dsa_reference(
            reference_layer,
            x,
            qr,
            q,
            latent_kv,
            sink,
            _CP8_CU_SEQLENS,
        )
        reference_scores = _reference_csa_index_scores(
            reference_layer,
            global_inputs,
            _CP8_CU_SEQLENS,
        )

    dsa_input, _ = _owner_input(
        config,
        rank,
        world_size,
        requires_grad=False,
        cu_seqlens=_CP8_CU_SEQLENS,
        local_counts=_CP8_LOCAL_COUNTS,
        seed=450,
    )
    runtime, handle = _prepare_runtime(
        sequential_layer,
        dsa_input,
        "sequential",
        health_check=True,
    )
    _cp2_record("prewarm_begin", case="cp8-topk-diagnostic")
    prewarm_started = time.monotonic()
    with torch.no_grad():
        runtime.calc_dsa(sequential_layer, dsa_input, handle)
    torch.cuda.synchronize()
    prewarm_seconds = time.monotonic() - prewarm_started
    _cp2_record(
        "prewarm_end",
        case="cp8-topk-diagnostic",
        elapsed_seconds=prewarm_seconds,
    )

    with dsa_phase("collective_actual_ready"):
        dist.barrier()
    _cp2_record(
        "execute_begin",
        case="cp8-topk-diagnostic",
        deadline_seconds=60,
    )
    execution_started = time.monotonic()
    with torch.no_grad():
        actual, captured = _capture_indexer_forward(
            sequential_layer,
            dsa_input,
            runtime,
            handle,
        )
    torch.cuda.synchronize()
    execution_seconds = time.monotonic() - execution_started
    _cp2_record(
        "execute_end",
        case="cp8-topk-diagnostic",
        elapsed_seconds=execution_seconds,
    )

    _cp2_record("verification_begin", case="cp8-topk-diagnostic")
    diagnostics = _topk_diagnostic_report(
        actual,
        expected,
        reference_scores,
        captured,
        handle,
        rank,
        world_size,
    )
    _save_topk_backend_capture(rank, captured, handle)
    _cp2_record("verification_end", case="cp8-topk-diagnostic")
    return {
        "case": "cp8-topk-diagnostic",
        "diagnostics": diagnostics,
        "execution_seconds": execution_seconds,
        "prewarm_seconds": prewarm_seconds,
        "rank": rank,
        "warm_calls": runtime.counters.warm_invocations,
    }


def _prewarm_cp8_candidate(
    label: str,
    layer: MagiDSALayer,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    rank: int,
    world_size: int,
    *,
    seed: int,
    gradient_names: tuple[str, ...],
) -> float:
    warm_input, warm_tensors = _owner_input(
        layer.config,
        rank,
        world_size,
        requires_grad=True,
        cu_seqlens=_CP8_CU_SEQLENS,
        local_counts=_CP8_LOCAL_COUNTS,
        seed=seed,
    )
    layer.zero_grad(set_to_none=True)
    _cp2_record("prewarm_begin", case="cp8-natural-backward", path=label)
    started = time.monotonic()
    with dsa_phase(f"cudnn_dsa_prewarm_{label}"):
        snapshots = _backward_snapshot(
            layer,
            warm_input,
            warm_tensors,
            runtime,
            handle,
            gradient_names,
        )
    elapsed = time.monotonic() - started
    _cp2_record(
        "prewarm_end",
        case="cp8-natural-backward",
        elapsed_seconds=elapsed,
        path=label,
    )
    layer.zero_grad(set_to_none=True)
    for tensor in warm_tensors:
        tensor.grad = None
    del snapshots, warm_input, warm_tensors
    with dsa_phase("collective_prewarm_ready"):
        dist.barrier()
    torch.cuda.synchronize()
    return elapsed


def _run_cp8_natural_backward(rank: int, world_size: int) -> dict[str, object]:
    if world_size != 8:
        raise ValueError("CP8 correctness requires exactly eight ranks")
    configs: dict[DsaRatio, MagiDSAConfig] = {
        0: MagiDSAConfig(ratio=0),
        4: MagiDSAConfig(ratio=4),
        128: MagiDSAConfig(ratio=128),
    }
    for config in configs.values():
        config.validate_release_contract()

    (
        csa_reference_layer,
        csa_sequential_layer,
        csa_balanced_layer,
    ) = _release_layer_copies(configs[4], seed=440, count=3)
    window_reference_layer, window_layer = _release_layer_copies(
        configs[0],
        seed=441,
        count=2,
    )
    hca_reference_layer, hca_layer = _release_layer_copies(
        configs[128],
        seed=442,
        count=2,
    )

    references: dict[str, dict[str, torch.Tensor]] = {}
    reference_seconds: dict[str, float] = {}
    reference_cases: tuple[tuple[str, MagiDSALayer, DsaRatio, int], ...] = (
        ("csa", csa_reference_layer, 4, 450),
        ("window", window_reference_layer, 0, 451),
        ("hca", hca_reference_layer, 128, 452),
    )
    for label, layer, ratio, seed in reference_cases:
        reference_inputs = _global_inputs(
            configs[ratio],
            requires_grad=True,
            total_tokens=_CP8_CU_SEQLENS[-1],
            seed=seed,
        )
        _cp2_record("reference_begin", case="cp8-natural-backward", path=label)
        started = time.monotonic()
        references[label] = _reference_snapshot(
            layer,
            reference_inputs,
            _CP8_CU_SEQLENS,
            _release_gradient_names(ratio),
        )
        reference_seconds[label] = time.monotonic() - started
        _cp2_record(
            "reference_end",
            case="cp8-natural-backward",
            elapsed_seconds=reference_seconds[label],
            path=label,
        )

    candidates: dict[
        str,
        tuple[
            MagiDSALayer,
            MagiDSAInput,
            tuple[torch.Tensor, ...],
            MagiDSARuntimeMgr,
            DsaExecutionHandle,
            tuple[str, ...],
            int,
        ],
    ] = {}
    candidate_cases: tuple[
        tuple[str, MagiDSALayer, DsaRatio, DsaPlanPolicy, int], ...
    ] = (
        ("csa_sequential", csa_sequential_layer, 4, "sequential", 450),
        ("csa_balanced", csa_balanced_layer, 4, "indexer_balanced", 450),
        ("window", window_layer, 0, "sequential", 451),
        ("hca", hca_layer, 128, "sequential", 452),
    )
    for label, layer, ratio, policy, seed in candidate_cases:
        dsa_input, input_tensors = _owner_input(
            configs[ratio],
            rank,
            world_size,
            requires_grad=True,
            cu_seqlens=_CP8_CU_SEQLENS,
            local_counts=_CP8_LOCAL_COUNTS,
            seed=seed,
        )
        runtime, handle = _prepare_runtime(
            layer,
            dsa_input,
            policy,
            health_check=True,
        )
        candidates[label] = (
            layer,
            dsa_input,
            input_tensors,
            runtime,
            handle,
            _release_gradient_names(ratio),
            seed,
        )

    prewarm_seconds: dict[str, float] = {}
    for label, candidate in candidates.items():
        layer, _, _, runtime, handle, gradient_names, seed = candidate
        prewarm_seconds[label] = _prewarm_cp8_candidate(
            label,
            layer,
            runtime,
            handle,
            rank,
            world_size,
            seed=seed,
            gradient_names=gradient_names,
        )
    prewarm_inventory = _cudnn_cache_inventory()
    prewarm_entries = prewarm_inventory["entries"]
    if not isinstance(prewarm_entries, dict) or any(
        int(count) < 1 for count in prewarm_entries.values()
    ):
        raise AssertionError(
            f"CP8 prewarm left an empty cuDNN cache: {prewarm_inventory}"
        )
    _save_cudnn_cache_inventory("cp8_after_prewarm", prewarm_inventory)

    for layer, _, _, _, _, _, _ in candidates.values():
        layer.zero_grad(set_to_none=True)
    with dsa_phase("collective_actual_ready"):
        dist.barrier()
    torch.cuda.synchronize()

    _cp2_record(
        "execute_begin",
        case="cp8-natural-backward",
        deadline_seconds=60,
        paths=tuple(candidates),
    )
    execution_started = time.monotonic()
    actual: dict[str, dict[str, torch.Tensor]] = {}
    for label, candidate in candidates.items():
        layer, dsa_input, input_tensors, runtime, handle, gradient_names, _ = candidate
        actual[label] = _backward_snapshot(
            layer,
            dsa_input,
            input_tensors,
            runtime,
            handle,
            gradient_names,
        )
    torch.cuda.synchronize()
    execution_seconds = time.monotonic() - execution_started
    _cp2_record(
        "execute_end",
        case="cp8-natural-backward",
        elapsed_seconds=execution_seconds,
        paths=tuple(candidates),
    )
    if execution_seconds >= 60.0:
        raise TimeoutError(
            f"CP8 post-compile natural execution took {execution_seconds:.6f}s"
        )

    final_inventory = _cudnn_cache_inventory()
    if final_inventory["entries"] != prewarm_entries:
        raise AssertionError(
            "CP8 actual execution compiled a new cuDNN specialization after prewarm"
        )
    _save_cudnn_cache_inventory("cp8_after_execute", final_inventory)

    _cp2_record("verification_begin", case="cp8-natural-backward")
    bounds = _owner_bounds(rank, world_size, _CP8_LOCAL_COUNTS)
    compare_parameter_values = rank == 0
    metrics: dict[str, dict[str, float]] = {}
    metrics["csa_plan_alignment"], csa_parameters = _compare_natural_snapshots(
        actual["csa_balanced"],
        actual["csa_sequential"],
        label="CP8 balanced vs sequential",
        compare_parameter_values=compare_parameter_values,
    )
    _cp2_record(
        "verification_checkpoint",
        case="cp8-natural-backward",
        checkpoint="csa_plan_alignment",
    )
    for label in ("csa_sequential", "csa_balanced"):
        metrics[f"{label}_reference"], _ = _compare_natural_snapshots(
            actual[label],
            references["csa"],
            label=f"CP8 {label} vs reference",
            reference_bounds=bounds,
            compare_parameter_values=compare_parameter_values,
            topk_comparison="canonical_reference",
        )
        _cp2_record(
            "verification_checkpoint",
            case="cp8-natural-backward",
            checkpoint=f"{label}_reference",
        )
    non_csa_cases: tuple[tuple[str, DsaRatio], ...] = (
        ("window", 0),
        ("hca", 128),
    )
    for label, ratio in non_csa_cases:
        metrics[f"{label}_reference"], _ = _compare_natural_snapshots(
            actual[label],
            references[label],
            label=f"CP8 {label} vs reference",
            reference_bounds=bounds,
            gradient_names=_release_gradient_names(ratio),
            compare_parameter_values=compare_parameter_values,
            topk_comparison="canonical_reference",
        )
        _cp2_record(
            "verification_checkpoint",
            case="cp8-natural-backward",
            checkpoint=f"{label}_reference",
        )

    _cp2_record("verification_end", case="cp8-natural-backward")

    return {
        "case": "cp8-natural-backward",
        "execution_seconds": execution_seconds,
        "local_rows": _CP8_LOCAL_COUNTS[rank],
        "metrics": metrics,
        "model_parameter_values_checked": compare_parameter_values,
        "parameter_gradients": len(csa_parameters),
        "prewarm_seconds": prewarm_seconds,
        "rank": rank,
        "reference_seconds": reference_seconds,
        "warm_calls": {
            label: candidate[3].counters.warm_invocations
            for label, candidate in candidates.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "smoke",
            "csa-natural",
            "csa-natural-backward",
            "cp8-topk-diagnostic",
            "cp8-natural-backward",
        ),
        required=True,
    )
    arguments = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    installed_wheel = _installed_wheel_metadata()
    torch.cuda.set_device(local_rank)
    process_group_timeout = (
        1800
        if arguments.case
        in (
            "csa-natural-backward",
            "cp8-topk-diagnostic",
            "cp8-natural-backward",
        )
        else 45
    )
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=process_group_timeout),
    )
    completed = False
    try:
        if arguments.case == "smoke":
            report = _run_route_smoke(rank, world_size)
        elif arguments.case == "csa-natural":
            report = _run_csa_natural(rank, world_size)
        elif arguments.case == "csa-natural-backward":
            report = _run_csa_natural_backward(rank, world_size)
        elif arguments.case == "cp8-topk-diagnostic":
            report = _run_cp8_topk_diagnostic(rank, world_size)
        elif arguments.case == "cp8-natural-backward":
            report = _run_cp8_natural_backward(rank, world_size)
        else:
            raise AssertionError("unreachable distributed case")
        if installed_wheel is not None:
            report["installed_wheel"] = installed_wheel
        if arguments.case == "smoke":
            hashes: list[str | None] = [None] * world_size
            dist.all_gather_object(hashes, report["plan_hash"])
            if len(set(hashes)) != 1:
                raise AssertionError("ranks received different frozen plan hashes")
        _save_rank_report(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        _wait_for_rank_reports(world_size)
        completed = True
    except BaseException as error:
        _save_rank_failure(error)
        raise
    finally:
        # A failed rank must exit immediately so torchrun can terminate its peers.
        if completed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
