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
import hashlib
import importlib
import json
import os
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any


def _configure_rank_local_caches() -> dict[str, str]:
    cache_root_value = os.environ.get("MAGI_DSA_RANK_CACHE_ROOT")
    if not cache_root_value:
        return {}
    rank_value = os.environ.get("RANK")
    local_rank_value = os.environ.get("LOCAL_RANK")
    if rank_value is None or local_rank_value is None:
        raise RuntimeError("rank-local cache setup requires RANK and LOCAL_RANK")
    rank = int(rank_value)
    local_rank = int(local_rank_value)
    root = Path(cache_root_value) / f"rank{rank}_local{local_rank}"
    cache_paths = {
        "QUACK_CACHE_DIR": root / "quack",
        "TEMP": root / "tmp",
        "TMP": root / "tmp",
        "TMPDIR": root / "tmp",
        "TORCH_EXTENSIONS_DIR": root / "torch-extensions",
        "TORCH_HOME": root / "torch-home",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "TRITON_CACHE_DIR": root / "triton",
        "XDG_CACHE_HOME": root / "xdg",
    }
    for directory in set(cache_paths.values()):
        directory.mkdir(parents=True, exist_ok=True)
    for variable, directory in cache_paths.items():
        os.environ[variable] = str(directory)
    # The worker configures TMPDIR before importing compiler stacks. Resetting
    # this module cache also protects against an earlier stdlib temp lookup.
    tempfile.tempdir = None
    return {variable: str(path) for variable, path in cache_paths.items()}


_RANK_LOCAL_CACHE_PATHS = _configure_rank_local_caches()

# These imports are intentionally delayed until rank-local compiler caches exist.
# isort: off
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from magi_attn_extensions.DSA.config import (  # noqa: E402
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
)
from magi_attn_extensions.DSA.modeling import MagiDSALayer  # noqa: E402
from magi_attn_extensions.DSA.model_adapter import (  # noqa: E402
    layout_and_project_dsa_input,
)
from magi_attn_extensions.DSA.nvtx import dsa_nvtx_range  # noqa: E402
from magi_attn_extensions.DSA.pro_runtime import MagiDSAProRuntimeMgr  # noqa: E402
from magi_attn_extensions.DSA.runtime import (  # noqa: E402
    DsaExecutionHandle,
    MagiDSARuntimeMgr,
)
from magi_attn_extensions.DSA.types import (  # noqa: E402
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)
from magi_attn_extensions.DSA.comm import (  # noqa: E402
    finish_dsa_tensor_route,
    start_dsa_tensor_route,
    unlayout_dsa_query_tensor,
)
from magi_attn_extensions.DSA.phase import dsa_phase  # noqa: E402
from magi_attn_extensions.DSA.reference import (  # noqa: E402
    assert_backend_native_topk_outputs_close,
    dsa_reference,
    validate_backend_native_topk,
    validate_backend_native_topk_pair,
)

# isort: on

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
    """Record where Core and Extension were actually imported from.

    Magi-DSA now ships in the magi_attn_extensions distribution, so checking the
    Core wheel alone would no longer prove that the DSA code under test came
    from an installed wheel rather than a mounted source tree.
    """

    if os.environ.get("MAGI_DSA_REQUIRE_INSTALLED_WHEEL") != "1":
        return None
    import magi_attn_extensions.DSA as magi_dsa

    import magi_attention

    expected_revision = os.environ.get("MAGI_DSA_EXPECTED_REVISION", "")
    if len(expected_revision) != 40:
        raise RuntimeError("installed-wheel validation requires an exact revision")
    package_path = str(Path(magi_attention.__file__).resolve())
    package_version = package_metadata.version("magi-attention")
    expected_version = f"1.1.1+g{expected_revision}"
    extension_path = str(Path(magi_dsa.__file__).resolve())
    extension_version = package_metadata.version("magi_attn_extensions")
    for label, path in (
        ("magi_attention", package_path),
        ("magi_attn_extensions.DSA", extension_path),
    ):
        if not set(Path(path).parts).intersection({"site-packages", "dist-packages"}):
            raise RuntimeError(
                f"{label} was not imported from a Python installation directory: {path}"
            )
    if package_version != expected_version:
        raise RuntimeError(
            f"installed magi_attention version is {package_version}, "
            f"expected {expected_version}"
        )
    return {
        "package_path": package_path,
        "package_version": package_version,
        "extension_path": extension_path,
        "extension_version": extension_version,
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


def _worker_device_metadata(rank: int, local_rank: int) -> dict[str, object]:
    current_device = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(current_device)
    raw_uuid = getattr(properties, "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("CUDA device properties do not expose a device UUID")
    if isinstance(raw_uuid, (bytes, bytearray)):
        device_uuid = bytes(raw_uuid).hex()
    else:
        device_uuid = str(raw_uuid)
    if not device_uuid:
        raise RuntimeError("CUDA device UUID is empty")
    return {
        "cache_paths": dict(_RANK_LOCAL_CACHE_PATHS),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": current_device,
        "device_name": properties.name,
        "device_uuid": device_uuid,
        "local_rank": local_rank,
        "rank": rank,
    }


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
    """Use the complete frozen DeepSeek-V4-Pro schema for CP2 correctness."""

    return MagiDSAConfig(ratio=4)


def _run_route_smoke(rank: int, world_size: int) -> dict[str, object]:
    if world_size != 2:
        raise ValueError("the CP2 smoke case requires exactly two ranks")
    # rank 0 owns no source rows; it still receives Query rows after the layout.
    smoke_counts = (0, 31)
    local_count = smoke_counts[rank]
    meta = MagiDSAPackedMeta((0, 7, 31), smoke_counts)
    runtime = MagiDSARuntimeMgr(
        _small_config(), dist.group.WORLD
    )
    handle = runtime.prepare_execution(
        meta,
        torch.device("cuda", rank),
        local_token_capacity=max(local_count, 16),
    )
    route = handle.device_plan.token_layout_route
    if route is None:
        raise RuntimeError("CSA smoke plan has no TOKEN_LAYOUT route")
    rank_plan = handle.plan.rank_plans[rank]
    global_ids = torch.arange(
        rank_plan.source_global_begin,
        rank_plan.source_global_end,
        dtype=torch.float32,
        device="cuda",
    )
    source_x = (
        global_ids.to(torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, runtime.config.hidden_size)
        .contiguous()
        .requires_grad_(True)
    )
    local_x = runtime.layout_hidden(source_x, handle)
    expected_local = route.consumer_global_rows.to(torch.bfloat16).unsqueeze(1)
    expected_local = expected_local.expand(-1, runtime.config.hidden_size)
    if not torch.equal(local_x, expected_local):
        raise AssertionError("TOKEN_LAYOUT does not match final Query global rows")
    source_order = unlayout_dsa_query_tensor(local_x.detach(), route, dist.group.WORLD)
    expected_source = global_ids.to(torch.bfloat16).unsqueeze(1)
    expected_source = expected_source.expand(-1, runtime.config.hidden_size)
    if not torch.equal(source_order, expected_source):
        raise AssertionError("diagnostic unlayout does not restore source-owner rows")
    gradient = torch.autograd.grad(local_x.float().sum(), source_x)[0]
    if not torch.equal(gradient, torch.ones_like(source_x)):
        raise AssertionError("TOKEN_LAYOUT backward is not a global bijection")

    first_source = source_x.detach().clone().requires_grad_(True)
    second_source = (source_x.detach() + 1).contiguous().requires_grad_(True)
    first_transfer = start_dsa_tensor_route(first_source, route, dist.group.WORLD)
    second_transfer = start_dsa_tensor_route(second_source, route, dist.group.WORLD)
    if first_transfer.received.data_ptr() == second_transfer.received.data_ptr():
        raise AssertionError("two in-flight routes reused one receive buffer")
    first_local = finish_dsa_tensor_route(first_transfer)
    second_local = finish_dsa_tensor_route(second_transfer)
    if not torch.equal(first_local, expected_local):
        raise AssertionError("the first in-flight route returned incorrect rows")
    if not torch.equal(second_local, expected_local + 1):
        raise AssertionError("the second in-flight route returned incorrect rows")
    accumulated_loss = first_local.float().sum() + second_local.float().sum()
    accumulated_loss.backward(retain_graph=True)
    accumulated_loss.backward()
    for source in (first_source, second_source):
        if source.grad is None or not torch.equal(
            source.grad, torch.full_like(source, 2)
        ):
            raise AssertionError(
                "reentrant route backward did not accumulate gradients"
            )
    torch.cuda.synchronize()
    counters = runtime.counters
    if counters.health_checks != 1 or counters.device_materializations != 1:
        raise AssertionError("cold route health check/materialization count mismatch")
    return {
        "case": "smoke",
        "final_query_rows": route.consumer_row_count,
        "source_rows": local_count,
        "two_inflight_reentrant": True,
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
    global_x, sink = _global_source(config, total_tokens=total_tokens, seed=seed)
    global_x.requires_grad_(requires_grad)
    positions = _packed_position_ids((0, total_tokens), device=global_x.device)
    global_qr, global_q, global_kv = _project_test_dsa_inputs(
        global_x, positions, config
    )
    sink.requires_grad_(requires_grad)
    projected = (global_qr, global_q, global_kv)
    if requires_grad:
        for tensor in projected:
            tensor.retain_grad()
    return (global_x, *projected, sink)


def _global_source(
    config: MagiDSAConfig,
    *,
    total_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    global_x = torch.randn(
        total_tokens,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sink = torch.randn(
        config.num_query_heads,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    return global_x, sink


def _packed_position_ids(
    cu_seqlens: tuple[int, ...], *, device: torch.device
) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.arange(end - begin, dtype=torch.int32, device=device)
            for begin, end in zip(cu_seqlens, cu_seqlens[1:])
        )
    )


def _repeat_feature(source: torch.Tensor, width: int) -> torch.Tensor:
    if width <= source.shape[1]:
        return source[:, :width].clone(memory_format=torch.contiguous_format)
    repeats = (width + source.shape[1] - 1) // source.shape[1]
    return source.repeat(1, repeats)[:, :width].contiguous()


def _project_test_dsa_inputs(
    local_x: torch.Tensor,
    position_ids: torch.Tensor,
    config: MagiDSAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic parameter-free model projection used only by test workers."""

    if position_ids.shape != (local_x.shape[0],):
        raise ValueError("test projector received invalid final-local position IDs")
    with dsa_nvtx_range("model_projection::test_qr", enabled=local_x.is_cuda):
        qr = _repeat_feature(local_x, config.q_lora_rank)
    with dsa_nvtx_range("model_projection::test_q", enabled=local_x.is_cuda):
        q_base = _repeat_feature(local_x, config.head_dim)
        q = (
            q_base.unsqueeze(1)
            .expand(-1, config.num_query_heads, -1)
            .contiguous()
            .mul_(0.25)
        )
    with dsa_nvtx_range("model_projection::test_kv", enabled=local_x.is_cuda):
        latent_kv = _repeat_feature(local_x, config.head_dim).mul_(0.25)
    return qr, q, latent_kv


def _owner_source(
    config: MagiDSAConfig,
    rank: int,
    world_size: int,
    *,
    requires_grad: bool,
    cu_seqlens: tuple[int, ...] = _CP2_CU_SEQLENS,
    local_counts: tuple[int, ...] = _CP2_LOCAL_COUNTS,
    seed: int = 411,
) -> tuple[torch.Tensor, torch.Tensor, MagiDSAPackedMeta]:
    begin, end = _owner_bounds(rank, world_size, local_counts)
    global_x, global_sink = _global_source(
        config,
        total_tokens=cu_seqlens[-1],
        seed=seed,
    )
    source_x = global_x[begin:end].clone().contiguous().requires_grad_(requires_grad)
    sink = global_sink.clone().contiguous().requires_grad_(requires_grad)
    # The planner is told the whole source split, so every rank derives the
    # same plan locally without an owner-layout collective.
    meta = MagiDSAPackedMeta(cu_seqlens, tuple(local_counts))
    return source_x, sink, meta


def _materialize_owner_input(
    source_x: torch.Tensor,
    sink: torch.Tensor,
    packed_meta: MagiDSAPackedMeta,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
) -> tuple[MagiDSAInput, tuple[torch.Tensor, ...]]:
    config = runtime.config

    def projector(
        local_x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _project_test_dsa_inputs(local_x, position_ids, config)

    dsa_input = layout_and_project_dsa_input(
        source_x,
        sink,
        packed_meta,
        runtime,
        handle,
        projector,
    )
    if source_x.requires_grad:
        for tensor in (dsa_input.qr, dsa_input.q, dsa_input.latent_kv):
            tensor.retain_grad()
    return dsa_input, (
        source_x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        sink,
    )


def _runtime_result(
    layer: MagiDSALayer,
    source_x: torch.Tensor,
    sink: torch.Tensor,
    packed_meta: MagiDSAPackedMeta,
) -> tuple[MagiDSAForwardResult, MagiDSARuntimeMgr, DsaExecutionHandle]:
    runtime, handle = _prepare_runtime(layer, packed_meta, source_x.device)
    dsa_input, _ = _materialize_owner_input(
        source_x, sink, packed_meta, runtime, handle
    )
    return runtime.calc_dsa(layer.projections(), dsa_input, handle), runtime, handle


def _prepare_runtime(
    layer: MagiDSALayer,
    packed_meta: MagiDSAPackedMeta,
    device: torch.device,
    *,
    health_check: bool = False,
    structural_layout_config: DsaStructuralLayoutConfig | None = None,
) -> tuple[MagiDSARuntimeMgr, DsaExecutionHandle]:
    if structural_layout_config is None:
        structural_layout_config = DsaStructuralLayoutConfig()
    runtime = MagiDSARuntimeMgr(
        layer.config,
        dist.group.WORLD,
        structural_layout_config=structural_layout_config,
    )
    final_query_capacity = (
        packed_meta.cu_seqlens[-1] + dist.get_world_size() - 1
    ) // dist.get_world_size()
    handle = runtime.prepare_execution(
        packed_meta,
        device,
        local_token_capacity=max(
            packed_meta.local_token_count(dist.get_rank(dist.group.WORLD)),
            final_query_capacity,
        ),
        health_check=health_check,
    )
    return runtime, handle


def _plan_evidence(handle: DsaExecutionHandle) -> dict[str, object]:
    plan = handle.plan
    rank_plan = plan.rank_plans[handle.rank]
    rank_layout_payload = {
        "local_query_global_rows": rank_plan.local_query_global_rows,
        "query_fragments": [asdict(fragment) for fragment in rank_plan.query_fragments],
        "token_layout_route": asdict(plan.token_layout_route),
    }
    rank_layout_signature = hashlib.sha256(
        json.dumps(
            rank_layout_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    structural_metrics: dict[str, object] | None = None
    structural_rank_cost: dict[str, object] | None = None
    if True:
        if plan.layout_metrics is None or plan.structural_layout_config is None:
            raise AssertionError("structural plan evidence is incomplete")
        structural_metrics = asdict(plan.layout_metrics)
        rank_costs = structural_metrics.get("rank_costs")
        if (
            not isinstance(rank_costs, (list, tuple))
            or len(rank_costs) != handle.world_size
        ):
            raise AssertionError("structural rank-cost evidence is incomplete")
        rank_cost = rank_costs[handle.rank]
        if not isinstance(rank_cost, dict):
            raise TypeError("structural rank-cost evidence must be a dictionary")
        structural_rank_cost = rank_cost
    return {
        "declared_local_token_capacity": handle.local_token_capacity,
        "fragment_count": len(rank_plan.query_fragments),
        "local_query_tokens": rank_plan.local_token_count,
        "local_source_tokens": rank_plan.source_token_count,
        "plan_hash": plan.plan_hash,
        "policy": "structural_balanced",
        "query_layout_hash": plan.query_layout_hash,
        "query_token_counts": list(plan.query_token_counts),
        "rank_query_layout_signature": rank_layout_signature,
        "ratio": plan.ratio,
        "source_token_counts": list(plan.source_token_counts),
        "structural_layout_config": (
            None
            if plan.structural_layout_config is None
            else asdict(plan.structural_layout_config)
        ),
        "structural_layout_metrics": structural_metrics,
        "structural_rank_cost": structural_rank_cost,
    }


def _prepare_owner_case(
    layer: MagiDSALayer,
    rank: int,
    world_size: int,
    *,
    requires_grad: bool,
    cu_seqlens: tuple[int, ...] = _CP2_CU_SEQLENS,
    local_counts: tuple[int, ...] = _CP2_LOCAL_COUNTS,
    seed: int = 411,
    health_check: bool = False,
    structural_layout_config: DsaStructuralLayoutConfig | None = None,
) -> tuple[
    MagiDSAInput,
    tuple[torch.Tensor, ...],
    MagiDSARuntimeMgr,
    DsaExecutionHandle,
]:
    source_x, sink, packed_meta = _owner_source(
        layer.config,
        rank,
        world_size,
        requires_grad=requires_grad,
        cu_seqlens=cu_seqlens,
        local_counts=local_counts,
        seed=seed,
    )
    runtime, handle = _prepare_runtime(
        layer,
        packed_meta,
        source_x.device,
        health_check=health_check,
        structural_layout_config=structural_layout_config,
    )
    dsa_input, input_tensors = _materialize_owner_input(
        source_x,
        sink,
        packed_meta,
        runtime,
        handle,
    )
    return dsa_input, input_tensors, runtime, handle


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    actual_float = actual.float()
    expected_float = expected.float()
    finite = torch.isfinite(actual_float) & torch.isfinite(expected_float)
    if not torch.any(finite):
        return 0.0
    return float((actual_float[finite] - expected_float[finite]).abs().max().item())


def _canonicalize_rows(
    local: torch.Tensor,
    global_rows: torch.Tensor,
    total_tokens: int,
    *,
    label: str,
) -> torch.Tensor:
    """Assemble one distributed exact-cover tensor in canonical global order."""

    global_rows = global_rows.to(device=local.device, dtype=torch.int64)
    if local.shape[0] != global_rows.numel():
        raise AssertionError(f"{label} does not match its global-row metadata")
    canonical = torch.zeros(
        (total_tokens, *local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    canonical.index_copy_(0, global_rows, local)
    with dsa_phase(f"collective_canonicalize::{label}"):
        dist.all_reduce(canonical)
    return canonical


def _canonicalize_query_tensor(
    local: torch.Tensor,
    handle: DsaExecutionHandle,
    *,
    label: str,
) -> torch.Tensor:
    global_rows = torch.tensor(
        handle.plan.rank_plans[handle.rank].local_query_global_rows,
        dtype=torch.int64,
        device=local.device,
    )
    return _canonicalize_rows(
        local,
        global_rows,
        handle.plan.total_tokens,
        label=label,
    )


def _canonicalize_source_tensor(
    local: torch.Tensor,
    handle: DsaExecutionHandle,
    *,
    label: str,
) -> torch.Tensor:
    rank_plan = handle.plan.rank_plans[handle.rank]
    global_rows = torch.arange(
        rank_plan.source_global_begin,
        rank_plan.source_global_end,
        dtype=torch.int64,
        device=local.device,
    )
    return _canonicalize_rows(
        local,
        global_rows,
        handle.plan.total_tokens,
        label=label,
    )


def _canonicalize_forward_result(
    local: MagiDSAForwardResult,
    handle: DsaExecutionHandle,
) -> MagiDSAForwardResult:
    global_kl = local.kl.detach().clone()
    with dsa_phase("collective_canonicalize::kl"):
        dist.all_reduce(global_kl)
    return MagiDSAForwardResult(
        output=_canonicalize_query_tensor(local.output, handle, label="output"),
        kl=global_kl,
        sparse_lse=_canonicalize_query_tensor(
            local.sparse_lse, handle, label="sparse_lse"
        ),
        topk_ids=_canonicalize_query_tensor(local.topk_ids, handle, label="topk_ids"),
        topk_length=_canonicalize_query_tensor(
            local.topk_length, handle, label="topk_length"
        ),
        indexer_lse=_canonicalize_query_tensor(
            local.indexer_lse, handle, label="indexer_lse"
        ),
    )


def _assert_forward_equal(
    actual: MagiDSAForwardResult, expected: MagiDSAForwardResult
) -> dict[str, object]:
    output_diagnostics = assert_backend_native_topk_outputs_close(
        actual.output,
        expected.output,
        actual.topk_ids,
        actual.topk_length,
        expected.topk_ids,
        expected.topk_length,
        label="balanced vs sequential",
    )
    torch.testing.assert_close(
        actual.sparse_lse, expected.sparse_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        actual.indexer_lse, expected.indexer_lse, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(actual.kl, expected.kl, atol=2e-2, rtol=2e-2)
    return output_diagnostics


def _run_csa_natural(rank: int, world_size: int) -> dict[str, object]:
    config = MagiDSAConfig(ratio=4)
    _, sequential_layer, balanced_layer = _release_layers(config)
    source_x, sink, packed_meta = _owner_source(
        config,
        rank,
        world_size,
        requires_grad=False,
    )
    with torch.no_grad():
        sequential_local, sequential_runtime, sequential_handle = _runtime_result(
            sequential_layer,
            source_x,
            sink,
            packed_meta,
            "sequential",
        )
        balanced_local, balanced_runtime, balanced_handle = _runtime_result(
            balanced_layer,
            source_x,
            sink,
            packed_meta,
            "structural_balanced",
        )
        sequential = _canonicalize_forward_result(sequential_local, sequential_handle)
        balanced = _canonicalize_forward_result(balanced_local, balanced_handle)
    torch.cuda.synchronize()
    _assert_forward_equal(balanced, sequential)
    return {
        "balanced_plan_evidence": _plan_evidence(balanced_handle),
        "balanced_policy": "structural_balanced",
        "case": "csa-natural",
        "indexer_lse_max_abs": _max_abs(balanced.indexer_lse, sequential.indexer_lse),
        "kl_abs": _max_abs(balanced.kl, sequential.kl),
        "local_rows": packed_meta.local_token_count(rank),
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
            captured_raw_scores["indexer_raw_scores"] = (
                backend_result["scores"].detach().clone()
            )
            return backend_result

        DSA.indexer_forward_wrapper = capture_score
    try:
        result = runtime.calc_dsa(layer.projections(), dsa_input, handle)
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
        "indexer_lse": _canonicalize_query_tensor(
            result.indexer_lse.detach(), handle, label="indexer_lse"
        ),
        "kl": result.kl.detach().clone(),
        "kl_global": global_kl,
        "output": _canonicalize_query_tensor(
            result.output.detach(), handle, label="output"
        ),
        "sparse_lse": _canonicalize_query_tensor(
            result.sparse_lse.detach(), handle, label="sparse_lse"
        ),
        "topk_ids": _canonicalize_query_tensor(
            result.topk_ids.detach(), handle, label="topk_ids"
        ),
        "topk_length": _canonicalize_query_tensor(
            result.topk_length.detach(), handle, label="topk_length"
        ),
    }
    if layer.indexer is not None:
        indexer_map = handle.device_plan.indexer
        rank_plan = handle.plan.rank_plans[handle.rank]
        if indexer_map is None:
            raise AssertionError("CSA snapshot is missing Indexer plan metadata")
        if "indexer_raw_scores" not in captured_raw_scores:
            if indexer_map.seq_lens.numel():
                raise AssertionError("CSA snapshot did not capture Indexer raw scores")
            captured_raw_scores["indexer_raw_scores"] = torch.empty(
                (0, indexer_map.backend_max_seqlen_k),
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
                    rank_plan.local_query_global_rows,
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
            gradient = tensor.grad.detach()
            if name == "x":
                gradient = _canonicalize_source_tensor(
                    gradient, handle, label=gradient_name
                )
            elif name in ("qr", "q", "kv"):
                gradient = _canonicalize_query_tensor(
                    gradient, handle, label=gradient_name
                )
            else:
                gradient = gradient.clone()
            snapshots[gradient_name] = gradient
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None
        snapshots[f"parameter::{name}"] = parameter.grad.detach().clone()
    return snapshots


def _reference_snapshot(
    layer: MagiDSALayer,
    input_tensors: tuple[torch.Tensor, ...],
    cu_seqlens: tuple[int, ...] = _CP2_CU_SEQLENS,
    gradient_names: tuple[str, ...] = _ALL_GRADIENT_NAMES,
    control_context: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    context = dict(control_context or {})
    x, qr, q, latent_kv, sink = input_tensors
    forward_started = time.monotonic()
    _cp2_record("reference_forward_begin", **context)
    result = dsa_reference(
        layer,
        x,
        qr,
        q,
        latent_kv,
        sink,
        cu_seqlens,
    )
    _cp2_record(
        "reference_forward_launch_end",
        elapsed_seconds=time.monotonic() - forward_started,
        **context,
    )
    torch.cuda.synchronize()
    _cp2_record(
        "reference_forward_end",
        elapsed_seconds=time.monotonic() - forward_started,
        **context,
    )

    raw_score_started = time.monotonic()
    _cp2_record("reference_raw_score_begin", **context)
    reference_raw_scores = None
    if layer.indexer is not None:
        reference_raw_scores = _reference_csa_index_scores(
            layer,
            input_tensors,
            cu_seqlens,
        )
    _cp2_record(
        "reference_raw_score_launch_end",
        elapsed_seconds=time.monotonic() - raw_score_started,
        skipped=layer.indexer is None,
        **context,
    )
    torch.cuda.synchronize()
    _cp2_record(
        "reference_raw_score_end",
        elapsed_seconds=time.monotonic() - raw_score_started,
        skipped=layer.indexer is None,
        **context,
    )

    backward_started = time.monotonic()
    _cp2_record("reference_backward_begin", **context)
    loss = result.output.float().square().mean() + result.kl
    loss.backward()
    _cp2_record(
        "reference_backward_launch_end",
        elapsed_seconds=time.monotonic() - backward_started,
        **context,
    )
    torch.cuda.synchronize()
    _cp2_record(
        "reference_backward_end",
        elapsed_seconds=time.monotonic() - backward_started,
        **context,
    )
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
    for local_row, (global_row_value, block_offset_value, length_value) in enumerate(
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
        backend_valid = raw_scores[local_row, :length]
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
        backend_padding = raw_scores[local_row, length:]
        if backend_padding.numel() and not bool(
            torch.all(torch.isneginf(backend_padding)).item()
        ):
            raise AssertionError(
                f"{label}: Indexer raw-score causal padding is not -inf"
            )
        max_abs = max(max_abs, _max_abs(backend_valid, reference_valid))
    return max_abs


def _compare_natural_snapshots(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    label: str,
    gradient_names: tuple[str, ...] = _ALL_GRADIENT_NAMES,
    compare_parameter_values: bool = True,
    compare_raw_indexer_scores: bool = False,
) -> tuple[dict[str, object], list[str]]:
    output_diagnostics = assert_backend_native_topk_outputs_close(
        actual["output"],
        expected["output"],
        actual["topk_ids"],
        actual["topk_length"],
        expected["topk_ids"],
        expected["topk_length"],
        label=label,
    )
    raw_score_max_abs = 0.0
    if compare_raw_indexer_scores:
        raw_score_max_abs = _assert_raw_indexer_scores(
            actual,
            expected,
            label=label,
        )
    for name in ("sparse_lse", "indexer_lse"):
        _assert_regular_close(
            actual[name],
            expected[name],
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
    gradient_failures: list[str] = []
    for name in gradient_names:
        if name == "grad_kv":
            continue
        try:
            _assert_regular_close(
                actual[name],
                expected[name],
                atol=2e-2,
                rtol=2e-2,
                name=f"{label}: {name}",
            )
        except AssertionError as error:
            gradient_failures.append(str(error))
    kv_mismatch = 0.0
    if "grad_kv" in gradient_names:
        try:
            kv_mismatch = _assert_bf16_reduction_close(
                actual["grad_kv"],
                expected["grad_kv"],
                name=f"{label}: latent_kv gradient",
            )
        except AssertionError as error:
            gradient_failures.append(str(error))
    parameter_names = sorted(name for name in actual if name.startswith("parameter::"))
    expected_parameter_names = sorted(
        name for name in expected if name.startswith("parameter::")
    )
    if parameter_names != expected_parameter_names:
        raise AssertionError(f"{label}: parameter gradient schemas differ")
    if compare_parameter_values:
        for name in parameter_names:
            try:
                _assert_regular_close(
                    actual[name],
                    expected[name],
                    atol=2e-2,
                    rtol=2e-2,
                    name=f"{label}: {name}",
                )
            except AssertionError as error:
                gradient_failures.append(str(error))
    if gradient_failures:
        raise AssertionError(" | ".join(gradient_failures))
    return (
        {
            **output_diagnostics,
            "indexer_lse_max_abs": _max_abs(
                actual["indexer_lse"],
                expected["indexer_lse"],
            ),
            "indexer_raw_score_max_abs": raw_score_max_abs,
            "kl_abs": _max_abs(actual["kl_global"], expected["kl_global"]),
            "kv_gradient_mismatch_ratio": kv_mismatch,
        },
        parameter_names,
    )


def _prewarm_natural_plan(
    layer: MagiDSALayer,
    runtime: MagiDSARuntimeMgr,
    handle: DsaExecutionHandle,
    rank: int,
    world_size: int,
) -> tuple[float, dict[str, object]]:
    source_x, sink, packed_meta = _owner_source(
        layer.config,
        rank,
        world_size,
        requires_grad=True,
    )
    warm_input, warm_tensors = _materialize_owner_input(
        source_x, sink, packed_meta, runtime, handle
    )
    layer.zero_grad(set_to_none=True)
    _cp2_record("prewarm_begin", policy="structural_balanced")
    started = time.monotonic()
    with dsa_phase("cudnn_dsa_prewarm_structural_balanced"):
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
    _save_cudnn_cache_inventory("after_prewarm_structural_balanced", inventory)
    _cp2_record(
        "prewarm_end",
        elapsed_seconds=elapsed,
        inventory=inventory,
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
    reference_context: dict[str, object] = {
        "case": "csa-natural-backward",
        "path": "csa",
    }
    _cp2_record("reference_begin", **reference_context)
    reference_started = time.monotonic()
    reference = _reference_snapshot(
        reference_layer,
        reference_inputs,
        control_context=reference_context,
    )
    reference_seconds = time.monotonic() - reference_started
    _cp2_record(
        "reference_end",
        elapsed_seconds=reference_seconds,
        **reference_context,
    )

    (
        sequential_input,
        sequential_tensors,
        sequential_runtime,
        sequential_handle,
    ) = _prepare_owner_case(
        sequential_layer,
        rank,
        world_size,
        "sequential",
        requires_grad=True,
    )
    (
        balanced_input,
        balanced_tensors,
        balanced_runtime,
        balanced_handle,
    ) = _prepare_owner_case(
        balanced_layer,
        rank,
        world_size,
        "structural_balanced",
        requires_grad=True,
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
        "structural_balanced",
    )
    sequential_layer.zero_grad(set_to_none=True)
    balanced_layer.zero_grad(set_to_none=True)
    with dsa_phase("collective_actual_ready"):
        dist.barrier()
    torch.cuda.synchronize()

    _cp2_record(
        "execute_begin",
        deadline_seconds=60,
        plans=("sequential", "structural_balanced"),
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
        plans=("sequential", "structural_balanced"),
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
    sequential_reference_metrics, _ = _compare_natural_snapshots(
        sequential,
        reference,
        label="sequential vs CP1 reference",
        compare_raw_indexer_scores=True,
    )
    balanced_reference_metrics, _ = _compare_natural_snapshots(
        balanced,
        reference,
        label="balanced vs CP1 reference",
        compare_raw_indexer_scores=True,
    )
    _cp2_record("verification_end", case="csa-natural-backward")
    return {
        "balanced_plan_evidence": _plan_evidence(balanced_handle),
        "balanced_policy": "structural_balanced",
        "balanced_prewarm_seconds": balanced_prewarm_seconds,
        "balanced_reference": balanced_reference_metrics,
        "balanced_warm_calls": balanced_runtime.counters.warm_invocations,
        "case": "csa-natural-backward",
        "execution_seconds": execution_seconds,
        "local_rows": sequential_input.packed_meta.local_token_count(rank),
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
        return ("grad_x", "grad_q", "grad_kv", "grad_sink")
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
    from magi_attn_extensions.DSA.reference import _compress_global, dsa_position_ids

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
    indexer_scale = layer.config.indexer_head_dim**-0.5
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
            dots.relu() * indexer_scale * weights[q_begin:q_end].float().unsqueeze(-1)
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
        captured["scores"] = result["scores"].detach().clone()
        return result

    def capture_topk(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_val") is not False:
            raise AssertionError("production cuDNN Top-K must run in IDs-only mode")
        result = original_topk(*args, **kwargs)
        if result["values"] is not None:
            raise AssertionError("IDs-only cuDNN Top-K unexpectedly returned values")
        captured["raw_topk_indices"] = result["indices"].detach()
        return result

    DSA.indexer_forward_wrapper = capture_score
    DSA.indexer_top_k_wrapper = capture_topk
    try:
        result = runtime.calc_dsa(layer.projections(), dsa_input, handle)
    finally:
        DSA.indexer_forward_wrapper = original_score
        DSA.indexer_top_k_wrapper = original_topk
    missing = {"scores", "raw_topk_indices"} - captured.keys()
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
    indexer_map = handle.device_plan.indexer
    if indexer_map is None:
        raise RuntimeError("CP8 diagnostic plan is missing Indexer metadata")
    payload: dict[str, object] = {
        name: value.detach().cpu() for name, value in captured.items()
    }
    payload.update(
        {
            "q_sample_block_offsets": indexer_map.q_sample_block_offsets.cpu(),
            "rank": rank,
            "seq_lens": indexer_map.seq_lens.cpu(),
            "query_global_rows": rank_plan.local_query_global_rows,
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
    del world_size
    rank_plan = handle.plan.rank_plans[rank]
    query_rows = torch.tensor(rank_plan.local_query_global_rows, dtype=torch.int64)
    expected_ids = expected.topk_ids.cpu().index_select(0, query_rows)
    expected_lengths = expected.topk_length.cpu().index_select(0, query_rows)
    actual_ids = actual.topk_ids.cpu()
    actual_lengths = actual.topk_length.cpu()
    validate_backend_native_topk_pair(
        actual_ids,
        actual_lengths,
        expected_ids,
        expected_lengths,
        label="CP8 diagnostic production vs pure-PyTorch reference",
    )
    reference_scores = reference_scores.cpu()
    backend_scores = captured["scores"].float().cpu()
    raw_backend_ids = captured["raw_topk_indices"].to(torch.int32).cpu()

    indexer_map = handle.device_plan.indexer
    if indexer_map is None:
        raise RuntimeError("CP8 diagnostic plan is missing Indexer metadata")
    offsets = indexer_map.q_sample_block_offsets.cpu()
    backend_lengths = indexer_map.seq_lens.cpu()

    mismatch_mask = torch.any(actual_ids != expected_ids, dim=1)
    mismatch_mask |= actual_lengths != expected_lengths
    mismatch_rows = torch.nonzero(mismatch_mask, as_tuple=False).flatten().tolist()
    details: list[dict[str, object]] = []
    score_max_abs = 0.0
    all_sets_equal = True
    all_actual_match_backend_output = True
    for local_row, global_row_value in enumerate(rank_plan.local_query_global_rows):
        global_row = int(global_row_value)
        backend_length = int(backend_lengths[local_row].item())
        block_offset = int(offsets[local_row].item())
        backend_visible = backend_scores[local_row, :backend_length]
        if backend_length:
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
        backend_padding = backend_scores[local_row, backend_length:]
        if backend_padding.numel() and not bool(
            torch.all(torch.isneginf(backend_padding)).item()
        ):
            raise AssertionError(
                "CP8 diagnostic Indexer raw-score causal padding is not -inf"
            )
        actual_length = int(actual_lengths[local_row].item())
        backend_output = (raw_backend_ids[local_row, :actual_length] + block_offset).to(
            torch.int32
        )
        actual_matches_backend_output = torch.equal(
            actual_ids[local_row, :actual_length],
            backend_output,
        )
        all_actual_match_backend_output &= actual_matches_backend_output
        validate_backend_native_topk(
            backend_visible,
            actual_ids[local_row, :actual_length],
            actual_length,
            global_offset=block_offset,
        )
        if local_row not in mismatch_rows:
            continue

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
        detail: dict[str, object] = {
            "actual_ids": actual_valid,
            "actual_length": actual_length,
            "actual_matches_backend_output": actual_matches_backend_output,
            "backend_output_ids": backend_output.tolist(),
            "backend_length": backend_length,
            "expected_ids": expected_valid,
            "expected_length": expected_length,
            "first_difference": first_difference,
            "global_row": global_row,
            "local_row": local_row,
            "sets_equal": sets_equal,
            "query_local_row": local_row,
        }
        if first_difference < min(actual_length, expected_length):
            actual_id = actual_valid[first_difference]
            expected_id = expected_valid[first_difference]
            actual_column = actual_id - block_offset
            expected_column = expected_id - block_offset
            detail["first_pair"] = {
                "actual_backend_score": float(
                    backend_scores[local_row, actual_column].item()
                ),
                "actual_id": actual_id,
                "actual_reference_score": float(
                    reference_scores[global_row, actual_id].item()
                ),
                "backend_gap_actual_minus_expected": float(
                    (
                        backend_scores[local_row, actual_column]
                        - backend_scores[local_row, expected_column]
                    ).item()
                ),
                "expected_backend_score": float(
                    backend_scores[local_row, expected_column].item()
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
        "all_actual_match_backend_output": all_actual_match_backend_output,
        "all_mismatch_sets_equal": all_sets_equal,
        "backend_reference_score_max_abs": score_max_abs,
        "backend_reference_score_tolerance": {"atol": 5e-3, "rtol": 5e-3},
        "backend_reference_score_tolerance_passed": True,
        "lengths_exact": torch.equal(actual_lengths, expected_lengths),
        "mismatch_row_count": len(mismatch_rows),
        "mismatch_rows": details,
        "query_global_rows": list(rank_plan.local_query_global_rows),
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
    reference_layer, structural_layer = _release_layer_copies(
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

    dsa_input, _, runtime, handle = _prepare_owner_case(
        structural_layer,
        rank,
        world_size,
        "structural_balanced",
        requires_grad=False,
        cu_seqlens=_CP8_CU_SEQLENS,
        local_counts=_CP8_LOCAL_COUNTS,
        seed=450,
        health_check=True,
    )
    _cp2_record("prewarm_begin", case="cp8-topk-diagnostic")
    prewarm_started = time.monotonic()
    with torch.no_grad():
        runtime.calc_dsa(structural_layer.projections(), dsa_input, handle)
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
            structural_layer,
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
        "plan_policy": "structural_balanced",
        "prewarm_seconds": prewarm_seconds,
        "rank": rank,
        "structural_plan_evidence": _plan_evidence(handle),
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
    source_x, sink, packed_meta = _owner_source(
        layer.config,
        rank,
        world_size,
        requires_grad=True,
        cu_seqlens=_CP8_CU_SEQLENS,
        local_counts=_CP8_LOCAL_COUNTS,
        seed=seed,
    )
    warm_input, warm_tensors = _materialize_owner_input(
        source_x, sink, packed_meta, runtime, handle
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
        4: MagiDSAConfig(ratio=4),
        128: MagiDSAConfig(ratio=128),
    }
    for config in configs.values():
        config.validate_release_contract()

    csa_reference_layer, csa_structural_layer = _release_layer_copies(
        configs[4], seed=440, count=2
    )
    hca_reference_layer, hca_structural_layer = _release_layer_copies(
        configs[128], seed=442, count=2
    )

    references: dict[str, dict[str, torch.Tensor]] = {}
    reference_seconds: dict[str, float] = {}
    reference_cases: tuple[tuple[str, MagiDSALayer, DsaRatio, int], ...] = (
        ("csa", csa_reference_layer, 4, 450),
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
            control_context={
                "case": "cp8-natural-backward",
                "path": label,
            },
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
    # One planner, one Query layout: the CP8 case is now exactly the Pro pair.
    candidate_cases: tuple[tuple[str, MagiDSALayer, DsaRatio, int], ...] = (
        ("csa_structural", csa_structural_layer, 4, 450),
        ("hca_structural", hca_structural_layer, 128, 452),
    )
    for label, layer, ratio, seed in candidate_cases:
        dsa_input, input_tensors, runtime, handle = _prepare_owner_case(
            layer,
            rank,
            world_size,
            requires_grad=True,
            cu_seqlens=_CP8_CU_SEQLENS,
            local_counts=_CP8_LOCAL_COUNTS,
            seed=seed,
            health_check=False,
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

    structural_layout_hashes = {
        candidates[label][4].plan.query_layout_hash
        for label in ("csa_structural", "hca_structural")
    }
    if len(structural_layout_hashes) != 1:
        raise AssertionError("CP8 structural CSA/HCA plans use different Query layouts")
    MagiDSAProRuntimeMgr._validate_shared_layout(
        candidates["csa_structural"][4],
        candidates["hca_structural"][4],
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
    compare_parameter_values = rank == 0
    metrics: dict[str, dict[str, object]] = {}
    # One planner remains, so there is no cross-plan alignment to check: the
    # CP8 contract is that each structural plan matches the CP1 reference.
    csa_parameters: tuple[str, ...] = ()
    for label in ("csa_structural",):
        metrics[f"{label}_reference"], csa_parameters = _compare_natural_snapshots(
            actual[label],
            references["csa"],
            label=f"CP8 {label} vs reference",
            compare_parameter_values=compare_parameter_values,
            compare_raw_indexer_scores=True,
        )
        _cp2_record(
            "verification_checkpoint",
            case="cp8-natural-backward",
            checkpoint=f"{label}_reference",
        )
    non_csa_cases: tuple[tuple[str, str, DsaRatio], ...] = (
        ("hca_structural", "hca", 128),
    )
    for label, reference_label, ratio in non_csa_cases:
        metrics[f"{label}_reference"], _ = _compare_natural_snapshots(
            actual[label],
            references[reference_label],
            label=f"CP8 {label} vs reference",
            gradient_names=_release_gradient_names(ratio),
            compare_parameter_values=compare_parameter_values,
        )
        _cp2_record(
            "verification_checkpoint",
            case="cp8-natural-backward",
            checkpoint=f"{label}_reference",
        )

    _cp2_record("verification_end", case="cp8-natural-backward")

    plan_evidence = {
        label: _plan_evidence(candidate[4]) for label, candidate in candidates.items()
    }
    return {
        "case": "cp8-natural-backward",
        "execution_seconds": execution_seconds,
        "local_rows": _CP8_LOCAL_COUNTS[rank],
        "metrics": metrics,
        "model_parameter_values_checked": compare_parameter_values,
        "parameter_gradients": len(csa_parameters),
        "plan_evidence": plan_evidence,
        "prewarm_seconds": prewarm_seconds,
        "rank": rank,
        "reference_seconds": reference_seconds,
        "structural_query_layout_hash": next(iter(structural_layout_hashes)),
        "structural_layout_shared": True,
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
    worker_device = _worker_device_metadata(rank, local_rank)
    _cp2_record("worker_device_ready", **worker_device)
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
        report["worker_device"] = worker_device
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
