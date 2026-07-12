#!/usr/bin/env python3
"""Magi_DSA V4 B300/SM103 CP8 calibration and performance driver.

The distributed subcommands are intended to run under ``torchrun`` with one
process per GPU.  Contract constants are imported from ``validate.py`` and are
not exposed as CLI overrides.  Workload execution uses only the public DSA API;
internal imports are limited to native-backend preflight and cache evidence.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate import (  # noqa: E402
    CALIBRATION_CASES,
    CALIBRATION_SCHEMA_VERSION,
    CASE_BY_ID,
    CHUNK_RATIO,
    COMPILE_ITERS,
    CP_SIZE,
    DATASET_RELATIVE_PATH,
    DATASET_SHA256,
    DTYPE_NAME,
    GLOBAL_TOKENS,
    GRAD_SEED,
    HIDDEN_SIZE,
    HF_CONFIG_SHA256,
    INDEXER_PHASES,
    INPUT_SEED,
    KERNEL_BACKEND,
    MEASURE_CASES,
    MEASURE_ITERS,
    MODEL_SEED,
    NATIVE_BUFFER_NAMES,
    NATIVE_NUM_NVL_BYTES,
    NATIVE_NUM_RDMA_BYTES,
    NATIVE_NUM_SMS,
    PACK_NUM,
    PACK_SEED,
    PLAN_FEATURE_NAMES,
    PROFILE_CASES,
    PROFILE_PACK_INDEX,
    Q_LORA_RANK,
    SCHEMA_VERSION,
    SOFTMAX_SCALE,
    TARGET_TOKENS_PER_RANK,
    WARMUP_ITERS,
    WORLD_SIZE,
    CaseSpec,
    atomic_write_json,
    pack_sha256,
    pack_suite_sha256,
    read_json,
    sha256_file,
    validate_packs,
    validate_run,
    write_jsonl,
)


HF_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
INPUT_CHUNK_ROWS = 128
CORRECTNESS_RTOL = 3e-2
CORRECTNESS_ATOL = 3e-3


def _run_command(
    command: Sequence[str], *, cwd: Path = REPO_ROOT, check: bool = True
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def generate_packs(output: Path, dataset: Path) -> dict[str, Any]:
    """Generate and freeze the exact 20 DatasetSampler packs."""

    dataset = dataset.resolve()
    _require(dataset.is_file(), f"dataset does not exist: {dataset}")
    actual_dataset_hash = sha256_file(dataset)
    _require(
        actual_dataset_hash == DATASET_SHA256,
        f"dataset SHA256 {actual_dataset_hash} != frozen {DATASET_SHA256}",
    )
    repo_path = str(REPO_ROOT)
    inserted_repo = repo_path not in sys.path
    if inserted_repo:
        sys.path.insert(0, repo_path)
    try:
        from exps.dist_attn.benchmark.utils import DatasetSampler
    finally:
        if inserted_repo:
            sys.path.remove(repo_path)

    sampler = DatasetSampler(
        data_path=str(dataset),
        pack_len=GLOBAL_TOKENS,
        chunk_ratio=CHUNK_RATIO,
        is_binned=True,
        seed=PACK_SEED,
        drop_thres=-1,
    )
    packs: list[dict[str, Any]] = []
    for index in range(PACK_NUM):
        lengths = [int(value) for value in sampler.generate_pack_samples()]
        _require(lengths and all(value > 0 for value in lengths), "empty pack")
        _require(
            sum(lengths) == GLOBAL_TOKENS,
            f"pack {index} contains {sum(lengths)} tokens, expected {GLOBAL_TOKENS}",
        )
        packs.append(
            {"index": index, "lengths": lengths, "sha256": pack_sha256(lengths)}
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "sampler": {
            "implementation": "exps.dist_attn.benchmark.utils.DatasetSampler",
            "dataset_path": DATASET_RELATIVE_PATH,
            "dataset_sha256": actual_dataset_hash,
            "seed": PACK_SEED,
            "pack_num": PACK_NUM,
            "chunk_ratio": CHUNK_RATIO,
            "pack_len": GLOBAL_TOKENS,
            "is_binned": True,
            "drop_thres": -1,
        },
        "packs": packs,
        "packs_sha256": pack_suite_sha256(packs),
    }
    validate_packs(payload)
    atomic_write_json(output, payload)
    return payload


def _source_preflight(expected_revision: str) -> dict[str, Any]:
    manifest_path = Path("/opt/magi-dsa-build-manifest.json")
    revision_path = REPO_ROOT / ".magi-source-revision"
    submodule_path = REPO_ROOT / ".magi-submodules"
    _require(
        manifest_path.is_file(),
        "distributed benchmark requires /opt/magi-dsa-build-manifest.json; "
        "a development Git checkout is not an acceptance image",
    )
    _require(
        revision_path.is_file(),
        f"immutable source revision marker is missing: {revision_path}",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "unsupported build manifest")
    revision = revision_path.read_text(encoding="utf-8").strip()
    _require(
        revision == expected_revision,
        f"source marker {revision} != {expected_revision}",
    )
    _require(
        manifest.get("magi_attention_revision") == expected_revision,
        "build manifest revision differs from --expected-revision",
    )
    build = manifest.get("build")
    _require(
        isinstance(build, dict)
        and build.get("target_runtime_capability") == "103"
        and build.get("nvshmem_disabled") is False,
        "build manifest is not the SM103 native-NVSHMEM image",
    )
    expected_submodules = manifest.get("submodules")
    _require(isinstance(expected_submodules, dict), "manifest submodules are missing")
    _require(submodule_path.is_file(), "immutable submodule marker is missing")
    actual_submodules: dict[str, str] = {}
    for line in submodule_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        _require(len(parts) == 2, f"malformed submodule marker line {line!r}")
        actual_submodules[parts[1]] = parts[0]
    _require(
        actual_submodules == expected_submodules,
        "source archive submodule markers differ from the build manifest",
    )
    artifacts = manifest.get("artifacts")
    _require(
        isinstance(artifacts, dict) and artifacts, "manifest artifacts are missing"
    )
    verified_artifacts: dict[str, str] = {}
    for name, expected_sha256 in artifacts.items():
        path = Path(str(name))
        _require(
            path.is_absolute() and path.is_file(), f"manifest artifact missing: {path}"
        )
        actual_sha256 = sha256_file(path)
        _require(
            actual_sha256 == expected_sha256,
            f"manifest artifact hash mismatch: {path}",
        )
        verified_artifacts[str(path)] = actual_sha256
    return {
        "revision": revision,
        "worktree_clean": True,
        "worktree_clean_definition": (
            "Git-archive source with no .git; revision/submodules and all build "
            "manifest artifacts verified"
        ),
        "source_mode": "immutable_git_archive",
        "source_revision_marker": str(revision_path),
        "build_manifest_path": str(manifest_path),
        "build_manifest_sha256": sha256_file(manifest_path),
        "build_manifest": manifest,
        "verified_build_artifacts": verified_artifacts,
    }


def _parse_nvlink_topology(text: str) -> bool:
    rows: dict[int, list[str]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("GPU"):
            continue
        suffix = fields[0][3:]
        if suffix.isdigit() and len(fields) >= WORLD_SIZE + 1:
            rows[int(suffix)] = fields[1 : WORLD_SIZE + 1]
    if set(rows) != set(range(WORLD_SIZE)):
        return False
    for source in range(WORLD_SIZE):
        for destination in range(WORLD_SIZE):
            value = rows[source][destination]
            if source == destination:
                if value != "X":
                    return False
            elif not value.startswith("NV"):
                return False
    return True


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _module_evidence(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    path = (
        Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    )
    return {
        "module": name,
        "path": None if path is None else str(path),
        "sha256": None if path is None or not path.is_file() else sha256_file(path),
    }


def _installed_wheel_evidence() -> dict[str, Any]:
    module = importlib.import_module("magi_attention")
    module_path = Path(module.__file__).resolve()
    source_root = REPO_ROOT.resolve()
    _require(
        not module_path.is_relative_to(source_root),
        f"magi_attention resolved to source checkout {module_path}, not installed wheel",
    )
    distribution = importlib.metadata.distribution("magi_attention")
    files = list(distribution.files or ())
    record_entry = next(
        (
            entry
            for entry in files
            if entry.name == "RECORD"
            and any(part.endswith(".dist-info") for part in entry.parts)
        ),
        None,
    )
    _require(record_entry is not None, "installed distribution has no RECORD entry")
    record_path = Path(distribution.locate_file(record_entry)).resolve()
    metadata_path = record_path.parent
    _require(record_path.is_file(), f"installed wheel RECORD is missing: {record_path}")
    return {
        "module_path": str(module_path),
        "distribution_path": str(metadata_path),
        "distribution_version": distribution.version,
        "wheel_record_path": str(record_path),
        "wheel_record_sha256": sha256_file(record_path),
        "outside_source_checkout": True,
    }


def _flash_mla_sm100_evidence() -> dict[str, Any]:
    module = importlib.import_module("flash_mla")
    module_path = Path(module.__file__).resolve()
    root = module_path.parent
    candidates = {module_path} if module_path.suffix == ".so" else set()
    for pattern in ("*.so", "*.cubin", "*.fatbin"):
        candidates.update(root.rglob(pattern))
    _require(candidates, f"flash_mla package at {root} has no CUDA binary objects")
    matching_objects: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in sorted(candidates):
        result = subprocess.run(
            ["cuobjdump", "--list-elf", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            failures.append(f"{path}: exit {result.returncode}")
            continue
        matching_lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if "sm_100" in line.lower()
        ]
        if matching_lines:
            matching_objects.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "sm100_entries": matching_lines,
                }
            )
    _require(
        matching_objects,
        "flash_mla contains no cuobjdump-visible sm_100 cubin; "
        f"examined {len(candidates)} objects ({'; '.join(failures[:3])})",
    )
    return {
        "verified": True,
        "architecture": "sm_100",
        "tool": "cuobjdump --list-elf",
        "objects": matching_objects,
    }


def _initialize_distributed():
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    _require("RANK" in os.environ and "LOCAL_RANK" in os.environ, "use torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _require(world_size == WORLD_SIZE, f"WORLD_SIZE={world_size}, expected 8")
    _require(
        torch.cuda.device_count() == WORLD_SIZE, "exactly eight visible GPUs required"
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    _require(dist.get_world_size() == WORLD_SIZE, "default process group is not CP8")
    _require(dist.get_rank() == rank, "rank mismatch")
    return torch, dist, rank, local_rank, dist.distributed_c10d._get_default_group()


def _native_probe(torch, dist, rank: int, group) -> dict[str, Any]:
    from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
    from magi_attention.comm.primitive.grpcoll._handle import GrpCollIntraHandle
    from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
    from magi_attention.functional.dsa_comm import (
        DsaPayloadKind,
        DsaTypedPayload,
        build_dsa_comm_plan,
        materialize_dsa_comm_plan,
        start_dsa_group_cast,
    )
    from magi_attention.meta.collection.dsa_meta import DsaFragmentSpec
    from magi_attention.meta.solver.dsa_dispatch import build_dsa_dispatch_plan

    config = GrpCollConfig(
        num_sms=NATIVE_NUM_SMS,
        nvl_chunk_size=8,
        nvl_buffer_size=256,
        rdma_chunk_size=8,
        rdma_buffer_size=256,
        num_nvl_bytes=NATIVE_NUM_NVL_BYTES,
        num_rdma_bytes=NATIVE_NUM_RDMA_BYTES,
    )
    grpcoll_buffer_mgr.initialize(group=group, config=config)
    buffers = {}
    for name in NATIVE_BUFFER_NAMES:
        buffers[name] = grpcoll_buffer_mgr.get_buffer(group, name)

    fragments = [
        [DsaFragmentSpec(0, current * 128, (current + 1) * 128)]
        for current in range(WORLD_SIZE)
    ]
    plan = build_dsa_dispatch_plan(
        [WORLD_SIZE * 128], fragments, compress_ratio=4, policy="sequential"
    )
    comm_plan = build_dsa_comm_plan(plan, rank, group)
    device_plan = materialize_dsa_comm_plan(comm_plan, torch.cuda.current_device())
    meta = comm_plan.compressed_kv
    device_map = device_plan.compressed_kv
    payload = DsaTypedPayload(
        DsaPayloadKind.COMPRESSED_KV,
        torch.zeros(
            (meta.local_row_count, 512),
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        ),
    )
    work = start_dsa_group_cast(payload, meta, device_map=device_map, async_op=True)
    work.wait()
    handle = work.native_handle_dict.get("group_cast")
    _require(
        isinstance(handle, GrpCollIntraHandle),
        f"native probe loaded {type(handle).__name__}, expected GrpCollIntraHandle",
    )
    rdma_counts = {
        int(buffer.runtime.get_num_rdma_ranks()) for buffer in buffers.values()
    }
    _require(rdma_counts == {1}, f"unexpected RDMA rank counts {rdma_counts}")
    return {
        "native_handle": type(handle).__name__,
        "num_rdma_ranks": 1,
        "native_buffers": sorted(buffers),
        "native_config": asdict(config),
    }


def _distributed_preflight(
    torch,
    dist,
    rank: int,
    local_rank: int,
    group,
    *,
    run_id: str,
    mode: str,
    expected_revision: str,
    expected_image_id: str,
) -> dict[str, Any]:
    _require(
        expected_image_id.startswith("sha256:") and len(expected_image_id) == 71,
        "--expected-image-id must be an immutable sha256:... image ID",
    )
    source: Any = None
    if rank == 0:
        source = _source_preflight(expected_revision)
    source_values = [source]
    dist.broadcast_object_list(source_values, src=0, group=group)
    source = source_values[0]
    _require(isinstance(source, dict), "immutable source evidence is missing")
    installed_wheel = _installed_wheel_evidence()
    _require(
        os.environ.get("MAGI_ATTENTION_NATIVE_GRPCOLL") == "1",
        "MAGI_ATTENTION_NATIVE_GRPCOLL=1 is mandatory; A2AV fallback is forbidden",
    )
    _require(
        os.environ.get("MAGI_ATTENTION_HIERARCHICAL_COMM", "0") != "1",
        "hierarchical communication is outside the frozen single-node path",
    )
    _require(
        "NVSHMEM_SYMMETRIC_SIZE" not in os.environ,
        "NVSHMEM_SYMMETRIC_SIZE must be unset for the intranode path",
    )
    _require(
        int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8")) > 1,
        "CUDA_DEVICE_MAX_CONNECTIONS=1 prevents the required overlap",
    )

    from magi_attention.api import (
        DsaOverlapConfig,
        DsaPackedMeta,
        MagiDSAConfig,
        MagiDSAInput,
        MagiDSARuntimeMgr,
        calc_dsa,
    )
    from magi_attention.meta.solver.dsa_calibration import calibration_manifest

    del (
        DsaOverlapConfig,
        DsaPackedMeta,
        MagiDSAConfig,
        MagiDSAInput,
        MagiDSARuntimeMgr,
        calc_dsa,
    )
    capability = list(torch.cuda.get_device_capability())
    name = torch.cuda.get_device_name()
    _require(
        capability == [10, 3], f"rank {rank} is capability {capability}, not SM103"
    )
    _require("B300" in name, f"rank {rank} device {name!r} is not B300")
    device_record = {
        "rank": rank,
        "local_rank": local_rank,
        "name": name,
        "capability": capability,
        "uuid": str(torch.cuda.get_device_properties(local_rank).uuid),
        "hostname": socket.gethostname(),
    }
    devices: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(devices, device_record, group=group)
    _require(len({item["uuid"] for item in devices}) == WORLD_SIZE, "GPU UUIDs repeat")
    _require(len({item["hostname"] for item in devices}) == 1, "ranks span hosts")

    topology = ""
    topology_ok = False
    if rank == 0:
        topology = _run_command(("nvidia-smi", "topo", "-m"))
        topology_ok = _parse_nvlink_topology(topology)
    values: list[Any] = [topology_ok, topology]
    dist.broadcast_object_list(values, src=0, group=group)
    topology_ok, topology = bool(values[0]), str(values[1])
    _require(topology_ok, "nvidia-smi topology is not all-pairs NVLink")

    flash_mla_sm100: Any = None
    if rank == 0:
        flash_mla_sm100 = _flash_mla_sm100_evidence()
    flash_values = [flash_mla_sm100]
    dist.broadcast_object_list(flash_values, src=0, group=group)
    flash_mla_sm100 = flash_values[0]
    _require(
        isinstance(flash_mla_sm100, dict) and flash_mla_sm100.get("verified") is True,
        "FlashMLA sm_100 cubin evidence is missing",
    )

    native = _native_probe(torch, dist, rank, group)
    calibration = calibration_manifest()
    environment = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "mode": mode,
        **source,
        "image_id": expected_image_id,
        "world_size": WORLD_SIZE,
        "cp_size": CP_SIZE,
        "global_tokens": GLOBAL_TOKENS,
        "target_tokens_per_rank": TARGET_TOKENS_PER_RANK,
        "dtype": DTYPE_NAME,
        "kernel_backend": KERNEL_BACKEND,
        "gpu_count": WORLD_SIZE,
        "gpu_capability": [10, 3],
        "devices": devices,
        "nvlink_all_pairs": topology_ok,
        "nvidia_smi_topology": topology,
        "native_backend": True,
        "native_handle": native["native_handle"],
        "num_rdma_ranks": native["num_rdma_ranks"],
        "native_buffers": native["native_buffers"],
        "num_sms": NATIVE_NUM_SMS,
        "num_nvl_bytes": NATIVE_NUM_NVL_BYTES,
        "num_rdma_bytes": NATIVE_NUM_RDMA_BYTES,
        "native_config": native["native_config"],
        "nvshmem_symmetric_size": "unset",
        "cuda_device_max_connections": int(
            os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8")
        ),
        "calibration_id": calibration["calibration_id"],
        "calibration_target": calibration["target"],
        "calibration": calibration,
        "hf_revision": HF_REVISION,
        "hf_config_sha256": HF_CONFIG_SHA256,
        "flash_mla_sm100_cubin": flash_mla_sm100,
        "installed_wheel": installed_wheel,
        "model_config": {
            "hidden_size": HIDDEN_SIZE,
            "q_lora_rank": Q_LORA_RANK,
            "softmax_scale": SOFTMAX_SCALE,
            "num_heads": 64,
            "kv_dim": 512,
            "indexer_dim": 128,
            "topk": 512,
        },
        "timing_protocol": {
            "compile_iters": COMPILE_ITERS,
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
            "aggregation": "median_per_pack_rank_then_rank_max_then_mean_over_packs",
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in (
                "nvidia-nvshmem-cu13",
                "nvidia-cudnn-frontend",
                "nvidia-cutlass-dsl",
            )
        },
        "modules": [
            _module_evidence("magi_attention.magi_attn_comm"),
            _module_evidence("magi_attention.magi_attn_ext"),
            _module_evidence("flash_mla"),
        ],
    }
    return environment


def _packed_meta(torch, lengths: Sequence[int]):
    from magi_attention.api import DsaPackedMeta

    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + int(length))
    return DsaPackedMeta(torch.tensor(bounds, dtype=torch.int32))


def _local_global_rows(torch, plan, rank: int):
    local = torch.empty(plan.ranks[rank].token_count, dtype=torch.int64)
    entries = sorted(
        (entry for entry in plan.restore_map if entry.rank == rank),
        key=lambda entry: entry.local_begin,
    )
    cursor = 0
    for entry in entries:
        _require(entry.local_begin == cursor, "restore map local order has a hole")
        local[entry.local_begin : entry.local_end] = torch.arange(
            entry.global_begin, entry.global_end, dtype=torch.int64
        )
        cursor = entry.local_end
    _require(cursor == local.numel(), "restore map does not cover local rows")
    return local


def _deterministic_rows(
    torch,
    global_rows,
    trailing_shape: Sequence[int],
    *,
    seed: int,
    dtype,
    device,
):
    """Generate a tensor whose logical row values are policy-independent."""

    row_values = [int(value) for value in global_rows.tolist()]
    output = torch.empty(
        (len(row_values), *tuple(trailing_shape)), dtype=dtype, device=device
    )
    by_chunk: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for local_index, global_index in enumerate(row_values):
        chunk, offset = divmod(global_index, INPUT_CHUNK_ROWS)
        by_chunk[chunk].append((local_index, offset))
    generator = torch.Generator(device=device)
    for chunk, positions in by_chunk.items():
        generator.manual_seed(seed + chunk)
        source = torch.randn(
            (INPUT_CHUNK_ROWS, *tuple(trailing_shape)),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        destination_index = torch.tensor(
            [item[0] for item in positions], dtype=torch.int64, device=device
        )
        source_index = torch.tensor(
            [item[1] for item in positions], dtype=torch.int64, device=device
        )
        output.index_copy_(0, destination_index, source.index_select(0, source_index))
    return output


def _make_local_input(torch, config, packed_meta, global_rows, pack_index: int):
    from magi_attention.api import MagiDSAInput

    device = torch.cuda.current_device()
    base = INPUT_SEED + pack_index * 1_000_003
    tensors = {
        "x": _deterministic_rows(
            torch,
            global_rows,
            (config.hidden_size,),
            seed=base + 10_000_019,
            dtype=torch.bfloat16,
            device=device,
        ),
        "qr": _deterministic_rows(
            torch,
            global_rows,
            (config.q_lora_rank,),
            seed=base + 20_000_033,
            dtype=torch.bfloat16,
            device=device,
        ),
        "q": _deterministic_rows(
            torch,
            global_rows,
            (config.num_heads, config.kv_dim),
            seed=base + 30_000_047,
            dtype=torch.bfloat16,
            device=device,
        ),
        "latent_kv": _deterministic_rows(
            torch,
            global_rows,
            (config.kv_dim,),
            seed=base + 40_000_063,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    sink_generator = torch.Generator(device=device)
    sink_generator.manual_seed(base + 50_000_081)
    sink = torch.randn(
        config.num_heads,
        dtype=torch.float32,
        device=device,
        generator=sink_generator,
    )
    for tensor in (*tensors.values(), sink):
        tensor.requires_grad_(True)
    dsa_input = MagiDSAInput(
        x=tensors["x"],
        qr=tensors["qr"],
        q=tensors["q"],
        latent_kv=tensors["latent_kv"],
        sink=sink,
        packed_meta=packed_meta,
    )
    output_gradient = _deterministic_rows(
        torch,
        global_rows,
        (config.num_heads, config.kv_dim),
        seed=GRAD_SEED + pack_index * 1_000_003,
        dtype=torch.bfloat16,
        device=device,
    )
    return dsa_input, output_gradient


def _clear_gradients(dsa_input, runtime) -> None:
    for tensor in (
        dsa_input.x,
        dsa_input.qr,
        dsa_input.q,
        dsa_input.latent_kv,
        dsa_input.sink,
    ):
        tensor.grad = None
    runtime.zero_grad(set_to_none=True)


def _run_one(torch, dsa_input, output_gradient, runtime, *, telemetry: bool):
    from magi_attention.api import calc_dsa
    from magi_attention.dsa import DsaTelemetry

    _clear_gradients(dsa_input, runtime)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    recorder = DsaTelemetry() if telemetry else None
    start.record()
    if recorder is None:
        output, kl_loss = calc_dsa(dsa_input, runtime)
        loss = torch.sum(output * output_gradient, dtype=torch.float32) / GLOBAL_TOKENS
        loss = loss + kl_loss
        loss.backward()
    else:
        with recorder:
            output, kl_loss = calc_dsa(dsa_input, runtime)
            loss = (
                torch.sum(output * output_gradient, dtype=torch.float32) / GLOBAL_TOKENS
                + kl_loss
            )
            loss.backward()
    end.record()
    torch.cuda.synchronize()
    e2e_ms = float(start.elapsed_time(end))
    phases = {} if recorder is None else recorder.durations_ms(synchronize=False)
    finite = bool(torch.isfinite(output).all()) and bool(torch.isfinite(kl_loss))
    return output, kl_loss, loss, e2e_ms, phases, finite


def _canonical_owner_tensor(torch, dist, tensor, global_rows, rank: int, group):
    """Redistribute policy-local rows to fixed contiguous global-row shards."""

    _require(tensor is not None, "canonical correctness tensor is missing")
    rows = global_rows.to(device=tensor.device, dtype=torch.int64)
    _require(rows.numel() == tensor.size(0), "global-row/tensor length mismatch")
    order = torch.argsort(rows)
    send_rows = rows.index_select(0, order).contiguous()
    send_tensor = tensor.detach().index_select(0, order).contiguous()
    destinations = torch.div(send_rows, TARGET_TOKENS_PER_RANK, rounding_mode="floor")
    _require(
        bool(((destinations >= 0) & (destinations < WORLD_SIZE)).all().item()),
        "global row maps outside the canonical CP8 partition",
    )
    send_counts_tensor = torch.bincount(destinations, minlength=WORLD_SIZE).to(
        dtype=torch.int64
    )
    receive_counts_tensor = torch.empty_like(send_counts_tensor)
    dist.all_to_all_single(receive_counts_tensor, send_counts_tensor, group=group)
    send_counts = [int(value) for value in send_counts_tensor.cpu().tolist()]
    receive_counts = [int(value) for value in receive_counts_tensor.cpu().tolist()]
    receive_count = sum(receive_counts)
    _require(
        receive_count == TARGET_TOKENS_PER_RANK,
        f"rank {rank} canonical shard has {receive_count} rows",
    )
    received_rows = torch.empty(receive_count, dtype=torch.int64, device=tensor.device)
    received_tensor = torch.empty(
        (receive_count, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    dist.all_to_all_single(
        received_rows,
        send_rows,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    dist.all_to_all_single(
        received_tensor,
        send_tensor,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    canonical_order = torch.argsort(received_rows)
    expected_rows = torch.arange(
        rank * TARGET_TOKENS_PER_RANK,
        (rank + 1) * TARGET_TOKENS_PER_RANK,
        dtype=torch.int64,
        device=tensor.device,
    )
    _require(
        bool(
            torch.equal(received_rows.index_select(0, canonical_order), expected_rows)
        ),
        f"rank {rank} canonical shard is not an exact global-row partition",
    )
    canonical = received_tensor.index_select(0, canonical_order)
    del rows, order, send_rows, send_tensor, destinations
    del send_counts_tensor, receive_counts_tensor, received_rows, received_tensor
    return canonical


def _assert_close_elementwise(torch, name: str, actual, expected) -> dict[str, Any]:
    _require(actual.shape == expected.shape, f"{name} shape mismatch")
    _require(actual.dtype == expected.dtype, f"{name} dtype mismatch")
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    block = 1 << 22
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for begin in range(0, actual_flat.numel(), block):
        actual_chunk = actual_flat[begin : begin + block]
        expected_chunk = expected_flat[begin : begin + block]
        try:
            torch.testing.assert_close(
                actual_chunk,
                expected_chunk,
                rtol=CORRECTNESS_RTOL,
                atol=CORRECTNESS_ATOL,
                equal_nan=False,
                msg=lambda message: f"{name} chunk at element {begin}: {message}",
            )
        except AssertionError as error:
            raise RuntimeError(f"elementwise correctness failed: {error}") from error
        difference = (actual_chunk.float() - expected_chunk.float()).abs()
        if difference.numel():
            maximum_absolute = max(maximum_absolute, float(difference.max().item()))
            denominator = expected_chunk.float().abs().clamp_min(CORRECTNESS_ATOL)
            maximum_relative = max(
                maximum_relative, float((difference / denominator).max().item())
            )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "numel": actual.numel(),
        "max_abs_diff": maximum_absolute,
        "max_relative_diff_at_atol_floor": maximum_relative,
        "elementwise_assert_close": True,
    }


def _assert_replicated(torch, dist, name: str, tensor, group) -> None:
    if tensor is None:
        return
    rank_zero = tensor.detach().clone()
    dist.broadcast(rank_zero, src=0, group=group)
    _assert_close_elementwise(torch, f"{name}/replicated_across_cp", tensor, rank_zero)


def _policy_correctness_against_sequential(
    torch,
    dist,
    *,
    case: CaseSpec,
    packed_meta,
    pack_index: int,
    rank: int,
    group,
    candidate_global_rows,
    candidate_input,
    candidate_output,
    candidate_kl,
    candidate_runtime,
) -> dict[str, Any]:
    """Run and compare a fresh sequential reference outside timed regions."""

    baseline_case = CASE_BY_ID[f"r{case.ratio}-sequential-00"]
    baseline_runtime = _make_runtime(torch, baseline_case, group)
    baseline_plan = baseline_runtime.get_dispatch_plan(packed_meta)
    baseline_global_rows = _local_global_rows(torch, baseline_plan, rank)
    baseline_input, baseline_output_gradient = _make_local_input(
        torch,
        baseline_runtime.config,
        packed_meta,
        baseline_global_rows,
        pack_index,
    )
    baseline_output, baseline_kl, baseline_loss, _, _, baseline_finite = _run_one(
        torch,
        baseline_input,
        baseline_output_gradient,
        baseline_runtime,
        telemetry=False,
    )
    _require(baseline_finite, "sequential correctness reference is non-finite")

    candidate_kl_global = candidate_kl.detach().float().clone()
    baseline_kl_global = baseline_kl.detach().float().clone()
    dist.all_reduce(candidate_kl_global, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(baseline_kl_global, op=dist.ReduceOp.SUM, group=group)
    metrics: dict[str, Any] = {
        "kl": _assert_close_elementwise(
            torch, "kl", candidate_kl_global, baseline_kl_global
        )
    }

    sharded = {
        "output": (candidate_output, baseline_output),
        "dx": (candidate_input.x.grad, baseline_input.x.grad),
        "dqr": (candidate_input.qr.grad, baseline_input.qr.grad),
        "dq": (candidate_input.q.grad, baseline_input.q.grad),
        "dkv": (
            candidate_input.latent_kv.grad,
            baseline_input.latent_kv.grad,
        ),
    }
    for name, (candidate_tensor, baseline_tensor) in sharded.items():
        _require(
            (candidate_tensor is None) == (baseline_tensor is None),
            f"{name} gradient presence differs",
        )
        if candidate_tensor is None:
            metrics[name] = {
                "none": True,
                "none_presence_equal": True,
                "elementwise_assert_close": True,
            }
            continue
        candidate_canonical = _canonical_owner_tensor(
            torch, dist, candidate_tensor, candidate_global_rows, rank, group
        )
        baseline_canonical = _canonical_owner_tensor(
            torch, dist, baseline_tensor, baseline_global_rows, rank, group
        )
        metrics[name] = _assert_close_elementwise(
            torch, name, candidate_canonical, baseline_canonical
        )
        del candidate_canonical, baseline_canonical

    candidate_parameters = dict(candidate_runtime.dsa_module.named_parameters())
    baseline_parameters = dict(baseline_runtime.dsa_module.named_parameters())
    _require(
        candidate_parameters.keys() == baseline_parameters.keys(),
        "candidate and sequential parameter sets differ",
    )
    replicated = {
        "d_sink": (candidate_input.sink.grad, baseline_input.sink.grad),
        **{
            f"parameter:{name}": (
                candidate_parameters[name].grad,
                baseline_parameters[name].grad,
            )
            for name in candidate_parameters
        },
    }
    for name, (candidate_tensor, baseline_tensor) in replicated.items():
        _require(
            (candidate_tensor is None) == (baseline_tensor is None),
            f"{name} gradient presence differs",
        )
        if candidate_tensor is None:
            metrics[name] = {"none": True, "elementwise_assert_close": True}
            continue
        _assert_replicated(torch, dist, f"{name}/candidate", candidate_tensor, group)
        _assert_replicated(torch, dist, f"{name}/baseline", baseline_tensor, group)
        metrics[name] = _assert_close_elementwise(
            torch, name, candidate_tensor, baseline_tensor
        )

    del baseline_output, baseline_kl, baseline_loss
    del baseline_input, baseline_output_gradient, baseline_global_rows
    del baseline_runtime, baseline_plan
    torch.cuda.empty_cache()
    return metrics


def _plan_features(plan, rank: int) -> dict[str, int]:
    rank_plan = plan.ranks[rank]
    local_blocks = len(rank_plan.compressed_block_ids)
    total_blocks = len(plan.compressed_blocks)
    return {
        "token_count": rank_plan.token_count,
        "indexer_cost": rank_plan.indexer_cost,
        "fragment_count": rank_plan.fragment_count,
        "window_transfer_rows": sum(
            route.row_count
            for route in plan.window_transfers
            if route.destination_rank == rank
        ),
        "overlap_transfer_rows": sum(
            route.row_count
            for route in plan.overlap_transfers
            if route.destination_rank == rank
        ),
        "compressed_owner_send_rows": local_blocks * (WORLD_SIZE - 1),
        "compressed_remote_receive_rows": total_blocks - local_blocks,
    }


def _cache_snapshot() -> set[str]:
    from magi_attention.kernel.cutedsl import dsa_pack

    snapshot: set[str] = set()
    for name in ("_copy_compile_cache", "_remap_compile_cache", "_csr_compile_cache"):
        cache = getattr(dsa_pack, name, {})
        snapshot.update(f"{name}:{key!r}" for key in cache)
    return snapshot


def _contains_architecture(value: Any, architecture: tuple[int, int]) -> bool:
    if value == architecture:
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_architecture(item, architecture) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_architecture(key, architecture)
            or _contains_architecture(item, architecture)
            for key, item in value.items()
        )
    return False


def _cache_key_evidence() -> list[dict[str, Any]]:
    from magi_attention.kernel.cutedsl import dsa_pack

    evidence: list[dict[str, Any]] = []
    architecture = (10, 3)
    for name in ("_copy_compile_cache", "_remap_compile_cache", "_csr_compile_cache"):
        cache = getattr(dsa_pack, name, {})
        for key in cache:
            _require(
                _contains_architecture(key, architecture),
                f"{name} cache key {key!r} is not SM103-specific",
            )
            evidence.append(
                {
                    "cache": name,
                    "key": repr(key),
                    "architecture": list(architecture),
                }
            )
    _require(evidence, "compile pass produced no dsa_pack cache keys")
    return sorted(evidence, key=lambda item: (item["cache"], item["key"]))


def _make_config(ratio: int):
    from magi_attention.api import MagiDSAConfig

    return MagiDSAConfig(
        compress_ratio=ratio,
        hidden_size=HIDDEN_SIZE,
        q_lora_rank=Q_LORA_RANK,
        softmax_scale=SOFTMAX_SCALE,
        backend=KERNEL_BACKEND,
    )


def _make_runtime(torch, case: CaseSpec, group):
    from magi_attention.api import DsaOverlapConfig, MagiDSARuntimeMgr

    torch.manual_seed(MODEL_SEED + case.ratio)
    torch.cuda.manual_seed_all(MODEL_SEED + case.ratio)
    runtime = MagiDSARuntimeMgr(
        _make_config(case.ratio),
        cp_group=group,
        dispatch_policy=case.policy,
        overlap_config=DsaOverlapConfig(
            compressed_cast_indexer=case.compressed_cast_indexer,
            dki_reduce_sparse_backward=case.dki_reduce_sparse_backward,
        ),
    ).cuda()
    runtime.train()
    return runtime


def _merge_rank_artifacts(
    run_dir: Path, suffix: str, world_size: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = run_dir / "_rank" / f"rank{rank}.{suffix}.jsonl"
        _require(path.is_file(), f"rank {rank} did not produce {suffix} evidence")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    records.sort(
        key=lambda item: (
            item.get("case_id", ""),
            item.get("pack_index", -1),
            item.get("rank", -1),
            item.get("iteration", -1),
        )
    )
    write_jsonl(run_dir / f"{suffix}.jsonl", records)
    return records


def _claim_fresh_run_dir(run_dir: Path, environment: Mapping[str, Any]) -> None:
    """Atomically reject reuse of canonical or partially written run evidence."""

    run_dir.mkdir(parents=True, exist_ok=True)
    canonical = (
        "environment.json",
        "packs.json",
        "report.schema.json",
        "raw_timing.jsonl",
        "plans.jsonl",
        "correctness.jsonl",
        "calibration.json",
        "summary.json",
        "validation.json",
        "artifact_manifest.sha256",
    )
    stale = [name for name in canonical if (run_dir / name).exists()]
    if (run_dir / "_rank").exists():
        stale.append("_rank")
    _require(not stale, f"run directory contains prior canonical evidence: {stale}")
    claim = run_dir / ".magi-dsa-run-claim.json"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"run directory is already claimed: {claim}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": environment["run_id"],
                "mode": environment["mode"],
                "revision": environment["revision"],
                "image_id": environment["image_id"],
                "calibration_id": environment["calibration_id"],
                "claimed_at_utc": _utc_now(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fit_calibration(
    run_dir: Path,
    environment: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    plan_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    plan_by_key = {
        (item["case_id"], item["pack_index"], item["rank"]): item
        for item in plan_records
    }
    timing: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for item in raw_records:
        timing[(item["case_id"], item["pack_index"], item["rank"])].append(
            float(item["e2e_ms"])
        )
    weights: dict[str, dict[str, float]] = {}
    fits: dict[str, Any] = {}
    for ratio in (0, 4, 128):
        rows: list[list[float]] = []
        targets: list[float] = []
        for key, values in timing.items():
            case_id, _, _ = key
            if CASE_BY_ID[case_id].ratio != ratio:
                continue
            plan = plan_by_key[key]
            rows.append([float(plan["features"][name]) for name in PLAN_FEATURE_NAMES])
            targets.append(float(np.median(np.asarray(values, dtype=np.float64))))
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        scales = np.maximum(np.linalg.norm(x, axis=0), 1.0)
        normalized = x / scales
        coefficients, *_ = np.linalg.lstsq(normalized, y, rcond=None)
        coefficients = np.maximum(coefficients, 0.0) / scales
        predicted = x @ coefficients
        residual = y - predicted
        denominator = float(np.sum((y - y.mean()) ** 2))
        r_squared = (
            1.0 - float(np.sum(residual**2)) / denominator if denominator else 1.0
        )
        ratio_weights = dict(zip(PLAN_FEATURE_NAMES, coefficients.tolist()))
        weights[str(ratio)] = {
            "token_weight": ratio_weights["token_count"],
            "indexer_weight": ratio_weights["indexer_cost"] if ratio == 4 else 0.0,
            "fragment_overhead": ratio_weights["fragment_count"],
            "window_transfer_weight": ratio_weights["window_transfer_rows"],
            "overlap_transfer_weight": (
                ratio_weights["overlap_transfer_rows"] if ratio == 4 else 0.0
            ),
            "compressed_owner_send_weight": (
                ratio_weights["compressed_owner_send_rows"] if ratio else 0.0
            ),
            "compressed_remote_receive_weight": (
                ratio_weights["compressed_remote_receive_rows"] if ratio else 0.0
            ),
        }
        fits[str(ratio)] = {
            "observation_count": len(y),
            "method": "column-normalized least-squares with non-negative projection",
            "r_squared": r_squared,
            "rmse_ms": float(np.sqrt(np.mean(residual**2))),
            "max_abs_residual_ms": float(np.max(np.abs(residual))),
            "feature_names": list(PLAN_FEATURE_NAMES),
        }
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "target": "b300-sm103",
        "created_at_utc": _utc_now(),
        "source_run_id": environment["run_id"],
        "source_revision": environment["revision"],
        "source_image_id": environment["image_id"],
        "packs_sha256": read_json(run_dir / "packs.json")["packs_sha256"],
        "weights": weights,
        "fits": fits,
    }
    atomic_write_json(run_dir / "calibration.json", calibration)
    return calibration


def _run_distributed(args: argparse.Namespace) -> int:
    torch, dist, rank, local_rank, group = _initialize_distributed()
    run_dir: Path = args.run_dir.resolve()
    packs_payload = read_json(args.packs.resolve())
    packs = validate_packs(packs_payload)
    mode = args.command
    cases = {
        "calibrate": CALIBRATION_CASES,
        "measure": MEASURE_CASES,
        "profile": PROFILE_CASES,
    }[mode]
    artifact_mode = "calibration" if mode == "calibrate" else mode
    environment = _distributed_preflight(
        torch,
        dist,
        rank,
        local_rank,
        group,
        run_id=args.run_id,
        mode=artifact_mode,
        expected_revision=args.expected_revision,
        expected_image_id=args.expected_image_id,
    )
    if mode in ("measure", "profile"):
        _require(
            environment["calibration_target"] == "b300-sm103",
            "formal measure/profile refuses bootstrap calibration coefficients",
        )
    if rank == 0:
        _claim_fresh_run_dir(run_dir, environment)
        (run_dir / "_rank").mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "environment.json", environment)
        atomic_write_json(run_dir / "packs.json", packs_payload)
        schema_source = SCRIPT_DIR / "report.schema.json"
        (run_dir / "report.schema.json").write_bytes(schema_source.read_bytes())
    dist.barrier(group=group)

    raw_records: list[dict[str, Any]] = []
    plan_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    selected_packs = [packs[PROFILE_PACK_INDEX]] if mode == "profile" else list(packs)

    for case in cases:
        runtime = _make_runtime(torch, case, group)
        config = runtime.config
        for pack in selected_packs:
            pack_index = int(pack["index"])
            packed_meta = _packed_meta(torch, pack["lengths"])
            plan = runtime.get_dispatch_plan(packed_meta)
            global_rows = _local_global_rows(torch, plan, rank)
            dsa_input, output_gradient = _make_local_input(
                torch, config, packed_meta, global_rows, pack_index
            )
            plan_record = {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "mode": artifact_mode,
                "case_id": case.case_id,
                "pack_index": pack_index,
                "pack_sha256": pack["sha256"],
                "rank": rank,
                "plan_sha256": plan.plan_hash,
                "calibration_id": runtime.solver_calibration_id,
                "features": _plan_features(plan, rank),
            }
            plan_records.append(plan_record)

            # The compile pass is deliberately separate from correctness,
            # warm-up and measurement. It also materializes every plan map.
            for _ in range(COMPILE_ITERS):
                output, kl_loss, loss, _, _, finite = _run_one(
                    torch, dsa_input, output_gradient, runtime, telemetry=False
                )
                _require(finite, f"non-finite compile result for {case.case_id}")
                del output, kl_loss, loss
            plan_record["dsa_pack_cache_keys"] = _cache_key_evidence()

            if mode != "profile":
                output, kl_loss, loss, _, _, finite = _run_one(
                    torch, dsa_input, output_gradient, runtime, telemetry=False
                )
                _require(finite, f"non-finite correctness result for {case.case_id}")
                if case.policy == "sequential" and case.overlap_code == "00":
                    local_comparisons = {
                        "baseline_reference": True,
                        "required_sharded_fields": [
                            "output",
                            "dx",
                            "dqr",
                            "dq",
                            "dkv",
                        ],
                    }
                else:
                    local_comparisons = _policy_correctness_against_sequential(
                        torch,
                        dist,
                        case=case,
                        packed_meta=packed_meta,
                        pack_index=pack_index,
                        rank=rank,
                        group=group,
                        candidate_global_rows=global_rows,
                        candidate_input=dsa_input,
                        candidate_output=output,
                        candidate_kl=kl_loss,
                        candidate_runtime=runtime,
                    )
                rank_comparisons: list[Any] = [None] * WORLD_SIZE
                dist.all_gather_object(rank_comparisons, local_comparisons, group=group)
                if rank == 0:
                    correctness_records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": args.run_id,
                            "mode": artifact_mode,
                            "case_id": case.case_id,
                            "calibration_id": runtime.solver_calibration_id,
                            "pack_index": pack_index,
                            "pack_sha256": pack["sha256"],
                            "baseline_case_id": f"r{case.ratio}-sequential-00",
                            "pass": True,
                            "rtol": CORRECTNESS_RTOL,
                            "atol": CORRECTNESS_ATOL,
                            "native_backend": True,
                            "method": (
                                "owner-local all_to_all_single to fixed contiguous "
                                "24576-row/rank global order, then chunked "
                                "torch.testing.assert_close"
                            ),
                            "required_sharded_fields": [
                                "output",
                                "dx",
                                "dqr",
                                "dq",
                                "dkv",
                            ],
                            "rank_comparisons": rank_comparisons,
                            "digests": {
                                "used_for_acceptance": False,
                                "note": "elementwise rank_comparisons are authoritative",
                            },
                        }
                    )
                del output, kl_loss, loss

            for _ in range(WARMUP_ITERS):
                output, kl_loss, loss, _, _, finite = _run_one(
                    torch, dsa_input, output_gradient, runtime, telemetry=False
                )
                _require(finite, f"non-finite warm-up result for {case.case_id}")
                del output, kl_loss, loss
            torch.cuda.synchronize()
            dist.barrier(group=group)
            cache_before = _cache_snapshot()

            iterations = 1 if mode == "profile" else MEASURE_ITERS
            if mode == "profile":
                torch.cuda.cudart().cudaProfilerStart()
            for iteration in range(iterations):
                dist.barrier(group=group)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(
                    f"magi_dsa::profile::{case.case_id}::pack{pack_index}::"
                    f"rank{rank}::iter{iteration}"
                )
                output, kl_loss, loss, e2e_ms, phases, finite = _run_one(
                    torch, dsa_input, output_gradient, runtime, telemetry=True
                )
                torch.cuda.nvtx.range_pop()
                cache_after = _cache_snapshot()
                miss_delta = len(cache_after - cache_before)
                indexer_ms = sum(
                    float(phases.get(name, 0.0)) for name in INDEXER_PHASES
                )
                raw_records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": args.run_id,
                        "mode": artifact_mode,
                        "case_id": case.case_id,
                        "ratio": case.ratio,
                        "policy": case.policy,
                        "overlap_code": case.overlap_code,
                        "pack_index": pack_index,
                        "pack_sha256": pack["sha256"],
                        "plan_sha256": plan.plan_hash,
                        "calibration_id": runtime.solver_calibration_id,
                        "iteration": iteration,
                        "rank": rank,
                        "e2e_ms": e2e_ms,
                        "indexer_ms": indexer_ms,
                        "phases_ms": phases,
                        "overlap_windows_ms": {
                            name: float(phases.get(name, 0.0))
                            for name in (
                                "overlap_compressed_cast_indexer",
                                "overlap_dki_reduce_sparse_backward",
                            )
                        },
                        "native_backend": True,
                        "finite": finite,
                        "jit_cache_miss_delta": miss_delta,
                    }
                )
                _require(finite, f"non-finite measured result for {case.case_id}")
                _require(
                    miss_delta == 0,
                    f"JIT/cache miss during {case.case_id}/pack{pack_index}/iter{iteration}",
                )
                del output, kl_loss, loss
            if mode == "profile":
                torch.cuda.cudart().cudaProfilerStop()
            del dsa_input, output_gradient, global_rows
            torch.cuda.empty_cache()
        del runtime
        torch.cuda.empty_cache()

    rank_dir = run_dir / "_rank"
    write_jsonl(rank_dir / f"rank{rank}.raw_timing.jsonl", raw_records)
    write_jsonl(rank_dir / f"rank{rank}.plans.jsonl", plan_records)
    write_jsonl(rank_dir / f"rank{rank}.correctness.jsonl", correctness_records)
    dist.barrier(group=group)
    exit_code = 0
    if rank == 0:
        merged_raw = _merge_rank_artifacts(run_dir, "raw_timing", WORLD_SIZE)
        merged_plans = _merge_rank_artifacts(run_dir, "plans", WORLD_SIZE)
        _merge_rank_artifacts(run_dir, "correctness", WORLD_SIZE)
        if mode == "calibrate":
            _fit_calibration(run_dir, environment, merged_raw, merged_plans)
        result = validate_run(
            run_dir,
            mode=artifact_mode,
            expected_revision=args.expected_revision,
            expected_image_id=args.expected_image_id,
            write_summary=mode == "measure",
        )
        if mode == "profile":
            atomic_write_json(
                run_dir / "profile_metadata.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": args.run_id,
                    "revision": args.expected_revision,
                    "image_id": args.expected_image_id,
                    "pack_index": PROFILE_PACK_INDEX,
                    "pack_sha256": selected_packs[0]["sha256"],
                    "cases": [case.case_id for case in PROFILE_CASES],
                    "nvtx_prefix": "magi_dsa::profile::",
                    "note": (
                        "DsaTelemetry overlap_* values are launch-to-wait windows; "
                        "the .nsys-rep timeline is the authoritative proof of GPU overlap."
                    ),
                },
            )
        if mode == "measure" and not result["summary"]["all_gates_pass"]:
            exit_code = 2
    value = torch.tensor(
        [exit_code], dtype=torch.int32, device=torch.cuda.current_device()
    )
    dist.broadcast(value, src=0, group=group)
    exit_code = int(value.item())
    dist.barrier(group=group)
    from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr

    grpcoll_buffer_mgr.release_group(group)
    dist.destroy_process_group()
    return exit_code


def _common_distributed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--packs", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-image-id", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    packs = subparsers.add_parser(
        "packs", help="generate the frozen DatasetSampler packs"
    )
    packs.add_argument("--output", required=True, type=Path)
    packs.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / DATASET_RELATIVE_PATH
    )
    for name in ("calibrate", "measure", "profile"):
        child = subparsers.add_parser(name)
        _common_distributed_arguments(child)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "packs":
        payload = generate_packs(args.output, args.dataset)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "pack_count": len(payload["packs"]),
                    "packs_sha256": payload["packs_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return _run_distributed(args)


if __name__ == "__main__":
    raise SystemExit(main())
