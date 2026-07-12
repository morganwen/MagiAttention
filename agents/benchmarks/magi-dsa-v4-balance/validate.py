#!/usr/bin/env python3
"""Validate and summarize Magi_DSA V4 B300/CP8 benchmark artifacts.

This module deliberately uses only the Python standard library.  It is both
the post-run validator and the source of frozen benchmark constants consumed
by ``driver.py``.  The validator recomputes every reported performance gate
from rank-level raw records; a precomputed ``summary.json`` is never trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "magi-dsa-v4-balance/v1"
CALIBRATION_SCHEMA_VERSION = 1

WORLD_SIZE = 8
CP_SIZE = 8
GLOBAL_TOKENS = 196_608
TARGET_TOKENS_PER_RANK = 24_576
PACK_SEED = 42
PACK_NUM = 20
CHUNK_RATIO = 0.25
MEASURE_ITERS = 10
COMPILE_ITERS = 1
WARMUP_ITERS = 2
PROFILE_PACK_INDEX = 0

DATASET_RELATIVE_PATH = (
    "exps/dist_attn/benchmark/datasets/default/doc_length_distribution.csv"
)
DATASET_SHA256 = "67fe5f333fa3775ba547319e897b3781fda74ff5e19f3a38dc042a2f10414593"
HF_CONFIG_SHA256 = "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"

DTYPE_NAME = "torch.bfloat16"
KERNEL_BACKEND = "kernel"
NATIVE_NUM_SMS = 20
NATIVE_NUM_NVL_BYTES = 1 << 30
NATIVE_NUM_RDMA_BYTES = 0
NATIVE_BUFFER_NAMES = (
    "dsa_window_kv",
    "dsa_overlap_x",
    "dsa_compressed_kv",
    "dsa_compressed_ki",
    "dsa_replicated_gradient",
)

# DeepSeek-V4-Flash model-side dimensions from the pinned reference recipe.
HIDDEN_SIZE = 4096
Q_LORA_RANK = 1024
SOFTMAX_SCALE = 512**-0.5
INPUT_SEED = 20260712
MODEL_SEED = 20260713
GRAD_SEED = 20260714

INDEXER_PHASES = (
    "indexer_projection",
    "indexer_topk",
    "indexer_score_recompute",
    "indexer_backward",
)
OVERLAP_PHASES = (
    "overlap_compressed_cast_indexer",
    "overlap_dki_reduce_sparse_backward",
)
PLAN_FEATURE_NAMES = (
    "token_count",
    "indexer_cost",
    "fragment_count",
    "window_transfer_rows",
    "overlap_transfer_rows",
    "compressed_owner_send_rows",
    "compressed_remote_receive_rows",
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    ratio: int
    policy: str
    compressed_cast_indexer: bool
    dki_reduce_sparse_backward: bool

    @property
    def overlap_code(self) -> str:
        return (
            f"{int(self.compressed_cast_indexer)}{int(self.dki_reduce_sparse_backward)}"
        )


def _case(ratio: int, policy: str, overlap: str) -> CaseSpec:
    if ratio not in (0, 4, 128):
        raise ValueError(f"unsupported ratio {ratio}")
    if policy not in ("sequential", "balanced"):
        raise ValueError(f"unsupported policy {policy}")
    if overlap not in ("00", "01", "10", "11"):
        raise ValueError(f"invalid overlap code {overlap}")
    return CaseSpec(
        case_id=f"r{ratio}-{policy}-{overlap}",
        ratio=ratio,
        policy=policy,
        compressed_cast_indexer=overlap[0] == "1",
        dki_reduce_sparse_backward=overlap[1] == "1",
    )


MEASURE_CASES = (
    _case(4, "sequential", "00"),
    _case(4, "balanced", "00"),
    _case(4, "balanced", "01"),
    _case(4, "balanced", "10"),
    _case(4, "balanced", "11"),
    _case(128, "sequential", "00"),
    _case(128, "balanced", "11"),
    _case(0, "sequential", "00"),
    _case(0, "balanced", "11"),
)
CALIBRATION_CASES = (
    _case(0, "sequential", "00"),
    _case(0, "balanced", "00"),
    _case(4, "sequential", "00"),
    _case(4, "balanced", "00"),
    _case(128, "sequential", "00"),
    _case(128, "balanced", "00"),
)
PROFILE_CASES = (_case(4, "balanced", "11"),)
ALL_CASES = tuple(
    {
        case.case_id: case
        for case in (*MEASURE_CASES, *CALIBRATION_CASES, *PROFILE_CASES)
    }.values()
)
CASE_BY_ID = {case.case_id: case for case in ALL_CASES}


class ValidationError(RuntimeError):
    """An artifact violates the frozen performance contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_sha256(lengths: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes([int(length) for length in lengths]))


def pack_suite_sha256(packs: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {"index": int(pack["index"]), "sha256": str(pack["sha256"])} for pack in packs
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON artifact {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSONL artifact {path}: {error}") from error
    return records


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_packs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "pack schema mismatch")
    sampler = payload.get("sampler")
    _require(isinstance(sampler, dict), "packs.sampler must be an object")
    expected_sampler = {
        "seed": PACK_SEED,
        "pack_num": PACK_NUM,
        "chunk_ratio": CHUNK_RATIO,
        "pack_len": GLOBAL_TOKENS,
        "dataset_sha256": DATASET_SHA256,
        "is_binned": True,
    }
    for name, expected in expected_sampler.items():
        _require(
            sampler.get(name) == expected,
            f"packs sampler {name}={sampler.get(name)!r}, expected {expected!r}",
        )
    packs = payload.get("packs")
    _require(isinstance(packs, list), "packs.packs must be a list")
    _require(len(packs) == PACK_NUM, f"expected {PACK_NUM} packs, got {len(packs)}")
    normalized: list[dict[str, Any]] = []
    for index, pack in enumerate(packs):
        _require(isinstance(pack, dict), f"pack {index} must be an object")
        _require(pack.get("index") == index, f"pack index {index} is not canonical")
        lengths = pack.get("lengths")
        _require(isinstance(lengths, list) and lengths, f"pack {index} is empty")
        _require(
            all(isinstance(length, int) and length > 0 for length in lengths),
            f"pack {index} has an invalid sample length",
        )
        _require(
            sum(lengths) == GLOBAL_TOKENS,
            f"pack {index} has {sum(lengths)} tokens, expected {GLOBAL_TOKENS}",
        )
        expected_hash = pack_sha256(lengths)
        _require(
            pack.get("sha256") == expected_hash,
            f"pack {index} SHA256 mismatch",
        )
        normalized.append(
            {"index": index, "lengths": list(lengths), "sha256": expected_hash}
        )
    _require(
        payload.get("packs_sha256") == pack_suite_sha256(normalized),
        "pack suite SHA256 mismatch",
    )
    return normalized


def validate_environment(
    environment: Mapping[str, Any],
    *,
    mode: str,
    expected_revision: str | None = None,
    expected_image_id: str | None = None,
) -> None:
    _require(
        environment.get("schema_version") == SCHEMA_VERSION,
        "environment schema mismatch",
    )
    expected_values = {
        "world_size": WORLD_SIZE,
        "cp_size": CP_SIZE,
        "global_tokens": GLOBAL_TOKENS,
        "target_tokens_per_rank": TARGET_TOKENS_PER_RANK,
        "dtype": DTYPE_NAME,
        "kernel_backend": KERNEL_BACKEND,
        "native_backend": True,
        "native_handle": "GrpCollIntraHandle",
        "num_rdma_ranks": 1,
        "num_sms": NATIVE_NUM_SMS,
        "num_nvl_bytes": NATIVE_NUM_NVL_BYTES,
        "num_rdma_bytes": NATIVE_NUM_RDMA_BYTES,
        "worktree_clean": True,
        "gpu_capability": [10, 3],
        "gpu_count": WORLD_SIZE,
        "nvlink_all_pairs": True,
        "nvshmem_symmetric_size": "unset",
        "hf_config_sha256": HF_CONFIG_SHA256,
        "source_mode": "immutable_git_archive",
    }
    for name, expected in expected_values.items():
        _require(
            environment.get(name) == expected,
            f"environment {name}={environment.get(name)!r}, expected {expected!r}",
        )
    if expected_revision is not None:
        _require(
            environment.get("revision") == expected_revision,
            "environment revision does not match --expected-revision",
        )
    if expected_image_id is not None:
        _require(
            environment.get("image_id") == expected_image_id,
            "environment image ID does not match --expected-image-id",
        )
    buffers = environment.get("native_buffers")
    _require(
        sorted(buffers or []) == sorted(NATIVE_BUFFER_NAMES),
        "native dry-allocation did not cover all five buffers",
    )
    flash_evidence = environment.get("flash_mla_sm100_cubin")
    _require(
        isinstance(flash_evidence, dict)
        and flash_evidence.get("verified") is True
        and flash_evidence.get("architecture") == "sm_100"
        and isinstance(flash_evidence.get("objects"), list)
        and bool(flash_evidence["objects"]),
        "FlashMLA sm_100 cubin evidence is missing",
    )
    _require(
        environment.get("build_manifest_path") == "/opt/magi-dsa-build-manifest.json"
        and isinstance(environment.get("build_manifest_sha256"), str)
        and len(environment["build_manifest_sha256"]) == 64,
        "immutable build manifest evidence is missing",
    )
    installed_wheel = environment.get("installed_wheel")
    _require(
        isinstance(installed_wheel, dict)
        and installed_wheel.get("outside_source_checkout") is True
        and ".dist-info" in str(installed_wheel.get("distribution_path", ""))
        and isinstance(installed_wheel.get("wheel_record_sha256"), str)
        and len(installed_wheel["wheel_record_sha256"]) == 64,
        "installed-wheel import evidence is missing",
    )
    if mode in ("measure", "profile"):
        target = str(environment.get("calibration_target", ""))
        _require(
            target == "b300-sm103",
            f"measure/profile uses unfrozen calibration {target!r}",
        )


def _expected_cases(mode: str) -> tuple[CaseSpec, ...]:
    if mode == "measure":
        return MEASURE_CASES
    if mode == "calibration":
        return CALIBRATION_CASES
    if mode == "profile":
        return PROFILE_CASES
    raise ValidationError(f"unknown run mode {mode!r}")


def validate_raw_records(
    records: Sequence[Mapping[str, Any]],
    packs: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    expected_run_id: str | None = None,
    expected_calibration_id: str | None = None,
) -> None:
    cases = _expected_cases(mode)
    expected_case_ids = {case.case_id for case in cases}
    expected_pack_indices = (
        {PROFILE_PACK_INDEX} if mode == "profile" else set(range(PACK_NUM))
    )
    expected_iterations = {0} if mode == "profile" else set(range(MEASURE_ITERS))
    expected_count = (
        len(cases) * len(expected_pack_indices) * WORLD_SIZE * len(expected_iterations)
    )
    _require(
        len(records) == expected_count,
        f"raw timing has {len(records)} records, expected {expected_count}",
    )
    pack_hashes = {int(pack["index"]): str(pack["sha256"]) for pack in packs}
    keys: set[tuple[str, int, int, int]] = set()
    observed_cases: set[str] = set()
    for record in records:
        _require(record.get("schema_version") == SCHEMA_VERSION, "raw schema mismatch")
        _require(record.get("mode") == mode, "raw record mode mismatch")
        if expected_run_id is not None:
            _require(record.get("run_id") == expected_run_id, "raw run_id mismatch")
        if expected_calibration_id is not None:
            _require(
                record.get("calibration_id") == expected_calibration_id,
                "raw calibration_id mismatch",
            )
        case_id = str(record.get("case_id"))
        _require(case_id in expected_case_ids, f"unexpected case {case_id}")
        observed_cases.add(case_id)
        case = CASE_BY_ID[case_id]
        _require(record.get("ratio") == case.ratio, f"{case_id}: ratio mismatch")
        _require(record.get("policy") == case.policy, f"{case_id}: policy mismatch")
        _require(
            record.get("overlap_code") == case.overlap_code,
            f"{case_id}: overlap mismatch",
        )
        pack_index = record.get("pack_index")
        iteration = record.get("iteration")
        rank = record.get("rank")
        _require(
            pack_index in expected_pack_indices, f"invalid pack index {pack_index}"
        )
        _require(iteration in expected_iterations, f"invalid iteration {iteration}")
        _require(
            isinstance(rank, int) and 0 <= rank < WORLD_SIZE, f"invalid rank {rank}"
        )
        _require(
            record.get("pack_sha256") == pack_hashes[pack_index],
            f"pack {pack_index} hash mismatch in raw timing",
        )
        _require(
            isinstance(record.get("plan_sha256"), str)
            and len(record["plan_sha256"]) == 64,
            f"{case_id}: invalid plan SHA256",
        )
        for name in ("e2e_ms", "indexer_ms"):
            value = record.get(name)
            _require(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"{case_id}: invalid {name}={value!r}",
            )
        phases = record.get("phases_ms")
        _require(isinstance(phases, dict), f"{case_id}: phases_ms must be an object")
        computed_indexer_ms = sum(
            float(phases.get(name, 0.0)) for name in INDEXER_PHASES
        )
        _require(
            math.isclose(
                float(record["indexer_ms"]),
                computed_indexer_ms,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            f"{case_id}: indexer_ms omits a frozen Indexer phase",
        )
        if case.ratio == 4:
            _require(
                "indexer_projection" in phases,
                f"{case_id}: Indexer projection timing is missing",
            )
        _require(
            record.get("native_backend") is True, f"{case_id}: native backend false"
        )
        _require(record.get("finite") is True, f"{case_id}: non-finite result")
        _require(
            record.get("jit_cache_miss_delta") == 0,
            f"{case_id}: JIT/cache miss during measure",
        )
        key = (case_id, int(pack_index), int(rank), int(iteration))
        _require(key not in keys, f"duplicate raw timing key {key}")
        keys.add(key)
    _require(
        observed_cases == expected_case_ids, "raw timing case matrix is incomplete"
    )


def validate_plan_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    expected_run_id: str | None = None,
    expected_calibration_id: str | None = None,
) -> int:
    cases = _expected_cases(mode)
    expected_case_ids = {case.case_id for case in cases}
    expected_pack_indices = (
        {PROFILE_PACK_INDEX} if mode == "profile" else set(range(PACK_NUM))
    )
    pack_count = len(expected_pack_indices)
    expected_count = len(cases) * pack_count * WORLD_SIZE
    _require(
        len(records) == expected_count,
        f"plans.jsonl has {len(records)} records, expected {expected_count}",
    )
    seen: set[tuple[str, int, int]] = set()
    cache_key_count = 0
    for record in records:
        _require(record.get("schema_version") == SCHEMA_VERSION, "plan schema mismatch")
        _require(record.get("mode") == mode, "plan record mode mismatch")
        if expected_run_id is not None:
            _require(record.get("run_id") == expected_run_id, "plan run_id mismatch")
        if expected_calibration_id is not None:
            _require(
                record.get("calibration_id") == expected_calibration_id,
                "plan calibration_id mismatch",
            )
        key = (
            str(record.get("case_id")),
            int(record.get("pack_index", -1)),
            int(record.get("rank", -1)),
        )
        _require(key[0] in expected_case_ids, f"unexpected plan case {key[0]}")
        _require(key[1] in expected_pack_indices, f"invalid plan pack index {key[1]}")
        _require(0 <= key[2] < WORLD_SIZE, f"invalid plan rank {key[2]}")
        _require(key not in seen, f"duplicate plan record {key}")
        seen.add(key)
        features = record.get("features")
        _require(isinstance(features, dict), f"plan {key} has no features")
        for name in PLAN_FEATURE_NAMES:
            value = features.get(name)
            _require(
                isinstance(value, (int, float)) and value >= 0,
                f"plan {key} has invalid feature {name}",
            )
        _require(
            features.get("token_count", 0) <= TARGET_TOKENS_PER_RANK + 256,
            f"plan {key} exceeds token balance bound",
        )
        cache_keys = record.get("dsa_pack_cache_keys")
        _require(
            isinstance(cache_keys, list) and cache_keys,
            f"plan {key} has no compiled dsa_pack cache evidence",
        )
        for entry in cache_keys:
            _require(isinstance(entry, dict), f"plan {key} cache key is not an object")
            _require(
                entry.get("architecture") == [10, 3],
                f"plan {key} cache key is not compiled for SM103",
            )
            _require(
                isinstance(entry.get("cache"), str)
                and isinstance(entry.get("key"), str)
                and entry["key"],
                f"plan {key} has malformed cache evidence",
            )
        cache_key_count += len(cache_keys)
    expected_keys = {
        (case.case_id, pack_index, rank)
        for case in cases
        for pack_index in expected_pack_indices
        for rank in range(WORLD_SIZE)
    }
    _require(seen == expected_keys, "plan matrix is incomplete")
    return cache_key_count


def _rank_medians(
    records: Sequence[Mapping[str, Any]], case_id: str, pack_index: int, field: str
) -> list[float]:
    values: list[list[float]] = [[] for _ in range(WORLD_SIZE)]
    for record in records:
        if record["case_id"] == case_id and record["pack_index"] == pack_index:
            values[int(record["rank"])].append(float(record[field]))
    medians: list[float] = []
    for rank, rank_values in enumerate(values):
        _require(
            len(rank_values) == MEASURE_ITERS,
            f"{case_id} pack {pack_index} rank {rank} has "
            f"{len(rank_values)} iterations",
        )
        medians.append(float(statistics.median(rank_values)))
    return medians


def summarize_records(
    records: Sequence[Mapping[str, Any]], packs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Recompute the formal measure summary from raw rank timing."""

    summaries: dict[str, Any] = {}
    for case in MEASURE_CASES:
        pack_metrics: list[dict[str, Any]] = []
        for pack in packs:
            pack_index = int(pack["index"])
            e2e = _rank_medians(records, case.case_id, pack_index, "e2e_ms")
            indexer = _rank_medians(records, case.case_id, pack_index, "indexer_ms")

            def metric(values: Sequence[float]) -> tuple[float, float, float]:
                maximum = max(values)
                mean = statistics.fmean(values)
                imbalance = maximum / mean - 1.0 if mean else 0.0
                return maximum, mean, imbalance

            e2e_max, e2e_mean, e2e_imbalance = metric(e2e)
            idx_max, idx_mean, idx_imbalance = metric(indexer)
            pack_metrics.append(
                {
                    "pack_index": pack_index,
                    "pack_sha256": pack["sha256"],
                    "rank_median_e2e_ms": e2e,
                    "max_rank_e2e_ms": e2e_max,
                    "mean_rank_e2e_ms": e2e_mean,
                    "e2e_max_over_mean_minus_one": e2e_imbalance,
                    "rank_median_indexer_ms": indexer,
                    "max_rank_indexer_ms": idx_max,
                    "mean_rank_indexer_ms": idx_mean,
                    "indexer_max_over_mean_minus_one": idx_imbalance,
                }
            )
        summaries[case.case_id] = {
            "case": asdict(case),
            "packs": pack_metrics,
            "mean_pack_max_rank_e2e_ms": statistics.fmean(
                item["max_rank_e2e_ms"] for item in pack_metrics
            ),
            "mean_pack_max_rank_indexer_ms": statistics.fmean(
                item["max_rank_indexer_ms"] for item in pack_metrics
            ),
        }

    def aggregate(case_id: str, field: str) -> float:
        return float(summaries[case_id][field])

    r4_baseline = "r4-sequential-00"
    r4_candidate = "r4-balanced-11"
    r128_baseline = "r128-sequential-00"
    r128_candidate = "r128-balanced-11"
    gates = {
        "ratio4_indexer_balanced_lt_sequential": {
            "baseline_ms": aggregate(r4_baseline, "mean_pack_max_rank_indexer_ms"),
            "candidate_ms": aggregate(r4_candidate, "mean_pack_max_rank_indexer_ms"),
        },
        "ratio4_e2e_balanced_lt_sequential": {
            "baseline_ms": aggregate(r4_baseline, "mean_pack_max_rank_e2e_ms"),
            "candidate_ms": aggregate(r4_candidate, "mean_pack_max_rank_e2e_ms"),
        },
        "ratio128_e2e_balanced_lt_sequential": {
            "baseline_ms": aggregate(r128_baseline, "mean_pack_max_rank_e2e_ms"),
            "candidate_ms": aggregate(r128_candidate, "mean_pack_max_rank_e2e_ms"),
        },
    }
    for gate in gates.values():
        gate["pass"] = gate["candidate_ms"] < gate["baseline_ms"]
    for ratio in (4, 128):
        case_id = f"r{ratio}-balanced-11"
        failures = [
            item["pack_index"]
            for item in summaries[case_id]["packs"]
            if item["e2e_max_over_mean_minus_one"] > 0.05 + 1e-12
        ]
        gates[f"ratio{ratio}_all_candidate_packs_within_5pct"] = {
            "failed_packs": failures,
            "pass": not failures,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation": "median_per_pack_rank_then_rank_max_then_mean_over_packs",
        "cases": summaries,
        "gates": gates,
        "all_gates_pass": all(bool(gate["pass"]) for gate in gates.values()),
    }


def validate_correctness_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    expected_run_id: str | None = None,
    expected_calibration_id: str | None = None,
) -> None:
    if mode == "profile":
        return
    expected_cases = {case.case_id for case in _expected_cases(mode)}
    expected_count = len(expected_cases) * PACK_NUM
    _require(
        len(records) == expected_count,
        f"correctness has {len(records)} records, expected {expected_count}",
    )
    observed: set[tuple[str, int]] = set()
    for record in records:
        _require(
            record.get("schema_version") == SCHEMA_VERSION,
            "correctness schema mismatch",
        )
        _require(record.get("mode") == mode, "correctness mode mismatch")
        if expected_run_id is not None:
            _require(
                record.get("run_id") == expected_run_id,
                "correctness run_id mismatch",
            )
        if expected_calibration_id is not None:
            _require(
                record.get("calibration_id") == expected_calibration_id,
                "correctness calibration_id mismatch",
            )
        case_id = str(record.get("case_id"))
        pack_index = int(record.get("pack_index", -1))
        _require(case_id in expected_cases, f"unexpected correctness case {case_id}")
        _require(0 <= pack_index < PACK_NUM, "invalid correctness pack index")
        _require(
            record.get("pass") is True, f"correctness failed for {case_id}/{pack_index}"
        )
        _require(
            record.get("native_backend") is True, "correctness used non-native backend"
        )
        expected_method = (
            "owner-local all_to_all_single to fixed contiguous 24576-row/rank "
            "global order, then chunked torch.testing.assert_close"
        )
        _require(
            record.get("method") == expected_method,
            f"{case_id}/{pack_index} did not use canonical elementwise comparison",
        )
        comparisons = record.get("rank_comparisons")
        _require(
            isinstance(comparisons, list) and len(comparisons) == WORLD_SIZE,
            f"{case_id}/{pack_index} lacks all-rank correctness evidence",
        )
        case = CASE_BY_ID[case_id]
        if case.policy == "balanced":
            required_fields = ("output", "dx", "dqr", "dq", "dkv", "d_sink", "kl")
            for rank, rank_comparison in enumerate(comparisons):
                _require(
                    isinstance(rank_comparison, dict),
                    f"{case_id}/{pack_index}/rank{rank} comparison is malformed",
                )
                for name in required_fields:
                    field = rank_comparison.get(name)
                    _require(
                        isinstance(field, dict)
                        and field.get("elementwise_assert_close") is True,
                        f"{case_id}/{pack_index}/rank{rank} lacks elementwise {name}",
                    )
                parameter_fields = [
                    name for name in rank_comparison if name.startswith("parameter:")
                ]
                if case.ratio:
                    _require(
                        parameter_fields,
                        f"{case_id}/{pack_index}/rank{rank} lacks parameter gradients",
                    )
                for name in parameter_fields:
                    _require(
                        rank_comparison[name].get("elementwise_assert_close") is True,
                        f"{case_id}/{pack_index}/rank{rank} lacks elementwise {name}",
                    )
        key = (case_id, pack_index)
        _require(key not in observed, f"duplicate correctness record {key}")
        observed.add(key)
    expected = {
        (case.case_id, pack_index)
        for case in _expected_cases(mode)
        for pack_index in range(PACK_NUM)
    }
    _require(observed == expected, "correctness matrix is incomplete")


def _merged_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _intersection_ns(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    left_merged = _merged_intervals(left)
    right_merged = _merged_intervals(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _profile_overlap_from_sqlite(
    sqlite_path: Path,
    *,
    run_id: str,
    slowest_rank: int,
    slowest_rank_e2e_ms: float,
) -> dict[str, Any]:
    """Recover true cross-stream GPU overlap from one Nsight SQLite export.

    CUDA API calls are assigned to their enclosing NVTX range on the profiled
    Python thread. CUPTI correlation IDs then map those calls to GPU kernels.
    This avoids treating a host launch-to-wait window as proof of overlap.
    """

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _table_names(connection)
        required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"}
        _require(
            required <= tables,
            f"Nsight SQLite is missing tables {sorted(required - tables)}",
        )
        activity_tables = [
            name
            for name in (
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_DRIVER",
            )
            if name in tables
        ]
        _require(activity_tables, "Nsight SQLite has no CUDA API activity table")
        strings = {
            int(row[0]): str(row[1])
            for row in connection.execute("SELECT id, value FROM StringIds")
        }
        ranges = []
        for start, end, global_tid, text, text_id in connection.execute(
            "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL"
        ):
            label = str(text) if text else strings.get(int(text_id or -1), "")
            if global_tid is None or not label:
                continue
            ranges.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "global_tid": int(global_tid),
                    "label": label,
                }
            )

        outer_label = (
            "magi_dsa::profile::r4-balanced-11::"
            f"pack{PROFILE_PACK_INDEX}::rank{slowest_rank}::iter0"
        )
        outer_candidates = [item for item in ranges if item["label"] == outer_label]
        _require(
            len(outer_candidates) == 1,
            f"expected one NVTX range {outer_label!r}, found {len(outer_candidates)}",
        )
        outer = outer_candidates[0]
        global_tid = int(outer["global_tid"])
        # Nsight encodes globalTid as globalPid plus the low 24-bit TID.
        global_pid = (global_tid >> 24) << 24
        nested_ranges = [
            item
            for item in ranges
            if item["global_tid"] == global_tid
            and item["start"] >= outer["start"]
            and item["end"] <= outer["end"]
        ]

        launches: list[dict[str, int]] = []
        for table in activity_tables:
            for start, end, tid, correlation_id in connection.execute(
                f"SELECT start, end, globalTid, correlationId FROM {table} "
                "WHERE globalTid = ? AND start >= ? AND end <= ? "
                "AND correlationId IS NOT NULL",
                (global_tid, outer["start"], outer["end"]),
            ):
                launches.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "global_tid": int(tid),
                        "correlation_id": int(correlation_id),
                    }
                )

        kernels: list[dict[str, Any]] = []
        for (
            start,
            end,
            stream_id,
            correlation_id,
            short_name,
            demangled_name,
        ) in connection.execute(
            "SELECT start, end, streamId, correlationId, shortName, demangledName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid = ?",
            (global_pid,),
        ):
            name = strings.get(int(short_name), "") or strings.get(
                int(demangled_name), ""
            )
            kernels.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "stream_id": int(stream_id),
                    "correlation_id": int(correlation_id),
                    "name": name,
                }
            )

        kernels_by_correlation: dict[int, list[dict[str, Any]]] = {}
        for kernel in kernels:
            kernels_by_correlation.setdefault(kernel["correlation_id"], []).append(
                kernel
            )

        def kernels_launched_in(selected_ranges: Sequence[Mapping[str, Any]]):
            selected: dict[tuple[int, int, int, int], dict[str, Any]] = {}
            for nvtx_range in selected_ranges:
                for launch in launches:
                    if (
                        launch["start"] >= nvtx_range["start"]
                        and launch["end"] <= nvtx_range["end"]
                    ):
                        for kernel in kernels_by_correlation.get(
                            launch["correlation_id"], ()
                        ):
                            key = (
                                kernel["start"],
                                kernel["end"],
                                kernel["stream_id"],
                                kernel["correlation_id"],
                            )
                            selected[key] = kernel
            return list(selected.values())

        def phase_evidence(
            *,
            phase_label: str,
            native_launch_suffix: str,
            immediately_preceding_native_launches: int,
        ) -> dict[str, Any]:
            phase_ranges = [
                item for item in nested_ranges if item["label"] == phase_label
            ]
            native_ranges = [
                item
                for item in nested_ranges
                if item["label"].endswith(native_launch_suffix)
            ]
            _require(phase_ranges, f"profile lacks NVTX phase {phase_label!r}")
            _require(
                native_ranges,
                f"profile lacks native launch range *{native_launch_suffix}",
            )
            first_phase_start = min(int(item["start"]) for item in phase_ranges)
            preceding_native_ranges = sorted(
                (
                    item
                    for item in native_ranges
                    if int(item["end"]) <= first_phase_start
                ),
                key=lambda item: int(item["end"]),
            )
            _require(
                len(preceding_native_ranges) >= immediately_preceding_native_launches,
                f"{phase_label!r} lacks {immediately_preceding_native_launches} "
                f"preceding native {native_launch_suffix} launches",
            )
            native_ranges = preceding_native_ranges[
                -immediately_preceding_native_launches:
            ]
            compute_kernels = kernels_launched_in(phase_ranges)
            native_kernels = kernels_launched_in(native_ranges)
            _require(
                compute_kernels,
                f"no GPU kernels correlate with NVTX phase {phase_label!r}",
            )
            concurrent_native = [
                communication
                for communication in native_kernels
                if any(
                    communication["stream_id"] != compute["stream_id"]
                    and min(communication["end"], compute["end"])
                    > max(communication["start"], compute["start"])
                    for compute in compute_kernels
                )
            ]
            compute_intervals = [
                (item["start"], item["end"]) for item in compute_kernels
            ]
            communication_intervals = [
                (item["start"], item["end"]) for item in concurrent_native
            ]
            intersection = _intersection_ns(compute_intervals, communication_intervals)
            compute_duration = sum(
                end - start for start, end in _merged_intervals(compute_intervals)
            )
            communication_duration = sum(
                end - start for start, end in _merged_intervals(communication_intervals)
            )
            passed = bool(intersection > 0 and concurrent_native)
            return {
                "compute_nvtx": phase_label,
                "native_launch_nvtx_suffix": native_launch_suffix,
                "native_launch_selector": (
                    f"last {immediately_preceding_native_launches} launch range(s) "
                    "ending before the compute phase"
                ),
                "assignment": "NVTX CUDA-launch range -> CUPTI correlationId -> GPU kernel",
                "compute_kernel_count": len(compute_kernels),
                "native_communication_kernel_count": len(concurrent_native),
                "compute_stream_ids": sorted(
                    {int(item["stream_id"]) for item in compute_kernels}
                ),
                "native_communication_stream_ids": sorted(
                    {int(item["stream_id"]) for item in concurrent_native}
                ),
                "compute_kernel_names": sorted(
                    {str(item["name"]) for item in compute_kernels}
                ),
                "native_communication_kernel_names": sorted(
                    {str(item["name"]) for item in concurrent_native}
                ),
                "compute_gpu_ms": compute_duration / 1_000_000.0,
                "native_communication_gpu_ms": communication_duration / 1_000_000.0,
                "intersection_gpu_ms": intersection / 1_000_000.0,
                "intersection_over_compute": (
                    intersection / compute_duration if compute_duration else 0.0
                ),
                "pass": passed,
            }

        phases = {
            "compressed_cast_intersects_indexer_projection": phase_evidence(
                phase_label="magi_dsa::indexer_projection",
                native_launch_suffix="native_group_cast_impl",
                immediately_preceding_native_launches=2,
            ),
            "dki_reduce_intersects_sparse_backward": phase_evidence(
                phase_label="magi_dsa::overlap_dki_reduce_sparse_backward",
                native_launch_suffix="native_group_reduce_impl",
                immediately_preceding_native_launches=1,
            ),
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "case_id": "r4-balanced-11",
            "pack_index": PROFILE_PACK_INDEX,
            "slowest_rank": slowest_rank,
            "slowest_rank_e2e_ms": slowest_rank_e2e_ms,
            "outer_nvtx": outer_label,
            "time_unit": "nanoseconds",
            "method": "NVTX/CUPTI correlation with cross-stream GPU interval intersection",
            "phases": phases,
            "all_required_overlaps_observed": all(
                bool(item["pass"]) for item in phases.values()
            ),
        }
    finally:
        connection.close()


def _one_artifact(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    _require(
        len(matches) == 1,
        f"expected exactly one {pattern} in {root}, found {len(matches)}",
    )
    return matches[0]


def _export_nsys_sqlite(report: Path, output: Path) -> Path:
    command = [
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={output}",
        str(report),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ValidationError(
            f"nsys export failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    candidates = (output, output.with_suffix(".sqlite"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(f"nsys export did not create {output}")


def build_profile_overlap(profile_dir: Path) -> dict[str, Any]:
    metadata = read_json(profile_dir / "profile_metadata.json")
    raw = read_jsonl(profile_dir / "raw_timing.jsonl")
    candidate = [
        item
        for item in raw
        if item.get("case_id") == "r4-balanced-11"
        and item.get("pack_index") == PROFILE_PACK_INDEX
        and item.get("iteration") == 0
    ]
    _require(
        len(candidate) == WORLD_SIZE,
        f"candidate profile has {len(candidate)} rank records, expected {WORLD_SIZE}",
    )
    slowest = max(candidate, key=lambda item: float(item["e2e_ms"]))
    report = _one_artifact(profile_dir, "*.nsys-rep")
    sqlite_path = _export_nsys_sqlite(report, profile_dir / f"{report.stem}-verified")
    payload = _profile_overlap_from_sqlite(
        sqlite_path,
        run_id=str(metadata["run_id"]),
        slowest_rank=int(slowest["rank"]),
        slowest_rank_e2e_ms=float(slowest["e2e_ms"]),
    )
    payload.update(
        {
            "revision": metadata["revision"],
            "image_id": metadata["image_id"],
            "pack_sha256": metadata["pack_sha256"],
            "nsys_report": report.name,
            "nsys_report_sha256": sha256_file(report),
            "nsys_sqlite": sqlite_path.name,
            "nsys_sqlite_sha256": sha256_file(sqlite_path),
        }
    )
    _require(
        payload["all_required_overlaps_observed"],
        "Nsight GPU timeline does not prove both required overlaps",
    )
    atomic_write_json(profile_dir / "profile_overlap.json", payload)
    return payload


def validate_profile_overlap(
    profile_dir: Path,
    *,
    expected_revision: str | None,
    expected_image_id: str | None,
) -> dict[str, Any]:
    payload = build_profile_overlap(profile_dir)
    expected_values = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "r4-balanced-11",
        "pack_index": PROFILE_PACK_INDEX,
        "all_required_overlaps_observed": True,
    }
    for name, expected in expected_values.items():
        _require(
            payload.get(name) == expected,
            f"profile overlap {name}={payload.get(name)!r}, expected {expected!r}",
        )
    if expected_revision is not None:
        _require(
            payload.get("revision") == expected_revision,
            "profile overlap revision differs from measure revision",
        )
    if expected_image_id is not None:
        _require(
            payload.get("image_id") == expected_image_id,
            "profile overlap image differs from measure image",
        )
    report = profile_dir / str(payload.get("nsys_report", ""))
    sqlite_path = profile_dir / str(payload.get("nsys_sqlite", ""))
    _require(
        report.is_file() and sha256_file(report) == payload.get("nsys_report_sha256"),
        "profile report hash mismatch",
    )
    _require(
        sqlite_path.is_file()
        and sha256_file(sqlite_path) == payload.get("nsys_sqlite_sha256"),
        "profile SQLite hash mismatch",
    )
    phases = payload.get("phases")
    _require(isinstance(phases, dict) and len(phases) == 2, "invalid overlap phases")
    for name, phase in phases.items():
        _require(
            isinstance(phase, dict)
            and phase.get("pass") is True
            and float(phase.get("intersection_gpu_ms", 0.0)) > 0,
            f"profile phase {name} has no GPU-kernel intersection",
        )
    return payload


def create_manifest(
    perf_dir: Path, profile_dir: Path | None = None
) -> list[tuple[str, str]]:
    roots = [("perf", perf_dir)]
    if profile_dir is not None:
        roots.append(("profile", profile_dir))
    entries: list[tuple[str, str]] = []
    for label, root in roots:
        _require(root.is_dir(), f"artifact root does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "artifact_manifest.sha256":
                continue
            if (
                path.name.endswith(".tmp")
                or ".rank" in path.name
                or "_rank" in path.parts
            ):
                continue
            entries.append((sha256_file(path), f"{label}/{path.relative_to(root)}"))
    entries.sort(key=lambda item: item[1])
    return entries


def write_manifest(perf_dir: Path, profile_dir: Path | None = None) -> Path:
    entries = create_manifest(perf_dir, profile_dir)
    path = perf_dir / "artifact_manifest.sha256"
    temporary = path.with_suffix(".sha256.tmp")
    temporary.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def verify_manifest(perf_dir: Path, profile_dir: Path | None = None) -> None:
    path = perf_dir / "artifact_manifest.sha256"
    _require(path.is_file(), "artifact_manifest.sha256 is missing")
    actual: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        _require(len(parts) == 2 and len(parts[0]) == 64, "invalid manifest line")
        actual.append((parts[0], parts[1]))
    expected = create_manifest(perf_dir, profile_dir)
    _require(actual == expected, "artifact manifest is stale or incomplete")


def validate_run(
    run_dir: Path,
    *,
    mode: str,
    expected_revision: str | None = None,
    expected_image_id: str | None = None,
    write_summary: bool = False,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    environment = read_json(run_dir / "environment.json")
    validate_environment(
        environment,
        mode=mode,
        expected_revision=expected_revision,
        expected_image_id=expected_image_id,
    )
    packs = validate_packs(read_json(run_dir / "packs.json"))
    raw = read_jsonl(run_dir / "raw_timing.jsonl")
    plans = read_jsonl(run_dir / "plans.jsonl")
    correctness = read_jsonl(run_dir / "correctness.jsonl")
    run_id = str(environment["run_id"])
    calibration_id = str(environment["calibration_id"])
    validate_raw_records(
        raw,
        packs,
        mode=mode,
        expected_run_id=run_id,
        expected_calibration_id=calibration_id,
    )
    cache_key_count = validate_plan_records(
        plans,
        mode=mode,
        expected_run_id=run_id,
        expected_calibration_id=calibration_id,
    )
    validate_correctness_records(
        correctness,
        mode=mode,
        expected_run_id=run_id,
        expected_calibration_id=calibration_id,
    )
    profile_overlap: dict[str, Any] | None = None
    if mode == "measure":
        summary = summarize_records(raw, packs)
        if profile_dir is not None:
            profile_environment = read_json(profile_dir / "environment.json")
            validate_environment(
                profile_environment,
                mode="profile",
                expected_revision=str(environment["revision"]),
                expected_image_id=str(environment["image_id"]),
            )
            _require(
                profile_environment["run_id"] == environment["run_id"],
                "profile and measure use different run IDs",
            )
            _require(
                profile_environment["calibration_id"] == environment["calibration_id"],
                "profile and measure use different calibration IDs",
            )
            profile_packs = validate_packs(read_json(profile_dir / "packs.json"))
            _require(
                pack_suite_sha256(profile_packs) == pack_suite_sha256(packs),
                "profile and measure use different pack suites",
            )
            validate_raw_records(
                read_jsonl(profile_dir / "raw_timing.jsonl"),
                profile_packs,
                mode="profile",
                expected_run_id=str(profile_environment["run_id"]),
                expected_calibration_id=str(profile_environment["calibration_id"]),
            )
            validate_plan_records(
                read_jsonl(profile_dir / "plans.jsonl"),
                mode="profile",
                expected_run_id=str(profile_environment["run_id"]),
                expected_calibration_id=str(profile_environment["calibration_id"]),
            )
            profile_overlap = validate_profile_overlap(
                profile_dir,
                expected_revision=str(environment["revision"]),
                expected_image_id=str(environment["image_id"]),
            )
        if write_summary:
            atomic_write_json(run_dir / "summary.json", summary)
    else:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "record_count": len(raw),
        }
    validation = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "valid": True,
        "revision": environment["revision"],
        "image_id": environment["image_id"],
        "packs_sha256": pack_suite_sha256(packs),
        "raw_record_count": len(raw),
        "dsa_pack_cache_key_count": cache_key_count,
        "dsa_pack_sm103_cache_keys_validated": True,
        "all_gates_pass": summary.get("all_gates_pass"),
        "orthogonal_profile_evidence": profile_overlap is not None,
        "profile_overlap_sha256": (
            sha256_file(profile_dir / "profile_overlap.json")
            if profile_overlap is not None and profile_dir is not None
            else None
        ),
        "formal_acceptance": bool(
            mode == "measure"
            and summary.get("all_gates_pass")
            and profile_overlap is not None
        ),
    }
    atomic_write_json(run_dir / "validation.json", validation)
    return {"summary": summary, "validation": validation}


def _profile_sqlite_self_test(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL, end INTEGER, text TEXT, textId INTEGER,
            globalTid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER NOT NULL, end INTEGER NOT NULL, globalTid INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER NOT NULL, end INTEGER NOT NULL, streamId INTEGER,
            correlationId INTEGER, shortName INTEGER, demangledName INTEGER,
            globalPid INTEGER
        );
        """
    )
    labels = {
        1: "magi_dsa::profile::r4-balanced-11::pack0::rank7::iter0",
        2: "magi_dsa::indexer_projection",
        3: "native_group_cast_impl",
        4: "magi_dsa::overlap_dki_reduce_sparse_backward",
        5: "native_group_reduce_impl",
        6: "projection_kernel",
        7: "group_cast_kernel",
        8: "sparse_backward_kernel",
        9: "group_reduce_kernel",
    }
    connection.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)", labels.items()
    )
    global_pid = 42 << 24
    global_tid = global_pid + 123
    ranges = (
        (0, 1_000, None, 1, global_tid),
        (120, 220, None, 2, global_tid),
        (20, 40, None, 3, global_tid),
        (50, 100, None, 3, global_tid),
        (480, 650, None, 4, global_tid),
        (400, 450, None, 5, global_tid),
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS(start, end, text, textId, globalTid) "
        "VALUES (?, ?, ?, ?, ?)",
        ranges,
    )
    launches = (
        (25, 30, global_tid, 100),
        (60, 70, global_tid, 101),
        (130, 140, global_tid, 102),
        (410, 420, global_tid, 103),
        (500, 510, global_tid, 104),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
        "(start, end, globalTid, correlationId) VALUES (?, ?, ?, ?)",
        launches,
    )
    kernels = (
        (140, 155, 9, 100, 7, 7, global_pid),
        (150, 210, 9, 101, 7, 7, global_pid),
        (160, 230, 7, 102, 6, 6, global_pid),
        (500, 600, 9, 103, 9, 9, global_pid),
        (520, 620, 7, 104, 8, 8, global_pid),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, streamId, correlationId, shortName, demangledName, globalPid) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        kernels,
    )
    connection.commit()
    connection.close()
    evidence = _profile_overlap_from_sqlite(
        path, run_id="self-test", slowest_rank=7, slowest_rank_e2e_ms=1.0
    )
    _require(
        evidence["all_required_overlaps_observed"],
        "synthetic Nsight overlap analysis failed",
    )


def _self_test() -> None:
    lengths = [GLOBAL_TOKENS]
    pack_hash = pack_sha256(lengths)
    packs = [
        {"index": index, "lengths": lengths, "sha256": pack_hash}
        for index in range(PACK_NUM)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sampler": {
            "seed": PACK_SEED,
            "pack_num": PACK_NUM,
            "chunk_ratio": CHUNK_RATIO,
            "pack_len": GLOBAL_TOKENS,
            "dataset_sha256": DATASET_SHA256,
            "is_binned": True,
        },
        "packs": packs,
        "packs_sha256": pack_suite_sha256(packs),
    }
    validate_packs(payload)
    records: list[dict[str, Any]] = []
    for case in MEASURE_CASES:
        candidate = case.policy == "balanced" and case.overlap_code == "11"
        for pack in packs:
            for rank in range(WORLD_SIZE):
                for iteration in range(MEASURE_ITERS):
                    baseline = 10.0 + rank * 0.001
                    e2e = baseline * (
                        0.8 if candidate and case.ratio in (4, 128) else 1.0
                    )
                    indexer = (
                        baseline * (0.7 if candidate else 1.0)
                        if case.ratio == 4
                        else 0.0
                    )
                    records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "mode": "measure",
                            "case_id": case.case_id,
                            "ratio": case.ratio,
                            "policy": case.policy,
                            "overlap_code": case.overlap_code,
                            "pack_index": pack["index"],
                            "pack_sha256": pack["sha256"],
                            "plan_sha256": "0" * 64,
                            "iteration": iteration,
                            "rank": rank,
                            "e2e_ms": e2e,
                            "indexer_ms": indexer,
                            "phases_ms": (
                                {"indexer_projection": indexer}
                                if case.ratio == 4
                                else {}
                            ),
                            "native_backend": True,
                            "finite": True,
                            "jit_cache_miss_delta": 0,
                        }
                    )
    validate_raw_records(records, packs, mode="measure")
    summary = summarize_records(records, packs)
    _require(summary["all_gates_pass"], "synthetic passing summary failed")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        atomic_write_json(root / "sample.json", payload)
        _profile_sqlite_self_test(root / "sample.sqlite")
        write_manifest(root)
        verify_manifest(root)
    print("validate.py self-test: PASS")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("calibration", "measure", "profile"), default="measure"
    )
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-image-id")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --self-test is used")
    if args.mode == "measure" and args.profile_dir is None:
        raise SystemExit("formal measure validation requires --profile-dir")
    result = validate_run(
        args.run_dir,
        mode=args.mode,
        expected_revision=args.expected_revision,
        expected_image_id=args.expected_image_id,
        write_summary=args.write_summary,
        profile_dir=args.profile_dir,
    )
    if args.write_manifest:
        write_manifest(args.run_dir, args.profile_dir)
    if args.verify_manifest:
        verify_manifest(args.run_dir, args.profile_dir)
    print(json.dumps(result["validation"], indent=2, sort_keys=True))
    if args.mode == "measure" and not result["validation"]["formal_acceptance"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
