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
import json
import math
import re
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

_FLASHMLA_DUAL_LSE_PATCH_REVISION = "13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece"
_FLASHMLA_DUAL_LSE_PATCH_SHA256 = (
    "6957dbde516c73066c5911108761325edc1bdcd8f62e15dc0a84f4f290118d4b"
)
_FLASHMLA_PRO_H128_PATCH_REVISION = "b7643bd54521f563b839b98289b5cd048c062ba2"
_FLASHMLA_PRO_H128_PATCH_SHA256 = (
    "c534e13ff432ac1c694cb24981826c11be26a2d9743d7175ddb05f887279461f"
)
# Magi-DSA ships as the magi_attn_extensions distribution, whose version is a
# plain literal in extensions/magi_attn_extensions/__init__.py rather than a
# versioningit-derived string like the Core wheel's.
_MAGI_ATTN_EXTENSIONS_VERSION = "1.1.0"
_PRO_MODEL_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
_PRO_FLASHMLA_FORWARD_KERNEL = "sparse_attn_fwd_for_small_topk_kernel"
_PRO_PAIR_WORLD_SIZE = 8
_PRO_PAIR_STEPS = 5
_PRO_PAIR_QUERY_TOKENS_PER_RANK = 16_384
_PRO_PAIR_GLOBAL_OUTPUT_ELEMENTS = 131_072 * 128 * 512
_PRO_PAIR_MAJOR_KERNEL_GROUPS = (
    "csa_flashmla_forward",
    "csa_indexer_score",
    "csa_indexer_topk",
    "csa_selected_indexer_backward",
    "csa_sparse_attention_backward_main",
    "hca_flashmla_forward",
    "hca_sparse_attention_backward_main",
)
_PRO_PAIR_HARD_GATE_GROUPS = (
    "csa_indexer_score",
    "csa_indexer_topk",
)
_PRO_PAIR_D2D_SCOPES = {
    "indexer_score": "magi_dsa::CUDNN_CALL::indexer_score",
    "indexer_topk": "magi_dsa::CUDNN_CALL::indexer_topk",
    "selected_attention_recompute": (
        "magi_dsa::CUDNN_CALL::selected_attention_recompute"
    ),
    "selected_indexer_backward": "magi_dsa::CUDNN_CALL::indexer_backward",
    "selected_indexer_recompute": ("magi_dsa::CUDNN_CALL::selected_indexer_recompute"),
}
_PRO_PAIR_D2D_GROUPS = tuple(_PRO_PAIR_D2D_SCOPES)
_PRO_PAIR_SUPPORT_GROUPS = (
    "csa_grouped_k_pack_forward",
    "csa_kv_bank_catarray",
    "csa_route_stage_copy",
    "csa_route_stage_csr_reduce",
    "hca_kv_bank_catarray",
    "hca_route_stage_copy",
    "hca_route_stage_csr_reduce",
)
_PRO_PAIR_ROUTE_ORDER: dict[str, dict[str, tuple[str, ...]]] = {
    "csa": {
        "forward": ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV"),
        "backward": (
            "COMPRESSED_KI",
            "COMPRESSED_KV",
            "OVERLAP_X",
            "WINDOW_KV",
        ),
    },
    "hca": {
        "forward": ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
        "backward": ("COMPRESSED_KV", "WINDOW_KV", "OVERLAP_X"),
    },
}
_PRO_PAIR_ROUTE_GROUPS = tuple(
    f"{mode}.{direction}.{route}"
    for mode, directions in _PRO_PAIR_ROUTE_ORDER.items()
    for direction, routes in directions.items()
    for route in routes
)
_PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS = tuple(
    group
    for group in _PRO_PAIR_ROUTE_GROUPS
    if group.startswith("csa.") or ".WINDOW_KV" in group
)
_PRO_PAIR_DEPENDENCY_BOUND_ROUTE_GROUPS = tuple(
    group
    for group in _PRO_PAIR_ROUTE_GROUPS
    if group not in _PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS
)
_PRO_PAIR_REQUIRED_ARTIFACTS = (
    "SUMMARY_PRO_PAIR.json",
    "MAJOR_KERNEL_BALANCE_PRO_PAIR.json",
    "INDEXER_D2D_PRO_PAIR.json",
    "SUPPORT_OVERHEAD_PRO_PAIR.json",
    "PRO_PAIR_COMMUNICATION_OVERLAP.json",
    "PRO_PAIR_ARTIFACTS.txt",
    "REPORT_PRO_PAIR.md",
    "WORKLOAD.json",
    "balanced/balanced_5steps_pro_pair.nsys-rep",
    "balanced/balanced_5steps_pro_pair.sqlite",
)
_STRUCTURAL_CONFIG = {
    "chunk_size": 512,
    "min_chunks_per_rank": 16,
    "uneven_shard": True,
}
_CP8_PLAN_POLICIES = {
    "csa_balanced": "indexer_balanced",
    "csa_sequential": "sequential",
    "csa_structural": "structural_balanced",
    "hca": "sequential",
    "hca_structural": "structural_balanced",
    "window": "sequential",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a Magi-DSA v4 release artifact"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--correctness-artifact", type=Path, required=True)
    parser.add_argument("--profile-artifact", type=Path, required=True)
    parser.add_argument("--cp1-artifact", type=Path, required=True)
    parser.add_argument("--cp2-artifact", type=Path, required=True)
    return parser.parse_args()


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 60,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {' '.join(command)}\n{output}"
        )
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite release artifact: {path}")
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite release artifact: {destination}")
    shutil.copy2(source, destination)


def _validate_manifest(artifact: Path) -> str:
    manifest = artifact / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing SHA-256 manifest: {manifest}")
    artifact_root = artifact.resolve()
    manifest_path = manifest.resolve()
    expected_files: set[Path] = set()
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact manifest cannot cover a symlink: {path}")
        if not path.is_file() or path.resolve() == manifest_path:
            continue
        expected_files.add(path.resolve())

    covered_files: set[Path] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"[0-9a-f]{64} [ *](.+)", line)
        if match is None:
            raise ValueError(f"invalid SHA-256 manifest line {line_number}: {line!r}")
        recorded = Path(match.group(1))
        resolved = (
            recorded.resolve()
            if recorded.is_absolute()
            else (artifact_root / recorded).resolve()
        )
        if not resolved.is_relative_to(artifact_root) or resolved == manifest_path:
            raise ValueError(f"SHA-256 manifest path escapes artifact: {recorded}")
        if resolved in covered_files:
            raise ValueError(f"duplicate SHA-256 manifest path: {recorded}")
        covered_files.add(resolved)
    if covered_files != expected_files:
        missing = sorted(str(path) for path in expected_files - covered_files)
        extra = sorted(str(path) for path in covered_files - expected_files)
        raise ValueError(
            "SHA-256 manifest file grid differs: " f"missing={missing}, extra={extra}"
        )
    return _run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=artifact,
        timeout_seconds=60,
    )


def _require_sha256(value: object, field: str) -> str:
    resolved = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", resolved) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return resolved


def _require_json_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: object,
    field: str,
    *,
    minimum: float = 0.0,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return resolved


def _interval_duration_ns(intervals: Sequence[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError(f"invalid time interval: start={start}, end={end}")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _require_artifact_provenance(path: Path, revision: str, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} artifact directory does not exist: {path}")
    revision_path = path / "SOURCE_REVISION.txt"
    if not revision_path.is_file():
        raise FileNotFoundError(f"missing {label} source revision: {revision_path}")
    recorded_revision = revision_path.read_text(encoding="utf-8").strip()
    if recorded_revision != revision:
        raise ValueError(
            f"{label} source revision {recorded_revision!r} does not match {revision}"
        )
    _validate_manifest(path)


def _require_clean_artifact(path: Path, label: str) -> None:
    dirty_status = path / "DIRTY_STATUS.txt"
    if not dirty_status.is_file():
        raise FileNotFoundError(f"missing {label} dirty status: {dirty_status}")
    if dirty_status.read_text(encoding="utf-8").strip():
        raise ValueError(f"{label} artifact was produced from a dirty worktree")


def _validate_pro_pair_workload(workload: dict[str, Any]) -> None:
    expected = {
        "attention_order": ["csa", "hca"],
        "backward_order": ["hca", "csa"],
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "capture_order": ["balanced"],
        "case": "dsv4-pro-128k",
        "cp_size": _PRO_PAIR_WORLD_SIZE,
        "cu_seqlens": [0, 131_072],
        "dtype": "BF16",
        "gpu_clock_lock_mhz": None,
        "gradient_accumulation": False,
        "independent_attention_graphs": True,
        "layout_policy": "structural-balanced",
        "local_improvement_passes": None,
        "loss_capture": "none",
        "mode_backward_completion_join": {
            "csa": [
                "sparse_backward_stream",
                "csa_main_stream",
                "csa_indexer_stream",
                "csa_route_stream",
            ],
            "hca": ["hca_main_stream", "hca_route_stream"],
        },
        "mode_serialization": "cuda_event_happens_before",
        "overlap_accounting": "same_mode_same_direction_non_route_compute",
        "parameter_gradient_allreduce": "one_unified_after_two_backwards",
        "plans": ["balanced"],
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "profiler_attach_warmup_steps": 0,
        "projection_capture": "pre_capture_once_per_attention",
        "rank_size": _PRO_PAIR_WORLD_SIZE,
        "ratio": None,
        "ratios": [4, 128],
        "representative_layer_ids": {"csa": 2, "hca": 3},
        "representative_pair_semantics": (
            "independent_post_projection_graphs_serialized_in_layer_order"
        ),
        "required_summary_artifacts": [
            "INDEXER_D2D_PRO_PAIR.json",
            "SUPPORT_OVERHEAD_PRO_PAIR.json",
            "PRO_PAIR_COMMUNICATION_OVERLAP.json",
        ],
        "seed": 0,
        "smoke": "skipped_by_user",
        "step_mode": "pro-pair",
        "steps": _PRO_PAIR_STEPS,
        "structural_layout_config": _STRUCTURAL_CONFIG,
        "token_layout_capture": "pre_capture_once_per_attention",
        "warmup_steps_per_plan": 3,
        "world_size": _PRO_PAIR_WORLD_SIZE,
    }
    for field, expected_value in expected.items():
        if workload.get(field) != expected_value:
            raise ValueError(
                f"Pro-pair workload {field} differs: {workload.get(field)!r}"
            )
    expected_scopes = {
        "magi_dsa::CUDNN_CALL::indexer_score",
        "magi_dsa::CUDNN_CALL::indexer_topk",
        "magi_dsa::CUDNN_CALL::selected_indexer_recompute",
        "magi_dsa::CUDNN_CALL::selected_attention_recompute",
        "magi_dsa::CUDNN_CALL::indexer_backward",
    }
    scopes = workload.get("cudnn_memcpy_attribution_scopes")
    if not isinstance(scopes, list) or set(scopes) != expected_scopes:
        raise ValueError("Pro-pair cuDNN D2D attribution scopes differ")
    dout_scale = _require_finite_number(workload.get("dout_scale"), "dout_scale")
    if dout_scale != 1.0 / _PRO_PAIR_GLOBAL_OUTPUT_ELEMENTS:
        raise ValueError("Pro-pair dout scale differs")
    expected_model = {
        "source_revision": _PRO_MODEL_REVISION,
        "main_layer_count": 61,
        "csa_layer_count": 30,
        "hca_layer_count": 31,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "num_query_heads": 128,
        "head_dim": 512,
        "indexer_heads": 64,
        "indexer_head_dim": 128,
        "indexer_topk": 1024,
        "window_size": 128,
    }
    if workload.get("pro_model_contract") != expected_model:
        raise ValueError("Pro-pair DeepSeek-V4-Pro model contract differs")


def _validate_pro_pair_layout(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("result") != "PASS":
        raise ValueError("Pro-pair structural layout did not pass")
    query_layout_hash = _require_sha256(
        value.get("query_layout_hash"), "layout.query_layout_hash"
    )
    expected_counts = [_PRO_PAIR_QUERY_TOKENS_PER_RANK] * _PRO_PAIR_WORLD_SIZE
    if value.get("query_token_counts") != expected_counts:
        raise ValueError("Pro-pair structural layout does not cover 128K exactly once")
    if value.get("rank_results") != _PRO_PAIR_WORLD_SIZE:
        raise ValueError("Pro-pair structural rank-result grid differs")

    rank_costs = value.get("rank_costs")
    if not isinstance(rank_costs, list) or len(rank_costs) != _PRO_PAIR_WORLD_SIZE:
        raise ValueError("Pro-pair structural rank costs are incomplete")
    for rank, cost in enumerate(rank_costs):
        if not isinstance(cost, dict):
            raise ValueError(f"Pro-pair rank {rank} structural cost is invalid")
        required = {
            "rank": rank,
            "query_tokens": _PRO_PAIR_QUERY_TOKENS_PER_RANK,
        }
        if any(cost.get(field) != expected for field, expected in required.items()):
            raise ValueError(f"Pro-pair rank {rank} structural cost differs")
        for field in (
            "native_causal_area",
            "chunk_count",
            "fragment_count",
            "csa_unique_indexer_k_rows",
            "csa_packed_indexer_k_rows",
            "csa_duplicate_indexer_k_rows",
        ):
            _require_json_int(cost.get(field), f"layout.rank_costs[{rank}].{field}")
        if cost["csa_packed_indexer_k_rows"] != (
            cost["csa_unique_indexer_k_rows"] + cost["csa_duplicate_indexer_k_rows"]
        ):
            raise ValueError(f"Pro-pair rank {rank} structural KI rows differ")

    packing = value.get("indexer_k_packing")
    if not isinstance(packing, list) or len(packing) != _PRO_PAIR_WORLD_SIZE:
        raise ValueError("Pro-pair Indexer K packing metadata is incomplete")
    for rank, record in enumerate(packing):
        if not isinstance(record, dict) or record.get("rank") != rank:
            raise ValueError(f"Pro-pair rank {rank} Indexer K packing differs")
        row_bytes = _require_json_int(
            record.get("indexer_k_row_bytes"),
            f"layout.indexer_k_packing[{rank}].indexer_k_row_bytes",
            minimum=1,
        )
        if row_bytes != 256:
            raise ValueError(f"Pro-pair rank {rank} Indexer K row bytes differ")
        unique_rows = _require_json_int(
            record.get("unique_indexer_k_rows"), f"packing[{rank}].unique_rows"
        )
        packed_rows = _require_json_int(
            record.get("packed_indexer_k_rows"), f"packing[{rank}].packed_rows"
        )
        duplicate_rows = _require_json_int(
            record.get("duplicate_indexer_k_rows"),
            f"packing[{rank}].duplicate_rows",
        )
        if packed_rows != unique_rows + duplicate_rows:
            raise ValueError(f"Pro-pair rank {rank} Indexer K row accounting differs")
        for prefix, rows in (
            ("unique", unique_rows),
            ("packed", packed_rows),
            ("duplicate", duplicate_rows),
        ):
            if record.get(f"{prefix}_indexer_k_bytes") != rows * row_bytes:
                raise ValueError(
                    f"Pro-pair rank {rank} Indexer K {prefix} bytes differ"
                )
        cost = rank_costs[rank]
        if any(
            record[f"{prefix}_indexer_k_rows"] != cost[f"csa_{prefix}_indexer_k_rows"]
            for prefix in ("unique", "packed", "duplicate")
        ):
            raise ValueError(
                f"Pro-pair rank {rank} structural cost and KI packing differ"
            )
    return {**value, "query_layout_hash": query_layout_hash}


def _validate_timing_grid(
    report: dict[str, Any],
    *,
    groups: Sequence[str],
    ranges_field: str,
    label: str,
    positive_time: bool,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label} records are missing")
    expected = {
        (group, step, rank)
        for group in groups
        for step in range(_PRO_PAIR_STEPS)
        for rank in range(_PRO_PAIR_WORLD_SIZE)
    }
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label} record {index} is not an object")
        group = str(record.get("group", ""))
        step = _require_json_int(record.get("step"), f"{label}[{index}].step")
        rank = _require_json_int(record.get("rank"), f"{label}[{index}].rank")
        timing_key = (group, step, rank)
        if timing_key in by_key:
            raise ValueError(f"duplicate {label} timing record: {timing_key}")
        by_key[timing_key] = record
        gpu_time = _require_finite_number(
            record.get("gpu_time_ms"), f"{label}[{index}].gpu_time_ms"
        )
        if positive_time and gpu_time <= 0.0:
            raise ValueError(f"{label} GPU time must be positive: {timing_key}")
    if set(by_key) != expected:
        raise ValueError(
            f"{label} timing grid differs: missing={sorted(expected - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected)}"
        )

    ranges = report.get(ranges_field)
    if not isinstance(ranges, list):
        raise ValueError(f"{label} rank ranges are missing")
    expected_range_keys = {
        (group, step) for group in groups for step in range(_PRO_PAIR_STEPS)
    }
    range_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(ranges):
        if not isinstance(record, dict):
            raise ValueError(f"{label} rank range {index} is invalid")
        range_key = (
            str(record.get("group", "")),
            _require_json_int(record.get("step"), f"{label}.ranges[{index}].step"),
        )
        if range_key in range_by_key:
            raise ValueError(f"duplicate {label} rank range: {range_key}")
        range_by_key[range_key] = record
    if set(range_by_key) != expected_range_keys:
        raise ValueError(f"{label} rank-range grid differs")

    for group, step in sorted(expected_range_keys):
        values = [
            float(by_key[(group, step, rank)]["gpu_time_ms"])
            for rank in range(_PRO_PAIR_WORLD_SIZE)
        ]
        expected_statistics = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "rank_range": max(values) - min(values),
        }
        expected_statistics["relative_rank_range"] = (
            expected_statistics["rank_range"] / expected_statistics["mean"]
            if expected_statistics["mean"] > 0.0
            else 0.0
        )
        rank_range = range_by_key[(group, step)]
        for field, expected_value in expected_statistics.items():
            actual = _require_finite_number(
                rank_range.get(field), f"{label}.{group}.{step}.{field}"
            )
            if not math.isclose(
                actual,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{label} {group} step {step} {field} differs from records"
                )
        if rank_range.get("unit") != "ms":
            raise ValueError(f"{label} {group} step {step} unit differs")
    return by_key


def _validate_major_kernel_balance(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("result") != "PASS"
        or report.get("balance_gate") != "indexer_score_topk_0.05_others_report_only"
        or report.get("flashmla_forward_kernel") != _PRO_FLASHMLA_FORWARD_KERNEL
        or report.get("flashmla_forward_same_exact_variant") is not True
        or report.get("hard_gate_groups") != list(_PRO_PAIR_HARD_GATE_GROUPS)
    ):
        raise ValueError("Pro-pair major-kernel balance summary did not pass")
    by_key = _validate_timing_grid(
        report,
        groups=_PRO_PAIR_MAJOR_KERNEL_GROUPS,
        ranges_field="rank_ranges",
        label="Pro-pair major-kernel",
        positive_time=True,
    )
    for key, record in by_key.items():
        launch_count = _require_json_int(
            record.get("kernel_launch_count"),
            f"major_kernel.{key}.kernel_launch_count",
            minimum=1,
        )
        names = record.get("kernel_names")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name for name in names)
        ):
            raise ValueError(f"Pro-pair major-kernel names are missing: {key}")
        group, _, _ = key
        if group in ("csa_flashmla_forward", "hca_flashmla_forward") and (
            launch_count != 1 or names != [_PRO_FLASHMLA_FORWARD_KERNEL]
        ):
            raise ValueError(f"Pro-pair FlashMLA forward variant differs: {key}")
    expected_names = {
        group: sorted(
            {
                str(name)
                for (record_group, _, _), record in by_key.items()
                if record_group == group
                for name in record["kernel_names"]
            }
        )
        for group in _PRO_PAIR_MAJOR_KERNEL_GROUPS
    }
    if report.get("kernel_names") != expected_names:
        raise ValueError("Pro-pair major-kernel name summary differs")
    for group in ("csa_flashmla_forward", "hca_flashmla_forward"):
        if expected_names[group] != [_PRO_FLASHMLA_FORWARD_KERNEL]:
            raise ValueError(f"Pro-pair FlashMLA forward group differs: {group}")
    ranges = {
        (str(record["group"]), int(record["step"])): record
        for record in report["rank_ranges"]
    }
    for group in _PRO_PAIR_HARD_GATE_GROUPS:
        for step in range(_PRO_PAIR_STEPS):
            record = ranges[(group, step)]
            relative = float(record["relative_rank_range"])
            if (
                record.get("threshold") != 0.05
                or record.get("passed") is not True
                or relative > 0.05
            ):
                raise ValueError(
                    f"Pro-pair {group} step {step} does not pass the 5% gate"
                )
    return report


def _validate_indexer_d2d(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("result") != "PASS"
        or report.get("accounting") != "separate_from_kernel_gpu_time"
    ):
        raise ValueError("Pro-pair Indexer D2D summary did not pass")
    by_key = _validate_timing_grid(
        report,
        groups=_PRO_PAIR_D2D_GROUPS,
        ranges_field="gpu_time_rank_ranges",
        label="Pro-pair Indexer D2D",
        positive_time=False,
    )
    total_bytes = 0
    total_copy_count = 0
    total_gpu_time_ms = 0.0
    memcpy_rowids: set[int] = set()
    for key, record in by_key.items():
        group, _, _ = key
        if record.get("wrapper_nvtx_name") != _PRO_PAIR_D2D_SCOPES[group]:
            raise ValueError(f"Pro-pair Indexer D2D wrapper scope differs: {key}")
        _require_json_int(
            record.get("wrapper_nvtx_rowid"),
            f"D2D.{key}.wrapper_nvtx_rowid",
        )
        bytes_count = _require_json_int(record.get("bytes"), f"D2D.{key}.bytes")
        copy_count = _require_json_int(
            record.get("copy_count"), f"D2D.{key}.copy_count"
        )
        activity_count = _require_json_int(
            record.get("memcpy_activity_count"), f"D2D.{key}.activity_count"
        )
        rows = record.get("rows")
        if not isinstance(rows, list) or len(rows) != activity_count:
            raise ValueError(f"Pro-pair Indexer D2D rows differ: {key}")
        row_bytes = 0
        row_copies = 0
        row_intervals: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Pro-pair Indexer D2D row is invalid: {key}")
            prefix = f"D2D.{key}.rows[{row_index}]"
            row_bytes += _require_json_int(row.get("bytes"), f"{prefix}.bytes")
            row_copies += _require_json_int(
                row.get("copy_count"), f"{prefix}.copy_count", minimum=1
            )
            start = _require_json_int(
                row.get("memcpy_start_ns"), f"{prefix}.memcpy_start_ns"
            )
            end = _require_json_int(row.get("memcpy_end_ns"), f"{prefix}.memcpy_end_ns")
            if end <= start:
                raise ValueError(f"Pro-pair Indexer D2D interval differs: {key}")
            rowid = _require_json_int(row.get("memcpy_rowid"), f"{prefix}.memcpy_rowid")
            if rowid in memcpy_rowids:
                raise ValueError(f"duplicate Pro-pair Indexer D2D rowid: {rowid}")
            memcpy_rowids.add(rowid)
            _require_json_int(row.get("runtime_rowid"), f"{prefix}.runtime_rowid")
            row_intervals.append((start, end))
        if row_bytes != bytes_count:
            raise ValueError(f"Pro-pair Indexer D2D byte total differs: {key}")
        if row_copies != copy_count:
            raise ValueError(f"Pro-pair Indexer D2D copy total differs: {key}")
        expected_gpu_time_ms = _interval_duration_ns(row_intervals) / 1_000_000.0
        if not math.isclose(
            float(record["gpu_time_ms"]),
            expected_gpu_time_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Pro-pair Indexer D2D GPU time differs: {key}")
        if (
            _require_json_int(
                record.get("wrapper_kernel_launch_count"),
                f"D2D.{key}.wrapper_kernel_launch_count",
                minimum=1,
            )
            < 1
        ):
            raise AssertionError("unreachable")
        for overlap in ("same_mode_external_compute", "same_wrapper_kernel"):
            fraction = _require_finite_number(
                record.get(f"{overlap}_overlap_fraction"),
                f"D2D.{key}.{overlap}_overlap_fraction",
            )
            overlap_ms = _require_finite_number(
                record.get(f"{overlap}_overlap_ms"),
                f"D2D.{key}.{overlap}_overlap_ms",
            )
            expected_fraction = (
                overlap_ms / expected_gpu_time_ms if expected_gpu_time_ms else 0.0
            )
            if (
                fraction > 1.0
                or overlap_ms > expected_gpu_time_ms + 1e-12
                or not math.isclose(
                    fraction,
                    expected_fraction,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Pro-pair Indexer D2D overlap differs: {key}")
        total_bytes += bytes_count
        total_copy_count += copy_count
        total_gpu_time_ms += float(record["gpu_time_ms"])

    outside = report.get("outside_known_scope")
    if not isinstance(outside, dict):
        raise ValueError("Pro-pair Indexer D2D outside-scope audit is missing")
    for field in ("bytes", "copy_count", "memcpy_activity_count"):
        if outside.get(field) != 0:
            raise ValueError("Pro-pair Indexer D2D exists outside known scopes")
    if float(outside.get("gpu_time_ms", -1.0)) != 0.0 or outside.get("rows") != []:
        raise ValueError("Pro-pair Indexer D2D outside-scope rows are non-empty")
    if (
        report.get("total_bytes") != total_bytes
        or report.get("total_copy_count") != total_copy_count
        or not math.isclose(
            _require_finite_number(
                report.get("total_gpu_time_ms"), "D2D.total_gpu_time_ms"
            ),
            total_gpu_time_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Pro-pair Indexer D2D aggregate totals differ")
    return report


def _validate_support_overhead(
    report: dict[str, Any],
    layout: dict[str, Any],
) -> dict[str, Any]:
    if (
        report.get("result") != "PASS"
        or report.get("balance_gate") != "report_only"
        or report.get("grouped_k_pack_backward_csr_reduce_launches") != 0
    ):
        raise ValueError("Pro-pair support-overhead summary did not pass")
    by_key = _validate_timing_grid(
        report,
        groups=_PRO_PAIR_SUPPORT_GROUPS,
        ranges_field="rank_ranges",
        label="Pro-pair support-overhead",
        positive_time=True,
    )
    packing_by_rank = {
        int(record["rank"]): record for record in layout["indexer_k_packing"]
    }
    for key, record in by_key.items():
        group, _, rank = key
        _require_json_int(
            record.get("kernel_launch_count"),
            f"support.{key}.kernel_launch_count",
            minimum=1,
        )
        if group != "csa_grouped_k_pack_forward":
            continue
        packing = packing_by_rank[rank]
        expected = {
            "unique_rows": packing["unique_indexer_k_rows"],
            "packed_rows": packing["packed_indexer_k_rows"],
            "duplicate_rows": packing["duplicate_indexer_k_rows"],
            "unique_bytes": packing["unique_indexer_k_bytes"],
            "packed_bytes": packing["packed_indexer_k_bytes"],
            "duplicate_bytes": packing["duplicate_indexer_k_bytes"],
            "read_bytes": packing["packed_indexer_k_bytes"],
            "write_bytes": packing["packed_indexer_k_bytes"],
            "traffic_bytes": packing["packed_indexer_k_bytes"] * 2,
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Pro-pair grouped Indexer K accounting differs: {key}")
        fraction = _require_finite_number(
            record.get("external_compute_overlap_fraction"),
            f"support.{key}.external_compute_overlap_fraction",
        )
        if fraction > 1.0:
            raise ValueError(f"Pro-pair grouped K overlap exceeds one: {key}")
    return report


def _validate_communication_overlap(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("result") != "PASS"
        or report.get("cross_mode_compute_overlap_is_hard_gate") is not True
        or report.get("overlap_gate") != "report_only"
        or report.get("expected_sendrecv") != "CSA=4F+4B,HCA=3F+3B,total=7F+7B"
    ):
        raise ValueError("Pro-pair communication-overlap summary did not pass")
    expected_classification_counts = {
        "dependency_bound": (
            len(_PRO_PAIR_DEPENDENCY_BOUND_ROUTE_GROUPS)
            * _PRO_PAIR_WORLD_SIZE
            * _PRO_PAIR_STEPS
        ),
        "overlap_capable": (
            len(_PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS)
            * _PRO_PAIR_WORLD_SIZE
            * _PRO_PAIR_STEPS
        ),
    }
    expected_contract = {
        "classification_counts": expected_classification_counts,
        "dependency_bound_requires_positive_overlap": False,
        "fraction_threshold": None,
        "overlap_capable_requires_positive_overlap": False,
        "positive_time_threshold_ns": None,
    }
    if report.get("overlap_contract") != expected_contract:
        raise ValueError("Pro-pair route overlap contract differs")
    by_key = _validate_timing_grid(
        report,
        groups=_PRO_PAIR_ROUTE_GROUPS,
        ranges_field="rank_ranges",
        label="Pro-pair route",
        positive_time=True,
    )
    for key, record in by_key.items():
        group, _, _ = key
        mode, direction, route = group.split(".")
        if (
            record.get("mode") != mode
            or record.get("direction") != direction
            or record.get("route") != route
        ):
            raise ValueError(f"Pro-pair route identity differs: {key}")
        start = _require_json_int(record.get("kernel_start_ns"), f"route.{key}.start")
        end = _require_json_int(record.get("kernel_end_ns"), f"route.{key}.end")
        _require_json_int(record.get("runtime_start_ns"), f"route.{key}.runtime_start")
        if end <= start:
            raise ValueError(f"Pro-pair route interval is invalid: {key}")
        expected_scope = (
            "magi_dsa::phase::collective_all2all_v::"
            f"attention::{mode}::{route}.{direction}"
        )
        path = record.get("nvtx_path")
        if not isinstance(path, list) or expected_scope not in {
            str(scope.get("name", "")) for scope in path if isinstance(scope, dict)
        }:
            raise ValueError(f"Pro-pair route NVTX path differs: {key}")
        if (
            float(record.get("other_mode_compute_overlap_ms", -1.0)) != 0.0
            or float(record.get("other_mode_compute_overlap_fraction", -1.0)) != 0.0
        ):
            raise ValueError(f"Pro-pair route overlaps other-mode compute: {key}")
        expected_classification = (
            "overlap_capable"
            if group in _PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS
            else "dependency_bound"
        )
        overlap_ns = _require_json_int(
            record.get("same_mode_compute_overlap_ns"),
            f"route.{key}.same_mode_compute_overlap_ns",
        )
        reason = record.get("overlap_contract_reason")
        if (
            record.get("overlap_classification") != expected_classification
            or not isinstance(reason, str)
            or not reason.strip()
            or record.get("observed_positive_same_mode_compute_overlap")
            is not (overlap_ns > 0)
        ):
            raise ValueError(f"Pro-pair route overlap classification differs: {key}")
        fraction = _require_finite_number(
            record.get("same_mode_compute_overlap_fraction"),
            f"route.{key}.same_mode_compute_overlap_fraction",
        )
        overlap_ms = _require_finite_number(
            record.get("same_mode_compute_overlap_ms"),
            f"route.{key}.same_mode_compute_overlap_ms",
        )
        if not math.isclose(
            overlap_ms,
            overlap_ns / 1_000_000.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Pro-pair route overlap ns/ms totals differ: {key}")
        if fraction > 1.0 or overlap_ms > float(record["gpu_time_ms"]) + 1e-12:
            raise ValueError(f"Pro-pair route overlap accounting differs: {key}")

    for rank in range(_PRO_PAIR_WORLD_SIZE):
        for step in range(_PRO_PAIR_STEPS):
            for mode, directions in _PRO_PAIR_ROUTE_ORDER.items():
                for direction, routes in directions.items():
                    launch_starts = [
                        int(
                            by_key[(f"{mode}.{direction}.{route}", step, rank)][
                                "runtime_start_ns"
                            ]
                        )
                        for route in routes
                    ]
                    if launch_starts != sorted(launch_starts):
                        raise ValueError(
                            "Pro-pair route launch order differs: "
                            f"rank={rank}, step={step}, mode={mode}, "
                            f"direction={direction}"
                        )
    return report


def _validate_pro_pair_summary(
    summary: dict[str, Any],
    *,
    major: dict[str, Any],
    d2d: dict[str, Any],
    support: dict[str, Any],
    communication: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "attention_order": ["csa", "hca"],
        "backward_order": ["hca", "csa"],
        "expected_sendrecv": {
            "backward": 7,
            "csa_backward": 4,
            "csa_forward": 4,
            "forward": 7,
            "hca_backward": 3,
            "hca_forward": 3,
        },
        "excluded_capture_ranges": {
            "loss": 0,
            "projection": 0,
            "token_layout": 0,
            "w_mode": 0,
        },
        "flashmla_forward_kernel": _PRO_FLASHMLA_FORWARD_KERNEL,
        "flashmla_forward_same_exact_variant": True,
        "independent_attention_graphs": True,
        "kernel_attribution_coverage": 1.0,
        "major_kernel_balance_gate": ("indexer_score_topk_0.05_others_report_only"),
        "major_kernel_groups": list(_PRO_PAIR_MAJOR_KERNEL_GROUPS),
        "memcpy_attribution_coverage": 1.0,
        "mode_backward_completion_join": {
            "csa": [
                "sparse_backward_stream",
                "csa_main_stream",
                "csa_indexer_stream",
                "csa_route_stream",
            ],
            "hca": ["hca_main_stream", "hca_route_stream"],
        },
        "mode_serialization": "cuda_event_happens_before",
        "parameter_gradient_allreduce": "one_unified_after_two_backwards",
        "parameter_gradient_allreduce_in_7f7b": False,
        "parameter_gradient_reducer_precision": (
            "fp32_cp_bucket_model_side_diagnostic"
        ),
        "pro_runtime_bundle": True,
        "ratios": [4, 128],
        "representative_layer_ids": {"csa": 2, "hca": 3},
        "representative_pair_semantics": (
            "independent_post_projection_graphs_serialized_in_layer_order"
        ),
        "result": "PASS",
        "runtime_parameter_gradient_communication": False,
        "shared_source_packed_meta": True,
        "shared_source_x": True,
        "step_mode": "pro-pair",
        "steps": _PRO_PAIR_STEPS,
        "token_layout_invocations": 1,
        "world_size": _PRO_PAIR_WORLD_SIZE,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(f"Pro-pair summary {field} differs")
    if summary.get("grouped_k_pack_backward_csr_reduce_launches") != 0:
        raise ValueError("Pro-pair grouped Indexer K backward CSR was not eliminated")
    if summary.get("support_overhead_groups") != list(_PRO_PAIR_SUPPORT_GROUPS):
        raise ValueError("Pro-pair support-overhead groups differ")
    if summary.get("route_timing_records") != len(communication["records"]):
        raise ValueError("Pro-pair route record total differs")
    _require_json_int(
        summary.get("kernel_attribution_records"),
        "summary.kernel_attribution_records",
        minimum=1,
    )
    _require_json_int(
        summary.get("memcpy_attribution_records"),
        "summary.memcpy_attribution_records",
    )

    layout = _validate_pro_pair_layout(summary.get("layout"))
    if summary.get("shared_bundle_query_layout_hash") != layout["query_layout_hash"]:
        raise ValueError("Pro-pair CSA/HCA shared Query layout hash differs")
    indexer_d2d = summary.get("indexer_d2d")
    expected_d2d = {
        "outside_known_scope": d2d["outside_known_scope"],
        "total_bytes": d2d["total_bytes"],
        "total_copy_count": d2d["total_copy_count"],
        "total_gpu_time_ms": d2d["total_gpu_time_ms"],
    }
    if indexer_d2d != expected_d2d:
        raise ValueError("Pro-pair summary and Indexer D2D artifact differ")
    if summary.get("major_kernel_groups") != sorted(
        {str(record["group"]) for record in major["records"]}
    ):
        raise ValueError("Pro-pair summary and major-kernel artifact differ")
    if summary.get("support_overhead_groups") != sorted(
        {str(record["group"]) for record in support["records"]}
    ):
        raise ValueError("Pro-pair summary and support-overhead artifact differ")
    return {**summary, "layout": layout}


def _validate_cp8_structural_reports(
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    query_hashes: set[str] = set()
    plan_hashes: dict[str, set[str]] = {label: set() for label in _CP8_PLAN_POLICIES}
    for rank, report in enumerate(reports):
        evidence = report.get("plan_evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(_CP8_PLAN_POLICIES):
            raise ValueError(f"rank {rank} CP8 plan evidence is incomplete")
        validated: dict[str, dict[str, Any]] = {}
        for label, policy in _CP8_PLAN_POLICIES.items():
            plan = evidence[label]
            if not isinstance(plan, dict) or plan.get("policy") != policy:
                raise ValueError(f"rank {rank} {label} plan policy differs")
            expected_ratio = (
                4 if label.startswith("csa_") else 128 if label.startswith("hca") else 0
            )
            if plan.get("ratio") != expected_ratio:
                raise ValueError(f"rank {rank} {label} ratio differs")
            _require_sha256(plan.get("plan_hash"), f"rank {rank}.{label}.plan_hash")
            _require_sha256(
                plan.get("query_layout_hash"),
                f"rank {rank}.{label}.query_layout_hash",
            )
            _require_sha256(
                plan.get("rank_query_layout_signature"),
                f"rank {rank}.{label}.rank_query_layout_signature",
            )
            source_counts = plan.get("source_token_counts")
            query_counts = plan.get("query_token_counts")
            if source_counts != [32] * 8 or query_counts != [32] * 8:
                raise ValueError(f"rank {rank} {label} token-count table differs")
            if (
                plan.get("local_source_tokens") != 32
                or plan.get("local_query_tokens") != 32
                or plan.get("declared_local_token_capacity") != 32
            ):
                raise ValueError(f"rank {rank} {label} local capacity differs")
            if policy == "structural_balanced":
                metrics = plan.get("structural_layout_metrics")
                rank_cost = plan.get("structural_rank_cost")
                if (
                    plan.get("structural_layout_config") != _STRUCTURAL_CONFIG
                    or not isinstance(metrics, dict)
                    or not str(metrics.get("solver_scheme", ""))
                    or not str(metrics.get("cost_model_version", ""))
                    or not isinstance(rank_cost, dict)
                    or rank_cost.get("rank") != rank
                    or rank_cost.get("query_tokens") != 32
                ):
                    raise ValueError(
                        f"rank {rank} {label} structural solver evidence differs"
                    )
            validated[label] = plan
            plan_hashes[label].add(str(plan["plan_hash"]))

        csa = validated["csa_structural"]
        hca = validated["hca_structural"]
        shared_fields = (
            "query_layout_hash",
            "rank_query_layout_signature",
            "source_token_counts",
            "query_token_counts",
            "structural_layout_config",
            "structural_layout_metrics",
            "structural_rank_cost",
        )
        if any(csa[field] != hca[field] for field in shared_fields):
            raise ValueError(f"rank {rank} structural CSA/HCA layouts differ")
        if (
            report.get("structural_layout_shared") is not True
            or report.get("structural_query_layout_hash") != csa["query_layout_hash"]
        ):
            raise ValueError(f"rank {rank} shared structural proof differs")
        query_hashes.add(str(csa["query_layout_hash"]))

    if len(query_hashes) != 1 or any(
        len(hashes) != 1 for hashes in plan_hashes.values()
    ):
        raise ValueError("CP8 ranks did not execute identical structural plans")
    return {
        "plan_hashes": {
            label: next(iter(hashes)) for label, hashes in plan_hashes.items()
        },
        "query_layout_hash": next(iter(query_hashes)),
        "query_token_counts": [32] * 8,
        "result": "PASS",
    }


def _validate_correctness_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if (
        summary.get("case") != "cp8-natural-backward"
        or summary.get("world_size") != 8
        or summary.get("result_count") != 8
        or summary.get("model_parameter_value_check_ranks") != [0]
    ):
        raise ValueError("installed-wheel CP8 summary is incomplete")
    execution = summary.get("execution_seconds")
    if not isinstance(execution, dict):
        raise ValueError("installed-wheel CP8 summary is missing execution timings")
    minimum = execution.get("min")
    maximum = execution.get("max")
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        raise ValueError("installed-wheel CP8 min timing is not numeric")
    if not isinstance(maximum, (int, float)) or isinstance(maximum, bool):
        raise ValueError("installed-wheel CP8 max timing is not numeric")
    if minimum < 0.0 or maximum < minimum or maximum >= 60.0:
        raise ValueError("installed-wheel CP8 execution timings violate the deadline")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 8:
        raise ValueError("installed-wheel CP8 summary does not contain eight results")
    structural = _validate_cp8_structural_reports(results)
    if summary.get("structural_contract") != structural:
        raise ValueError("installed-wheel CP8 structural summary differs")
    validated = dict(summary)
    validated["result"] = "PASS"
    return validated


def _validate_correctness(path: Path, revision: str) -> dict[str, Any]:
    _require_artifact_provenance(path, revision, "installed-wheel CP8")
    _require_clean_artifact(path, "installed-wheel CP8")
    _validate_phase_audit(
        _read_json(path / "PHASE_AUDIT.json"),
        world_size=8,
        label="installed-wheel CP8",
    )
    summary = _validate_correctness_summary(_read_json(path / "SUMMARY.json"))
    reports = sorted(path.glob("result_rank*.json"))
    if len(reports) != 8:
        raise ValueError(f"installed-wheel CP8 produced {len(reports)} rank reports")
    expected_version = f"1.1.1+g{revision}"
    raw_reports: list[dict[str, Any]] = []
    for rank, report_path in enumerate(reports):
        report = _read_json(report_path)
        raw_reports.append(report)
        execution_seconds = report.get("execution_seconds")
        if (
            report.get("case") != "cp8-natural-backward"
            or report.get("rank") != rank
            or not isinstance(execution_seconds, (int, float))
            or isinstance(execution_seconds, bool)
            or execution_seconds < 0.0
            or execution_seconds >= 60.0
        ):
            raise ValueError(f"rank {rank} CP8 result violates the execution contract")
        installed = report.get("installed_wheel")
        if not isinstance(installed, dict):
            raise ValueError(f"rank {rank} did not report installed-wheel provenance")
        if installed.get("source_revision") != revision:
            raise ValueError(f"rank {rank} installed-wheel revision mismatch")
        if installed.get("package_version") != expected_version:
            raise ValueError(f"rank {rank} installed-wheel version mismatch")
        package_parts = set(Path(str(installed.get("package_path"))).parts)
        if not package_parts.intersection({"site-packages", "dist-packages"}):
            raise ValueError(
                f"rank {rank} did not import Magi-DSA from a Python installation directory"
            )
    structural = _validate_cp8_structural_reports(raw_reports)
    if summary.get("structural_contract") != structural:
        raise ValueError("installed-wheel CP8 report structural evidence differs")
    return summary


def _validate_phase_audit(
    value: dict[str, Any],
    *,
    world_size: int,
    label: str,
) -> None:
    if (
        value.get("world_size") != world_size
        or value.get("result") != "PASS"
        or value.get("failure_reasons") != []
        or value.get("error_count") != 0
        or value.get("open_phase_count") != 0
        or value.get("timed_out") is not False
        or value.get("collective_stall_confirmed") is not False
        or value.get("all_ranks_execute_started") is not True
        or value.get("all_ranks_execute_ended") is not True
        or value.get("latest_open_phases") != [None] * world_size
    ):
        raise ValueError(f"{label} phase audit did not pass")
    ranks = value.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError(f"{label} phase audit rank grid differs")
    for rank, record in enumerate(ranks):
        if (
            not isinstance(record, dict)
            or record.get("rank") != rank
            or record.get("errors") != []
            or record.get("open_phases") != []
            or record.get("execute_started") is not True
            or record.get("execute_ended") is not True
        ):
            raise ValueError(f"{label} rank {rank} phase audit did not pass")


def _read_key_value_contract(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise ValueError(f"invalid key-value contract line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate key-value contract field in {path}: {key!r}")
        result[key] = value
    return result


def _validate_cp1(path: Path, revision: str) -> dict[str, Any]:
    _require_artifact_provenance(path, revision, "CP1")
    required = (
        "COMMAND.txt",
        "DIRTY_STATUS.txt",
        "IMAGE.json",
        "IMAGE_CONTRACT.txt",
        "INSTALLED_PACKAGE.json",
        "PYTEST.xml",
        "RAW.json",
        "STDERR.txt",
        "STDOUT.txt",
        "SUMMARY.json",
    )
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(f"missing CP1 artifact: {path / name}")
    if (path / "DIRTY_STATUS.txt").read_text(encoding="utf-8").strip():
        raise ValueError("CP1 artifact was produced from a dirty worktree")
    command = _read_key_value_contract(path / "COMMAND.txt")
    expected_command = {
        "case": "cp1-kernel",
        "package_import": "installed-wheel",
        "pytest": "tests/dsa_v4/test_cp1_kernel.py",
        "pytest_import_mode": "importlib",
        "source_mount": "read-only",
        "source_revision": revision,
    }
    if any(command.get(field) != value for field, value in expected_command.items()):
        raise ValueError("CP1 command contract differs")

    image_contract = _read_key_value_contract(path / "IMAGE_CONTRACT.txt")
    expected_image_contract = {
        "flashmla_base_revision": "9241ae3ef9bac614dd25e45e507e089f888280e0",
        "flashmla_dual_lse_patch_revision": _FLASHMLA_DUAL_LSE_PATCH_REVISION,
        "flashmla_dual_lse_patch_sha256": _FLASHMLA_DUAL_LSE_PATCH_SHA256,
        "flashmla_pro_h128_patch_revision": _FLASHMLA_PRO_H128_PATCH_REVISION,
        "flashmla_pro_h128_patch_sha256": _FLASHMLA_PRO_H128_PATCH_SHA256,
        "cudnn_backend_version": "9.24.0.43",
        "cudnn_frontend_version": "1.26.0",
        "cudnn_frontend_revision": "35fd7b0d0e1d4952b904c79341c5e84e3af0a328",
        "cudnn_frontend_source": "official-unmodified",
        "cudnn_frontend_local_patches": "none",
        "cutlass_dsl_version": "4.5.0",
        "quack_version": "0.4.1",
        "tvm_ffi_version": "0.1.8.post0",
        "magi_source_revision": revision,
        "install_mode": "python-wheel",
        "validation": "all_required_image_labels_exact",
    }
    if image_contract != expected_image_contract:
        raise ValueError("CP1 image contract differs")

    summary = _read_json(path / "SUMMARY.json")
    expected_summary = {
        "case": "cp1-kernel",
        "errors": 0,
        "failures": 0,
        "image_contract": "PASS",
        "pytest_exit_status": 0,
        "result": "PASS",
        "skipped": 0,
        "source_dirty": False,
        "source_revision": revision,
        "tests": 6,
    }
    if any(summary.get(field) != value for field, value in expected_summary.items()):
        raise ValueError("CP1 summary did not pass the cp1-kernel contract")
    if not str(summary.get("image", "")) or not str(summary.get("image_id", "")):
        raise ValueError("CP1 image provenance is missing")
    installed = _read_json(path / "INSTALLED_PACKAGE.json")
    if summary.get("installed_package") != installed:
        raise ValueError("CP1 installed-package summary differs")
    package_parts = set(Path(str(installed.get("package_path", ""))).parts)
    if not package_parts.intersection({"site-packages", "dist-packages"}):
        raise ValueError("CP1 did not import the installed magi_attention wheel")
    if installed.get("package_version") != f"1.1.1+g{revision}":
        raise ValueError("CP1 installed magi_attention version differs")

    xml_path = path / "PYTEST.xml"
    xml_root = ET.parse(xml_path).getroot()
    xml_counts = {
        field: int(xml_root.attrib.get(field, 0))
        for field in ("tests", "failures", "errors", "skipped")
    }
    if xml_root.tag == "testsuites" and xml_counts["tests"] == 0:
        for suite in xml_root.findall("testsuite"):
            for field in xml_counts:
                xml_counts[field] += int(suite.attrib.get(field, 0))
    expected_counts = {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}
    if xml_counts != expected_counts:
        raise ValueError("CP1 JUnit result differs")
    raw = _read_json(path / "RAW.json")
    expected_raw = {
        "case": "cp1-kernel",
        "pytest_exit_status": 0,
        **expected_counts,
    }
    if any(raw.get(field) != value for field, value in expected_raw.items()):
        raise ValueError("CP1 raw result differs")
    junit_sha256 = _require_sha256(raw.get("junit_sha256"), "CP1.junit_sha256")
    if junit_sha256 != _sha256(xml_path):
        raise ValueError("CP1 JUnit digest differs")
    return summary


def _validate_cp2_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if (
        summary.get("case") != "csa-natural-backward"
        or summary.get("world_size") != 2
        or summary.get("result_count") != 2
    ):
        raise ValueError("CP2 summary case or rank grid differs")
    execution = summary.get("execution_seconds")
    if not isinstance(execution, dict):
        raise ValueError("CP2 execution timing summary is missing")
    minimum = _require_finite_number(execution.get("min"), "CP2.execution.min")
    maximum = _require_finite_number(execution.get("max"), "CP2.execution.max")
    if maximum < minimum or maximum >= 60.0:
        raise ValueError("CP2 execution timings violate the 60-second deadline")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("CP2 summary does not contain two rank results")
    plan_hashes: set[str] = set()
    query_layout_hashes: set[str] = set()
    query_count_tables: set[tuple[int, ...]] = set()
    source_count_tables: set[tuple[int, ...]] = set()
    metric_payloads: set[str] = set()
    rank_execution_seconds: list[float] = []
    for rank, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"CP2 rank {rank} result is invalid")
        elapsed = _require_finite_number(
            result.get("execution_seconds"), f"CP2.rank{rank}.execution_seconds"
        )
        if (
            result.get("rank") != rank
            or result.get("case") != "csa-natural-backward"
            or elapsed >= 60.0
            or result.get("balanced_policy") != "structural_balanced"
            or _require_json_int(
                result.get("parameter_gradients"),
                f"CP2.rank{rank}.parameter_gradients",
                minimum=1,
            )
            < 1
        ):
            raise ValueError(f"CP2 rank {rank} result did not pass")
        evidence = result.get("balanced_plan_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"CP2 rank {rank} structural evidence is missing")
        if (
            evidence.get("policy") != "structural_balanced"
            or evidence.get("ratio") != 4
            or evidence.get("structural_layout_config") != _STRUCTURAL_CONFIG
        ):
            raise ValueError(f"CP2 rank {rank} structural policy differs")
        plan_hashes.add(
            _require_sha256(evidence.get("plan_hash"), f"CP2.rank{rank}.plan_hash")
        )
        query_layout_hashes.add(
            _require_sha256(
                evidence.get("query_layout_hash"),
                f"CP2.rank{rank}.query_layout_hash",
            )
        )
        _require_sha256(
            evidence.get("rank_query_layout_signature"),
            f"CP2.rank{rank}.rank_query_layout_signature",
        )
        source_counts = evidence.get("source_token_counts")
        query_counts = evidence.get("query_token_counts")
        if (
            not isinstance(source_counts, list)
            or not isinstance(query_counts, list)
            or len(source_counts) != 2
            or len(query_counts) != 2
            or any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in (*source_counts, *query_counts)
            )
            or sum(source_counts) != sum(query_counts)
            or evidence.get("local_source_tokens") != source_counts[rank]
            or evidence.get("local_query_tokens") != query_counts[rank]
            or _require_json_int(
                evidence.get("declared_local_token_capacity"),
                f"CP2.rank{rank}.capacity",
            )
            < max(source_counts[rank], query_counts[rank])
            or _require_json_int(
                evidence.get("fragment_count"),
                f"CP2.rank{rank}.fragment_count",
                minimum=1,
            )
            < 1
        ):
            raise ValueError(f"CP2 rank {rank} Query coverage differs")
        source_count_tables.add(tuple(source_counts))
        query_count_tables.add(tuple(query_counts))
        rank_execution_seconds.append(elapsed)
        metrics = evidence.get("structural_layout_metrics")
        cost = evidence.get("structural_rank_cost")
        if (
            not isinstance(metrics, dict)
            or not str(metrics.get("solver_scheme", ""))
            or not str(metrics.get("cost_model_version", ""))
            or metrics.get("uneven_shard") is not True
            or not isinstance(cost, dict)
            or cost.get("rank") != rank
            or cost.get("query_tokens") != query_counts[rank]
        ):
            raise ValueError(f"CP2 rank {rank} structural solver evidence differs")
        metric_payloads.add(json.dumps(metrics, sort_keys=True))
    if (
        len(plan_hashes) != 1
        or len(query_layout_hashes) != 1
        or len(source_count_tables) != 1
        or len(query_count_tables) != 1
        or len(metric_payloads) != 1
    ):
        raise ValueError("CP2 ranks did not execute one structural plan")
    if minimum != min(rank_execution_seconds) or maximum != max(rank_execution_seconds):
        raise ValueError("CP2 execution timing summary differs from rank results")
    validated = dict(summary)
    validated["result"] = "PASS"
    return validated


def _validate_cp2(path: Path, revision: str) -> dict[str, Any]:
    _require_artifact_provenance(path, revision, "CP2")
    _require_clean_artifact(path, "CP2")
    summary = _validate_cp2_summary(_read_json(path / "SUMMARY.json"))
    _validate_phase_audit(
        _read_json(path / "PHASE_AUDIT.json"), world_size=2, label="CP2"
    )
    return summary


def _validate_profile(path: Path, revision: str) -> dict[str, Any]:
    _require_artifact_provenance(path, revision, "Pro-pair profile")
    _require_clean_artifact(path, "Pro-pair profile")
    for relative in _PRO_PAIR_REQUIRED_ARTIFACTS:
        artifact = path / relative
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise FileNotFoundError(f"missing Pro-pair profile artifact: {artifact}")
    _validate_pro_pair_workload(_read_json(path / "WORKLOAD.json"))
    inventory = (path / "PRO_PAIR_ARTIFACTS.txt").read_text(encoding="utf-8")
    for marker in (
        "capture_source=balanced/balanced_5steps_pro_pair.nsys-rep",
        "capture_replay=none",
        "extractor=single_export_from_same_aggregate_capture",
    ):
        if marker not in inventory.splitlines():
            raise ValueError(f"Pro-pair artifact inventory is missing {marker}")

    major = _validate_major_kernel_balance(
        _read_json(path / "MAJOR_KERNEL_BALANCE_PRO_PAIR.json")
    )
    d2d = _validate_indexer_d2d(_read_json(path / "INDEXER_D2D_PRO_PAIR.json"))
    communication = _validate_communication_overlap(
        _read_json(path / "PRO_PAIR_COMMUNICATION_OVERLAP.json")
    )
    raw_summary = _read_json(path / "SUMMARY_PRO_PAIR.json")
    layout = _validate_pro_pair_layout(raw_summary.get("layout"))
    support = _validate_support_overhead(
        _read_json(path / "SUPPORT_OVERHEAD_PRO_PAIR.json"), layout
    )
    return _validate_pro_pair_summary(
        raw_summary,
        major=major,
        d2d=d2d,
        support=support,
        communication=communication,
    )


def _profile_rows(major: dict[str, Any]) -> list[str]:
    ranges = {
        (str(record["group"]), int(record["step"])): record
        for record in major["rank_ranges"]
    }
    lines = [
        "| Step | Kernel group | Min ms | Max ms | Relative rank range | Gate |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for step in range(5):
        for group in _PRO_PAIR_MAJOR_KERNEL_GROUPS:
            record = ranges[(group, step)]
            gate = "PASS" if group in _PRO_PAIR_HARD_GATE_GROUPS else "report-only"
            lines.append(
                "| {step} | {group} | {minimum:.6f} | {maximum:.6f} | "
                "{relative:.6f} | {gate} |".format(
                    step=step,
                    group=group,
                    minimum=float(record["min"]),
                    maximum=float(record["max"]),
                    relative=float(record["relative_rank_range"]),
                    gate=gate,
                )
            )
    return lines


def _report(
    *,
    revision: str,
    image: str,
    image_id: str,
    correctness: Path,
    profile: Path,
    cp1: Path,
    cp2: Path,
    major: dict[str, Any],
    d2d: dict[str, Any],
    communication: dict[str, Any],
) -> str:
    ranges = major["rank_ranges"]
    worst_score = max(
        record["relative_rank_range"]
        for record in ranges
        if record["group"] == "csa_indexer_score"
    )
    worst_topk = max(
        record["relative_rank_range"]
        for record in ranges
        if record["group"] == "csa_indexer_topk"
    )
    overlap_capable_records = sum(
        record["overlap_classification"] == "overlap_capable"
        and int(record["same_mode_compute_overlap_ns"]) > 0
        for record in communication["records"]
    )
    expected_overlap_capable_records = (
        len(_PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS)
        * _PRO_PAIR_WORLD_SIZE
        * _PRO_PAIR_STEPS
    )
    lines = [
        "# Magi-DSA DeepSeek-V4-Pro Release Report",
        "",
        "## 结论",
        "",
        "- DeepSeek-V4-Pro 31×HCA + 30×CSA implementation、CP correctness、正式五步",
        "  Pro-pair profile、",
        "  installed-wheel 镜像与追溯证据均已完成。",
        f"- Clean revision：`{revision}`。",
        f"- Release image：`{image}`（`{image_id}`）。",
        "- 本次未重复用户取消的 smoke；沿用已封存 smoke 证据，release 镜像改用 CP8 natural",
        "  forward/backward 验证 installed wheel。",
        "",
        "## 实现摘要",
        "",
        "- parameter-free `MagiDSARuntimeMgr` 与模型侧 `MagiDSALayer` 参数归属。",
        "- CSA/HCA 共享一次 `structural_balanced` Query layout；代表层按 CSA→HCA forward、",
        "  HCA→CSA backward 连续执行五轮，每轮通信总账为 7F+7B。",
        "- CSA/HCA FlashMLA forward 每个 rank/step 均固定单次调用",
        f"  `{_PRO_FLASHMLA_FORWARD_KERNEL}`，两个 mode 使用完全相同的 kernel variant。",
        "- Q16 直接保留 cuDNN backend-native Top-K IDs 与位序，只做 sample-local 到",
        "  canonical-global offset、effective length 和负值 padding。exact-cutoff tie",
        "  允许集合或位序不同；仅 canonical 集合相同的 Query 行执行 output 数值 gate。",
        "- Indexer D2D、support pack/cat/CSR 与 14 条 route 的通信/计算 overlap 均从",
        "  同一份 balanced aggregate capture 独立归因，没有 capture replay。",
        "",
        "## 实际验证与结果路径",
        "",
        f"- CP1 reference/kernel 通过证据：`{cp1}`。",
        f"- CP2 natural backward 通过证据：`{cp2}`。",
        f"- Installed-wheel CP8 natural backward：`{correctness}`。",
        f"- 正式 Pro-pair profile：`{profile}`。",
        "- Balanced Pro-pair Nsight："
        f"`{profile / 'balanced' / 'balanced_5steps_pro_pair.nsys-rep'}`。",
        "- 完整命令见本目录 `COMMAND.txt`，构建输出见 `BUILD.log`，CP8 输出见 `CP8.log`。",
        "",
        "## 大 kernel 负载均衡",
        "",
        *_profile_rows(major),
        "",
        f"- CSA Indexer score 五步最差 relative rank range：`{worst_score:.6f}`。",
        f"- CSA Indexer Top-K 五步最差 relative rank range：`{worst_topk:.6f}`。",
        "- 10/10 Indexer step/group 均满足 `<=0.05`；其余大 kernel 为 report-only。",
        "",
        "## D2D 与通信 overlap",
        "",
        f"- Indexer/selected-KL D2D：copies=`{d2d['total_copy_count']}`，",
        f"  bytes=`{d2d['total_bytes']}`，GPU time=`{d2d['total_gpu_time_ms']:.6f} ms`。",
        "- 已知 cuDNN scope 外的 Indexer/selected-KL D2D 为 0。",
        f"- 14 routes × 8 ranks × 5 steps 共 `{len(communication['records'])}` 条记录；",
        f"  `{overlap_capable_records}/{expected_overlap_capable_records}` 条 overlap-capable",
        "  route 观察到同 mode/同 direction compute 正时间交集；该数量只报告、不设门槛。",
        "- HCA OX/CKV 四条 dependency-bound route 不借用其他 mode 伪造 overlap；跨 mode",
        "  compute overlap=0 仍为 release hard gate。",
        "",
        "## 已知风险",
        "",
        "- Release wheel 为 DSA Python runtime 范围，不编译 Magi 的 legacy CUDA extension；",
        "  DSA 使用的 torch NCCL All2AllV、CuTe pack、cuDNN DSA 与 FlashMLA 路径已由",
        "  installed-wheel CP8 验证。",
        "- 代表性 Pro-pair 是两张独立 post-projection Attention 图，不冒充具有跨层",
        "  activation 依赖的完整 61 层 Transformer 训练 profile。",
        "",
        "## 未完成项",
        "",
        "- Magi-DSA v4 已批准范围内：无。",
        "- decode/cache、TP、FP8/FP4、CUDA Graph 与 selected-KV routing 属于明确排除范围。",
        "",
    ]
    return "\n".join(lines)


def _write_manifest(artifact_dir: Path) -> None:
    entries: list[str] = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(artifact_dir)}")
    _write_text(artifact_dir / "SHA256SUMS", "\n".join(entries) + "\n")


def _write_release_summary_and_manifest(
    artifact_dir: Path, summary: dict[str, object]
) -> str:
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    _write_text(artifact_dir / "SUMMARY.json", payload)
    _write_text(artifact_dir / "FINALIZE.stdout", payload)
    _write_manifest(artifact_dir)
    return payload


def _expected_release_image_labels(revision: str) -> dict[str, str]:
    return {
        "org.magi-dsa.cudnn-backend": "9.24.0.43",
        "org.magi-dsa.cudnn-frontend": "1.26.0",
        "org.magi-dsa.cudnn-frontend-local-patches": "none",
        "org.magi-dsa.cudnn-frontend-revision": (
            "35fd7b0d0e1d4952b904c79341c5e84e3af0a328"
        ),
        "org.magi-dsa.cudnn-frontend-source": "official-unmodified",
        "org.magi-dsa.cutlass-dsl": "4.5.0",
        "org.magi-dsa.flashmla-dual-lse-patch-revision": (
            _FLASHMLA_DUAL_LSE_PATCH_REVISION
        ),
        "org.magi-dsa.flashmla-dual-lse-patch-sha256": (
            _FLASHMLA_DUAL_LSE_PATCH_SHA256
        ),
        "org.magi-dsa.flashmla-pro-h128-patch-revision": (
            _FLASHMLA_PRO_H128_PATCH_REVISION
        ),
        "org.magi-dsa.flashmla-pro-h128-patch-sha256": (
            _FLASHMLA_PRO_H128_PATCH_SHA256
        ),
        "org.magi-dsa.flashmla-revision": ("9241ae3ef9bac614dd25e45e507e089f888280e0"),
        "org.magi-dsa.install-mode": "python-wheel",
        "org.magi-dsa.magi-attention-revision": revision,
        "org.magi-dsa.magi-attn-extensions-revision": revision,
        "org.magi-dsa.magi-attn-extensions-version": _MAGI_ATTN_EXTENSIONS_VERSION,
        "org.magi-dsa.quack-kernels": "0.4.1",
        "org.magi-dsa.tvm-ffi": "0.1.8.post0",
    }


def _validate_release_image_labels(labels: object, revision: str) -> None:
    if not isinstance(labels, dict):
        raise ValueError("release image labels are missing")
    expected_labels = _expected_release_image_labels(revision)
    label_mismatches = {
        label: {"actual": labels.get(label), "expected": expected}
        for label, expected in expected_labels.items()
        if labels.get(label) != expected
    }
    if label_mismatches:
        raise ValueError(f"release image labels differ: {label_mismatches}")


def main() -> None:
    args = _parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("--revision must be an exact 40-character commit")
    artifact_dir = args.artifact_dir.resolve()
    correctness = args.correctness_artifact.resolve()
    profile = args.profile_artifact.resolve()
    cp1 = args.cp1_artifact.resolve()
    cp2 = args.cp2_artifact.resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"release artifact directory does not exist: {artifact_dir}"
        )

    repo_root = Path(
        _run(["git", "rev-parse", "--show-toplevel"], timeout_seconds=30).strip()
    )
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_seconds=30).strip()
    if head != args.revision:
        raise ValueError(f"HEAD {head} does not match release revision {args.revision}")
    dirty = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    if dirty:
        raise ValueError("release finalization requires a clean worktree")

    correctness_summary = _validate_correctness(correctness, args.revision)
    profile_summary = _validate_profile(profile, args.revision)
    cp1_summary = _validate_cp1(cp1, args.revision)
    cp2_summary = _validate_cp2(cp2, args.revision)
    major = _read_json(profile / "MAJOR_KERNEL_BALANCE_PRO_PAIR.json")
    d2d = _read_json(profile / "INDEXER_D2D_PRO_PAIR.json")
    communication = _read_json(profile / "PRO_PAIR_COMMUNICATION_OVERLAP.json")

    inspect_output = _run(
        ["docker", "image", "inspect", args.image],
        cwd=repo_root,
        timeout_seconds=30,
    )
    inspect = json.loads(inspect_output)
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise ValueError("docker image inspect returned an unexpected payload")
    image_record = inspect[0]
    if cp1_summary.get("image") != args.image or cp1_summary.get(
        "image_id"
    ) != image_record.get("Id"):
        raise ValueError("CP1 artifact was not produced by the release image")
    labels = image_record.get("Config", {}).get("Labels", {})
    _validate_release_image_labels(labels, args.revision)

    environment = _run(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "60s",
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "bash",
            "--workdir",
            "/tmp",
            args.image,
            "-lc",
            "python3 -VV; python3 -m pip freeze; nsys --version; "
            "sha256sum /opt/magi-wheels/magi_attention-*.whl "
            "/opt/magi-wheels/magi_attn_extensions-*.whl; "
            "python3 -c 'from importlib import metadata; import magi_attention; "
            "import magi_attn_extensions.DSA as magi_dsa; "
            'print(metadata.version("magi-attention")); print(magi_attention.__file__); '
            'print(metadata.version("magi_attn_extensions")); print(magi_dsa.__file__)\'',
        ],
        cwd=repo_root,
        timeout_seconds=70,
    )

    _write_text(artifact_dir / "SOURCE_REVISION.txt", args.revision + "\n")
    _write_text(
        artifact_dir / "SOURCE_REMOTE.txt",
        _run(["git", "remote", "get-url", "origin"], cwd=repo_root, timeout_seconds=30),
    )
    _write_text(
        artifact_dir / "SUBMODULES.txt",
        _run(
            ["git", "submodule", "status", "--recursive"],
            cwd=repo_root,
            timeout_seconds=30,
        ),
    )
    _write_text(artifact_dir / "DIRTY_STATUS.txt", dirty)
    _write_text(artifact_dir / "ENVIRONMENT.txt", environment)
    _write_text(
        artifact_dir / "HARDWARE.txt",
        _run(["nvidia-smi", "-q"], cwd=repo_root, timeout_seconds=30),
    )
    _write_json(artifact_dir / "IMAGE.json", image_record)

    references = {
        "correctness_artifact": str(correctness),
        "correctness_manifest_sha256": _sha256(correctness / "SHA256SUMS"),
        "cp1_artifact": str(cp1),
        "cp1_manifest_sha256": _sha256(cp1 / "SHA256SUMS"),
        "cp2_artifact": str(cp2),
        "cp2_manifest_sha256": _sha256(cp2 / "SHA256SUMS"),
        "profile_artifact": str(profile),
        "profile_manifest_sha256": _sha256(profile / "SHA256SUMS"),
        "profile_report_sha256": _sha256(profile / "REPORT_PRO_PAIR.md"),
    }
    _write_json(artifact_dir / "REFERENCES.json", references)
    _copy(correctness / "SUMMARY.json", artifact_dir / "CP8_SUMMARY.json")
    _copy(correctness / "PHASE_AUDIT.json", artifact_dir / "CP8_PHASE_AUDIT.json")
    _copy(correctness / "SEEDS.json", artifact_dir / "CP8_SEEDS.json")
    _copy(cp1 / "SUMMARY.json", artifact_dir / "CP1_SUMMARY.json")
    _copy(cp1 / "RAW.json", artifact_dir / "CP1_RAW.json")
    _copy(cp2 / "SUMMARY.json", artifact_dir / "CP2_SUMMARY.json")
    _copy(
        profile / "SUMMARY_PRO_PAIR.json",
        artifact_dir / "PROFILE_SUMMARY_PRO_PAIR.json",
    )
    _copy(profile / "WORKLOAD.json", artifact_dir / "PROFILE_WORKLOAD.json")
    for source_name, destination_name in (
        ("MAJOR_KERNEL_BALANCE_PRO_PAIR.json", "PROFILE_MAJOR_KERNEL_BALANCE.json"),
        ("INDEXER_D2D_PRO_PAIR.json", "PROFILE_INDEXER_D2D.json"),
        ("SUPPORT_OVERHEAD_PRO_PAIR.json", "PROFILE_SUPPORT_OVERHEAD.json"),
        (
            "PRO_PAIR_COMMUNICATION_OVERLAP.json",
            "PROFILE_COMMUNICATION_OVERLAP.json",
        ),
        ("REPORT_PRO_PAIR.md", "PROFILE_REPORT.md"),
    ):
        _copy(profile / source_name, artifact_dir / destination_name)
    _write_text(
        artifact_dir / "REPORT.md",
        _report(
            revision=args.revision,
            image=args.image,
            image_id=str(image_record["Id"]),
            correctness=correctness,
            profile=profile,
            cp1=cp1,
            cp2=cp2,
            major=major,
            d2d=d2d,
            communication=communication,
        ),
    )
    summary = {
        "cp1": cp1_summary["result"],
        "cp2": cp2_summary["result"],
        "correctness": correctness_summary["result"],
        "image": args.image,
        "image_id": image_record["Id"],
        "indexer_d2d": d2d["result"],
        "installed_wheel_cp8": "PASS",
        "major_kernel_balance": major["result"],
        "profile": profile_summary["result"],
        "profile_step_mode": profile_summary["step_mode"],
        "pro_pair_communication_overlap": communication["result"],
        "result": "PASS",
        "revision": args.revision,
        "structural_layout_shared": "PASS",
    }
    print(_write_release_summary_and_manifest(artifact_dir, summary), end="")


if __name__ == "__main__":
    main()
