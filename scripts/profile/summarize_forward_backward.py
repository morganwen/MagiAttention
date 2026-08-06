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
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

_PHASES = (
    "backward",
    "forward",
    "indexer_score",
    "indexer_topk",
    "parameter_gradient_allreduce",
)
_NVTX_NAMES = {phase: f"magi_dsa::{phase}" for phase in _PHASES}
_CSA_BACKWARD_SCOPE = "magi_dsa::backward"
_CSA_BACKWARD_ROUTES = (
    "COMPRESSED_KI",
    "COMPRESSED_KV",
    "OVERLAP_X",
    "WINDOW_KV",
)
_CSA_PROJECTION_SUPPORT_RELEASE = (
    "magi_dsa::module::attention::csa::" "backward_overlap::projection_support_release"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Magi-DSA balanced forward-backward profile"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected an object at {path}:{line_number}")
            records.append(value)
    return records


def read_nvtx_range_counts(sqlite_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT COALESCE(ranges.text, strings.value), COUNT(*)
            FROM NVTX_EVENTS AS ranges
            LEFT JOIN StringIds AS strings ON strings.id = ranges.textId
            WHERE ranges.end IS NOT NULL
              AND COALESCE(ranges.text, strings.value) IS NOT NULL
            GROUP BY COALESCE(ranges.text, strings.value)
            ORDER BY COALESCE(ranges.text, strings.value)
            """
        ).fetchall()
        return {str(name): int(count) for name, count in rows}
    finally:
        connection.close()


def count_token_layout_nvtx_ranges(sqlite_path: Path) -> int:
    return sum(
        count
        for name, count in read_nvtx_range_counts(sqlite_path).items()
        if "TOKEN_LAYOUT" in name
    )


def count_model_projection_nvtx_ranges(sqlite_path: Path) -> int:
    return sum(
        count
        for name, count in read_nvtx_range_counts(sqlite_path).items()
        if "magi_dsa::module::model_projection::" in name
    )


def count_scalar_loss_nvtx_ranges(sqlite_path: Path) -> int:
    return int(read_nvtx_range_counts(sqlite_path).get("magi_dsa::loss", 0))


def validate_forward_backward_records(
    records: list[dict[str, Any]], world_size: int, steps: int
) -> list[dict[str, Any]]:
    expected = {
        (step, rank, phase)
        for step in range(steps)
        for rank in range(world_size)
        for phase in _PHASES
    }
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record.get("step", -1)),
            int(record.get("rank", -1)),
            str(record.get("phase")),
        )
        if key in by_key:
            raise ValueError(f"duplicate forward-backward profile record: {key}")
        by_key[key] = record
        step, rank, phase = key
        if record.get("plan") != "balanced" or phase not in _PHASES:
            raise ValueError(f"invalid forward-backward profile record: {key}")
        if record.get("nvtx_name") != _NVTX_NAMES[phase]:
            raise ValueError(f"NVTX name mismatch: {key}")
        if int(record.get("logical_call_count", 0)) != 1:
            raise ValueError(f"logical call count is not one: {key}")
        gpu_time_ms = float(record.get("gpu_time_ms", float("nan")))
        if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
            raise ValueError(f"GPU time is not finite and positive: {key}")
        if int(record.get("kernel_launch_count", 0)) <= 0:
            raise ValueError(f"phase has no correlated kernels: {key}")
        if not (0 <= step < steps and 0 <= rank < world_size):
            raise ValueError(f"rank or step is outside the profile grid: {key}")
    if set(by_key) != expected:
        raise ValueError(
            "forward-backward profile grid mismatch: "
            f"missing={sorted(expected - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected)}"
        )
    return [by_key[key] for key in sorted(expected)]


def compute_forward_backward_rank_ranges(
    records: list[dict[str, Any]], world_size: int, steps: int
) -> list[dict[str, Any]]:
    validated = validate_forward_backward_records(records, world_size, steps)
    ranges: list[dict[str, Any]] = []
    for step in range(steps):
        for phase in _PHASES:
            values = [
                float(record["gpu_time_ms"])
                for record in validated
                if int(record["step"]) == step and record["phase"] == phase
            ]
            mean_ms = statistics.fmean(values)
            minimum = min(values)
            maximum = max(values)
            ranges.append(
                {
                    "max_rank_time_ms": maximum,
                    "mean_rank_time_ms": mean_ms,
                    "min_rank_time_ms": minimum,
                    "phase": phase,
                    "plan": "balanced",
                    "rank_range_ms": maximum - minimum,
                    "relative_rank_range": (maximum - minimum) / mean_ms,
                    "step": step,
                    "threshold": None,
                }
            )
    return ranges


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("kernel attribution contains an invalid GPU interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_overlap_ns(
    interval: tuple[int, int],
    others: list[tuple[int, int]],
) -> int:
    start, end = interval
    return sum(
        max(0, min(end, other_end) - max(start, other_start))
        for other_start, other_end in others
    )


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "max": max(values),
        "mean": statistics.fmean(values),
        "min": min(values),
    }


def compute_csa_backward_route_overlap(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Require every CSA reverse All2AllV to overlap non-route GPU compute."""

    expected = {(rank, step) for rank in range(world_size) for step in range(steps)}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in expected}
    for record in attribution_records:
        key = (int(record.get("rank", -1)), int(record.get("step", -1)))
        if key in grouped:
            grouped[key].append(record)

    route_records: dict[str, list[dict[str, Any]]] = {
        route: [] for route in _CSA_BACKWARD_ROUTES
    }
    for key in sorted(expected):
        rank, step = key
        records = grouped[key]
        compute_intervals = _merge_intervals(
            [
                (
                    int(record["kernel_start_ns"]),
                    int(record["kernel_end_ns"]),
                )
                for record in records
                if str(record.get("kernel_name", "")) != "ncclDevKernel_SendRecv"
                and "DsaRowCopy" not in str(record.get("kernel_name", ""))
                and "DsaRowCsrReduce" not in str(record.get("kernel_name", ""))
                and any(
                    str(scope.get("name", "")) == _CSA_BACKWARD_SCOPE
                    for scope in record.get("attribution_path", [])
                )
            ]
        )
        if not compute_intervals:
            raise ValueError(
                f"CSA backward compute kernels are missing: rank={rank}, step={step}"
            )

        for route in _CSA_BACKWARD_ROUTES:
            route_scope = (
                "magi_dsa::phase::collective_all2all_v::"
                f"attention::csa::{route}.backward"
            )
            communication = [
                record
                for record in records
                if str(record.get("kernel_name", "")) == "ncclDevKernel_SendRecv"
                and any(
                    str(scope.get("name", "")) == route_scope
                    for scope in record.get("attribution_path", [])
                )
            ]
            if len(communication) != 1:
                raise ValueError(
                    "CSA backward route SendRecv count differs: "
                    f"route={route}, rank={rank}, step={step}, "
                    f"count={len(communication)}"
                )
            record = communication[0]
            interval = (
                int(record["kernel_start_ns"]),
                int(record["kernel_end_ns"]),
            )
            duration_ns = interval[1] - interval[0]
            if duration_ns <= 0:
                raise ValueError(
                    "CSA backward route has an invalid GPU interval: "
                    f"route={route}, rank={rank}, step={step}"
                )
            overlap_ns = _interval_overlap_ns(interval, compute_intervals)
            if overlap_ns <= 0:
                raise ValueError(
                    "CSA backward route has no GPU compute overlap: "
                    f"route={route}, rank={rank}, step={step}"
                )
            route_records[route].append(
                {
                    "duration_us": duration_ns / 1_000.0,
                    "kernel_end_ns": interval[1],
                    "kernel_start_ns": interval[0],
                    "overlap_fraction": overlap_ns / duration_ns,
                    "overlap_us": overlap_ns / 1_000.0,
                    "rank": rank,
                    "step": step,
                }
            )

    routes: dict[str, Any] = {}
    for route, records in route_records.items():
        durations = [float(record["duration_us"]) for record in records]
        overlaps = [float(record["overlap_us"]) for record in records]
        fractions = [float(record["overlap_fraction"]) for record in records]
        routes[route] = {
            "duration_us": _stats(durations),
            "overlap_fraction": _stats(fractions),
            "overlap_us": _stats(overlaps),
            "positive_gpu_overlap_records": sum(value > 0.0 for value in overlaps),
            "records": records,
            "total_records": len(records),
        }
    return {
        "actual_gpu_overlap_is_hard_gate": True,
        "result": "PASS",
        "routes": routes,
    }


def _validate_rank_results(plan_dir: Path, world_size: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    expected_delta = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": 5,
    }
    for rank in range(world_size):
        metadata = _read_json(plan_dir / f"metadata_rank{rank}.json")
        result = _read_json(plan_dir / f"result_rank{rank}.json")
        if metadata.get("step_mode") != "forward-backward":
            raise ValueError(f"rank {rank} metadata has the wrong step mode")
        if (
            metadata.get("token_layout_capture") != "pre_capture_once"
            or metadata.get("projection_capture") != "pre_capture_once"
            or metadata.get("profile_gradient_boundary")
            != "post_projection_magi_dsa_input"
            or metadata.get("backward_seed")
            != "precomputed_global_mean_scaled_dout_and_unit_dkl"
            or float(metadata.get("dout_scale", float("nan"))) != 1.0 / 4_294_967_296
            or metadata.get("loss_capture") != "none"
        ):
            raise ValueError(f"rank {rank} metadata has the wrong DSA input boundary")
        if (
            result.get("result") != "PASS"
            or result.get("plan") != "balanced"
            or result.get("shadow_plan") != "sequential"
            or result.get("step_mode") != "forward-backward"
            or result.get("backward_seed")
            != "precomputed_global_mean_scaled_dout_and_unit_dkl"
            or float(result.get("dout_scale", float("nan"))) != 1.0 / 4_294_967_296
            or result.get("loss_capture") != "none"
            or result.get("projection_capture") != "pre_capture_once"
            or result.get("token_layout_capture") != "pre_capture_once"
        ):
            raise ValueError(f"rank {rank} forward-backward result did not pass")
        if result.get("capture_counter_delta") != expected_delta:
            raise ValueError(f"rank {rank} warm-path counter delta differs")
        metrics = result.get("metrics")
        backend_native_valid = (
            metrics.get(
                "topk_backend_native_valid",
                metrics.get("ordered_topk_exact"),
            )
            if isinstance(metrics, dict)
            else False
        )
        if (
            not isinstance(metrics, dict)
            or backend_native_valid is not True
            or not all(
                metrics.get(name) is True
                for name in (
                    "output_finite",
                    "topk_length_exact",
                    "topk_unique",
                )
            )
        ):
            raise ValueError(f"rank {rank} forward shadow comparison failed")
        compared_rows = int(metrics.get("output_compared_rows", -1))
        exempt_rows = int(metrics.get("output_tie_exempt_rows", -1))
        if (
            compared_rows < 0
            or exempt_rows < 0
            or compared_rows + exempt_rows
            != int(metadata.get("local_source_tokens", -1))
        ):
            raise ValueError(f"rank {rank} tie-aware output row accounting differs")
        for name in ("output_max_abs", "output_non_tie_max_abs"):
            if not math.isfinite(float(metrics.get(name, float("nan")))):
                raise ValueError(f"rank {rank} {name} is not finite")
        gradient_metrics = result.get("gradient_metrics")
        if (
            not isinstance(gradient_metrics, dict)
            or gradient_metrics.get("all_close") is not True
        ):
            raise ValueError(f"rank {rank} gradient shadow comparison failed")
        results.append(result)
    return results


def main() -> None:
    args = _parse_args()
    if args.world_size != 8 or args.steps != 5:
        raise ValueError("forward-backward summary requires 8 ranks and five steps")
    workload = _read_json(args.artifact_dir / "WORKLOAD.json")
    expected_workload = {
        "cp_size": 8,
        "cu_seqlens": [0, 131072],
        "dtype": "BF16",
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "dout_scale": 1.0 / 4_294_967_296,
        "loss_capture": "none",
        "plans": ["balanced"],
        "rank_size": 8,
        "ratio": 4,
        "seed": 0,
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "projection_capture": "pre_capture_once",
        "step_mode": "forward-backward",
        "steps": 5,
        "token_layout_capture": "pre_capture_once",
        "world_size": 8,
    }
    for name, expected in expected_workload.items():
        if workload.get(name) != expected:
            raise ValueError(
                f"forward-backward workload mismatch: {name}={workload.get(name)!r}"
            )

    plan_dir = args.artifact_dir / "balanced"
    report = plan_dir / "balanced_5steps_forward_backward.nsys-rep"
    sqlite_path = plan_dir / "balanced_5steps_forward_backward.sqlite"
    if not report.is_file() or report.stat().st_size == 0:
        raise FileNotFoundError(f"missing forward-backward report: {report}")
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing forward-backward SQLite: {sqlite_path}")
    nvtx_range_counts = read_nvtx_range_counts(sqlite_path)
    token_layout_nvtx_ranges = sum(
        count for name, count in nvtx_range_counts.items() if "TOKEN_LAYOUT" in name
    )
    if token_layout_nvtx_ranges != 0:
        raise ValueError(
            "DSA-core forward-backward capture contains "
            f"{token_layout_nvtx_ranges} TOKEN_LAYOUT NVTX ranges"
        )
    model_projection_nvtx_ranges = sum(
        count
        for name, count in nvtx_range_counts.items()
        if "magi_dsa::module::model_projection::" in name
    )
    if model_projection_nvtx_ranges != 0:
        raise ValueError(
            "DSA-core forward-backward capture contains "
            f"{model_projection_nvtx_ranges} model projection NVTX ranges"
        )
    loss_nvtx_ranges = int(nvtx_range_counts.get("magi_dsa::loss", 0))
    if loss_nvtx_ranges != 0:
        raise ValueError(
            "MSA-style DSA-core capture contains "
            f"{loss_nvtx_ranges} scalar loss NVTX ranges"
        )
    expected_schedule_ranges = args.world_size * args.steps
    required_schedule_ranges = (
        _CSA_PROJECTION_SUPPORT_RELEASE,
        (
            "magi_dsa::module::attention::csa::"
            "stream_overlap::indexer_projection_launch"
        ),
        ("magi_dsa::module::attention::csa::" "stream_overlap::main_compressor_launch"),
        ("magi_dsa::module::attention::csa::" "stream_overlap::support_routes_launch"),
    )
    for name in required_schedule_ranges:
        count = int(nvtx_range_counts.get(name, 0))
        if count != expected_schedule_ranges:
            raise ValueError(
                "CSA overlap schedule NVTX count differs: "
                f"name={name}, count={count}, expected={expected_schedule_ranges}"
            )
    (plan_dir / "NVTX_RANGE_COUNTS.json").write_text(
        json.dumps(
            {
                "counts": nvtx_range_counts,
                "range_names": len(nvtx_range_counts),
                "ranges": sum(nvtx_range_counts.values()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit = _read_json(plan_dir / "NSYS_AUDIT.json")
    if (
        audit.get("result") != "PASS"
        or audit.get("step_mode") != "forward-backward"
        or int(audit.get("logical_phase_records", 0))
        != args.world_size * args.steps * len(_PHASES)
    ):
        raise ValueError("forward-backward Nsight audit did not pass")
    attribution_audit = _read_json(plan_dir / "NSYS_ATTRIBUTION.json")
    if (
        attribution_audit.get("result") != "PASS"
        or int(attribution_audit.get("unattributed_kernel_count", -1)) != 0
        or float(attribution_audit.get("attribution_coverage", 0.0)) != 1.0
        or int(attribution_audit.get("kernel_records", 0)) <= 0
    ):
        raise ValueError("forward-backward Nsight kernel attribution did not pass")
    attribution_records = _read_jsonl(plan_dir / "nsys_kernel_attribution.jsonl")
    if len(attribution_records) != int(attribution_audit["kernel_records"]):
        raise ValueError("forward-backward kernel attribution record count differs")
    token_layout_records = [
        record
        for record in attribution_records
        if any(
            "TOKEN_LAYOUT" in str(scope.get("name", ""))
            for scope in record.get("attribution_path", [])
        )
    ]
    if token_layout_records:
        raise ValueError(
            "DSA-core forward-backward capture contains TOKEN_LAYOUT kernels"
        )
    model_projection_records = [
        record
        for record in attribution_records
        if any(
            "magi_dsa::module::model_projection::" in str(scope.get("name", ""))
            for scope in record.get("attribution_path", [])
        )
    ]
    if model_projection_records:
        raise ValueError(
            "DSA-core forward-backward capture contains model projection kernels"
        )
    backward_route_overlap = compute_csa_backward_route_overlap(
        attribution_records,
        args.world_size,
        args.steps,
    )
    (args.artifact_dir / "CSA_BACKWARD_COMMUNICATION_OVERLAP.json").write_text(
        json.dumps(backward_route_overlap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    records = validate_forward_backward_records(
        _read_jsonl(plan_dir / "nsys_phase_records.jsonl"),
        args.world_size,
        args.steps,
    )
    for record in records:
        phase = str(record["phase"])
        if phase not in ("forward", "backward"):
            continue
        sendrecv_count = int(
            record.get("kernel_name_counts", {}).get("ncclDevKernel_SendRecv", 0)
        )
        if sendrecv_count != 4:
            raise ValueError(
                f"DSA-core {phase} must contain exactly four All2AllV kernels: "
                f"rank={record['rank']}, step={record['step']}, count={sendrecv_count}"
            )
    for rank in range(args.world_size):
        raw = _read_jsonl(plan_dir / f"rank{rank}_nsys_raw.jsonl")
        if len(raw) != args.steps * len(_PHASES):
            raise ValueError(f"rank {rank} raw Nsight record count differs")
        rank_attribution = _read_jsonl(
            plan_dir / f"rank{rank}_nsys_kernel_attribution.jsonl"
        )
        if not rank_attribution or any(
            int(record.get("rank", -1)) != rank for record in rank_attribution
        ):
            raise ValueError(
                f"rank {rank} kernel attribution record set is empty or misrouted"
            )
    results = _validate_rank_results(plan_dir, args.world_size)
    ranges = compute_forward_backward_rank_ranges(records, args.world_size, args.steps)
    metrics = [result["metrics"] for result in results]
    gradient_metrics = [result["gradient_metrics"] for result in results]
    output_compared_rows = sum(int(item["output_compared_rows"]) for item in metrics)
    output_tie_exempt_rows = sum(
        int(item["output_tie_exempt_rows"]) for item in metrics
    )
    output_max_abs = max(float(item["output_max_abs"]) for item in metrics)
    output_non_tie_max_abs = max(
        float(item["output_non_tie_max_abs"]) for item in metrics
    )
    gradient_max_abs = max(float(item["max_abs"]) for item in gradient_metrics)
    latent_kv_gradient_mismatch_ratio = max(
        float(item["latent_kv_mismatch_ratio"]) for item in gradient_metrics
    )
    process_temporal_kernel_launches = sum(
        int(record.get("process_temporal_kernel_launch_count", 0)) for record in records
    )

    (args.artifact_dir / "profile_forward_backward.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.artifact_dir / "rank_ranges_forward_backward.json").write_text(
        json.dumps(ranges, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "canonical_topk_exact": all(
            bool(item.get("canonical_topk_exact", item["ordered_topk_exact"]))
            for item in metrics
        ),
        "csa_backward_route_overlap": {
            route: {
                key: value
                for key, value in backward_route_overlap["routes"][route].items()
                if key != "records"
            }
            for route in _CSA_BACKWARD_ROUTES
        },
        "logical_phase_records": len(records),
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "dout_scale": 1.0 / 4_294_967_296,
        "kernel_attribution_coverage": 1.0,
        "kernel_attribution_records": len(attribution_records),
        "gradient_max_abs": gradient_max_abs,
        "latent_kv_gradient_mismatch_ratio": latent_kv_gradient_mismatch_ratio,
        "ordered_topk_exact": all(bool(item["ordered_topk_exact"]) for item in metrics),
        "output_compared_rows": output_compared_rows,
        "output_max_abs": output_max_abs,
        "output_non_tie_max_abs": output_non_tie_max_abs,
        "output_tie_exempt_rows": output_tie_exempt_rows,
        "plan": "balanced",
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "projection_capture": "pre_capture_once",
        "model_projection_nvtx_ranges": model_projection_nvtx_ranges,
        "loss_capture": "none",
        "loss_nvtx_ranges": loss_nvtx_ranges,
        "process_temporal_kernel_launches": process_temporal_kernel_launches,
        "rank_results": len(results),
        "result": "PASS",
        "step_mode": "forward-backward",
        "steps": args.steps,
        "token_layout_capture": "pre_capture_once",
        "token_layout_nvtx_ranges": token_layout_nvtx_ranges,
        "nvtx_range_names": len(nvtx_range_counts),
        "nvtx_ranges": sum(nvtx_range_counts.values()),
        "topk_backend_native_valid": True,
        "world_size": args.world_size,
    }
    (args.artifact_dir / "SUMMARY_FORWARD_BACKWARD.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Magi-DSA balanced forward+backward profile",
        "",
        "- 结果：PASS",
        "- Workload：8×B300，BF16，CSA ratio=4，单条 128K，5 steps",
        "- Capture：balanced DSA-core forward/backward/parameter-gradient all-reduce；无 scalar loss",
        "- Boundary：每个 plan 在 capture/prewarm 前执行一次 TOKEN_LAYOUT + profile projection；"
        "capture 内为 post-projection MagiDSAInput leaf，严格 4F+4B",
        "- Backward seed：capture 外生成同一 global BF16 randn dout，以 "
        "1/global_output_elements=2^-32 缩放后按各 plan local_query_global_rows 排列；"
        "dkl=FP32 scalar one",
        "- Shadow：capture 外 sequential DSA-core 前后向；Top-K backend-native length/结构、tie-aware output 与全部 DSA 输入/参数梯度通过",
        f"- Tie-aware output：豁免 {output_tie_exempt_rows} / "
        f"{output_compared_rows + output_tie_exempt_rows} 个 canonical-set mismatch Query；"
        f"非 tie max-abs={output_non_tie_max_abs:.9g}，"
        f"全量诊断 max-abs={output_max_abs:.9g}",
        f"- Gradient：gradient max-abs={gradient_max_abs:.9g}，latent_kv mismatch ratio={latent_kv_gradient_mismatch_ratio:.9g}",
        "- Backward/gradient-allreduce：仅报告逐 rank 时间，不设置新硬门槛",
        f"- Kernel 归因：{len(attribution_records)} 个 CUDA runtime-correlated kernels，"
        "覆盖率 100%，未归因为 0；跨 autograd thread 的 process-temporal "
        f"launches={process_temporal_kernel_launches}",
        f"- 详细 NVTX：{sum(nvtx_range_counts.values())} 个 ranges、"
        f"{len(nvtx_range_counts)} 个名称；TOKEN_LAYOUT=0，model projection=0，"
        "scalar loss=0",
        "- CSA reverse overlap：四条 All2AllV 均要求每个 rank/step 与非 route GPU compute "
        "存在实际时间线交集；完整逐记录证据见 "
        "`CSA_BACKWARD_COMMUNICATION_OVERLAP.json`",
        "",
        "| reverse route | NCCL duration mean (us) | compute overlap mean (us) | overlap fraction mean | positive records |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for route in _CSA_BACKWARD_ROUTES:
        route_overlap = backward_route_overlap["routes"][route]
        report_lines.append(
            f"| `{route}.backward` | "
            f"{float(route_overlap['duration_us']['mean']):.3f} | "
            f"{float(route_overlap['overlap_us']['mean']):.3f} | "
            f"{float(route_overlap['overlap_fraction']['mean']):.3f} | "
            f"{int(route_overlap['positive_gpu_overlap_records'])}/"
            f"{int(route_overlap['total_records'])} |"
        )
    report_lines.extend(
        [
            "",
            "| step | forward mean/range (ms) | backward mean/range (ms) | "
            "grad-allreduce mean/range (ms) | indexer-score mean/range (ms) | "
            "indexer-topk mean/range (ms) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ranges_by_key = {
        (int(record["step"]), str(record["phase"])): record for record in ranges
    }
    for step in range(args.steps):
        values = []
        for phase in (
            "forward",
            "backward",
            "parameter_gradient_allreduce",
            "indexer_score",
            "indexer_topk",
        ):
            record = ranges_by_key[(step, phase)]
            values.append(
                f"{float(record['mean_rank_time_ms']):.6f}/"
                f"{float(record['rank_range_ms']):.6f}"
            )
        report_lines.append(f"| {step} | " + " | ".join(values) + " |")
    report_lines.append("")
    (args.artifact_dir / "REPORT_FORWARD_BACKWARD.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
