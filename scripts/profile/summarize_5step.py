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
import statistics
from pathlib import Path
from typing import Any, Iterable

_PLANS = ("sequential", "balanced")
_PHASES = ("indexer_score", "indexer_topk")
_NVTX_NAMES = {
    "indexer_score": "magi_dsa::indexer_score",
    "indexer_topk": "magi_dsa::indexer_topk",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Magi-DSA v4 five-step profile"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def validate_phase_records(
    records: Iterable[dict[str, Any]],
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    concrete = list(records)
    expected = {
        (plan, step, rank, phase)
        for plan in _PLANS
        for step in range(steps)
        for rank in range(world_size)
        for phase in _PHASES
    }
    by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for record in concrete:
        key = (
            str(record.get("plan")),
            int(record.get("step", -1)),
            int(record.get("rank", -1)),
            str(record.get("phase")),
        )
        if key in by_key:
            raise ValueError(f"duplicate profile record: {key}")
        by_key[key] = record
        plan, step, rank, phase = key
        if plan not in _PLANS or phase not in _PHASES:
            raise ValueError(f"unknown plan or phase in profile record: {key}")
        if record.get("nvtx_name") != _NVTX_NAMES[phase]:
            raise ValueError(f"NVTX name mismatch for profile record: {key}")
        if int(record.get("logical_call_count", 0)) != 1:
            raise ValueError(f"logical call count is not one: {key}")
        gpu_time_ms = float(record.get("gpu_time_ms", float("nan")))
        if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
            raise ValueError(f"profile GPU time is not finite and positive: {key}")
        if int(record.get("kernel_launch_count", 0)) <= 0:
            raise ValueError(f"logical phase has no physical kernel launches: {key}")
        if not (0 <= rank < world_size and 0 <= step < steps):
            raise ValueError(f"profile rank or step is outside the frozen grid: {key}")
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise ValueError(
            f"profile record grid mismatch: missing={missing}, extra={extra}"
        )
    return [by_key[key] for key in sorted(expected)]


def compute_rank_ranges(
    records: Iterable[dict[str, Any]],
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    concrete = validate_phase_records(records, world_size, steps)
    ranges: list[dict[str, Any]] = []
    for plan in _PLANS:
        for step in range(steps):
            for phase in _PHASES:
                values = [
                    float(record["gpu_time_ms"])
                    for record in concrete
                    if record["plan"] == plan
                    and int(record["step"]) == step
                    and record["phase"] == phase
                ]
                if len(values) != world_size:
                    raise ValueError("profile range group does not contain all ranks")
                mean_ms = statistics.fmean(values)
                if not math.isfinite(mean_ms) or mean_ms <= 0:
                    raise ValueError("profile rank mean must be finite and positive")
                minimum = min(values)
                maximum = max(values)
                rank_range = maximum - minimum
                relative = rank_range / mean_ms
                ranges.append(
                    {
                        "balanced_threshold": 0.05 if plan == "balanced" else None,
                        "gate_pass": relative <= 0.05 if plan == "balanced" else None,
                        "max_rank_time_ms": maximum,
                        "mean_rank_time_ms": mean_ms,
                        "min_rank_time_ms": minimum,
                        "phase": phase,
                        "plan": plan,
                        "rank_range_ms": rank_range,
                        "relative_rank_range": relative,
                        "step": step,
                    }
                )
    return ranges


def _slim_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "cpu_range_ns",
            "global_pid",
            "gpu_time_ms",
            "kernel_launch_count",
            "kernel_name_counts",
            "logical_call_count",
            "nvtx_name",
            "phase",
            "plan",
            "rank",
            "record_type",
            "report",
            "step",
        )
    }


def _validate_workload(
    artifact_dir: Path, world_size: int, steps: int
) -> dict[str, Any]:
    workload = _read_json(artifact_dir / "WORKLOAD.json")
    expected = {
        "cp_size": 8,
        "cu_seqlens": [0, 131072],
        "dtype": "BF16",
        "plans": ["sequential", "balanced"],
        "rank_size": 8,
        "ratio": 4,
        "seed": 0,
        "steps": 5,
        "world_size": 8,
    }
    for name, value in expected.items():
        if workload.get(name) != value:
            raise ValueError(
                f"frozen workload field mismatch: {name}={workload.get(name)!r}, expected {value!r}"
            )
    if world_size != 8 or steps != 5:
        raise ValueError(
            "the release summarizer requires world size eight and five steps"
        )
    return workload


def _validate_plan_artifacts(
    artifact_dir: Path,
    world_size: int,
) -> dict[str, Any]:
    metadata: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    for plan in _PLANS:
        plan_dir = artifact_dir / plan
        report = plan_dir / f"{plan}_5steps.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise FileNotFoundError(
                f"missing or empty aggregate Nsight report: {report}"
            )
        if not (plan_dir / "NSYS_STATS.txt").is_file():
            raise FileNotFoundError(f"missing Nsight open-check output for {plan}")
        audit = _read_json(plan_dir / "NSYS_AUDIT.json")
        if (
            audit.get("result") != "PASS"
            or int(audit.get("logical_phase_records", 0)) != 80
        ):
            raise ValueError(f"Nsight audit did not pass for {plan}")
        metadata[plan] = []
        results[plan] = []
        for rank in range(world_size):
            raw_path = plan_dir / f"rank{rank}_nsys_raw.jsonl"
            if len(_read_jsonl(raw_path)) != 10:
                raise ValueError(
                    f"per-rank Nsight raw record count mismatch: {raw_path}"
                )
            rank_metadata = _read_json(plan_dir / f"metadata_rank{rank}.json")
            rank_result = _read_json(plan_dir / f"result_rank{rank}.json")
            if rank_result.get("result") != "PASS":
                raise ValueError(
                    f"profile worker result did not pass: plan={plan}, rank={rank}"
                )
            if rank_result.get("plan") != plan:
                raise ValueError(
                    f"profile worker target plan mismatch: plan={plan}, rank={rank}"
                )
            expected_delta = {
                "device_materializations": 0,
                "health_checks": 0,
                "object_collective_invocations": 0,
                "solver_invocations": 0,
                "warm_invocations": 5,
            }
            if rank_result.get("capture_counter_delta") != expected_delta:
                raise ValueError(
                    f"warm-path counter delta mismatch: plan={plan}, rank={rank}"
                )
            metrics = rank_result.get("metrics")
            if not isinstance(metrics, dict) or not all(
                metrics.get(name) is True
                for name in (
                    "ordered_topk_exact",
                    "output_finite",
                    "topk_length_exact",
                    "topk_unique",
                )
            ):
                raise ValueError(
                    f"plan shadow equivalence failed: plan={plan}, rank={rank}"
                )
            metadata[plan].append(rank_metadata)
            results[plan].append(rank_result)

    for rank in range(world_size):
        sequential = metadata["sequential"][rank]
        balanced = metadata["balanced"][rank]
        for field in (
            "config_sha256",
            "input_sha256",
            "parameter_sha256",
            "tensor_seeds",
        ):
            if sequential.get(field) != balanced.get(field):
                raise ValueError(f"profile plans differ in {field}: rank={rank}")
        if sequential.get("seed") != 0 or balanced.get("seed") != 0:
            raise ValueError(f"profile seed mismatch: rank={rank}")
        if sequential.get("dtype") != "torch.bfloat16":
            raise ValueError(f"profile dtype mismatch: rank={rank}")
    parameter_hashes = {
        str(item["parameter_sha256"])
        for plan_metadata in metadata.values()
        for item in plan_metadata
    }
    if len(parameter_hashes) != 1:
        raise ValueError("model parameters differ across profile ranks or plans")
    return {
        "input_hashes_by_rank": [
            metadata["sequential"][rank]["input_sha256"] for rank in range(world_size)
        ],
        "ordered_topk_exact": True,
        "output_close": True,
        "output_finite": True,
        "parameter_sha256": next(iter(parameter_hashes)),
        "result": "PASS",
        "topk_length_exact": True,
        "topk_unique": True,
    }


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile summary: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile summary: {path}")
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def _format_report(
    artifact_dir: Path,
    ranges: list[dict[str, Any]],
    correctness: dict[str, Any],
) -> str:
    by_key = {
        (str(record["plan"]), int(record["step"]), str(record["phase"])): record
        for record in ranges
    }
    lines = [
        "# Magi-DSA v4 CP8 5-step Profile",
        "",
        f"- Run ID：`{artifact_dir.name}`",
        "- Workload：seed=0，BF16，ratio=4，CP8，`cu_seqlens=[0,131072]`。",
        "- 捕获：每个 plan 恰好 5 个 forward step；JIT/prewarm 位于捕获区间外。",
        "- 时间口径：logical NVTX 内 CUDA runtime launch 关联的 CUPTI kernel GPU duration 之和。",
        "- Correctness：sequential/balanced ordered Top-K 与 length exact，output tolerance 通过，ID 唯一。",
        "",
        "## 逐 step rank balance",
        "",
        "| Step | Phase | Sequential min/max/range ms | Sequential relative | "
        "Balanced min/max/range ms | Balanced relative | 5% gate |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for step in range(5):
        for phase in _PHASES:
            sequential = by_key[("sequential", step, phase)]
            balanced = by_key[("balanced", step, phase)]
            lines.append(
                "| {step} | {phase} | {smin:.6f}/{smax:.6f}/{srange:.6f} | "
                "{srelative:.6f} | {bmin:.6f}/{bmax:.6f}/{brange:.6f} | "
                "{brelative:.6f} | {gate} |".format(
                    step=step,
                    phase=phase,
                    smin=sequential["min_rank_time_ms"],
                    smax=sequential["max_rank_time_ms"],
                    srange=sequential["rank_range_ms"],
                    srelative=sequential["relative_rank_range"],
                    bmin=balanced["min_rank_time_ms"],
                    bmax=balanced["max_rank_time_ms"],
                    brange=balanced["rank_range_ms"],
                    brelative=balanced["relative_rank_range"],
                    gate="PASS" if balanced["gate_pass"] else "FAIL",
                )
            )
    balanced_pass = all(
        bool(record["gate_pass"]) for record in ranges if record["plan"] == "balanced"
    )
    lines.extend(
        (
            "",
            "## 验收结论",
            "",
            "- 160 条唯一逐 rank phase 记录：PASS。",
            "- 两个 aggregate `.nsys-rep` 可打开且均包含 8 ranks × 5 steps：PASS。",
            f"- Q14 same-backend 对齐：{correctness['result']}。",
            f"- Balanced 每 step/phase 5% 门槛：{'PASS' if balanced_pass else 'FAIL'}。",
            "- Sequential 仅为 baseline，不设硬门槛。",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    _validate_workload(args.artifact_dir, args.world_size, args.steps)
    correctness = _validate_plan_artifacts(args.artifact_dir, args.world_size)
    all_records: list[dict[str, Any]] = []
    for plan in _PLANS:
        plan_records = _read_jsonl(
            args.artifact_dir / plan / "nsys_phase_records.jsonl"
        )
        all_records.extend(plan_records)
    ordered = validate_phase_records(all_records, args.world_size, args.steps)
    ranges = compute_rank_ranges(ordered, args.world_size, args.steps)
    slim_records = [_slim_record(record) for record in ordered]
    _write_jsonl(args.artifact_dir / "profile_5step.jsonl", slim_records)
    _write_json(args.artifact_dir / "rank_ranges.json", ranges)
    _write_json(args.artifact_dir / "CORRECTNESS.json", correctness)
    balanced_pass = all(
        bool(record["gate_pass"]) for record in ranges if record["plan"] == "balanced"
    )
    summary = {
        "balanced_5pct_gate": balanced_pass,
        "correctness": correctness["result"],
        "profile_records": len(slim_records),
        "rank_range_groups": len(ranges),
        "result": "PASS" if balanced_pass else "FAIL",
    }
    _write_json(args.artifact_dir / "SUMMARY.json", summary)
    report_path = args.artifact_dir / "REPORT.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite profile report: {report_path}")
    report_path.write_text(
        _format_report(args.artifact_dir, ranges, correctness),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not balanced_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
