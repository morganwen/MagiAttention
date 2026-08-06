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
from typing import Any

_PROFILE_PHASES = (
    "w_forward",
    "w_backward",
    "csa_forward",
    "csa_backward",
    "hca_forward",
    "hca_backward",
    "parameter_gradient_allreduce",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize shared-greedy local-pass and clock/warmup ablations"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=ARTIFACT_DIR",
        help="Add one validated attention-suite artifact",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    if len(values) < 2:
        raise ValueError("dispatch ablation requires at least two runs")
    runs: list[tuple[str, Path]] = []
    labels: set[str] = set()
    paths: set[Path] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid --run value: {value!r}")
        path = Path(raw_path).resolve()
        if label in labels:
            raise ValueError(f"duplicate dispatch-ablation label: {label}")
        if path in paths:
            raise ValueError(f"duplicate dispatch-ablation artifact: {path}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        labels.add(label)
        paths.add(path)
        runs.append((label, path))
    return runs


def _stats(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("dispatch-ablation metric is empty or non-finite")
    mean = statistics.fmean(values)
    if mean <= 0.0:
        raise ValueError("dispatch-ablation metric mean must be positive")
    return {
        "max": max(values),
        "mean": mean,
        "min": min(values),
        "relative_rank_range": (max(values) - min(values)) / mean,
    }


def _aggregate_step_stats(
    rows: list[dict[str, float | int]],
    field: str,
) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    for step in range(5):
        values = [float(row[field]) for row in rows if int(row["step"]) == step]
        if len(values) != 8:
            raise ValueError(
                f"HCA forward {field} step {step} has {len(values)} rank records"
            )
        per_step.append({"step": step, **_stats(values)})

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "max_relative_rank_range": max(
                float(record["relative_rank_range"]) for record in selected
            ),
            "mean_rank_time_ms": statistics.fmean(
                float(record["mean"]) for record in selected
            ),
            "mean_relative_rank_range": statistics.fmean(
                float(record["relative_rank_range"]) for record in selected
            ),
        }

    return {
        "all_steps": summarize(per_step),
        "per_step": per_step,
        "steady_steps_1_to_4": summarize(per_step[1:]),
    }


def _hca_forward_breakdown(path: Path) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    with (path / "profile_attention_suite.jsonl").open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("phase") != "hca_forward":
                continue
            kernels = record.get("kernels")
            if not isinstance(kernels, list) or not kernels:
                raise ValueError("HCA forward record is missing kernels")
            sparse_ns = 0
            nccl_ns = 0
            other_ns = 0
            for kernel in kernels:
                name = str(kernel.get("name", ""))
                duration_ns = int(kernel.get("duration_ns", 0))
                if duration_ns <= 0:
                    raise ValueError("HCA forward kernel duration is invalid")
                if name == "sparse_attn_fwd_for_small_topk_kernel":
                    sparse_ns += duration_ns
                elif name == "ncclDevKernel_SendRecv":
                    nccl_ns += duration_ns
                else:
                    other_ns += duration_ns
            total_ms = float(record.get("gpu_time_ms", float("nan")))
            decomposed_ms = (sparse_ns + nccl_ns + other_ns) / 1_000_000.0
            if not math.isclose(total_ms, decomposed_ms, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("HCA forward kernel decomposition is incomplete")
            rows.append(
                {
                    "nccl_ms": nccl_ns / 1_000_000.0,
                    "other_ms": other_ns / 1_000_000.0,
                    "rank": int(record["rank"]),
                    "sparse_ms": sparse_ns / 1_000_000.0,
                    "step": int(record["step"]),
                    "total_ms": total_ms,
                }
            )
    if len(rows) != 40:
        raise ValueError(f"expected 40 HCA forward records, found {len(rows)}")
    return {
        "nccl": _aggregate_step_stats(rows, "nccl_ms"),
        "other": _aggregate_step_stats(rows, "other_ms"),
        "sparse_attention": _aggregate_step_stats(rows, "sparse_ms"),
        "total": _aggregate_step_stats(rows, "total_ms"),
    }


def _identity(path: Path) -> dict[str, Any]:
    rank_metadata = []
    for rank in range(8):
        metadata = _read_json(path / "balanced" / f"metadata_rank{rank}.json")
        rank_metadata.append(
            {
                "config_sha256": metadata.get("config_sha256"),
                "input_sha256": metadata.get("input_sha256"),
                "input_tensor_sha256": metadata.get("input_tensor_sha256"),
                "parameter_sha256": metadata.get("parameter_sha256"),
                "rank": rank,
                "seed": metadata.get("seed"),
                "tensor_seeds": metadata.get("tensor_seeds"),
            }
        )
    image = json.loads((path / "IMAGE.json").read_text(encoding="utf-8"))
    if not isinstance(image, list) or len(image) != 1:
        raise ValueError("profile IMAGE.json has an unexpected schema")
    return {
        "image_id": image[0].get("Id"),
        "rank_metadata": rank_metadata,
        "source_revision": (path / "SOURCE_REVISION.txt")
        .read_text(encoding="utf-8")
        .strip(),
    }


def _dsa_core_mean_ms(summary: dict[str, Any]) -> float:
    phase_mean = summary.get("phase_mean_ms")
    if not isinstance(phase_mean, dict):
        raise ValueError("attention-suite summary is missing phase means")
    values = [float(phase_mean[phase]) for phase in _PROFILE_PHASES]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("attention-suite phase mean is invalid")
    return sum(values)


def _summarize_run(label: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _read_json(path / "SUMMARY_ATTENTION_SUITE.json")
    workload = _read_json(path / "WORKLOAD.json")
    if summary.get("result") != "PASS":
        raise ValueError(f"dispatch-ablation run did not pass: {path}")
    if (
        int(summary.get("world_size", 0)) != 8
        or int(summary.get("steps", 0)) != 5
        or summary.get("step_mode") != "attention-suite"
        or workload.get("layout_policy") != "shared-greedy"
    ):
        raise ValueError(f"dispatch-ablation workload differs: {path}")
    dispatch = summary.get("dispatch_ablation")
    layout = summary.get("layout")
    if not isinstance(dispatch, dict) or not isinstance(layout, dict):
        raise ValueError(f"dispatch-ablation metadata is missing: {path}")
    passes = int(dispatch.get("local_improvement_passes", -1))
    if passes not in (0, 1, 4, 8):
        raise ValueError(f"dispatch-ablation pass count is invalid: {passes}")
    layout_key = [int(value) for value in layout.get("layout_key", [])]
    if len(layout_key) != 4:
        raise ValueError("dispatch-ablation layout key is invalid")
    prepare = layout.get("prepare_seconds")
    if not isinstance(prepare, dict) or not isinstance(prepare.get("w"), dict):
        raise ValueError("dispatch-ablation prepare timing is missing")
    record = {
        "artifact_dir": str(path),
        "dispatch": dispatch,
        "dsa_core_phase_sum_mean_ms": _dsa_core_mean_ms(summary),
        "hca_forward": _hca_forward_breakdown(path),
        "label": label,
        "layout_key": layout_key,
        "local_improvement_passes": passes,
        "prepare_seconds": prepare,
        "query_layout_hash": layout.get("query_layout_hash"),
        "solver": layout.get("layout_solver"),
        "token_layout_forward_ms": layout.get("token_layout_forward_ms"),
        "token_layout_remote_rows": layout.get("token_layout_remote_rows"),
    }
    return record, _identity(path)


def _validate_pass_trajectory(records: list[dict[str, Any]]) -> dict[str, Any]:
    representative: dict[int, dict[str, Any]] = {}
    hashes_by_pass: dict[int, set[str]] = {}
    for record in records:
        passes = int(record["local_improvement_passes"])
        hashes_by_pass.setdefault(passes, set()).add(str(record["query_layout_hash"]))
        representative.setdefault(passes, record)
    for passes, hashes in hashes_by_pass.items():
        if len(hashes) != 1:
            raise ValueError(
                f"deterministic shared-greedy layout differs for passes={passes}"
            )
    required = {0, 1, 4, 8}
    if not required.issubset(representative):
        return {
            "complete_pass_matrix": False,
            "present_passes": sorted(representative),
        }
    trajectory = [representative[passes]["layout_key"] for passes in sorted(required)]
    if any(
        tuple(next_key) > tuple(key)
        for key, next_key in zip(trajectory, trajectory[1:])
    ):
        raise ValueError("shared-greedy layout key worsened with additional passes")
    pass4 = representative[4]
    pass8 = representative[8]
    indexer4 = int(pass4["layout_key"][0])
    indexer8 = int(pass8["layout_key"][0])
    return {
        "complete_pass_matrix": True,
        "pass4_to_pass8_indexer_relative_improvement": (
            (indexer4 - indexer8) / indexer4
        ),
        "present_passes": sorted(representative),
        "trajectory": [
            {"layout_key": representative[passes]["layout_key"], "passes": passes}
            for passes in sorted(required)
        ],
    }


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Magi-DSA shared-greedy dispatch ablation",
        "",
        f"- 结果：{summary['result']}",
        f"- Runs：{len(summary['runs'])}",
        f"- 输入/参数/镜像一致：{str(summary['identity_equal']).lower()}",
        "- HCA forward 分解口径：正式 step logical NVTX 内的 CUPTI kernel duration；"
        "sparse core、NCCL SendRecv 与其余 kernel 分开统计。",
        "",
        "| run | passes | key | W prepare mean (s) | HCA F steady mean/range | "
        "sparse range | NCCL range | DSA-core phase sum (ms) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary["runs"]:
        total = record["hca_forward"]["total"]["steady_steps_1_to_4"]
        sparse = record["hca_forward"]["sparse_attention"]["steady_steps_1_to_4"]
        nccl = record["hca_forward"]["nccl"]["steady_steps_1_to_4"]
        lines.append(
            f"| {record['label']} | {record['local_improvement_passes']} "
            f"| `{tuple(record['layout_key'])}` "
            f"| {record['prepare_seconds']['w']['mean']:.3f} "
            f"| {total['mean_rank_time_ms']:.3f}/"
            f"{total['mean_relative_rank_range'] * 100:.2f}% "
            f"| {sparse['mean_relative_rank_range'] * 100:.2f}% "
            f"| {nccl['mean_relative_rank_range'] * 100:.2f}% "
            f"| {record['dsa_core_phase_sum_mean_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`range` 为 steps 1--4 的逐 step relative rank range 平均；step 0 仍在 JSON 的 "
            "`all_steps/per_step` 中完整保留。",
            "",
        ]
    )
    (output_dir / "REPORT_DISPATCH_ABLATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    runs = _parse_runs(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "SUMMARY_DISPATCH_ABLATION.json"
    report_path = args.output_dir / "REPORT_DISPATCH_ABLATION.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite dispatch-ablation summary")

    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for label, path in runs:
        record, identity = _summarize_run(label, path)
        records.append(record)
        identities.append(identity)
    identity_equal = all(identity == identities[0] for identity in identities[1:])
    if not identity_equal:
        raise ValueError("dispatch-ablation input, parameter, image, or source differs")
    summary = {
        "identity": identities[0],
        "identity_equal": True,
        "pass_trajectory": _validate_pass_trajectory(records),
        "result": "PASS",
        "runs": records,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir, summary)


if __name__ == "__main__":
    main()
