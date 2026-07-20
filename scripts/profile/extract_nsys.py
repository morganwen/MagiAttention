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
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

_PHASE_NAMES = {
    "indexer_score": "magi_dsa::indexer_score",
    "indexer_topk": "magi_dsa::indexer_topk",
}
_STEP_PATTERN = re.compile(
    r"^(?P<plan>sequential|balanced)/rank_(?P<rank>\d+)/"
    r"training_step_(?P<step>\d+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Magi-DSA logical GPU ranges")
    parser.add_argument("--plan", choices=("sequential", "balanced"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _nvtx_ranges(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT n.rowid,
               n.start,
               n.end,
               n.globalTid,
               COALESCE(n.text, strings.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS strings ON strings.id = n.textId
        WHERE n.end IS NOT NULL
        ORDER BY n.start, n.rowid
        """
    )
    ranges: list[dict[str, Any]] = []
    for rowid, start, end, global_tid, text in rows:
        if text is None or global_tid is None:
            continue
        ranges.append(
            {
                "end": int(end),
                "global_tid": int(global_tid),
                "rowid": int(rowid),
                "start": int(start),
                "text": str(text),
            }
        )
    return ranges


def _phase_kernels(
    connection: sqlite3.Connection,
    phase_range: dict[str, Any],
) -> list[dict[str, Any]]:
    global_tid = int(phase_range["global_tid"])
    global_pid = global_tid & ~((1 << 24) - 1)
    rows = connection.execute(
        """
        SELECT DISTINCT kernels.rowid,
                        kernels.start,
                        kernels.end,
                        COALESCE(strings.value, CAST(kernels.shortName AS TEXT))
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernels
          ON kernels.correlationId = runtime.correlationId
         AND kernels.globalPid = ?
        LEFT JOIN StringIds AS strings ON strings.id = kernels.shortName
        WHERE runtime.globalTid = ?
          AND runtime.start >= ?
          AND runtime.end <= ?
        ORDER BY kernels.start, kernels.rowid
        """,
        (
            global_pid,
            global_tid,
            int(phase_range["start"]),
            int(phase_range["end"]),
        ),
    )
    return [
        {
            "duration_ns": int(end) - int(start),
            "end_ns": int(end),
            "name": str(name),
            "rowid": int(rowid),
            "start_ns": int(start),
        }
        for rowid, start, end, name in rows
    ]


def extract_profile_records(
    sqlite_path: Path,
    report_path: Path,
    plan: str,
    world_size: int,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan not in ("sequential", "balanced"):
        raise ValueError(f"unknown profile plan: {plan}")
    if world_size <= 0 or steps <= 0:
        raise ValueError("world size and step count must be positive")
    if not sqlite_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Nsight report or exported SQLite file is missing")
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        required_tables = {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "NVTX_EVENTS",
            "StringIds",
        }
        missing = required_tables - _table_names(connection)
        if missing:
            raise ValueError(f"Nsight SQLite is missing tables: {sorted(missing)}")
        ranges = _nvtx_ranges(connection)
        steps_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for candidate in ranges:
            match = _STEP_PATTERN.fullmatch(candidate["text"])
            if match is None or match.group("plan") != plan:
                continue
            rank = int(match.group("rank"))
            step = int(match.group("step"))
            key = (rank, step)
            if key in steps_by_key:
                raise ValueError(
                    f"duplicate Nsight step range: plan={plan}, rank={rank}, step={step}"
                )
            steps_by_key[key] = candidate
        expected_step_keys = {
            (rank, step) for rank in range(world_size) for step in range(steps)
        }
        if set(steps_by_key) != expected_step_keys:
            missing_steps = sorted(expected_step_keys - set(steps_by_key))
            extra_steps = sorted(set(steps_by_key) - expected_step_keys)
            raise ValueError(
                f"Nsight step grid mismatch: missing={missing_steps}, extra={extra_steps}"
            )

        logical_ranges: dict[tuple[int, int, str], list[dict[str, Any]]] = {
            (rank, step, phase): []
            for rank, step in expected_step_keys
            for phase in _PHASE_NAMES
        }
        phase_text_to_name = {value: key for key, value in _PHASE_NAMES.items()}
        for candidate in ranges:
            phase = phase_text_to_name.get(candidate["text"])
            if phase is None:
                continue
            enclosing = [
                (rank, step)
                for (rank, step), step_range in steps_by_key.items()
                if step_range["global_tid"] == candidate["global_tid"]
                and step_range["start"] <= candidate["start"]
                and candidate["end"] <= step_range["end"]
            ]
            if not enclosing:
                continue
            if len(enclosing) != 1:
                raise ValueError(
                    f"logical phase has {len(enclosing)} enclosing step ranges: {candidate}"
                )
            rank, step = enclosing[0]
            logical_ranges[(rank, step, phase)].append(candidate)

        records: list[dict[str, Any]] = []
        global_pids: set[int] = set()
        for rank in range(world_size):
            for step in range(steps):
                for phase, nvtx_name in _PHASE_NAMES.items():
                    matches = logical_ranges[(rank, step, phase)]
                    if len(matches) != 1:
                        raise ValueError(
                            "logical range count mismatch: "
                            f"plan={plan}, rank={rank}, step={step}, phase={phase}, count={len(matches)}"
                        )
                    phase_range = matches[0]
                    kernels = _phase_kernels(connection, phase_range)
                    if not kernels:
                        raise ValueError(
                            "logical range has no correlated GPU kernels: "
                            f"plan={plan}, rank={rank}, step={step}, phase={phase}"
                        )
                    gpu_time_ms = (
                        sum(kernel["duration_ns"] for kernel in kernels) / 1_000_000.0
                    )
                    if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
                        raise ValueError("logical GPU time must be finite and positive")
                    global_pid = int(phase_range["global_tid"]) & ~((1 << 24) - 1)
                    global_pids.add(global_pid)
                    name_counts = Counter(str(kernel["name"]) for kernel in kernels)
                    records.append(
                        {
                            "cpu_range_ns": int(phase_range["end"])
                            - int(phase_range["start"]),
                            "global_pid": global_pid,
                            "global_tid": int(phase_range["global_tid"]),
                            "gpu_time_ms": gpu_time_ms,
                            "kernel_launch_count": len(kernels),
                            "kernel_name_counts": dict(sorted(name_counts.items())),
                            "kernels": kernels,
                            "logical_call_count": 1,
                            "nvtx_name": nvtx_name,
                            "nvtx_rowid": int(phase_range["rowid"]),
                            "phase": phase,
                            "plan": plan,
                            "rank": rank,
                            "record_type": "magi_dsa_indexer_phase",
                            "report": report_path.name,
                            "step": step,
                        }
                    )
        summary = {
            "global_pids": sorted(global_pids),
            "logical_phase_records": len(records),
            "logical_step_ranges": len(steps_by_key),
            "plan": plan,
            "report": report_path.name,
            "result": "PASS",
            "steps": steps,
            "world_size": world_size,
        }
        if len(global_pids) != world_size:
            raise ValueError(
                f"Nsight report exposes {len(global_pids)} worker processes, expected {world_size}"
            )
        return records, summary
    finally:
        connection.close()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile extraction: {path}")
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = extract_profile_records(
        args.sqlite,
        args.report,
        args.plan,
        args.world_size,
        args.steps,
    )
    _write_jsonl(args.output_dir / "nsys_phase_records.jsonl", records)
    for rank in range(args.world_size):
        _write_jsonl(
            args.output_dir / f"rank{rank}_nsys_raw.jsonl",
            [record for record in records if int(record["rank"]) == rank],
        )
    summary_path = args.output_dir / "NSYS_AUDIT.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite Nsight audit: {summary_path}")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
