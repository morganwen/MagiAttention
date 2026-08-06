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

_PHASE_NAMES_BY_STEP_MODE = {
    "forward": {
        "indexer_score": "magi_dsa::indexer_score",
        "indexer_topk": "magi_dsa::indexer_topk",
    },
    "forward-backward": {
        "backward": "magi_dsa::backward",
        "forward": "magi_dsa::forward",
        "indexer_score": "magi_dsa::indexer_score",
        "indexer_topk": "magi_dsa::indexer_topk",
        "parameter_gradient_allreduce": "magi_dsa::parameter_gradient_allreduce",
    },
    "attention-suite": {
        "backward": "magi_dsa::backward",
        "csa_backward": "magi_dsa::attention_suite::csa::backward",
        "csa_forward": "magi_dsa::attention_suite::csa::forward",
        "forward": "magi_dsa::forward",
        "hca_backward": "magi_dsa::attention_suite::hca::backward",
        "hca_forward": "magi_dsa::attention_suite::hca::forward",
        "indexer_score": "magi_dsa::indexer_score",
        "indexer_topk": "magi_dsa::indexer_topk",
        "parameter_gradient_allreduce": "magi_dsa::parameter_gradient_allreduce",
        "w_backward": "magi_dsa::attention_suite::w::backward",
        "w_forward": "magi_dsa::attention_suite::w::forward",
    },
    "pro-pair": {
        "backward": "magi_dsa::backward",
        "csa_backward": "magi_dsa::pro_pair::csa::backward",
        "csa_forward": "magi_dsa::pro_pair::csa::forward",
        "forward": "magi_dsa::forward",
        "hca_backward": "magi_dsa::pro_pair::hca::backward",
        "hca_forward": "magi_dsa::pro_pair::hca::forward",
        "indexer_score": "magi_dsa::indexer_score",
        "indexer_topk": "magi_dsa::indexer_topk",
        "parameter_gradient_allreduce": "magi_dsa::parameter_gradient_allreduce",
    },
}
_STEP_PATTERN = re.compile(
    r"^(?P<plan>sequential|balanced)/rank_(?P<rank>\d+)/"
    r"training_step_(?P<step>\d+)$"
)
_MODULE_NVTX_PREFIX = "magi_dsa::module::"
_CUDNN_CALL_NVTX_PREFIX = "magi_dsa::CUDNN_CALL::"
_FORMAL_LOGICAL_NVTX_NAMES = frozenset(
    name
    for phase_names in _PHASE_NAMES_BY_STEP_MODE.values()
    for name in phase_names.values()
)
_ANALYSIS_INDEXES = {
    "CUPTI_ACTIVITY_KIND_RUNTIME": (
        "CREATE INDEX IF NOT EXISTS magi_dsa_runtime_pid_time_corr "
        "ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start, end, correlationId)"
    ),
    "CUPTI_ACTIVITY_KIND_KERNEL": (
        "CREATE INDEX IF NOT EXISTS magi_dsa_kernel_pid_corr "
        "ON CUPTI_ACTIVITY_KIND_KERNEL(globalPid, correlationId)"
    ),
    "CUPTI_ACTIVITY_KIND_MEMCPY": (
        "CREATE INDEX IF NOT EXISTS magi_dsa_memcpy_pid_kind_corr "
        "ON CUPTI_ACTIVITY_KIND_MEMCPY(globalPid, copyKind, correlationId)"
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Magi-DSA logical GPU ranges")
    parser.add_argument("--plan", choices=("sequential", "balanced"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--step-mode",
        choices=tuple(_PHASE_NAMES_BY_STEP_MODE),
        default="forward",
    )
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def _prepare_analysis_indexes(sqlite_path: Path) -> None:
    """Add analysis-only indexes to the completed Nsight SQLite export."""

    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-262144")
        tables = _table_names(connection)
        for table, statement in _ANALYSIS_INDEXES.items():
            if table in tables:
                connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    print("nsys extraction indexes ready", flush=True)


def _open_analysis_connection(sqlite_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def _report_extraction_progress(stage: str, plan: str, rank: int, step: int) -> None:
    print(
        f"nsys extraction stage={stage} plan={plan} rank={rank} step={step}",
        flush=True,
    )


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
    global_pid = _global_pid(global_tid)
    rows = connection.execute(
        """
        SELECT runtime.rowid,
               runtime.globalTid,
               runtime.correlationId,
               kernels.rowid,
                        kernels.start,
                        kernels.end,
                        COALESCE(strings.value, CAST(kernels.shortName AS TEXT))
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernels
          ON kernels.correlationId = runtime.correlationId
         AND kernels.globalPid = ?
        LEFT JOIN StringIds AS strings ON strings.id = kernels.shortName
        WHERE runtime.globalTid >= ?
          AND runtime.globalTid < ?
          AND runtime.start >= ?
          AND runtime.end <= ?
        ORDER BY runtime.start, runtime.rowid, kernels.start, kernels.rowid
        """,
        (
            global_pid,
            global_pid,
            global_pid + (1 << 24),
            int(phase_range["start"]),
            int(phase_range["end"]),
        ),
    )
    kernels: list[dict[str, Any]] = []
    seen_kernel_rowids: dict[int, int] = {}
    for (
        runtime_rowid,
        runtime_global_tid,
        correlation_id,
        kernel_rowid,
        start,
        end,
        name,
    ) in rows:
        runtime_rowid = int(runtime_rowid)
        kernel_rowid = int(kernel_rowid)
        previous_runtime = seen_kernel_rowids.get(kernel_rowid)
        if previous_runtime is not None:
            raise ValueError(
                "one phase kernel matched multiple runtime launches: "
                f"kernel_rowid={kernel_rowid}, "
                f"runtimes=({previous_runtime}, {runtime_rowid})"
            )
        seen_kernel_rowids[kernel_rowid] = runtime_rowid
        kernels.append(
            {
                "attribution_thread_scope": (
                    "same_thread"
                    if int(runtime_global_tid) == global_tid
                    else "process_temporal"
                ),
                "correlation_id": int(correlation_id),
                "duration_ns": int(end) - int(start),
                "end_ns": int(end),
                "name": str(name),
                "rowid": kernel_rowid,
                "runtime_global_tid": int(runtime_global_tid),
                "runtime_rowid": runtime_rowid,
                "start_ns": int(start),
            }
        )
    return kernels


def _global_pid(global_tid: int) -> int:
    return global_tid & ~((1 << 24) - 1)


def _collect_step_ranges(
    ranges: list[dict[str, Any]],
    plan: str,
    world_size: int,
    steps: int,
) -> dict[tuple[int, int], dict[str, Any]]:
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
    return steps_by_key


def _attribution_kind(text: str) -> str | None:
    if text.startswith(_CUDNN_CALL_NVTX_PREFIX):
        return "cudnn_call"
    if text.startswith(_MODULE_NVTX_PREFIX):
        return "module"
    if text.startswith("magi_dsa::"):
        return "logical"
    return None


def _step_kernel_rows(
    connection: sqlite3.Connection,
    step_range: dict[str, Any],
) -> list[dict[str, Any]]:
    global_pid = _global_pid(int(step_range["global_tid"]))
    rows = connection.execute(
        """
        SELECT runtime.rowid,
               runtime.start,
               runtime.end,
               runtime.globalTid,
               runtime.correlationId,
               kernels.rowid,
               kernels.start,
               kernels.end,
               COALESCE(strings.value, CAST(kernels.shortName AS TEXT))
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernels
          ON kernels.correlationId = runtime.correlationId
         AND kernels.globalPid = ?
        LEFT JOIN StringIds AS strings ON strings.id = kernels.shortName
        WHERE runtime.start >= ?
          AND runtime.end <= ?
          AND runtime.globalTid >= ?
          AND runtime.globalTid < ?
        ORDER BY runtime.start, runtime.rowid, kernels.start, kernels.rowid
        """,
        (
            global_pid,
            int(step_range["start"]),
            int(step_range["end"]),
            global_pid,
            global_pid + (1 << 24),
        ),
    )
    kernels: list[dict[str, Any]] = []
    seen_kernel_rowids: dict[int, int] = {}
    for (
        runtime_rowid,
        runtime_start,
        runtime_end,
        runtime_global_tid,
        correlation_id,
        kernel_rowid,
        kernel_start,
        kernel_end,
        kernel_name,
    ) in rows:
        kernel_rowid = int(kernel_rowid)
        runtime_rowid = int(runtime_rowid)
        previous_runtime = seen_kernel_rowids.get(kernel_rowid)
        if previous_runtime is not None:
            raise ValueError(
                "one CUPTI kernel row matched multiple runtime launches: "
                f"kernel_rowid={kernel_rowid}, runtimes=({previous_runtime}, {runtime_rowid})"
            )
        seen_kernel_rowids[kernel_rowid] = runtime_rowid
        duration_ns = int(kernel_end) - int(kernel_start)
        if duration_ns <= 0:
            raise ValueError(
                f"CUPTI kernel duration must be positive: rowid={kernel_rowid}"
            )
        kernels.append(
            {
                "correlation_id": int(correlation_id),
                "duration_ns": duration_ns,
                "end_ns": int(kernel_end),
                "name": str(kernel_name),
                "rowid": kernel_rowid,
                "runtime_end_ns": int(runtime_end),
                "runtime_global_tid": int(runtime_global_tid),
                "runtime_rowid": runtime_rowid,
                "runtime_start_ns": int(runtime_start),
                "start_ns": int(kernel_start),
            }
        )
    return kernels


def _range_descriptor(
    candidate: dict[str, Any],
    runtime_global_tid: int,
) -> dict[str, Any]:
    kind = _attribution_kind(str(candidate["text"]))
    if kind is None:
        raise ValueError(f"range is not a DSA attribution range: {candidate}")
    return {
        "duration_ns": int(candidate["end"]) - int(candidate["start"]),
        "end_ns": int(candidate["end"]),
        "global_tid": int(candidate["global_tid"]),
        "kind": kind,
        "name": str(candidate["text"]),
        "rowid": int(candidate["rowid"]),
        "start_ns": int(candidate["start"]),
        "thread_scope": (
            "same_thread"
            if int(candidate["global_tid"]) == runtime_global_tid
            else "process_temporal"
        ),
    }


def _runtime_activity_attribution(
    activity: dict[str, Any],
    process_ranges: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    runtime_start = int(activity["runtime_start_ns"])
    runtime_end = int(activity["runtime_end_ns"])
    runtime_global_tid = int(activity["runtime_global_tid"])
    same_thread = [
        candidate
        for candidate in process_ranges
        if int(candidate["global_tid"]) == runtime_global_tid
        and int(candidate["start"]) <= runtime_start
        and runtime_end <= int(candidate["end"])
        and _attribution_kind(str(candidate["text"])) is not None
    ]
    formal_context = [
        candidate
        for candidate in process_ranges
        if str(candidate["text"]) in _FORMAL_LOGICAL_NVTX_NAMES
        and int(candidate["start"]) <= runtime_start
        and runtime_end <= int(candidate["end"])
    ]
    candidates_by_rowid = {
        int(candidate["rowid"]): candidate
        for candidate in (*formal_context, *same_thread)
    }
    descriptors = [
        _range_descriptor(candidate, runtime_global_tid)
        for candidate in candidates_by_rowid.values()
    ]
    descriptors.sort(
        key=lambda item: (
            -int(item["duration_ns"]),
            int(item["start_ns"]),
            int(item["rowid"]),
        )
    )
    primary_candidates = [
        descriptor
        for descriptor in descriptors
        if descriptor["thread_scope"] == "same_thread"
    ]
    if not primary_candidates:
        primary_candidates = descriptors
    if not primary_candidates:
        return None, descriptors
    kind_priority = {"logical": 0, "module": 1, "cudnn_call": 2}
    primary = min(
        primary_candidates,
        key=lambda item: (
            int(item["duration_ns"]),
            -kind_priority[str(item["kind"])],
            int(item["rowid"]),
        ),
    )
    return primary, descriptors


def _d2d_copy_kind(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT id
        FROM ENUM_CUDA_MEMCPY_OPER
        WHERE name = 'CUDA_MEMCPY_KIND_DTOD'
        """
    ).fetchone()
    if row is None:
        raise ValueError("Nsight SQLite does not define CUDA D2D copy kind")
    return int(row[0])


def _step_d2d_rows(
    connection: sqlite3.Connection,
    step_range: dict[str, Any],
    d2d_kind: int,
) -> list[dict[str, Any]]:
    global_pid = _global_pid(int(step_range["global_tid"]))
    rows = connection.execute(
        """
        SELECT runtime.rowid,
               runtime.start,
               runtime.end,
               runtime.globalTid,
               runtime.correlationId,
               copies.rowid,
               copies.start,
               copies.end,
               copies.bytes,
               COALESCE(copies.copyCount, 1),
               copies.copyKind,
               copies.globalPid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN CUPTI_ACTIVITY_KIND_MEMCPY AS copies
          ON copies.correlationId = runtime.correlationId
         AND copies.globalPid = ?
        WHERE copies.copyKind = ?
          AND runtime.start >= ?
          AND runtime.end <= ?
          AND runtime.globalTid >= ?
          AND runtime.globalTid < ?
        ORDER BY runtime.start, runtime.rowid, copies.start, copies.rowid
        """,
        (
            global_pid,
            d2d_kind,
            int(step_range["start"]),
            int(step_range["end"]),
            global_pid,
            global_pid + (1 << 24),
        ),
    )
    copies: list[dict[str, Any]] = []
    seen_copy_rowids: dict[int, int] = {}
    for (
        runtime_rowid,
        runtime_start,
        runtime_end,
        runtime_global_tid,
        correlation_id,
        copy_rowid,
        copy_start,
        copy_end,
        copy_bytes,
        copy_count,
        copy_kind,
        copy_global_pid,
    ) in rows:
        runtime_rowid = int(runtime_rowid)
        copy_rowid = int(copy_rowid)
        previous_runtime = seen_copy_rowids.get(copy_rowid)
        if previous_runtime is not None:
            raise ValueError(
                "one CUPTI memcpy row matched multiple runtime launches: "
                f"memcpy_rowid={copy_rowid}, "
                f"runtimes=({previous_runtime}, {runtime_rowid})"
            )
        seen_copy_rowids[copy_rowid] = runtime_rowid
        duration_ns = int(copy_end) - int(copy_start)
        if duration_ns <= 0:
            raise ValueError(
                f"CUPTI memcpy duration must be positive: rowid={copy_rowid}"
            )
        if int(copy_bytes) < 0:
            raise ValueError(
                f"CUPTI memcpy bytes must be non-negative: rowid={copy_rowid}"
            )
        if int(copy_count) <= 0:
            raise ValueError(f"CUPTI memcpy count must be positive: rowid={copy_rowid}")
        if int(copy_kind) != d2d_kind or int(copy_global_pid) != global_pid:
            raise ValueError(f"CUPTI D2D memcpy ownership differs: rowid={copy_rowid}")
        copies.append(
            {
                "bytes": int(copy_bytes),
                "copy_count": int(copy_count),
                "copy_kind_id": int(copy_kind),
                "correlation_id": int(correlation_id),
                "duration_ns": duration_ns,
                "end_ns": int(copy_end),
                "rowid": copy_rowid,
                "runtime_end_ns": int(runtime_end),
                "runtime_global_tid": int(runtime_global_tid),
                "runtime_rowid": runtime_rowid,
                "runtime_start_ns": int(runtime_start),
                "start_ns": int(copy_start),
            }
        )
    return copies


def extract_kernel_attribution_records(
    sqlite_path: Path,
    report_path: Path,
    plan: str,
    world_size: int,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attribute every step-local CUPTI kernel through its CUDA runtime launch."""

    if plan not in ("sequential", "balanced"):
        raise ValueError(f"unknown profile plan: {plan}")
    if world_size <= 0 or steps <= 0:
        raise ValueError("world size and step count must be positive")
    if not sqlite_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Nsight report or exported SQLite file is missing")
    connection = _open_analysis_connection(sqlite_path)
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
        steps_by_key = _collect_step_ranges(
            ranges,
            plan,
            world_size,
            steps,
        )
        ranges_by_pid: dict[int, list[dict[str, Any]]] = {}
        for candidate in ranges:
            global_pid = _global_pid(int(candidate["global_tid"]))
            ranges_by_pid.setdefault(global_pid, []).append(candidate)

        records: list[dict[str, Any]] = []
        global_pids: set[int] = set()
        step_summaries: list[dict[str, Any]] = []
        kind_counts: Counter[str] = Counter()
        kind_duration_ns: Counter[str] = Counter()
        unattributed_name_counts: Counter[str] = Counter()
        for rank in range(world_size):
            for step in range(steps):
                step_range = steps_by_key[(rank, step)]
                global_pid = _global_pid(int(step_range["global_tid"]))
                global_pids.add(global_pid)
                process_ranges = [
                    candidate
                    for candidate in ranges_by_pid.get(global_pid, [])
                    if int(step_range["start"]) <= int(candidate["start"])
                    and int(candidate["end"]) <= int(step_range["end"])
                ]
                step_kernels = _step_kernel_rows(connection, step_range)
                if not step_kernels:
                    raise ValueError(
                        "Nsight step has no runtime-correlated GPU kernels: "
                        f"plan={plan}, rank={rank}, step={step}"
                    )
                step_kind_counts: Counter[str] = Counter()
                step_kind_duration_ns: Counter[str] = Counter()
                for kernel in step_kernels:
                    primary, path = _runtime_activity_attribution(
                        kernel,
                        process_ranges,
                    )
                    kind = "unattributed" if primary is None else str(primary["kind"])
                    duration_ns = int(kernel["duration_ns"])
                    kind_counts[kind] += 1
                    kind_duration_ns[kind] += duration_ns
                    step_kind_counts[kind] += 1
                    step_kind_duration_ns[kind] += duration_ns
                    if primary is None:
                        unattributed_name_counts[str(kernel["name"])] += 1
                    records.append(
                        {
                            "attribution_kind": kind,
                            "attribution_name": (
                                None if primary is None else primary["name"]
                            ),
                            "attribution_nvtx_rowid": (
                                None if primary is None else primary["rowid"]
                            ),
                            "attribution_path": path,
                            "attribution_thread_scope": (
                                None if primary is None else primary["thread_scope"]
                            ),
                            "correlation_id": int(kernel["correlation_id"]),
                            "global_pid": global_pid,
                            "gpu_time_ms": duration_ns / 1_000_000.0,
                            "kernel_end_ns": int(kernel["end_ns"]),
                            "kernel_name": str(kernel["name"]),
                            "kernel_rowid": int(kernel["rowid"]),
                            "kernel_start_ns": int(kernel["start_ns"]),
                            "plan": plan,
                            "rank": rank,
                            "record_type": "magi_dsa_kernel_attribution",
                            "report": report_path.name,
                            "runtime_end_ns": int(kernel["runtime_end_ns"]),
                            "runtime_global_tid": int(kernel["runtime_global_tid"]),
                            "runtime_rowid": int(kernel["runtime_rowid"]),
                            "runtime_start_ns": int(kernel["runtime_start_ns"]),
                            "step": step,
                        }
                    )
                step_summaries.append(
                    {
                        "attribution_kind_counts": dict(
                            sorted(step_kind_counts.items())
                        ),
                        "attribution_kind_gpu_time_ms": {
                            kind: duration / 1_000_000.0
                            for kind, duration in sorted(step_kind_duration_ns.items())
                        },
                        "kernel_count": len(step_kernels),
                        "plan": plan,
                        "rank": rank,
                        "step": step,
                    }
                )
                _report_extraction_progress("kernel", plan, rank, step)

        if len(global_pids) != world_size:
            raise ValueError(
                f"Nsight report exposes {len(global_pids)} worker processes, expected {world_size}"
            )
        total_duration_ns = sum(kind_duration_ns.values())
        unattributed_count = kind_counts["unattributed"]
        unattributed_duration_ns = kind_duration_ns["unattributed"]
        summary = {
            "attributed_kernel_count": len(records) - unattributed_count,
            "attribution_coverage": (
                (len(records) - unattributed_count) / len(records) if records else 0.0
            ),
            "attribution_kind_counts": dict(sorted(kind_counts.items())),
            "attribution_kind_gpu_time_ms": {
                kind: duration / 1_000_000.0
                for kind, duration in sorted(kind_duration_ns.items())
            },
            "global_pids": sorted(global_pids),
            "kernel_gpu_time_ms": total_duration_ns / 1_000_000.0,
            "kernel_records": len(records),
            "logical_step_ranges": len(steps_by_key),
            "plan": plan,
            "report": report_path.name,
            "result": "PASS" if unattributed_count == 0 else "FAIL",
            "step_records": step_summaries,
            "steps": steps,
            "unattributed_gpu_time_ms": unattributed_duration_ns / 1_000_000.0,
            "unattributed_kernel_count": unattributed_count,
            "unattributed_kernel_name_counts": dict(
                sorted(unattributed_name_counts.items())
            ),
            "world_size": world_size,
        }
        return records, summary
    finally:
        connection.close()


def extract_memcpy_attribution_records(
    sqlite_path: Path,
    report_path: Path,
    plan: str,
    world_size: int,
    steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attribute every step-local CUDA D2D activity through its runtime launch."""

    if plan not in ("sequential", "balanced"):
        raise ValueError(f"unknown profile plan: {plan}")
    if world_size <= 0 or steps <= 0:
        raise ValueError("world size and step count must be positive")
    if not sqlite_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Nsight report or exported SQLite file is missing")
    connection = _open_analysis_connection(sqlite_path)
    try:
        required_tables = {
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "ENUM_CUDA_MEMCPY_OPER",
            "NVTX_EVENTS",
            "StringIds",
        }
        missing = required_tables - _table_names(connection)
        if missing:
            raise ValueError(
                f"Nsight SQLite is missing memcpy tables: {sorted(missing)}"
            )
        d2d_kind = _d2d_copy_kind(connection)
        ranges = _nvtx_ranges(connection)
        steps_by_key = _collect_step_ranges(
            ranges,
            plan,
            world_size,
            steps,
        )
        ranges_by_pid: dict[int, list[dict[str, Any]]] = {}
        for candidate in ranges:
            global_pid = _global_pid(int(candidate["global_tid"]))
            ranges_by_pid.setdefault(global_pid, []).append(candidate)

        records: list[dict[str, Any]] = []
        global_pids: set[int] = set()
        step_summaries: list[dict[str, Any]] = []
        kind_activity_counts: Counter[str] = Counter()
        kind_byte_counts: Counter[str] = Counter()
        kind_copy_counts: Counter[str] = Counter()
        kind_duration_ns: Counter[str] = Counter()
        for rank in range(world_size):
            for step in range(steps):
                step_range = steps_by_key[(rank, step)]
                global_pid = _global_pid(int(step_range["global_tid"]))
                global_pids.add(global_pid)
                process_ranges = [
                    candidate
                    for candidate in ranges_by_pid.get(global_pid, [])
                    if int(step_range["start"]) <= int(candidate["start"])
                    and int(candidate["end"]) <= int(step_range["end"])
                ]
                step_copies = _step_d2d_rows(
                    connection,
                    step_range,
                    d2d_kind,
                )
                step_kind_activity_counts: Counter[str] = Counter()
                step_kind_byte_counts: Counter[str] = Counter()
                step_kind_copy_counts: Counter[str] = Counter()
                step_kind_duration_ns: Counter[str] = Counter()
                for copy in step_copies:
                    primary, path = _runtime_activity_attribution(
                        copy,
                        process_ranges,
                    )
                    kind = "unattributed" if primary is None else str(primary["kind"])
                    copy_bytes = int(copy["bytes"])
                    copy_count = int(copy["copy_count"])
                    duration_ns = int(copy["duration_ns"])
                    kind_activity_counts[kind] += 1
                    kind_byte_counts[kind] += copy_bytes
                    kind_copy_counts[kind] += copy_count
                    kind_duration_ns[kind] += duration_ns
                    step_kind_activity_counts[kind] += 1
                    step_kind_byte_counts[kind] += copy_bytes
                    step_kind_copy_counts[kind] += copy_count
                    step_kind_duration_ns[kind] += duration_ns
                    records.append(
                        {
                            "attribution_kind": kind,
                            "attribution_name": (
                                None if primary is None else primary["name"]
                            ),
                            "attribution_nvtx_rowid": (
                                None if primary is None else primary["rowid"]
                            ),
                            "attribution_path": path,
                            "attribution_thread_scope": (
                                None if primary is None else primary["thread_scope"]
                            ),
                            # Nsight defines bytes as the activity's total transfer
                            # size. copyCount reports batched operations and must not
                            # multiply this value.
                            "bytes": copy_bytes,
                            "copy_count": copy_count,
                            "copy_kind": "CUDA_MEMCPY_KIND_DTOD",
                            "copy_kind_id": int(copy["copy_kind_id"]),
                            "correlation_id": int(copy["correlation_id"]),
                            "global_pid": global_pid,
                            "gpu_time_ms": duration_ns / 1_000_000.0,
                            "memcpy_end_ns": int(copy["end_ns"]),
                            "memcpy_rowid": int(copy["rowid"]),
                            "memcpy_start_ns": int(copy["start_ns"]),
                            "plan": plan,
                            "rank": rank,
                            "record_type": "magi_dsa_memcpy_attribution",
                            "report": report_path.name,
                            "runtime_end_ns": int(copy["runtime_end_ns"]),
                            "runtime_global_tid": int(copy["runtime_global_tid"]),
                            "runtime_rowid": int(copy["runtime_rowid"]),
                            "runtime_start_ns": int(copy["runtime_start_ns"]),
                            "step": step,
                        }
                    )
                step_activity_count = len(step_copies)
                step_byte_count = sum(step_kind_byte_counts.values())
                step_copy_count = sum(step_kind_copy_counts.values())
                step_unattributed_activity_count = step_kind_activity_counts[
                    "unattributed"
                ]
                step_unattributed_byte_count = step_kind_byte_counts["unattributed"]
                step_unattributed_copy_count = step_kind_copy_counts["unattributed"]
                step_summaries.append(
                    {
                        "attributed_bytes": (
                            step_byte_count - step_unattributed_byte_count
                        ),
                        "attributed_copy_count": (
                            step_copy_count - step_unattributed_copy_count
                        ),
                        "attributed_memcpy_activity_count": (
                            step_activity_count - step_unattributed_activity_count
                        ),
                        "attribution_byte_coverage": (
                            (step_byte_count - step_unattributed_byte_count)
                            / step_byte_count
                            if step_byte_count
                            else 1.0
                        ),
                        "attribution_copy_count_coverage": (
                            (step_copy_count - step_unattributed_copy_count)
                            / step_copy_count
                            if step_copy_count
                            else 1.0
                        ),
                        "attribution_coverage": (
                            (step_activity_count - step_unattributed_activity_count)
                            / step_activity_count
                            if step_activity_count
                            else 1.0
                        ),
                        "attribution_kind_activity_counts": dict(
                            sorted(step_kind_activity_counts.items())
                        ),
                        "attribution_kind_bytes": dict(
                            sorted(step_kind_byte_counts.items())
                        ),
                        "attribution_kind_copy_counts": dict(
                            sorted(step_kind_copy_counts.items())
                        ),
                        "attribution_kind_gpu_time_ms": {
                            kind: duration / 1_000_000.0
                            for kind, duration in sorted(step_kind_duration_ns.items())
                        },
                        "bytes": step_byte_count,
                        "copy_count": step_copy_count,
                        "gpu_time_ms": (
                            sum(step_kind_duration_ns.values()) / 1_000_000.0
                        ),
                        "memcpy_activity_count": step_activity_count,
                        "plan": plan,
                        "rank": rank,
                        "result": (
                            "PASS" if step_unattributed_activity_count == 0 else "FAIL"
                        ),
                        "step": step,
                        "unattributed_bytes": step_unattributed_byte_count,
                        "unattributed_copy_count": step_unattributed_copy_count,
                        "unattributed_memcpy_activity_count": (
                            step_unattributed_activity_count
                        ),
                    }
                )
                _report_extraction_progress("memcpy", plan, rank, step)

        if len(global_pids) != world_size:
            raise ValueError(
                f"Nsight report exposes {len(global_pids)} worker processes, expected {world_size}"
            )
        activity_count = len(records)
        byte_count = sum(kind_byte_counts.values())
        copy_count = sum(kind_copy_counts.values())
        unattributed_activity_count = kind_activity_counts["unattributed"]
        unattributed_byte_count = kind_byte_counts["unattributed"]
        unattributed_copy_count = kind_copy_counts["unattributed"]
        unattributed_duration_ns = kind_duration_ns["unattributed"]
        summary = {
            "activity_kind": "CUDA_MEMCPY_KIND_DTOD",
            "attributed_bytes": byte_count - unattributed_byte_count,
            "attributed_copy_count": copy_count - unattributed_copy_count,
            "attributed_memcpy_activity_count": (
                activity_count - unattributed_activity_count
            ),
            "attribution_byte_coverage": (
                (byte_count - unattributed_byte_count) / byte_count
                if byte_count
                else 1.0
            ),
            "attribution_copy_count_coverage": (
                (copy_count - unattributed_copy_count) / copy_count
                if copy_count
                else 1.0
            ),
            "attribution_coverage": (
                (activity_count - unattributed_activity_count) / activity_count
                if activity_count
                else 1.0
            ),
            "attribution_kind_activity_counts": dict(
                sorted(kind_activity_counts.items())
            ),
            "attribution_kind_bytes": dict(sorted(kind_byte_counts.items())),
            "attribution_kind_copy_counts": dict(sorted(kind_copy_counts.items())),
            "attribution_kind_gpu_time_ms": {
                kind: duration / 1_000_000.0
                for kind, duration in sorted(kind_duration_ns.items())
            },
            "bytes": byte_count,
            "copy_count": copy_count,
            "global_pids": sorted(global_pids),
            "logical_step_ranges": len(steps_by_key),
            "memcpy_activity_records": activity_count,
            "memcpy_gpu_time_ms": sum(kind_duration_ns.values()) / 1_000_000.0,
            "plan": plan,
            "report": report_path.name,
            "result": "PASS" if unattributed_activity_count == 0 else "FAIL",
            "step_records": step_summaries,
            "steps": steps,
            "unattributed_bytes": unattributed_byte_count,
            "unattributed_copy_count": unattributed_copy_count,
            "unattributed_gpu_time_ms": (unattributed_duration_ns / 1_000_000.0),
            "unattributed_memcpy_activity_count": unattributed_activity_count,
            "world_size": world_size,
        }
        return records, summary
    finally:
        connection.close()


def extract_profile_records(
    sqlite_path: Path,
    report_path: Path,
    plan: str,
    world_size: int,
    steps: int,
    step_mode: str = "forward",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan not in ("sequential", "balanced"):
        raise ValueError(f"unknown profile plan: {plan}")
    if world_size <= 0 or steps <= 0:
        raise ValueError("world size and step count must be positive")
    if step_mode not in _PHASE_NAMES_BY_STEP_MODE:
        raise ValueError(f"unknown profile step mode: {step_mode}")
    phase_names = _PHASE_NAMES_BY_STEP_MODE[step_mode]
    if not sqlite_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Nsight report or exported SQLite file is missing")
    connection = _open_analysis_connection(sqlite_path)
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
        steps_by_key = _collect_step_ranges(ranges, plan, world_size, steps)
        expected_step_keys = {
            (rank, step) for rank in range(world_size) for step in range(steps)
        }

        logical_ranges: dict[tuple[int, int, str], list[dict[str, Any]]] = {
            (rank, step, phase): []
            for rank, step in expected_step_keys
            for phase in phase_names
        }
        phase_text_to_name = {value: key for key, value in phase_names.items()}
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
                for phase, nvtx_name in phase_names.items():
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
                    global_pid = _global_pid(int(phase_range["global_tid"]))
                    global_pids.add(global_pid)
                    name_counts = Counter(str(kernel["name"]) for kernel in kernels)
                    process_temporal_count = sum(
                        kernel["attribution_thread_scope"] == "process_temporal"
                        for kernel in kernels
                    )
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
                            "process_temporal_kernel_launch_count": process_temporal_count,
                            "rank": rank,
                            "record_type": (
                                "magi_dsa_indexer_phase"
                                if phase in ("indexer_score", "indexer_topk")
                                else "magi_dsa_training_phase"
                            ),
                            "report": report_path.name,
                            "step": step,
                            "same_thread_kernel_launch_count": len(kernels)
                            - process_temporal_count,
                        }
                    )
                _report_extraction_progress("logical", plan, rank, step)
        summary = {
            "global_pids": sorted(global_pids),
            "logical_phase_records": len(records),
            "logical_step_ranges": len(steps_by_key),
            "plan": plan,
            "report": report_path.name,
            "result": "PASS",
            "steps": steps,
            "step_mode": step_mode,
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
    _prepare_analysis_indexes(args.sqlite)
    records, summary = extract_profile_records(
        args.sqlite,
        args.report,
        args.plan,
        args.world_size,
        args.steps,
        args.step_mode,
    )
    attribution_records, attribution_summary = extract_kernel_attribution_records(
        args.sqlite,
        args.report,
        args.plan,
        args.world_size,
        args.steps,
    )
    memcpy_records, memcpy_summary = extract_memcpy_attribution_records(
        args.sqlite,
        args.report,
        args.plan,
        args.world_size,
        args.steps,
    )
    _write_jsonl(args.output_dir / "nsys_phase_records.jsonl", records)
    _write_jsonl(
        args.output_dir / "nsys_kernel_attribution.jsonl",
        attribution_records,
    )
    _write_jsonl(
        args.output_dir / "nsys_memcpy_attribution.jsonl",
        memcpy_records,
    )
    for rank in range(args.world_size):
        _write_jsonl(
            args.output_dir / f"rank{rank}_nsys_raw.jsonl",
            [record for record in records if int(record["rank"]) == rank],
        )
        _write_jsonl(
            args.output_dir / f"rank{rank}_nsys_kernel_attribution.jsonl",
            [record for record in attribution_records if int(record["rank"]) == rank],
        )
        _write_jsonl(
            args.output_dir / f"rank{rank}_nsys_memcpy_attribution.jsonl",
            [record for record in memcpy_records if int(record["rank"]) == rank],
        )
    summary_path = args.output_dir / "NSYS_AUDIT.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite Nsight audit: {summary_path}")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attribution_summary_path = args.output_dir / "NSYS_ATTRIBUTION.json"
    if attribution_summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Nsight attribution audit: {attribution_summary_path}"
        )
    attribution_summary_path.write_text(
        json.dumps(attribution_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    memcpy_summary_path = args.output_dir / "NSYS_MEMCPY_ATTRIBUTION.json"
    if memcpy_summary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Nsight memcpy attribution audit: {memcpy_summary_path}"
        )
    memcpy_summary_path.write_text(
        json.dumps(memcpy_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if attribution_summary["result"] != "PASS":
        raise SystemExit(
            "Nsight kernel attribution has unowned kernels; inspect "
            f"{attribution_summary_path}"
        )
    if memcpy_summary["result"] != "PASS":
        raise SystemExit(
            "Nsight D2D memcpy attribution has unowned activities; inspect "
            f"{memcpy_summary_path}"
        )


if __name__ == "__main__":
    main()
