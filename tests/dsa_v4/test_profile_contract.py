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

import math
import sqlite3
from pathlib import Path

import pytest
import torch

from benchmarks.dsa_v4.profile_5step import _topk_diagnostics
from magi_attention.dsa_types import MagiDSAForwardResult
from scripts.image.finalize_release import _validate_correctness_summary
from scripts.profile.extract_nsys import extract_profile_records
from scripts.profile.summarize_5step import compute_rank_ranges, validate_phase_records


def _phase_records(world_size: int = 2, steps: int = 2) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for plan in ("sequential", "balanced"):
        for step in range(steps):
            for rank in range(world_size):
                for phase in ("indexer_score", "indexer_topk"):
                    base = 1.0 if rank == 0 else 1.04
                    records.append(
                        {
                            "gpu_time_ms": base,
                            "kernel_launch_count": 1,
                            "logical_call_count": 1,
                            "nvtx_name": f"magi_dsa::{phase}",
                            "phase": phase,
                            "plan": plan,
                            "rank": rank,
                            "record_type": "magi_dsa_indexer_phase",
                            "step": step,
                        }
                    )
    return records


def test_profile_grid_and_rank_range_contract() -> None:
    records = _phase_records()
    assert len(validate_phase_records(records, world_size=2, steps=2)) == 16
    ranges = compute_rank_ranges(records, world_size=2, steps=2)
    assert len(ranges) == 8
    balanced = [record for record in ranges if record["plan"] == "balanced"]
    assert all(record["gate_pass"] is True for record in balanced)
    assert all(math.isclose(record["rank_range_ms"], 0.04) for record in ranges)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda records: records.append(dict(records[0])), "duplicate profile record"),
        (lambda records: records.pop(), "profile record grid mismatch"),
        (lambda records: records[0].update(logical_call_count=2), "logical call count"),
        (lambda records: records[0].update(gpu_time_ms=float("nan")), "GPU time"),
        (lambda records: records[0].update(kernel_launch_count=0), "physical kernel"),
    ),
)
def test_profile_grid_rejects_invalid_records(mutation, message: str) -> None:
    records = _phase_records()
    mutation(records)
    with pytest.raises(ValueError, match=message):
        validate_phase_records(records, world_size=2, steps=2)


def test_balanced_threshold_is_per_step_and_phase() -> None:
    records = _phase_records()
    for record in records:
        if (
            record["plan"] == "balanced"
            and record["step"] == 1
            and record["phase"] == "indexer_topk"
            and record["rank"] == 1
        ):
            record["gpu_time_ms"] = 1.20
    ranges = compute_rank_ranges(records, world_size=2, steps=2)
    failures = [record for record in ranges if record["gate_pass"] is False]
    assert [(record["step"], record["phase"]) for record in failures] == [
        (1, "indexer_topk")
    ]


def test_release_finalizer_derives_pass_from_distributed_summary_contract() -> None:
    summary = {
        "case": "cp8-natural-backward",
        "execution_seconds": {"max": 0.08, "min": 0.07},
        "model_parameter_value_check_ranks": [0],
        "result_count": 8,
        "results": [{"rank": rank} for rank in range(8)],
        "world_size": 8,
    }
    assert _validate_correctness_summary(summary)["result"] == "PASS"
    summary["execution_seconds"] = {"max": 60.0, "min": 0.07}
    with pytest.raises(ValueError, match="deadline"):
        _validate_correctness_summary(summary)


def _topk_result(ids: list[list[int]], lengths: list[int]) -> MagiDSAForwardResult:
    rows = len(ids)
    return MagiDSAForwardResult(
        output=torch.zeros(rows, 1, 1),
        kl=torch.zeros(()),
        sparse_lse=torch.zeros(rows, 1),
        topk_ids=torch.tensor(ids, dtype=torch.int32),
        topk_length=torch.tensor(lengths, dtype=torch.int32),
        indexer_lse=torch.zeros(rows),
    )


def test_topk_diagnostic_distinguishes_order_from_canonical_set() -> None:
    target = _topk_result([[7, 3, -1], [8, 2, -1]], [2, 2])
    reordered = _topk_result([[3, 7, -1], [8, 2, -1]], [2, 2])
    changed = _topk_result([[3, 6, -1], [8, 2, -1]], [2, 2])
    order_only = _topk_diagnostics(target, reordered)
    assert order_only["ordered_exact"] is False
    assert order_only["canonical_exact"] is True
    assert order_only["order_only_mismatch_rows"] == 1
    set_change = _topk_diagnostics(target, changed)
    assert set_change["ordered_exact"] is False
    assert set_change["canonical_exact"] is False
    assert set_change["canonical_mismatch_rows"] == 1


def _create_nsys_fixture(path: Path, plan: str, world_size: int, steps: int) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(
            start INTEGER NOT NULL,
            end INTEGER,
            text TEXT,
            globalTid INTEGER,
            textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            globalTid INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            correlationId INTEGER,
            globalPid INTEGER,
            shortName INTEGER
        );
        """
    )
    connection.execute("INSERT INTO StringIds(id, value) VALUES(1, 'fixture_kernel')")
    correlation = 1
    for rank in range(world_size):
        global_pid = (1000 + rank) << 24
        global_tid = global_pid + 123
        for step in range(steps):
            step_start = (rank * steps + step) * 10_000 + 1_000
            step_end = step_start + 8_000
            connection.execute(
                "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) VALUES(?, ?, ?, ?, NULL)",
                (
                    step_start,
                    step_end,
                    f"{plan}/rank_{rank}/training_step_{step}",
                    global_tid,
                ),
            )
            for phase_index, phase in enumerate(("indexer_score", "indexer_topk")):
                phase_start = step_start + 1_000 + phase_index * 3_000
                phase_end = phase_start + 2_000
                connection.execute(
                    "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) "
                    "VALUES(?, ?, ?, ?, NULL)",
                    (phase_start, phase_end, f"magi_dsa::{phase}", global_tid),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start, end, globalTid, correlationId) "
                    "VALUES(?, ?, ?, ?)",
                    (phase_start + 100, phase_start + 200, global_tid, correlation),
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, end, correlationId, globalPid, shortName) "
                    "VALUES(?, ?, ?, ?, 1)",
                    (phase_start + 300, phase_start + 800, correlation, global_pid),
                )
                correlation += 1
    connection.commit()
    connection.close()


def test_nsys_sqlite_extracts_exact_logical_grid(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "fixture.sqlite"
    report_path = tmp_path / "fixture.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(sqlite_path, "balanced", world_size=2, steps=2)
    records, summary = extract_profile_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=2,
        steps=2,
    )
    assert len(records) == 8
    assert summary["global_pids"] == [1000 << 24, 1001 << 24]
    assert all(record["logical_call_count"] == 1 for record in records)
    assert all(record["kernel_launch_count"] == 1 for record in records)
    assert all(record["gpu_time_ms"] == 0.0005 for record in records)
