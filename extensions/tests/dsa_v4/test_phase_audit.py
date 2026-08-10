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

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scripts.test.audit_dsa_phases import audit
from scripts.test.summarize_distributed import (
    _validate_cp8_structural_results,
    summarize,
)

from .conftest import find_repo_root


def _record(prefix: str, rank: int, event: str, name: str | None = None) -> str:
    payload: dict[str, object] = {"event": event, "rank": rank}
    if name is not None:
        payload["name"] = name
    return prefix + json.dumps(payload)


def test_audit_decodes_concatenated_rank_records(tmp_path) -> None:
    control = "MAGI_DSA_CP2 "
    phase = "MAGI_DSA_PHASE "
    lines = [
        _record(control, 0, "execute_begin") + _record(control, 1, "execute_begin"),
        _record(phase, 0, "begin", "forward") + _record(phase, 1, "begin", "forward"),
        _record(phase, 1, "end", "forward") + _record(phase, 0, "end", "forward"),
        _record(control, 0, "execute_end") + _record(control, 1, "execute_end"),
    ]
    log = tmp_path / "cp2.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = audit(log, world_size=2, timed_out=False)

    assert result["all_ranks_execute_started"] is True
    assert result["all_ranks_execute_ended"] is True
    assert result["collective_stall_confirmed"] is False
    assert result["failure_reasons"] == []
    assert result["result"] == "PASS"
    rank_reports = cast(list[dict[str, object]], result["ranks"])
    assert all(not rank_report["errors"] for rank_report in rank_reports)


def _run_phase_audit_cli(
    tmp_path: Path,
    lines: list[str],
    *,
    timed_out: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    log = tmp_path / "phase.log"
    output = tmp_path / "audit.json"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    repo_root = find_repo_root()
    command = [
        sys.executable,
        str(repo_root / "scripts/test/audit_dsa_phases.py"),
        "--log",
        str(log),
        "--output",
        str(output),
        "--world-size",
        "2",
    ]
    if timed_out:
        command.append("--timed-out")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_phase_audit_cli_returns_zero_only_for_complete_execution(tmp_path) -> None:
    control = "MAGI_DSA_CP2 "
    phase = "MAGI_DSA_PHASE "
    completed, report = _run_phase_audit_cli(
        tmp_path,
        [
            _record(control, 0, "execute_begin"),
            _record(control, 1, "execute_begin"),
            _record(phase, 0, "begin", "forward"),
            _record(phase, 0, "end", "forward"),
            _record(phase, 1, "begin", "forward"),
            _record(phase, 1, "end", "forward"),
            _record(control, 0, "execute_end"),
            _record(control, 1, "execute_end"),
        ],
    )

    assert completed.returncode == 0
    assert report["failure_reasons"] == []
    assert report["result"] == "PASS"


@pytest.mark.parametrize(
    ("lines", "timed_out", "expected_reasons"),
    [
        (
            [
                _record("MAGI_DSA_CP2 ", 0, "execute_begin"),
                _record("MAGI_DSA_CP2 ", 0, "execute_end"),
            ],
            False,
            {
                "not_all_ranks_execute_ended",
                "not_all_ranks_execute_started",
            },
        ),
        (
            [
                _record("MAGI_DSA_CP2 ", 0, "execute_begin"),
                _record("MAGI_DSA_CP2 ", 1, "execute_begin"),
                _record("MAGI_DSA_PHASE ", 0, "begin", "forward"),
                _record("MAGI_DSA_PHASE ", 0, "error", "forward"),
                _record("MAGI_DSA_CP2 ", 0, "execute_end"),
                _record("MAGI_DSA_CP2 ", 1, "execute_end"),
            ],
            False,
            {"phase_errors"},
        ),
        (
            [
                _record("MAGI_DSA_CP2 ", 0, "execute_begin"),
                _record("MAGI_DSA_CP2 ", 1, "execute_begin"),
                _record(
                    "MAGI_DSA_PHASE ",
                    0,
                    "begin",
                    "collective_actual_ready",
                ),
                _record(
                    "MAGI_DSA_PHASE ",
                    1,
                    "begin",
                    "collective_actual_ready",
                ),
            ],
            True,
            {
                "collective_stall_confirmed",
                "not_all_ranks_execute_ended",
                "open_phases",
                "timed_out",
            },
        ),
    ],
)
def test_phase_audit_cli_rejects_fabricated_failures(
    tmp_path: Path,
    lines: list[str],
    timed_out: bool,
    expected_reasons: set[str],
) -> None:
    completed, report = _run_phase_audit_cli(
        tmp_path,
        lines,
        timed_out=timed_out,
    )

    assert completed.returncode != 0
    assert report["result"] == "FAIL"
    assert set(cast(list[str], report["failure_reasons"])) == expected_reasons


def test_multigpu_runner_propagates_phase_audit_failure() -> None:
    repo_root = find_repo_root()
    source = (repo_root / "scripts/test/run_multigpu.sh").read_text(encoding="utf-8")

    assert "phase_audit_status=0" in source
    assert "if ((runner_status == 0 && phase_audit_status != 0)); then" in source
    assert 'runner_status="$phase_audit_status"' in source
    assert source.index("phase_audit_status=0") < source.index('exit "$runner_status"')


def _write_summary_fixture(tmp_path, execution_seconds: float) -> None:
    plan_evidence = {
        "declared_local_token_capacity": 1,
        "fragment_count": 1,
        "local_query_tokens": 1,
        "local_source_tokens": 1,
        "plan_hash": "a" * 64,
        "policy": "structural_balanced",
        "query_layout_hash": "b" * 64,
        "query_token_counts": [1],
        "rank_query_layout_signature": "c" * 64,
        "ratio": 4,
        "source_token_counts": [1],
        "structural_layout_config": {
            "chunk_size": 512,
            "min_chunks_per_rank": 16,
            "uneven_shard": True,
        },
        "structural_layout_metrics": {
            "chunk_size": 1,
            "cost_model_version": "native_causal_area_v1",
            "num_chunks": 1,
            "solver_scheme": "packed_global_minheap_v1",
            "uneven_shard": True,
        },
        "structural_rank_cost": {
            "query_tokens": 1,
            "rank": 0,
        },
    }
    (tmp_path / "result_rank0.json").write_text(
        json.dumps(
            {
                "balanced_plan_evidence": plan_evidence,
                "balanced_policy": "structural_balanced",
                "case": "csa-natural-backward",
                "execution_seconds": execution_seconds,
                "rank": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "control_rank0.jsonl").write_text(
        "\n".join(
            json.dumps({"event": event})
            for event in (
                "execute_begin",
                "execute_end",
                "verification_begin",
                "verification_end",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_distributed_summary_accepts_execution_inside_60_second_deadline(
    tmp_path,
) -> None:
    _write_summary_fixture(tmp_path, 59.999)
    summary = summarize(tmp_path, world_size=1, case="csa-natural-backward")
    assert summary["execution_seconds"] == {"max": 59.999, "min": 59.999}


def test_distributed_summary_rejects_execution_at_60_second_deadline(
    tmp_path,
) -> None:
    _write_summary_fixture(tmp_path, 60.0)
    with pytest.raises(TimeoutError, match="60s deadline"):
        summarize(tmp_path, world_size=1, case="csa-natural-backward")


def _cp8_plan_evidence(label: str, rank: int) -> dict[str, object]:
    policy = {
        "csa_balanced": "indexer_balanced",
        "csa_sequential": "sequential",
        "csa_structural": "structural_balanced",
        "hca": "sequential",
        "hca_structural": "structural_balanced",
        "window": "sequential",
    }[label]
    ratio = 4 if label.startswith("csa_") else 128 if label.startswith("hca") else 0
    structural = policy == "structural_balanced"
    result: dict[str, object] = {
        "declared_local_token_capacity": 32,
        "fragment_count": 1,
        "local_query_tokens": 32,
        "local_source_tokens": 32,
        "plan_hash": format(abs(hash(label)) % (1 << 256), "064x"),
        "policy": policy,
        "query_layout_hash": "d" * 64,
        "query_token_counts": [32] * 8,
        "rank_query_layout_signature": format(rank, "064x"),
        "ratio": ratio,
        "source_token_counts": [32] * 8,
        "structural_layout_config": None,
        "structural_layout_metrics": None,
        "structural_rank_cost": None,
    }
    if structural:
        result.update(
            structural_layout_config={
                "chunk_size": 512,
                "min_chunks_per_rank": 16,
                "uneven_shard": True,
            },
            structural_layout_metrics={
                "chunk_size": 2,
                "cost_model_version": "native_causal_area_v1",
                "num_chunks": 128,
                "solver_scheme": "packed_global_minheap_v1",
                "uneven_shard": True,
            },
            structural_rank_cost={
                "query_tokens": 32,
                "rank": rank,
            },
        )
    return result


def test_cp8_summary_requires_actual_structural_plan_evidence() -> None:
    results = []
    for rank in range(8):
        plans = {
            label: _cp8_plan_evidence(label, rank)
            for label in ("csa_structural", "hca_structural")
        }
        results.append(
            {
                "plan_evidence": plans,
                "structural_layout_shared": True,
                "structural_query_layout_hash": "d" * 64,
            }
        )
    structural = _validate_cp8_structural_results(results)
    assert structural["query_token_counts"] == [32] * 8
    assert structural["result"] == "PASS"

    final_plans = results[7]["plan_evidence"]
    assert isinstance(final_plans, dict)
    csa = final_plans["csa_structural"]
    assert isinstance(csa, dict)
    csa["policy"] = "not_structural_balanced"
    with pytest.raises(ValueError, match="did not execute structural_balanced"):
        _validate_cp8_structural_results(results)


def test_distributed_summary_accepts_smoke_report(tmp_path) -> None:
    report = {
        "case": "smoke",
        "final_query_rows": 7,
        "rank": 0,
        "source_rows": 7,
    }
    (tmp_path / "result_rank0.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    summary = summarize(tmp_path, world_size=1, case="smoke")

    assert summary["result_count"] == 1
    assert summary["results"] == [report]
