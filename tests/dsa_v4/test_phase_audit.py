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
from typing import cast

import pytest

from scripts.test.audit_dsa_phases import audit
from scripts.test.summarize_distributed import summarize


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
    rank_reports = cast(list[dict[str, object]], result["ranks"])
    assert all(not rank_report["errors"] for rank_report in rank_reports)


def _write_summary_fixture(tmp_path, execution_seconds: float) -> None:
    (tmp_path / "result_rank0.json").write_text(
        json.dumps(
            {
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
