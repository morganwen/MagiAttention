#!/usr/bin/env python3
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
from pathlib import Path


def _json_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be a JSON integer")
    return value


def _json_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a JSON number")
    return float(value)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return value


def _control_events(path: Path) -> list[str]:
    events: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path} contains a non-object control record")
        events.append(str(value["event"]))
    return events


def summarize(
    artifact_dir: Path,
    world_size: int,
    case: str,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for rank in range(world_size):
        path = artifact_dir / f"result_rank{rank}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank result: {path}")
        result = _read_json(path)
        if _json_int(result["rank"], "rank") != rank:
            raise AssertionError(f"{path} has an incorrect rank")
        if str(result["case"]) != case:
            raise AssertionError(f"{path} has an incorrect case")
        results.append(result)

    full_backward = case in ("csa-natural-backward", "cp8-natural-backward")
    if full_backward:
        for rank in range(world_size):
            control_path = artifact_dir / f"control_rank{rank}.jsonl"
            if not control_path.is_file():
                raise FileNotFoundError(f"missing rank control log: {control_path}")
            events = _control_events(control_path)
            for required in (
                "execute_begin",
                "execute_end",
                "verification_begin",
                "verification_end",
            ):
                if events.count(required) != 1:
                    raise AssertionError(
                        f"rank {rank} has {events.count(required)} {required} records"
                    )

    execution_seconds = [
        _json_float(result["execution_seconds"], "execution_seconds")
        for result in results
        if "execution_seconds" in result
    ]
    if full_backward:
        if len(execution_seconds) != world_size:
            raise AssertionError(
                "full natural backward must report execution_seconds on every rank"
            )
        overdue = [
            (
                _json_int(result["rank"], "rank"),
                _json_float(result["execution_seconds"], "execution_seconds"),
            )
            for result in results
            if _json_float(result["execution_seconds"], "execution_seconds") >= 60.0
        ]
        if overdue:
            raise TimeoutError(
                f"post-compile natural execution exceeded the 60s deadline: {overdue}"
            )
    summary: dict[str, object] = {
        "case": case,
        "result_count": len(results),
        "results": results,
        "world_size": world_size,
    }
    if execution_seconds:
        summary["execution_seconds"] = {
            "max": max(execution_seconds),
            "min": min(execution_seconds),
        }
    if case == "cp8-natural-backward":
        checked_ranks = [
            _json_int(result["rank"], "rank")
            for result in results
            if bool(result.get("model_parameter_values_checked", False))
        ]
        if checked_ranks != [0]:
            raise AssertionError(
                "CP8 must check globally AllReduced model parameter values on rank 0"
            )
        summary["model_parameter_value_check_ranks"] = checked_ranks
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    arguments = parser.parse_args()
    summary = summarize(arguments.artifact_dir, arguments.world_size, arguments.case)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
