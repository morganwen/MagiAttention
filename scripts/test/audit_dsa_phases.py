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

_PHASE_PREFIX = "MAGI_DSA_PHASE "
_CONTROL_PREFIX = "MAGI_DSA_CP2 "


def _json_rank(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("phase rank must be a JSON integer")
    return value


def _records(line: str) -> list[tuple[str, dict[str, object]]]:
    positions: list[tuple[int, str, str]] = []
    for record_type, prefix in (
        ("control", _CONTROL_PREFIX),
        ("phase", _PHASE_PREFIX),
    ):
        begin = 0
        while True:
            position = line.find(prefix, begin)
            if position < 0:
                break
            positions.append((position, record_type, prefix))
            begin = position + len(prefix)
    decoder = json.JSONDecoder()
    records: list[tuple[str, dict[str, object]]] = []
    for position, record_type, prefix in sorted(positions):
        value, _ = decoder.raw_decode(line[position + len(prefix) :])
        if not isinstance(value, dict):
            raise TypeError("phase log payload is not a JSON object")
        records.append((record_type, value))
    return records


def _is_collective(name: str) -> bool:
    return name.startswith("collective_") or name == "gradient_allreduce"


def audit(path: Path, world_size: int, timed_out: bool) -> dict[str, object]:
    execute_started: set[int] = set()
    execute_ended: set[int] = set()
    stacks: dict[int, list[str]] = {rank: [] for rank in range(world_size)}
    errors: dict[int, list[str]] = {rank: [] for rank in range(world_size)}
    last_event: dict[int, dict[str, object] | None] = {
        rank: None for rank in range(world_size)
    }

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for record_type, payload in _records(line):
            rank = _json_rank(payload["rank"])
            event = str(payload["event"])
            if record_type == "control":
                if event == "execute_begin":
                    execute_started.add(rank)
                elif event == "execute_end":
                    execute_ended.add(rank)
                continue
            if rank not in execute_started or rank in execute_ended:
                continue
            name = str(payload["name"])
            last_event[rank] = {"event": event, "name": name}
            if event == "begin":
                stacks[rank].append(name)
            elif event in ("end", "error"):
                if stacks[rank] and stacks[rank][-1] == name:
                    stacks[rank].pop()
                else:
                    errors[rank].append(f"unmatched {event} for {name}")

    rank_reports: list[dict[str, object]] = []
    latest_open: list[str | None] = []
    for rank in range(world_size):
        open_phase = stacks[rank][-1] if stacks[rank] else None
        latest_open.append(open_phase)
        rank_reports.append(
            {
                "errors": errors[rank],
                "execute_ended": rank in execute_ended,
                "execute_started": rank in execute_started,
                "last_event": last_event[rank],
                "open_phases": stacks[rank],
                "rank": rank,
            }
        )

    all_started = len(execute_started) == world_size
    all_open_collective = all(
        name is not None and _is_collective(name) for name in latest_open
    )
    same_collective = len(set(latest_open)) == 1 if all_open_collective else False
    return {
        "all_ranks_execute_ended": len(execute_ended) == world_size,
        "all_ranks_execute_started": all_started,
        "collective_stall_confirmed": bool(
            timed_out and all_started and all_open_collective and same_collective
        ),
        "latest_open_phases": latest_open,
        "ranks": rank_reports,
        "timed_out": timed_out,
        "world_size": world_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--timed-out", action="store_true")
    arguments = parser.parse_args()
    report = audit(arguments.log, arguments.world_size, arguments.timed_out)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
