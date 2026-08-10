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
import re
from pathlib import Path

_STRUCTURAL_CONFIG = {
    "chunk_size": 512,
    "min_chunks_per_rank": 16,
    "uneven_shard": True,
}
# One planner, and the main stack is CSA plus HCA only.
_CP8_PLAN_POLICIES = {
    "csa_structural": "structural_balanced",
    "hca_structural": "structural_balanced",
}


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


def _require_sha256(value: object, field: str) -> str:
    resolved = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", resolved) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return resolved


def _validate_plan_evidence(
    value: object,
    *,
    field: str,
    expected_policy: str,
    expected_ratio: int,
    rank: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    if value.get("policy") != expected_policy:
        raise ValueError(f"{field} did not execute {expected_policy}")
    if _json_int(value.get("ratio"), f"{field}.ratio") != expected_ratio:
        raise ValueError(f"{field} has an incorrect ratio")
    _require_sha256(value.get("plan_hash"), f"{field}.plan_hash")
    _require_sha256(value.get("query_layout_hash"), f"{field}.query_layout_hash")
    _require_sha256(
        value.get("rank_query_layout_signature"),
        f"{field}.rank_query_layout_signature",
    )
    source_counts = value.get("source_token_counts")
    query_counts = value.get("query_token_counts")
    if not isinstance(source_counts, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in source_counts
    ):
        raise TypeError(f"{field}.source_token_counts must be non-negative integers")
    if not isinstance(query_counts, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in query_counts
    ):
        raise TypeError(f"{field}.query_token_counts must be non-negative integers")
    if len(source_counts) != len(query_counts) or not 0 <= rank < len(query_counts):
        raise ValueError(f"{field} has inconsistent CP token-count tables")
    local_source = _json_int(
        value.get("local_source_tokens"), f"{field}.local_source_tokens"
    )
    local_query = _json_int(
        value.get("local_query_tokens"), f"{field}.local_query_tokens"
    )
    capacity = _json_int(
        value.get("declared_local_token_capacity"),
        f"{field}.declared_local_token_capacity",
    )
    if local_source != source_counts[rank] or local_query != query_counts[rank]:
        raise ValueError(f"{field} local token counts disagree with the global table")
    if capacity < max(local_source, local_query):
        raise ValueError(f"{field} declared capacity does not cover its local rows")
    if sum(query_counts) != sum(source_counts):
        raise ValueError(f"{field} does not cover every source Query exactly once")
    if _json_int(value.get("fragment_count"), f"{field}.fragment_count") <= 0:
        raise ValueError(f"{field} must contain at least one Query fragment")

    if expected_policy == "structural_balanced":
        if value.get("structural_layout_config") != _STRUCTURAL_CONFIG:
            raise ValueError(f"{field} structural config differs from the contract")
        metrics = value.get("structural_layout_metrics")
        rank_cost = value.get("structural_rank_cost")
        if (
            not isinstance(metrics, dict)
            or not str(metrics.get("solver_scheme", ""))
            or not str(metrics.get("cost_model_version", ""))
            or _json_int(metrics.get("num_chunks"), f"{field}.num_chunks") <= 0
            or _json_int(metrics.get("chunk_size"), f"{field}.chunk_size") <= 0
            or metrics.get("uneven_shard") is not True
        ):
            raise ValueError(f"{field} structural solver evidence is incomplete")
        if (
            not isinstance(rank_cost, dict)
            or _json_int(rank_cost.get("rank"), f"{field}.rank_cost.rank") != rank
            or _json_int(
                rank_cost.get("query_tokens"), f"{field}.rank_cost.query_tokens"
            )
            != local_query
        ):
            raise ValueError(f"{field} structural rank cost is incomplete")
    return value


def _validate_cp8_structural_results(
    results: list[dict[str, object]],
) -> dict[str, object]:
    query_hashes: set[str] = set()
    source_count_tables: set[tuple[int, ...]] = set()
    query_count_tables: set[tuple[int, ...]] = set()
    plan_hashes: dict[str, set[str]] = {label: set() for label in _CP8_PLAN_POLICIES}
    for rank, result in enumerate(results):
        evidence_by_label = result.get("plan_evidence")
        if not isinstance(evidence_by_label, dict) or set(evidence_by_label) != set(
            _CP8_PLAN_POLICIES
        ):
            raise ValueError(f"rank {rank} CP8 plan evidence is incomplete")
        validated: dict[str, dict[str, object]] = {}
        for label, policy in _CP8_PLAN_POLICIES.items():
            ratio = 4 if label.startswith("csa_") else 128
            plan = _validate_plan_evidence(
                evidence_by_label[label],
                field=f"rank {rank}.{label}",
                expected_policy=policy,
                expected_ratio=ratio,
                rank=rank,
            )
            validated[label] = plan
            plan_hashes[label].add(str(plan["plan_hash"]))
        csa = validated["csa_structural"]
        hca = validated["hca_structural"]
        shared_fields = (
            "query_layout_hash",
            "rank_query_layout_signature",
            "source_token_counts",
            "query_token_counts",
            "structural_layout_config",
            "structural_layout_metrics",
            "structural_rank_cost",
        )
        if any(csa[field] != hca[field] for field in shared_fields):
            raise ValueError(f"rank {rank} structural CSA/HCA layouts differ")
        if result.get("structural_layout_shared") is not True:
            raise ValueError(f"rank {rank} did not validate the shared Pro layout")
        if result.get("structural_query_layout_hash") != csa["query_layout_hash"]:
            raise ValueError(f"rank {rank} reported a stale structural layout hash")
        query_hashes.add(str(csa["query_layout_hash"]))
        source_counts = csa["source_token_counts"]
        query_counts = csa["query_token_counts"]
        if not isinstance(source_counts, list) or not isinstance(query_counts, list):
            raise TypeError(f"rank {rank} structural token counts are invalid")
        source_count_tables.add(
            tuple(
                _json_int(item, f"rank {rank}.source_token_counts")
                for item in source_counts
            )
        )
        query_count_tables.add(
            tuple(
                _json_int(item, f"rank {rank}.query_token_counts")
                for item in query_counts
            )
        )

    if len(query_hashes) != 1:
        raise ValueError("CP8 ranks did not execute one structural Query layout")
    if source_count_tables != {(32,) * 8} or query_count_tables != {(32,) * 8}:
        raise ValueError(
            "CP8 structural token-count tables differ from the frozen case"
        )
    if any(len(hashes) != 1 for hashes in plan_hashes.values()):
        raise ValueError("CP8 ranks did not execute identical global plans")
    return {
        "plan_hashes": {
            label: next(iter(hashes)) for label, hashes in plan_hashes.items()
        },
        "query_layout_hash": next(iter(query_hashes)),
        "query_token_counts": list(next(iter(query_count_tables))),
        "result": "PASS",
    }


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

    structural_contract: dict[str, object] | None = None
    if case in ("csa-natural", "csa-natural-backward"):
        for rank, result in enumerate(results):
            plan = _validate_plan_evidence(
                result.get("balanced_plan_evidence"),
                field=f"rank {rank}.balanced_plan_evidence",
                expected_policy="structural_balanced",
                expected_ratio=4,
                rank=rank,
            )
            if result.get("balanced_policy") != plan["policy"]:
                raise ValueError(f"rank {rank} reported a stale balanced policy")
    elif case == "cp8-topk-diagnostic":
        for rank, result in enumerate(results):
            plan = _validate_plan_evidence(
                result.get("structural_plan_evidence"),
                field=f"rank {rank}.structural_plan_evidence",
                expected_policy="structural_balanced",
                expected_ratio=4,
                rank=rank,
            )
            if result.get("plan_policy") != plan["policy"]:
                raise ValueError(f"rank {rank} reported a stale Top-K plan policy")
    elif case == "cp8-natural-backward":
        structural_contract = _validate_cp8_structural_results(results)

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
        assert structural_contract is not None
        summary["structural_contract"] = structural_contract
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
