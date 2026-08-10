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
from typing import Any, Callable

if __package__:
    from .summarize_forward_backward import _read_json as _package_read_json
    from .summarize_forward_backward import _read_jsonl as _package_read_jsonl
    from .summarize_forward_backward import (
        read_nvtx_range_counts as _package_read_nvtx_range_counts,
    )

    _read_json = _package_read_json
    _read_jsonl = _package_read_jsonl
    read_nvtx_range_counts = _package_read_nvtx_range_counts
else:
    from summarize_forward_backward import _read_json as _script_read_json
    from summarize_forward_backward import _read_jsonl as _script_read_jsonl
    from summarize_forward_backward import (
        read_nvtx_range_counts as _script_read_nvtx_range_counts,
    )

    _read_json = _script_read_json
    _read_jsonl = _script_read_jsonl
    read_nvtx_range_counts = _script_read_nvtx_range_counts

_PHASE_NVTX_NAMES = {
    "backward": "magi_dsa::backward",
    "csa_backward": "magi_dsa::pro_pair::csa::backward",
    "csa_forward": "magi_dsa::pro_pair::csa::forward",
    "forward": "magi_dsa::forward",
    "hca_backward": "magi_dsa::pro_pair::hca::backward",
    "hca_forward": "magi_dsa::pro_pair::hca::forward",
    "indexer_score": "magi_dsa::indexer_score",
    "indexer_topk": "magi_dsa::indexer_topk",
    "parameter_gradient_allreduce": "magi_dsa::parameter_gradient_allreduce",
}
_PHASES = tuple(_PHASE_NVTX_NAMES)
_EXPECTED_SENDRECV = {
    "backward": 7,
    "csa_backward": 4,
    "csa_forward": 4,
    "forward": 7,
    "hca_backward": 3,
    "hca_forward": 3,
}
_MODE_ROUTE_ORDER: dict[str, dict[str, tuple[str, ...]]] = {
    "csa": {
        "forward": ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV"),
        "backward": ("COMPRESSED_KI", "COMPRESSED_KV", "OVERLAP_X", "WINDOW_KV"),
    },
    "hca": {
        "forward": ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
        "backward": ("COMPRESSED_KV", "WINDOW_KV", "OVERLAP_X"),
    },
}
_ROUTE_OVERLAP_CONTRACT = {
    ("csa", "forward", "WINDOW_KV"): {
        "classification": "overlap_capable",
        "reason": "CSA Indexer projection and compression work are independent of the in-flight Window route.",
    },
    ("csa", "forward", "OVERLAP_X"): {
        "classification": "overlap_capable",
        "reason": "CSA Indexer projection runs independently while compression support is in flight.",
    },
    ("csa", "forward", "COMPRESSED_KI"): {
        "classification": "overlap_capable",
        "reason": "CSA Main Compressor runs after the Indexer Compressor starts the KI route.",
    },
    ("csa", "forward", "COMPRESSED_KV"): {
        "classification": "overlap_capable",
        "reason": "CSA grouped Indexer score and Top-K run while compressed KV is in flight.",
    },
    ("csa", "backward", "COMPRESSED_KI"): {
        "classification": "overlap_capable",
        "reason": "CSA sparse-attention backward runs on an independent stream after KI reverse starts.",
    },
    ("csa", "backward", "COMPRESSED_KV"): {
        "classification": "overlap_capable",
        "reason": "CSA Indexer backward is independent of the compressed-KV reverse route.",
    },
    ("csa", "backward", "OVERLAP_X"): {
        "classification": "overlap_capable",
        "reason": "CSA projection and support backward work remains independent of the support reverse route.",
    },
    ("csa", "backward", "WINDOW_KV"): {
        "classification": "overlap_capable",
        "reason": "CSA projection backward remains available after the late join releases Window reverse.",
    },
    ("hca", "forward", "OVERLAP_X"): {
        "classification": "dependency_bound",
        "reason": "HCA compression-support packing and Main Compressor require OVERLAP_X to finish first.",
    },
    ("hca", "forward", "WINDOW_KV"): {
        "classification": "overlap_capable",
        "reason": "HCA Main Compressor runs independently while Window KV is in flight.",
    },
    ("hca", "forward", "COMPRESSED_KV"): {
        "classification": "dependency_bound",
        "reason": "HCA compressed KV is produced by Main Compressor and must finish before KV-bank assembly and attention.",
    },
    ("hca", "backward", "COMPRESSED_KV"): {
        "classification": "dependency_bound",
        "reason": "HCA compressed-KV reverse supplies the Main Compressor gradient before its backward work can run.",
    },
    ("hca", "backward", "WINDOW_KV"): {
        "classification": "overlap_capable",
        "reason": "HCA Window reverse runs on the route stream while Main Compressor backward runs independently.",
    },
    ("hca", "backward", "OVERLAP_X"): {
        "classification": "dependency_bound",
        "reason": "HCA OVERLAP_X reverse consumes the Main Compressor support gradient and is the terminal support route.",
    },
}
_MODE_SERIAL_ORDER = (
    ("csa", "forward"),
    ("hca", "forward"),
    ("hca", "backward"),
    ("csa", "backward"),
)
_PRO_FLASHMLA_FORWARD_KERNEL = "sparse_attn_fwd_for_small_topk_kernel"
_MAJOR_KERNEL_SPECS = {
    "csa_indexer_score": {
        "phase": "indexer_score",
        "predicate": lambda name: "indexer_forward" in name.lower(),
        "expected_count": 1,
    },
    "csa_indexer_topk": {
        "phase": "indexer_topk",
        "predicate": lambda name: "indexer_topk_kernel" in name.lower(),
        "expected_count": 1,
    },
    "csa_selected_indexer_backward": {
        "phase": "csa_forward",
        "predicate": lambda name: (
            "indexer_backward" in name.lower()
            and "dense_indexer_backward" not in name.lower()
        ),
        "expected_count": None,
    },
    "csa_flashmla_forward": {
        "phase": "csa_forward",
        "predicate": lambda name: name == _PRO_FLASHMLA_FORWARD_KERNEL,
        "expected_count": 1,
    },
    "hca_flashmla_forward": {
        "phase": "hca_forward",
        "predicate": lambda name: name == _PRO_FLASHMLA_FORWARD_KERNEL,
        "expected_count": 1,
    },
    "csa_sparse_attention_backward_main": {
        "phase": "csa_backward",
        "predicate": lambda name: (
            "kernel_cutlass_bwd_" in name.lower()
            and "sparse_attention_backward" in name.lower()
        ),
        "expected_count": 1,
    },
    "hca_sparse_attention_backward_main": {
        "phase": "hca_backward",
        "predicate": lambda name: (
            "kernel_cutlass_bwd_" in name.lower()
            and "sparse_attention_backward" in name.lower()
        ),
        "expected_count": 1,
    },
}
_COMPRESSOR_FORWARD_SCOPES = {
    "csa_compressor_forward": (
        "csa",
        (
            "magi_dsa::module::compressor::indexer",
            "magi_dsa::module::compressor::main",
        ),
    ),
    "csa_indexer_compressor_forward": (
        "csa",
        ("magi_dsa::module::compressor::indexer",),
    ),
    "csa_main_compressor_forward": (
        "csa",
        ("magi_dsa::module::compressor::main",),
    ),
    "hca_compressor_forward": (
        "hca",
        ("magi_dsa::module::compressor::main",),
    ),
}
_CSA_EXPLICIT_COMPRESSOR_BACKWARD_KERNELS = {
    "_csa_compressor_ape_backward_kernel": 2,
    "_csa_compressor_backward_kernel": 2,
}
_PRO_REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
_GLOBAL_OUTPUT_ELEMENTS = 131072 * 128 * 512
_CUDNN_D2D_SCOPES = {
    "indexer_score": "magi_dsa::CUDNN_CALL::indexer_score",
    "indexer_topk": "magi_dsa::CUDNN_CALL::indexer_topk",
    "selected_attention_recompute": (
        "magi_dsa::CUDNN_CALL::selected_attention_recompute"
    ),
    "selected_indexer_backward": "magi_dsa::CUDNN_CALL::indexer_backward",
    "selected_indexer_recompute": ("magi_dsa::CUDNN_CALL::selected_indexer_recompute"),
}
# Route staging is Core range ops now, not the retired DSA pack kernels.
_NON_COMPUTE_KERNEL_TOKENS = (
    "ncclDevKernel_SendRecv",
    "range_gather",
    "range_sum_reduce",
    "range_avg_reduce",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the representative DeepSeek-V4-Pro CSA+HCA profile"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _scope_names(record: dict[str, Any]) -> set[str]:
    return {
        str(scope.get("name", ""))
        for scope in record.get("attribution_path", [])
        if isinstance(scope, dict)
    }


def _scope_rowids(record: dict[str, Any], scope_name: str) -> set[int]:
    return {
        int(scope["rowid"])
        for scope in record.get("attribution_path", [])
        if isinstance(scope, dict)
        and str(scope.get("name", "")) == scope_name
        and scope.get("rowid") is not None
    }


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized = sorted(
        (int(start), int(end)) for start, end in intervals if int(end) > int(start)
    )
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_duration_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _interval_overlap_ns(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> int:
    left_merged = _merge_intervals(left)
    right_merged = _merge_intervals(right)
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _is_non_route_compute(record: dict[str, Any]) -> bool:
    kernel_name = str(record.get("kernel_name", ""))
    if any(
        token.lower() in kernel_name.lower() for token in _NON_COMPUTE_KERNEL_TOKENS
    ):
        return False
    return not any(
        scope.startswith("magi_dsa::phase::collective_all2all_v::")
        for scope in _scope_names(record)
    )


def _rank_ranges(
    records: list[dict[str, Any]],
    *,
    group_field: str,
    value_field: str,
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    groups = sorted({str(record[group_field]) for record in records})
    ranges: list[dict[str, Any]] = []
    for step in range(steps):
        for group in groups:
            selected = [
                record
                for record in records
                if int(record["step"]) == step and str(record[group_field]) == group
            ]
            if len(selected) != world_size:
                raise ValueError(
                    f"Pro pair timing grid differs: group={group}, step={step}, "
                    f"records={len(selected)}, expected={world_size}"
                )
            values = [float(record[value_field]) for record in selected]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Pro pair timing contains invalid values: {group}")
            mean_value = statistics.fmean(values)
            minimum = min(values)
            maximum = max(values)
            ranges.append(
                {
                    "group": group,
                    "max": maximum,
                    "mean": mean_value,
                    "min": minimum,
                    "rank_range": maximum - minimum,
                    "relative_rank_range": (
                        (maximum - minimum) / mean_value if mean_value > 0 else 0.0
                    ),
                    "step": step,
                    "threshold": None,
                    "unit": "ms" if value_field.endswith("time_ms") else value_field,
                }
            )
    return ranges


def validate_pro_pair_records(
    records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    expected = {
        (step, rank, phase)
        for step in range(steps)
        for rank in range(world_size)
        for phase in _PHASES
    }
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record.get("step", -1)),
            int(record.get("rank", -1)),
            str(record.get("phase")),
        )
        if key in by_key:
            raise ValueError(f"duplicate Pro pair profile record: {key}")
        by_key[key] = record
        step, rank, phase = key
        if (
            record.get("plan") != "balanced"
            or phase not in _PHASES
            or not 0 <= step < steps
            or not 0 <= rank < world_size
        ):
            raise ValueError(f"invalid Pro pair profile record: {key}")
        if record.get("nvtx_name") != _PHASE_NVTX_NAMES[phase]:
            raise ValueError(f"Pro pair NVTX name differs: {key}")
        if int(record.get("logical_call_count", 0)) != 1:
            raise ValueError(f"Pro pair logical call count differs: {key}")
        gpu_time_ms = float(record.get("gpu_time_ms", float("nan")))
        if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
            raise ValueError(f"Pro pair phase GPU time is invalid: {key}")
        if int(record.get("kernel_launch_count", 0)) <= 0:
            raise ValueError(f"Pro pair phase has no kernels: {key}")
    if set(by_key) != expected:
        raise ValueError(
            "Pro pair profile grid mismatch: "
            f"missing={sorted(expected - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected)}"
        )
    return [by_key[key] for key in sorted(expected)]


def compute_pro_pair_phase_rank_ranges(
    records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    validated = validate_pro_pair_records(records, world_size, steps)
    normalized = [
        {
            "gpu_time_ms": float(record["gpu_time_ms"]),
            "phase": str(record["phase"]),
            "rank": int(record["rank"]),
            "step": int(record["step"]),
        }
        for record in validated
    ]
    return _rank_ranges(
        normalized,
        group_field="phase",
        value_field="gpu_time_ms",
        world_size=world_size,
        steps=steps,
    )


def validate_pro_pair_sendrecv(records: list[dict[str, Any]]) -> None:
    for record in records:
        phase = str(record["phase"])
        if phase not in _EXPECTED_SENDRECV:
            continue
        count = int(
            record.get("kernel_name_counts", {}).get("ncclDevKernel_SendRecv", 0)
        )
        expected = _EXPECTED_SENDRECV[phase]
        if count != expected:
            raise ValueError(
                f"Pro pair {phase} SendRecv count differs: rank={record['rank']}, "
                f"step={record['step']}, count={count}, expected={expected}"
            )


def compute_major_kernel_timings(
    records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    by_phase_key = {
        (str(record["phase"]), int(record["rank"]), int(record["step"])): record
        for record in records
    }
    timing_records: list[dict[str, Any]] = []
    kernel_names: dict[str, set[str]] = {group: set() for group in _MAJOR_KERNEL_SPECS}
    for group, spec in _MAJOR_KERNEL_SPECS.items():
        phase = str(spec["phase"])
        predicate = spec["predicate"]
        if not callable(predicate):
            raise TypeError(f"major kernel predicate is not callable: {group}")
        typed_predicate: Callable[[str], bool] = predicate
        expected_count = spec["expected_count"]
        for rank in range(world_size):
            for step in range(steps):
                try:
                    phase_record = by_phase_key[(phase, rank, step)]
                except KeyError as error:
                    raise ValueError(
                        f"major kernel phase is missing: {group}, rank={rank}, step={step}"
                    ) from error
                matching = [
                    kernel
                    for kernel in phase_record.get("kernels", [])
                    if typed_predicate(str(kernel.get("name", "")))
                ]
                if not matching:
                    raise ValueError(
                        f"major kernel group is empty: {group}, rank={rank}, step={step}"
                    )
                if expected_count is not None:
                    if not isinstance(expected_count, int):
                        raise TypeError(
                            f"major kernel expected count is not an integer: {group}"
                        )
                    if len(matching) != expected_count:
                        raise ValueError(
                            f"major kernel count differs: {group}, rank={rank}, step={step}, "
                            f"count={len(matching)}, expected={expected_count}"
                        )
                duration_ns = sum(int(kernel["duration_ns"]) for kernel in matching)
                if duration_ns <= 0:
                    raise ValueError(f"major kernel duration is invalid: {group}")
                names = {str(kernel["name"]) for kernel in matching}
                kernel_names[group].update(names)
                timing_records.append(
                    {
                        "gpu_time_ms": duration_ns / 1_000_000.0,
                        "group": group,
                        "kernel_launch_count": len(matching),
                        "kernel_names": sorted(names),
                        "phase": phase,
                        "rank": rank,
                        "step": step,
                    }
                )

    if any(
        "dense_indexer_backward" in name.lower()
        for names in kernel_names.values()
        for name in names
    ):
        raise ValueError("Pro pair contains a dense Indexer backward kernel")
    flashmla_groups = ("csa_flashmla_forward", "hca_flashmla_forward")
    for group in flashmla_groups:
        if kernel_names[group] != {_PRO_FLASHMLA_FORWARD_KERNEL}:
            raise ValueError(
                "Pro pair FlashMLA forward variant differs: "
                f"group={group}, names={sorted(kernel_names[group])}, "
                f"expected={_PRO_FLASHMLA_FORWARD_KERNEL}"
            )
    rank_ranges = _rank_ranges(
        timing_records,
        group_field="group",
        value_field="gpu_time_ms",
        world_size=world_size,
        steps=steps,
    )
    hard_gate_groups = {"csa_indexer_score", "csa_indexer_topk"}
    for item in rank_ranges:
        if item["group"] not in hard_gate_groups:
            continue
        item["threshold"] = 0.05
        item["passed"] = float(item["relative_rank_range"]) <= 0.05
        if not item["passed"]:
            raise ValueError(
                "Pro pair Indexer rank imbalance exceeds 5%: "
                f"group={item['group']}, step={item['step']}, "
                f"relative_rank_range={item['relative_rank_range']}"
            )
    return {
        "balance_gate": "indexer_score_topk_0.05_others_report_only",
        "flashmla_forward_kernel": _PRO_FLASHMLA_FORWARD_KERNEL,
        "flashmla_forward_same_exact_variant": True,
        "hard_gate_groups": sorted(hard_gate_groups),
        "kernel_names": {
            group: sorted(names) for group, names in sorted(kernel_names.items())
        },
        "rank_ranges": rank_ranges,
        "records": timing_records,
        "result": "PASS",
    }


def compute_compressor_timings(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Report exact forward scopes and only unambiguous backward kernels."""

    timing_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        for step in range(steps):
            grid_records = [
                record
                for record in attribution_records
                if int(record["rank"]) == rank and int(record["step"]) == step
            ]
            for group, (mode, compressor_scopes) in _COMPRESSOR_FORWARD_SCOPES.items():
                mode_scope = f"magi_dsa::pro_pair::{mode}::forward"
                matching = [
                    record
                    for record in grid_records
                    if mode_scope in _scope_names(record)
                    and any(
                        scope in _scope_names(record) for scope in compressor_scopes
                    )
                ]
                if not matching:
                    raise ValueError(
                        "Pro compressor forward group is empty: "
                        f"{group}, rank={rank}, step={step}"
                    )
                timing_records.append(
                    {
                        "attribution": "module_nvtx_runtime_launch",
                        "gpu_time_ms": sum(
                            float(record["gpu_time_ms"]) for record in matching
                        ),
                        "group": group,
                        "kernel_launch_count": len(matching),
                        "kernel_names": sorted(
                            {str(record["kernel_name"]) for record in matching}
                        ),
                        "rank": rank,
                        "step": step,
                    }
                )

            csa_backward_scope = "magi_dsa::pro_pair::csa::backward"
            backward_matching = [
                record
                for record in grid_records
                if csa_backward_scope in _scope_names(record)
                and str(record.get("kernel_name", ""))
                in _CSA_EXPLICIT_COMPRESSOR_BACKWARD_KERNELS
            ]
            backward_counts = {
                name: sum(
                    str(record["kernel_name"]) == name for record in backward_matching
                )
                for name in _CSA_EXPLICIT_COMPRESSOR_BACKWARD_KERNELS
            }
            if backward_counts != _CSA_EXPLICIT_COMPRESSOR_BACKWARD_KERNELS:
                raise ValueError(
                    "Pro CSA explicit compressor backward kernel counts differ: "
                    f"rank={rank}, step={step}, counts={backward_counts}"
                )
            timing_records.append(
                {
                    "attribution": "mode_backward_plus_exact_kernel_name",
                    "gpu_time_ms": sum(
                        float(record["gpu_time_ms"]) for record in backward_matching
                    ),
                    "group": "csa_compressor_backward_explicit",
                    "kernel_launch_count": len(backward_matching),
                    "kernel_names": sorted(backward_counts),
                    "rank": rank,
                    "step": step,
                }
            )

    return {
        "attribution_complete": False,
        "balance_gate": "report_only",
        "coverage": {
            "backward": ("CSA fused overlap-compressor kernels with exact names only"),
            "forward": "complete module-NVTX runtime-launch attribution",
        },
        "coverage_gap": [
            (
                "Compressor linear dgrad/wgrad kernels do not yet have "
                "compressor-specific launch-thread NVTX"
            ),
            (
                "HCA non-overlap Compressor backward uses generic autograd "
                "kernels that cannot be separated from the mode backward range"
            ),
        ],
        "rank_ranges": _rank_ranges(
            timing_records,
            group_field="group",
            value_field="gpu_time_ms",
            world_size=world_size,
            steps=steps,
        ),
        "records": timing_records,
        "result": "PASS_WITH_DOCUMENTED_COVERAGE_GAP",
    }


def compute_support_overhead_timings(
    attribution_records: list[dict[str, Any]],
    indexer_k_packing: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Account packing, route staging, and KV-bank assembly separately."""

    def in_mode(record: dict[str, Any], mode: str) -> bool:
        scopes = _scope_names(record)
        return any(
            f"magi_dsa::pro_pair::{mode}::{direction}" in scopes
            for direction in ("forward", "backward")
        )

    def select(record: dict[str, Any], group: str) -> bool:
        scopes = _scope_names(record)
        kernel_name = str(record.get("kernel_name", "")).lower()
        if group == "csa_grouped_k_pack_forward":
            return (
                "magi_dsa::module::packing::csa::indexer_key_support::forward_copy"
                in scopes
                and "range_gather" in kernel_name
            )
        if group.endswith("_route_stage_copy"):
            mode = group.split("_", 1)[0]
            return "range_gather" in kernel_name and any(
                scope.startswith(f"magi_dsa::module::route::attention::{mode}::")
                for scope in scopes
            )
        if group.endswith("_route_stage_csr_reduce"):
            mode = group.split("_", 1)[0]
            return "range_sum_reduce" in kernel_name and any(
                scope.startswith(f"magi_dsa::module::route::attention::{mode}::")
                for scope in scopes
            )
        if group.endswith("_kv_bank_catarray"):
            mode = group.split("_", 1)[0]
            return (
                f"magi_dsa::module::attention::{mode}::kv_bank_assembly" in scopes
                and "catarray" in kernel_name
                and in_mode(record, mode)
            )
        raise ValueError(f"unknown Pro support-overhead group: {group}")

    groups = (
        "csa_grouped_k_pack_forward",
        "csa_route_stage_copy",
        "csa_route_stage_csr_reduce",
        "hca_route_stage_copy",
        "hca_route_stage_csr_reduce",
        "csa_kv_bank_catarray",
        "hca_kv_bank_catarray",
    )
    grouped_k_scope_prefix = "magi_dsa::module::packing::csa::indexer_key_support::"
    unexpected_grouped_k_backward = [
        record
        for record in attribution_records
        if any(
            scope.startswith(grouped_k_scope_prefix) for scope in _scope_names(record)
        )
        and "range_sum_reduce" in str(record.get("kernel_name", "")).lower()
    ]
    if unexpected_grouped_k_backward:
        raise ValueError(
            "Pro pair grouped Indexer K unexpectedly ran a backward reduction"
        )
    packing_by_rank: dict[int, dict[str, Any]] = {}
    for packing in indexer_k_packing:
        rank = int(packing.get("rank", -1))
        if rank in packing_by_rank:
            raise ValueError(f"duplicate Pro Indexer K packing rank: {rank}")
        packing_by_rank[rank] = packing
    if set(packing_by_rank) != set(range(world_size)):
        raise ValueError("Pro Indexer K packing metadata rank grid differs")

    timing_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        for step in range(steps):
            grid_records = [
                record
                for record in attribution_records
                if int(record["rank"]) == rank and int(record["step"]) == step
            ]
            for group in groups:
                matching = [record for record in grid_records if select(record, group)]
                if group == "csa_grouped_k_pack_forward":
                    module_scope = (
                        "magi_dsa::module::packing::csa::"
                        "indexer_key_support::forward_copy"
                    )
                    module_rowids = {
                        rowid
                        for record in grid_records
                        for rowid in _scope_rowids(record, module_scope)
                    }
                    if len(module_rowids) != 1:
                        raise ValueError(
                            "Pro grouped Indexer K pack module range count differs: "
                            f"rank={rank}, step={step}, rowids={sorted(module_rowids)}"
                        )
                    if len(matching) != 1:
                        raise ValueError(
                            "Pro grouped Indexer K pack range_gather count differs: "
                            f"rank={rank}, step={step}, count={len(matching)}"
                        )
                    module_rowid = next(iter(module_rowids))
                    if module_rowid not in _scope_rowids(matching[0], module_scope):
                        raise ValueError(
                            "Pro grouped Indexer K pack kernel does not belong to "
                            "the unique module range"
                        )
                elif not matching:
                    raise ValueError(
                        f"Pro support-overhead group is empty: {group}, "
                        f"rank={rank}, step={step}"
                    )
                record_payload: dict[str, Any] = {
                    "gpu_time_ms": sum(
                        float(record["gpu_time_ms"]) for record in matching
                    ),
                    "group": group,
                    "kernel_launch_count": len(matching),
                    "kernel_names": sorted(
                        {str(record["kernel_name"]) for record in matching}
                    ),
                    "rank": rank,
                    "step": step,
                }
                if group == "csa_grouped_k_pack_forward":
                    pack_intervals = [
                        (
                            int(record["kernel_start_ns"]),
                            int(record["kernel_end_ns"]),
                        )
                        for record in matching
                    ]
                    external_compute = [
                        (
                            int(record["kernel_start_ns"]),
                            int(record["kernel_end_ns"]),
                        )
                        for record in grid_records
                        if "magi_dsa::pro_pair::csa::forward" in _scope_names(record)
                        and _is_non_route_compute(record)
                    ]
                    overlap_ns = _interval_overlap_ns(
                        pack_intervals,
                        external_compute,
                    )
                    duration_ns = _interval_duration_ns(pack_intervals)
                    packing = packing_by_rank[rank]
                    packed_rows = int(packing["packed_indexer_k_rows"])
                    unique_rows = int(packing["unique_indexer_k_rows"])
                    duplicate_rows = int(packing["duplicate_indexer_k_rows"])
                    row_bytes = int(packing["indexer_k_row_bytes"])
                    packed_bytes = int(packing["packed_indexer_k_bytes"])
                    unique_bytes = int(packing["unique_indexer_k_bytes"])
                    duplicate_bytes = int(packing["duplicate_indexer_k_bytes"])
                    if (
                        packed_rows != unique_rows + duplicate_rows
                        or packed_bytes != packed_rows * row_bytes
                        or unique_bytes != unique_rows * row_bytes
                        or duplicate_bytes != duplicate_rows * row_bytes
                    ):
                        raise ValueError(
                            f"rank {rank} Pro grouped Indexer K static accounting differs"
                        )
                    record_payload.update(
                        {
                            "duplicate_bytes": duplicate_bytes,
                            "duplicate_rows": duplicate_rows,
                            "external_compute_overlap_fraction": (
                                overlap_ns / duration_ns if duration_ns else 0.0
                            ),
                            "external_compute_overlap_ms": overlap_ns / 1_000_000.0,
                            "module_nvtx_rowid": module_rowid,
                            "packed_bytes": packed_bytes,
                            "packed_rows": packed_rows,
                            "read_bytes": packed_bytes,
                            "traffic_bytes": packed_bytes * 2,
                            "unique_bytes": unique_bytes,
                            "unique_rows": unique_rows,
                            "write_bytes": packed_bytes,
                        }
                    )
                timing_records.append(record_payload)
    return {
        "balance_gate": "report_only",
        "grouped_k_pack_backward_csr_reduce_launches": 0,
        "rank_ranges": _rank_ranges(
            timing_records,
            group_field="group",
            value_field="gpu_time_ms",
            world_size=world_size,
            steps=steps,
        ),
        "records": timing_records,
        "result": "PASS",
    }


def compute_route_timings(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    expected_contract_keys = {
        (mode, direction, route)
        for mode, directions in _MODE_ROUTE_ORDER.items()
        for direction, routes in directions.items()
        for route in routes
    }
    if set(_ROUTE_OVERLAP_CONTRACT) != expected_contract_keys:
        raise ValueError("Pro pair route-overlap contract does not cover every route")
    timing_records: list[dict[str, Any]] = []
    classification_counts = {
        "dependency_bound": 0,
        "overlap_capable": 0,
    }
    by_grid: dict[tuple[int, int, str, str, str], list[dict[str, Any]]] = {}
    for record in attribution_records:
        if "ncclDevKernel_SendRecv" not in str(record.get("kernel_name", "")):
            continue
        scopes = _scope_names(record)
        for mode, directions in _MODE_ROUTE_ORDER.items():
            for direction, routes in directions.items():
                for route in routes:
                    scope = (
                        "magi_dsa::phase::collective_all2all_v::"
                        f"attention::{mode}::{route}.{direction}"
                    )
                    if scope in scopes:
                        key = (
                            int(record["rank"]),
                            int(record["step"]),
                            mode,
                            direction,
                            route,
                        )
                        by_grid.setdefault(key, []).append(record)

    for rank in range(world_size):
        for step in range(steps):
            for mode, directions in _MODE_ROUTE_ORDER.items():
                for direction, routes in directions.items():
                    launch_starts: list[int] = []
                    for route in routes:
                        key = (rank, step, mode, direction, route)
                        matching = by_grid.get(key, [])
                        if len(matching) != 1:
                            raise ValueError(
                                "Pro pair route SendRecv count differs: "
                                f"key={key}, count={len(matching)}"
                            )
                        record = matching[0]
                        launch_starts.append(int(record["runtime_start_ns"]))
                        route_intervals = [
                            (
                                int(record["kernel_start_ns"]),
                                int(record["kernel_end_ns"]),
                            )
                        ]
                        phase_scope = f"magi_dsa::pro_pair::{mode}::{direction}"
                        same_mode_compute = [
                            (
                                int(candidate["kernel_start_ns"]),
                                int(candidate["kernel_end_ns"]),
                            )
                            for candidate in attribution_records
                            if int(candidate["rank"]) == rank
                            and int(candidate["step"]) == step
                            and phase_scope in _scope_names(candidate)
                            and _is_non_route_compute(candidate)
                        ]
                        other_mode_compute = [
                            (
                                int(candidate["kernel_start_ns"]),
                                int(candidate["kernel_end_ns"]),
                            )
                            for candidate in attribution_records
                            if int(candidate["rank"]) == rank
                            and int(candidate["step"]) == step
                            and _is_non_route_compute(candidate)
                            and any(
                                f"magi_dsa::pro_pair::{other_mode}::{other_direction}"
                                in _scope_names(candidate)
                                for other_mode, other_direction in _MODE_SERIAL_ORDER
                                if (other_mode, other_direction) != (mode, direction)
                            )
                        ]
                        overlap_ns = _interval_overlap_ns(
                            route_intervals,
                            same_mode_compute,
                        )
                        contract = _ROUTE_OVERLAP_CONTRACT[(mode, direction, route)]
                        classification = str(contract["classification"])
                        if classification not in classification_counts:
                            raise ValueError(
                                "Pro pair route has an invalid overlap classification: "
                                f"key={key}, classification={classification}"
                            )
                        classification_counts[classification] += 1
                        other_mode_overlap_ns = _interval_overlap_ns(
                            route_intervals,
                            other_mode_compute,
                        )
                        if other_mode_overlap_ns:
                            raise ValueError(
                                "Pro pair route overlaps other-mode compute: "
                                f"key={key}, overlap_ns={other_mode_overlap_ns}"
                            )
                        route_duration_ns = _interval_duration_ns(route_intervals)
                        timing_records.append(
                            {
                                "direction": direction,
                                "gpu_time_ms": float(record["gpu_time_ms"]),
                                "group": f"{mode}.{direction}.{route}",
                                "kernel_end_ns": int(record["kernel_end_ns"]),
                                "kernel_start_ns": int(record["kernel_start_ns"]),
                                "mode": mode,
                                "nvtx_path": record.get("attribution_path", []),
                                "overlap_classification": classification,
                                "overlap_contract_reason": str(contract["reason"]),
                                "observed_positive_same_mode_compute_overlap": (
                                    overlap_ns > 0
                                ),
                                "other_mode_compute_overlap_fraction": 0.0,
                                "other_mode_compute_overlap_ms": 0.0,
                                "rank": rank,
                                "route": route,
                                "runtime_start_ns": int(record["runtime_start_ns"]),
                                "same_mode_compute_overlap_fraction": (
                                    overlap_ns / route_duration_ns
                                    if route_duration_ns
                                    else 0.0
                                ),
                                "same_mode_compute_overlap_ms": (
                                    overlap_ns / 1_000_000.0
                                ),
                                "same_mode_compute_overlap_ns": overlap_ns,
                                "step": step,
                            }
                        )
                    if launch_starts != sorted(launch_starts):
                        raise ValueError(
                            "Pro pair route launch order differs: "
                            f"rank={rank}, step={step}, mode={mode}, "
                            f"direction={direction}"
                        )
    return {
        "cross_mode_compute_overlap_is_hard_gate": True,
        "expected_sendrecv": "CSA=4F+4B,HCA=3F+3B,total=7F+7B",
        "overlap_contract": {
            "classification_counts": classification_counts,
            "dependency_bound_requires_positive_overlap": False,
            "fraction_threshold": None,
            "overlap_capable_requires_positive_overlap": False,
            "positive_time_threshold_ns": None,
        },
        "overlap_gate": "report_only",
        "rank_ranges": _rank_ranges(
            timing_records,
            group_field="group",
            value_field="gpu_time_ms",
            world_size=world_size,
            steps=steps,
        ),
        "records": timing_records,
        "result": "PASS",
    }


def validate_mode_serialization(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    for rank in range(world_size):
        for step in range(steps):
            step_spans: list[tuple[int, int, str, str]] = []
            for mode, direction in _MODE_SERIAL_ORDER:
                scope = f"magi_dsa::pro_pair::{mode}::{direction}"
                matching = [
                    record
                    for record in attribution_records
                    if int(record["rank"]) == rank
                    and int(record["step"]) == step
                    and scope in _scope_names(record)
                ]
                if not matching:
                    raise ValueError(
                        f"Pro pair mode has no attributed kernels: {scope}, "
                        f"rank={rank}, step={step}"
                    )
                start_ns = min(int(record["kernel_start_ns"]) for record in matching)
                end_ns = max(int(record["kernel_end_ns"]) for record in matching)
                step_spans.append((start_ns, end_ns, mode, direction))
                spans.append(
                    {
                        "direction": direction,
                        "end_ns": end_ns,
                        "gpu_span_ms": (end_ns - start_ns) / 1_000_000.0,
                        "mode": mode,
                        "rank": rank,
                        "start_ns": start_ns,
                        "step": step,
                    }
                )
            for left, right in zip(step_spans, step_spans[1:]):
                if left[1] > right[0]:
                    raise ValueError(
                        "Pro pair cross-mode GPU overlap exists: "
                        f"rank={rank}, step={step}, left={left[2:]}, right={right[2:]}"
                    )
    return {
        "order": [f"{mode}.{direction}" for mode, direction in _MODE_SERIAL_ORDER],
        "records": spans,
        "result": "PASS",
    }


def extract_indexer_d2d(
    memcpy_records: list[dict[str, Any]],
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Account every known cuDNN Indexer/selected-KL D2D wrapper."""

    wrapper_rowids: dict[tuple[int, int, str], int] = {}
    for rank in range(world_size):
        for step in range(steps):
            grid_kernels = [
                record
                for record in attribution_records
                if int(record["rank"]) == rank and int(record["step"]) == step
            ]
            for group, scope_name in _CUDNN_D2D_SCOPES.items():
                rowids = {
                    rowid
                    for record in grid_kernels
                    for rowid in _scope_rowids(record, scope_name)
                }
                if len(rowids) != 1:
                    raise ValueError(
                        "Pro cuDNN wrapper NVTX range count differs: "
                        f"group={group}, rank={rank}, step={step}, "
                        f"rowids={sorted(rowids)}"
                    )
                wrapper_rowids[(rank, step, group)] = next(iter(rowids))

    grouped_memcpy: dict[tuple[int, int, str], list[dict[str, Any]]] = {
        key: [] for key in wrapper_rowids
    }
    outside_known_scopes: list[dict[str, Any]] = []
    seen_memcpy_rowids: set[int] = set()
    for record in memcpy_records:
        rank = int(record.get("rank", -1))
        step = int(record.get("step", -1))
        rowid = int(record.get("memcpy_rowid", -1))
        if (
            record.get("record_type") != "magi_dsa_memcpy_attribution"
            or record.get("copy_kind") != "CUDA_MEMCPY_KIND_DTOD"
            or not 0 <= rank < world_size
            or not 0 <= step < steps
            or rowid < 0
            or rowid in seen_memcpy_rowids
            or int(record.get("bytes", -1)) < 0
            or int(record.get("copy_count", 0)) <= 0
            or int(record.get("memcpy_end_ns", 0))
            <= int(record.get("memcpy_start_ns", 0))
        ):
            raise ValueError(f"invalid Pro D2D attribution record: {record}")
        seen_memcpy_rowids.add(rowid)
        matched_groups = [
            group
            for group, scope_name in _CUDNN_D2D_SCOPES.items()
            if wrapper_rowids[(rank, step, group)] in _scope_rowids(record, scope_name)
        ]
        if len(matched_groups) > 1:
            raise ValueError(
                f"Pro D2D activity belongs to multiple known wrappers: rowid={rowid}"
            )
        if matched_groups:
            grouped_memcpy[(rank, step, matched_groups[0])].append(record)
            continue
        scope_names = {name.lower() for name in _scope_names(record)}
        if any(
            "indexer" in name or "selected_kl" in name or "selected-kl" in name
            for name in scope_names
        ):
            outside_known_scopes.append(record)

    records: list[dict[str, Any]] = []
    for key in sorted(grouped_memcpy):
        rank, step, group = key
        scope_name = _CUDNN_D2D_SCOPES[group]
        wrapper_rowid = wrapper_rowids[key]
        copies = grouped_memcpy[key]
        copy_intervals = [
            (int(record["memcpy_start_ns"]), int(record["memcpy_end_ns"]))
            for record in copies
        ]
        wrapper_kernels = [
            record
            for record in attribution_records
            if int(record["rank"]) == rank
            and int(record["step"]) == step
            and wrapper_rowid in _scope_rowids(record, scope_name)
        ]
        wrapper_kernel_intervals = [
            (int(record["kernel_start_ns"]), int(record["kernel_end_ns"]))
            for record in wrapper_kernels
        ]
        external_compute_intervals = [
            (int(record["kernel_start_ns"]), int(record["kernel_end_ns"]))
            for record in attribution_records
            if int(record["rank"]) == rank
            and int(record["step"]) == step
            and "magi_dsa::pro_pair::csa::forward" in _scope_names(record)
            and _is_non_route_compute(record)
            and wrapper_rowid not in _scope_rowids(record, scope_name)
        ]
        duration_ns = _interval_duration_ns(copy_intervals)
        wrapper_overlap_ns = _interval_overlap_ns(
            copy_intervals,
            wrapper_kernel_intervals,
        )
        external_overlap_ns = _interval_overlap_ns(
            copy_intervals,
            external_compute_intervals,
        )
        records.append(
            {
                "bytes": sum(int(record["bytes"]) for record in copies),
                "copy_count": sum(int(record["copy_count"]) for record in copies),
                "gpu_time_ms": duration_ns / 1_000_000.0,
                "group": group,
                "memcpy_activity_count": len(copies),
                "rank": rank,
                "rows": [
                    {
                        "bytes": int(record["bytes"]),
                        "copy_count": int(record["copy_count"]),
                        "memcpy_end_ns": int(record["memcpy_end_ns"]),
                        "memcpy_rowid": int(record["memcpy_rowid"]),
                        "memcpy_start_ns": int(record["memcpy_start_ns"]),
                        "runtime_rowid": int(record["runtime_rowid"]),
                    }
                    for record in copies
                ],
                "same_mode_external_compute_overlap_fraction": (
                    external_overlap_ns / duration_ns if duration_ns else 0.0
                ),
                "same_mode_external_compute_overlap_ms": (
                    external_overlap_ns / 1_000_000.0
                ),
                "same_wrapper_kernel_overlap_fraction": (
                    wrapper_overlap_ns / duration_ns if duration_ns else 0.0
                ),
                "same_wrapper_kernel_overlap_ms": (wrapper_overlap_ns / 1_000_000.0),
                "step": step,
                "wrapper_kernel_launch_count": len(wrapper_kernels),
                "wrapper_nvtx_name": scope_name,
                "wrapper_nvtx_rowid": wrapper_rowid,
            }
        )

    return {
        "accounting": "separate_from_kernel_gpu_time",
        "gpu_time_rank_ranges": _rank_ranges(
            records,
            group_field="group",
            value_field="gpu_time_ms",
            world_size=world_size,
            steps=steps,
        ),
        "records": records,
        "result": "PASS",
        "outside_known_scope": {
            "bytes": sum(int(record["bytes"]) for record in outside_known_scopes),
            "copy_count": sum(
                int(record["copy_count"]) for record in outside_known_scopes
            ),
            "gpu_time_ms": _interval_duration_ns(
                [
                    (
                        int(record["memcpy_start_ns"]),
                        int(record["memcpy_end_ns"]),
                    )
                    for record in outside_known_scopes
                ]
            )
            / 1_000_000.0,
            "memcpy_activity_count": len(outside_known_scopes),
            "rows": outside_known_scopes,
        },
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "total_copy_count": sum(int(record["copy_count"]) for record in records),
        "total_gpu_time_ms": sum(float(record["gpu_time_ms"]) for record in records),
    }


def _validate_workload(workload: dict[str, Any]) -> None:
    expected = {
        "attention_order": ["csa", "hca"],
        "backward_order": ["hca", "csa"],
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "case": "dsv4-pro-128k",
        "cp_size": 8,
        "cu_seqlens": [0, 131072],
        "dtype": "BF16",
        "gradient_accumulation": False,
        "independent_attention_graphs": True,
        "layout_policy": "structural-balanced",
        "local_improvement_passes": None,
        "loss_capture": "none",
        "mode_backward_completion_join": {
            "csa": [
                "sparse_backward_stream",
                "csa_main_stream",
                "csa_indexer_stream",
                "csa_route_stream",
            ],
            "hca": ["hca_main_stream", "hca_route_stream"],
        },
        "mode_serialization": "cuda_event_happens_before",
        "parameter_gradient_allreduce": "one_unified_after_two_backwards",
        "plans": ["balanced"],
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "projection_capture": "pre_capture_once_per_attention",
        "ratios": [4, 128],
        "representative_layer_ids": {"csa": 2, "hca": 3},
        "representative_pair_semantics": (
            "independent_post_projection_graphs_serialized_in_layer_order"
        ),
        "seed": 0,
        "step_mode": "pro-pair",
        "steps": 5,
        "structural_layout_config": {
            "chunk_size": 512,
            "min_chunks_per_rank": 16,
            "uneven_shard": True,
        },
        "token_layout_capture": "pre_capture_once_per_attention",
        "world_size": 8,
    }
    for name, expected_value in expected.items():
        if workload.get(name) != expected_value:
            raise ValueError(
                f"Pro pair workload mismatch: {name}={workload.get(name)!r}"
            )
    if float(workload.get("dout_scale", float("nan"))) != 1.0 / _GLOBAL_OUTPUT_ELEMENTS:
        raise ValueError("Pro pair dout scale does not match the Pro output shape")
    model = workload.get("pro_model_contract")
    expected_model = {
        "source_revision": _PRO_REVISION,
        "main_layer_count": 61,
        "csa_layer_count": 30,
        "hca_layer_count": 31,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "num_query_heads": 128,
        "head_dim": 512,
        "indexer_heads": 64,
        "indexer_head_dim": 128,
        "indexer_topk": 1024,
        "window_size": 128,
    }
    if model != expected_model:
        raise ValueError(f"Pro model contract differs: {model!r}")


def _validate_pro_runtime_bundle_payload(
    payload: dict[str, Any],
    *,
    rank: int,
    label: str,
) -> str:
    expected_handles = {
        "csa": {"layer_id": 2, "ratio": 4},
        "hca": {"layer_id": 3, "ratio": 128},
    }
    query_layout_hash = str(payload.get("shared_bundle_query_layout_hash", ""))
    handles = payload.get("pro_bundle_handles")
    if (
        payload.get("pro_runtime_bundle") is not True
        or int(payload.get("token_layout_invocations", -1)) != 1
        or payload.get("shared_source_x") is not True
        or payload.get("shared_source_packed_meta") is not True
        or not query_layout_hash
        or not isinstance(handles, dict)
        or set(handles) != set(expected_handles)
        or payload.get("parameter_gradient_reducer_precision")
        != "fp32_cp_bucket_model_side_diagnostic"
        or payload.get("parameter_gradient_allreduce_in_7f7b") is not False
        or payload.get("runtime_parameter_gradient_communication") is not False
    ):
        raise ValueError(f"rank {rank} {label} Pro runtime bundle contract differs")
    for mode, expected in expected_handles.items():
        handle = handles[mode]
        if (
            not isinstance(handle, dict)
            or handle.get("policy") != "structural_balanced"
            or int(handle.get("ratio", -1)) != expected["ratio"]
            or int(handle.get("layer_id", -1)) != expected["layer_id"]
            or int(handle.get("declared_local_token_capacity", -1)) != 16384
            or handle.get("query_layout_hash") != query_layout_hash
            or not str(handle.get("plan_hash", ""))
        ):
            raise ValueError(f"rank {rank} {label} {mode} bundle handle differs")
    return query_layout_hash


def _validate_rank_results(
    plan_dir: Path,
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    expected_delta = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": steps,
    }
    expected_backward_completion_join = {
        "csa": [
            "sparse_backward_stream",
            "csa_main_stream",
            "csa_indexer_stream",
            "csa_route_stream",
        ],
        "hca": ["hca_main_stream", "hca_route_stream"],
    }
    query_hashes: set[str] = set()
    query_counts: dict[int, int] = {}
    indexer_packing: list[dict[str, Any]] = []
    rank_costs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for rank in range(world_size):
        metadata = _read_json(plan_dir / f"metadata_rank{rank}.json")
        result = _read_json(plan_dir / f"result_rank{rank}.json")
        metadata_bundle_hash = _validate_pro_runtime_bundle_payload(
            metadata,
            rank=rank,
            label="metadata",
        )
        result_bundle_hash = _validate_pro_runtime_bundle_payload(
            result,
            rank=rank,
            label="result",
        )
        if metadata_bundle_hash != result_bundle_hash:
            raise ValueError(f"rank {rank} Pro bundle Query hashes differ")
        if (
            metadata.get("step_mode") != "pro-pair"
            or metadata.get("attention_order") != ["csa", "hca"]
            or metadata.get("backward_order") != ["hca", "csa"]
            or metadata.get("representative_layer_ids") != {"csa": 2, "hca": 3}
            or metadata.get("representative_pair_semantics")
            != "independent_post_projection_graphs_serialized_in_layer_order"
            or metadata.get("layout_policy") != "structural-balanced"
            or metadata.get("mode_backward_completion_join")
            != expected_backward_completion_join
            or metadata.get("structural_layout_config")
            != {"chunk_size": 512, "min_chunks_per_rank": 16, "uneven_shard": True}
            or int(metadata.get("dout_global_output_elements", 0))
            != _GLOBAL_OUTPUT_ELEMENTS
            or float(metadata.get("dout_scale", float("nan")))
            != 1.0 / _GLOBAL_OUTPUT_ELEMENTS
        ):
            raise ValueError(f"rank {rank} Pro pair metadata differs")
        modes = metadata.get("attention_modes")
        if not isinstance(modes, dict) or set(modes) != {"csa", "hca"}:
            raise ValueError(f"rank {rank} Pro pair mode metadata differs")
        mode_hashes: set[str] = set()
        mode_query_counts: set[int] = set()
        mode_costs: list[dict[str, Any]] = []
        for mode, ratio in (("csa", 4), ("hca", 128)):
            mode_metadata = modes[mode]
            bundle_handle = metadata["pro_bundle_handles"][mode]
            solver = mode_metadata.get("layout_solver")
            if (
                mode_metadata.get("policy") != "structural_balanced"
                or int(mode_metadata.get("layer_id", -1)) != {"csa": 2, "hca": 3}[mode]
                or int(mode_metadata.get("ratio", -1)) != ratio
                or int(mode_metadata.get("declared_local_token_capacity", -1)) != 16384
                or int(mode_metadata.get("final_query_tokens", -1)) != 16384
                or int(mode_metadata.get("source_tokens", -1)) != 16384
                or not isinstance(solver, dict)
                or int(solver.get("chunk_size", 0)) != 512
                or int(solver.get("num_chunks", 0)) <= 0
                or solver.get("uneven_shard") is not True
                or not str(solver.get("solver_scheme", ""))
                or not str(solver.get("cost_model_version", ""))
                or mode_metadata.get("policy") != bundle_handle.get("policy")
                or mode_metadata.get("plan_hash") != bundle_handle.get("plan_hash")
                or mode_metadata.get("query_layout_hash")
                != bundle_handle.get("query_layout_hash")
                or int(mode_metadata.get("declared_local_token_capacity", -1))
                != int(bundle_handle.get("declared_local_token_capacity", -2))
                or int(mode_metadata.get("layer_id", -1))
                != int(bundle_handle.get("layer_id", -2))
                or int(mode_metadata.get("ratio", -1))
                != int(bundle_handle.get("ratio", -2))
            ):
                raise ValueError(f"rank {rank} {mode} structural metadata differs")
            cost = mode_metadata.get("layout_rank_cost")
            if not isinstance(cost, dict) or int(cost.get("rank", -1)) != rank:
                raise ValueError(f"rank {rank} {mode} structural cost is missing")
            mode_costs.append(cost)
            mode_hashes.add(str(mode_metadata["query_layout_hash"]))
            mode_query_counts.add(int(mode_metadata["final_query_tokens"]))
        if (
            len(mode_hashes) != 1
            or len(mode_query_counts) != 1
            or mode_costs[0] != mode_costs[1]
            or next(iter(mode_hashes)) != metadata_bundle_hash
        ):
            raise ValueError(f"rank {rank} CSA/HCA do not share one structural layout")
        packing = modes["csa"].get("indexer_k_packing")
        if not isinstance(packing, dict):
            raise ValueError(f"rank {rank} CSA Indexer packing metadata is missing")
        packed_rows = int(packing.get("packed_indexer_k_rows", -1))
        unique_rows = int(packing.get("unique_indexer_k_rows", -1))
        duplicate_rows = int(packing.get("duplicate_indexer_k_rows", -1))
        row_bytes = int(packing.get("indexer_k_row_bytes", -1))
        if (
            packed_rows != unique_rows + duplicate_rows
            or packed_rows != int(mode_costs[0].get("csa_packed_indexer_k_rows", -2))
            or unique_rows != int(mode_costs[0].get("csa_unique_indexer_k_rows", -2))
            or duplicate_rows
            != int(mode_costs[0].get("csa_duplicate_indexer_k_rows", -2))
            or row_bytes != 256
            or int(packing.get("packed_indexer_k_bytes", -1)) != packed_rows * row_bytes
            or int(packing.get("unique_indexer_k_bytes", -1)) != unique_rows * row_bytes
            or int(packing.get("duplicate_indexer_k_bytes", -1))
            != duplicate_rows * row_bytes
        ):
            raise ValueError(f"rank {rank} CSA Indexer packing accounting differs")
        expected_amplification = packed_rows / unique_rows if unique_rows else 1.0
        if (
            float(packing.get("packing_amplification", float("nan")))
            != expected_amplification
        ):
            raise ValueError(f"rank {rank} CSA Indexer packing ratio differs")
        indexer_packing.append({"rank": rank, **packing})
        query_hashes.add(metadata_bundle_hash)
        query_counts[rank] = next(iter(mode_query_counts))
        rank_costs.append(mode_costs[0])

        if (
            result.get("result") != "PASS"
            or result.get("step_mode") != "pro-pair"
            or result.get("attention_order") != ["csa", "hca"]
            or result.get("backward_order") != ["hca", "csa"]
            or result.get("ratios") != [4, 128]
            or result.get("layout_policy") != "structural-balanced"
            or result.get("mode_backward_completion_join")
            != expected_backward_completion_join
            or result.get("parameter_gradient_allreduce")
            != "one_unified_after_two_backwards"
            or result.get("representative_pair_semantics")
            != "independent_post_projection_graphs_serialized_in_layer_order"
        ):
            raise ValueError(f"rank {rank} Pro pair result did not pass")
        deltas = result.get("attention_counter_delta")
        if not isinstance(deltas, dict) or any(
            deltas.get(mode) != expected_delta for mode in ("csa", "hca")
        ):
            raise ValueError(f"rank {rank} Pro pair counter delta differs")
        mode_metrics = result.get("mode_metrics")
        if not isinstance(mode_metrics, dict) or set(mode_metrics) != {"csa", "hca"}:
            raise ValueError(f"rank {rank} Pro pair result metrics differ")
        for mode in ("csa", "hca"):
            metrics = mode_metrics.get(mode)
            if not isinstance(metrics, dict) or not all(
                metrics.get(name) is True
                for name in (
                    "output_finite",
                    "sparse_lse_finite",
                    "topk_backend_native_valid",
                )
            ):
                raise ValueError(f"rank {rank} Pro pair {mode} validation failed")
            if metrics.get("indexer") is not (mode == "csa"):
                raise ValueError(f"rank {rank} Pro pair {mode} schema differs")
        results.append(result)
    if len(query_hashes) != 1 or sum(query_counts.values()) != 131072:
        raise ValueError("Pro pair structural Query layout does not cover 128K once")
    return {
        "query_layout_hash": next(iter(query_hashes)),
        "query_token_counts": [query_counts[rank] for rank in range(world_size)],
        "indexer_k_packing": indexer_packing,
        "rank_costs": rank_costs,
        "rank_results": len(results),
        "result": "PASS",
    }


def _validate_nvtx_counts(
    counts: dict[str, int],
    world_size: int,
    steps: int,
) -> None:
    invocations = world_size * steps
    expected = {
        "$Magi_DSA/capture_five_pro_pair_steps": world_size,
        **{name: invocations for name in _PHASE_NVTX_NAMES.values()},
        "magi_dsa::CUDNN_CALL::indexer_backward": invocations,
        "magi_dsa::CUDNN_CALL::selected_attention_recompute": invocations,
        "magi_dsa::CUDNN_CALL::selected_indexer_recompute": invocations,
        "magi_dsa::CUDNN_CALL::sparse_attention_backward": invocations * 2,
        "magi_dsa::module::pro_pair::gradient_clear": invocations,
        "magi_dsa::module::pro_pair::stream_overlap::gradient_join": invocations,
        "magi_dsa::module::pro_pair::mode_serial::csa_forward_to_hca_forward": invocations,
        "magi_dsa::module::pro_pair::mode_serial::hca_backward_to_csa_backward": invocations,
    }
    for mode, stream_fields in {
        "csa": (
            "sparse_backward_stream",
            "csa_main_stream",
            "csa_indexer_stream",
            "csa_route_stream",
        ),
        "hca": ("hca_main_stream", "hca_route_stream"),
    }.items():
        parent = (
            "magi_dsa::module::pro_pair::stream_overlap::"
            f"backward_completion_join::{mode}"
        )
        expected[parent] = invocations
        for field in stream_fields:
            expected[f"{parent}::{field}"] = invocations
    for mode, directions in _MODE_ROUTE_ORDER.items():
        expected[
            f"magi_dsa::module::attention::{mode}::sparse_attention::flashmla_forward"
        ] = invocations
        expected[
            f"magi_dsa::module::attention::{mode}::sparse_attention::cudnn_backward"
        ] = invocations
        for direction, routes in directions.items():
            for route in routes:
                expected[
                    "magi_dsa::phase::collective_all2all_v::"
                    f"attention::{mode}::{route}.{direction}"
                ] = invocations
    for name, expected_count in expected.items():
        if int(counts.get(name, 0)) != expected_count:
            raise ValueError(
                f"Pro pair NVTX count differs: {name}={counts.get(name, 0)}, "
                f"expected={expected_count}"
            )


def main() -> None:
    args = _parse_args()
    if args.world_size != 8 or args.steps != 5:
        raise ValueError("the formal Pro pair summary requires 8 ranks and five steps")
    workload = _read_json(args.artifact_dir / "WORKLOAD.json")
    _validate_workload(workload)
    plan_dir = args.artifact_dir / "balanced"
    report = plan_dir / "balanced_5steps_pro_pair.nsys-rep"
    sqlite_path = plan_dir / "balanced_5steps_pro_pair.sqlite"
    if not report.is_file() or report.stat().st_size == 0:
        raise FileNotFoundError(f"missing Pro pair report: {report}")
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing Pro pair SQLite: {sqlite_path}")

    nvtx_counts = read_nvtx_range_counts(sqlite_path)
    excluded = {
        "loss": int(nvtx_counts.get("magi_dsa::loss", 0)),
        "projection": sum(
            count
            for name, count in nvtx_counts.items()
            if "magi_dsa::module::model_projection::" in name
        ),
        "token_layout": sum(
            count for name, count in nvtx_counts.items() if "TOKEN_LAYOUT" in name
        ),
        "w_mode": sum(
            count
            for name, count in nvtx_counts.items()
            if "magi_dsa::pro_pair::w::" in name
        ),
    }
    if any(excluded.values()):
        raise ValueError(f"Pro pair capture contains excluded work: {excluded}")
    _validate_nvtx_counts(nvtx_counts, args.world_size, args.steps)

    audit = _read_json(plan_dir / "NSYS_AUDIT.json")
    if (
        audit.get("result") != "PASS"
        or audit.get("step_mode") != "pro-pair"
        or int(audit.get("logical_phase_records", 0))
        != args.world_size * args.steps * len(_PHASES)
    ):
        raise ValueError("Pro pair Nsight phase audit did not pass")
    attribution_audit = _read_json(plan_dir / "NSYS_ATTRIBUTION.json")
    if (
        attribution_audit.get("result") != "PASS"
        or int(attribution_audit.get("unattributed_kernel_count", -1)) != 0
        or float(attribution_audit.get("attribution_coverage", 0.0)) != 1.0
    ):
        raise ValueError("Pro pair kernel attribution did not pass")
    memcpy_attribution_audit = _read_json(plan_dir / "NSYS_MEMCPY_ATTRIBUTION.json")
    if (
        memcpy_attribution_audit.get("result") != "PASS"
        or int(memcpy_attribution_audit.get("world_size", -1)) != args.world_size
        or int(memcpy_attribution_audit.get("steps", -1)) != args.steps
        or float(memcpy_attribution_audit.get("attribution_coverage", 0.0)) != 1.0
        or float(memcpy_attribution_audit.get("attribution_byte_coverage", 0.0)) != 1.0
        or float(memcpy_attribution_audit.get("attribution_copy_count_coverage", 0.0))
        != 1.0
        or int(
            memcpy_attribution_audit.get(
                "unattributed_memcpy_activity_count",
                -1,
            )
        )
        != 0
    ):
        raise ValueError("Pro pair D2D attribution did not pass")

    records = validate_pro_pair_records(
        _read_jsonl(plan_dir / "nsys_phase_records.jsonl"),
        args.world_size,
        args.steps,
    )
    validate_pro_pair_sendrecv(records)
    attribution_records = _read_jsonl(plan_dir / "nsys_kernel_attribution.jsonl")
    if len(attribution_records) != int(attribution_audit.get("kernel_records", -1)):
        raise ValueError("Pro pair attribution record count differs")
    memcpy_records = _read_jsonl(plan_dir / "nsys_memcpy_attribution.jsonl")
    if len(memcpy_records) != int(
        memcpy_attribution_audit.get("memcpy_activity_records", -1)
    ):
        raise ValueError("Pro pair D2D attribution record count differs")
    phase_ranges = compute_pro_pair_phase_rank_ranges(
        records,
        args.world_size,
        args.steps,
    )
    major_kernels = compute_major_kernel_timings(
        records,
        args.world_size,
        args.steps,
    )
    compressors = compute_compressor_timings(
        attribution_records,
        args.world_size,
        args.steps,
    )
    layout = _validate_rank_results(plan_dir, args.world_size, args.steps)
    support_overheads = compute_support_overhead_timings(
        attribution_records,
        layout["indexer_k_packing"],
        args.world_size,
        args.steps,
    )
    routes = compute_route_timings(
        attribution_records,
        args.world_size,
        args.steps,
    )
    serialization = validate_mode_serialization(
        attribution_records,
        args.world_size,
        args.steps,
    )
    d2d = extract_indexer_d2d(
        memcpy_records,
        attribution_records,
        args.world_size,
        args.steps,
    )

    for rank in range(args.world_size):
        raw = _read_jsonl(plan_dir / f"rank{rank}_nsys_raw.jsonl")
        if len(raw) != args.steps * len(_PHASES):
            raise ValueError(f"rank {rank} Pro pair raw phase grid differs")
        rank_attribution = _read_jsonl(
            plan_dir / f"rank{rank}_nsys_kernel_attribution.jsonl"
        )
        if not rank_attribution or any(
            int(record.get("rank", -1)) != rank for record in rank_attribution
        ):
            raise ValueError(f"rank {rank} Pro pair attribution grid differs")
        rank_memcpy = _read_jsonl(
            plan_dir / f"rank{rank}_nsys_memcpy_attribution.jsonl"
        )
        expected_rank_memcpy = [
            record for record in memcpy_records if int(record["rank"]) == rank
        ]
        if rank_memcpy != expected_rank_memcpy:
            raise ValueError(f"rank {rank} Pro pair D2D attribution grid differs")

    _write_jsonl(args.artifact_dir / "profile_pro_pair.jsonl", records)
    _write_json(args.artifact_dir / "rank_ranges_pro_pair.json", phase_ranges)
    _write_json(args.artifact_dir / "MAJOR_KERNEL_BALANCE_PRO_PAIR.json", major_kernels)
    _write_json(args.artifact_dir / "COMPRESSOR_TIMINGS_PRO_PAIR.json", compressors)
    _write_json(
        args.artifact_dir / "SUPPORT_OVERHEAD_PRO_PAIR.json",
        support_overheads,
    )
    _write_json(args.artifact_dir / "PRO_PAIR_ROUTE_TIMINGS.json", routes)
    _write_json(
        args.artifact_dir / "PRO_PAIR_COMMUNICATION_OVERLAP.json",
        routes,
    )
    _write_json(args.artifact_dir / "MODE_SERIALIZATION_PRO_PAIR.json", serialization)
    _write_json(args.artifact_dir / "INDEXER_D2D_PRO_PAIR.json", d2d)

    summary = {
        "attention_order": ["csa", "hca"],
        "backward_order": ["hca", "csa"],
        "dout_scale": 1.0 / _GLOBAL_OUTPUT_ELEMENTS,
        "expected_sendrecv": _EXPECTED_SENDRECV,
        "excluded_capture_ranges": excluded,
        "flashmla_forward_kernel": major_kernels["flashmla_forward_kernel"],
        "flashmla_forward_same_exact_variant": major_kernels[
            "flashmla_forward_same_exact_variant"
        ],
        "independent_attention_graphs": True,
        "indexer_d2d": {
            "outside_known_scope": d2d["outside_known_scope"],
            "total_bytes": d2d["total_bytes"],
            "total_copy_count": d2d["total_copy_count"],
            "total_gpu_time_ms": d2d["total_gpu_time_ms"],
        },
        "kernel_attribution_coverage": 1.0,
        "kernel_attribution_records": len(attribution_records),
        "memcpy_attribution_coverage": 1.0,
        "memcpy_attribution_records": len(memcpy_records),
        "layout": layout,
        "compressor_attribution_complete": compressors["attribution_complete"],
        "compressor_coverage_gap": compressors["coverage_gap"],
        "compressor_timing_records": len(compressors["records"]),
        "major_kernel_balance_gate": ("indexer_score_topk_0.05_others_report_only"),
        "major_kernel_groups": sorted(_MAJOR_KERNEL_SPECS),
        "mode_backward_completion_join": {
            "csa": [
                "sparse_backward_stream",
                "csa_main_stream",
                "csa_indexer_stream",
                "csa_route_stream",
            ],
            "hca": ["hca_main_stream", "hca_route_stream"],
        },
        "mode_serialization": "cuda_event_happens_before",
        "parameter_gradient_allreduce": "one_unified_after_two_backwards",
        "parameter_gradient_allreduce_in_7f7b": False,
        "parameter_gradient_reducer_precision": (
            "fp32_cp_bucket_model_side_diagnostic"
        ),
        "pro_runtime_bundle": True,
        "ratios": [4, 128],
        "representative_layer_ids": {"csa": 2, "hca": 3},
        "representative_pair_semantics": (
            "independent_post_projection_graphs_serialized_in_layer_order"
        ),
        "result": "PASS",
        "runtime_parameter_gradient_communication": False,
        "shared_bundle_query_layout_hash": layout["query_layout_hash"],
        "shared_source_packed_meta": True,
        "shared_source_x": True,
        "route_timing_records": len(routes["records"]),
        "support_overhead_groups": sorted(
            {str(record["group"]) for record in support_overheads["records"]}
        ),
        "grouped_k_pack_backward_csr_reduce_launches": 0,
        "step_mode": "pro-pair",
        "steps": args.steps,
        "token_layout_invocations": 1,
        "world_size": args.world_size,
    }
    _write_json(args.artifact_dir / "SUMMARY_PRO_PAIR.json", summary)

    major_max = sorted(
        major_kernels["rank_ranges"],
        key=lambda item: float(item["relative_rank_range"]),
        reverse=True,
    )
    report_lines = [
        "# Magi-DSA DeepSeek-V4-Pro representative CSA + HCA profile",
        "",
        "- 结果：PASS",
        "- Workload：8×B300、BF16、单条 128K、5 个 forward/backward steps",
        "- 代表层：main layer 2 (CSA) → layer 3 (HCA)；forward=CSA→HCA，backward=HCA→CSA",
        "- 图语义：两张独立 post-projection autograd graph，通过 CUDA event 串行；不冒充完整 Transformer 两层依赖",
        "- Backward completion：CSA/HCA mode stream 在 mode-done event 前 join 全部 handle 内部 streams；无全局 synchronize",
        "- 主干合同：官方 Pro 61 层由 stack 支持，本 capture 不复制 61 层参数或 activation",
        "- Query layout：structural-balanced，CSA/HCA 共享同一最终 Query layout",
        "- Pro runtime：一个 execution bundle；source hidden 只执行一次 TOKEN_LAYOUT，再分别构造 layer 2/3 post-projection leaves",
        "- 参数梯度：模型侧诊断 reducer 使用 FP32 CP bucket，不属于 DSA runtime，也不计入 7F+7B",
        "- 通信：每 rank/step 固定 CSA 4F+4B、HCA 3F+3B，总计 7F+7B；14 条 route 均单列 GPU 时间",
        "- 大 kernel：逐 rank/step 报告 min/max/range/relative range；仅沿用 Indexer score/top-k 5% 门槛，其余 report-only",
        "- FlashMLA forward：CSA/HCA 每个 rank/step 都精确调用一次 "
        "`sparse_attn_fwd_for_small_topk_kernel`；两个 mode 的 variant 名称一致",
        "- Compressor：forward 按 module NVTX 完整归因；backward 只报可明确识别的 CSA fused kernels",
        "- Compressor coverage gap：linear dgrad/wgrad 与 HCA generic backward 尚无 compressor-specific launch-thread NVTX",
        "- Grouped Indexer K：只计 forward pack；scorer 为 no_grad，backward CSR reduce 硬校验为 0",
        "- Indexer D2D：与 kernel GPU time 分账，完整单列 score/top-k、selected "
        "recompute 与 selected backward 的 copy/字节/union GPU 时长及 overlap",
        "- Indexer/selected-KL known CUDNN_CALL 外 D2D："
        f"activities={d2d['outside_known_scope']['memcpy_activity_count']}，"
        f"bytes={d2d['outside_known_scope']['bytes']}",
        "- Indexer D2D 合计："
        f"copies={d2d['total_copy_count']}，bytes={d2d['total_bytes']}，"
        f"GPU time={float(d2d['total_gpu_time_ms']):.6f} ms",
        "",
        "## 最大的大-kernel relative ranges（report-only）",
        "",
        "| Step | Kernel group | Min (ms) | Max (ms) | Relative range |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in major_max[: min(20, len(major_max))]:
        report_lines.append(
            f"| {item['step']} | `{item['group']}` | {float(item['min']):.6f} | "
            f"{float(item['max']):.6f} | {float(item['relative_rank_range']):.6f} |"
        )
    (args.artifact_dir / "REPORT_PRO_PAIR.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
