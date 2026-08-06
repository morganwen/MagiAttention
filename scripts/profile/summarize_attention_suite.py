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
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

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
}
_PHASES = tuple(_PHASE_NVTX_NAMES)
_EXPECTED_SENDRECV = {
    "backward": 8,
    "csa_backward": 4,
    "csa_forward": 4,
    "forward": 8,
    "hca_backward": 3,
    "hca_forward": 3,
    "w_backward": 1,
    "w_forward": 1,
}
_MODE_PHASES = {
    "w": ("w_forward", "w_backward"),
    "csa": ("csa_forward", "csa_backward"),
    "hca": ("hca_forward", "hca_backward"),
}
_MODE_ROUTES = {
    "w": ("WINDOW_KV",),
    "csa": ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV"),
    "hca": ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
}
_MODE_ROUTE_ORDER: dict[str, dict[str, tuple[str, ...]]] = {
    "w": {
        "forward": ("WINDOW_KV",),
        "backward": ("WINDOW_KV",),
    },
    "csa": {
        "forward": ("WINDOW_KV", "OVERLAP_X", "COMPRESSED_KI", "COMPRESSED_KV"),
        "backward": ("COMPRESSED_KI", "COMPRESSED_KV", "OVERLAP_X", "WINDOW_KV"),
    },
    "hca": {
        "forward": ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
        "backward": ("COMPRESSED_KV", "WINDOW_KV", "OVERLAP_X"),
    },
}
_MODE_SERIAL_ORDER = (
    ("w", "forward"),
    ("csa", "forward"),
    ("hca", "forward"),
    ("hca", "backward"),
    ("csa", "backward"),
    ("w", "backward"),
)
_MODE_LOCAL_OVERLAP_EXPLANATIONS = {
    ("w", "forward", "WINDOW_KV"): {
        "candidate": False,
        "reason_code": "no_independent_w_forward_compute",
        "reason": (
            "W 没有 Compressor/Indexer；sparse attention 必须等待 WINDOW_KV "
            "接收完成，因此没有可并行的 W 自身计算。"
        ),
    },
    ("w", "backward", "WINDOW_KV"): {
        "candidate": False,
        "reason_code": "terminal_w_reverse_route",
        "reason": (
            "WINDOW_KV reverse 依赖 W sparse-attention backward 产生 dKV，"
            "且它是 W backward 的末端 route，之后没有 W 自身计算。"
        ),
    },
    ("csa", "forward", "WINDOW_KV"): {
        "candidate": True,
        "reason_code": "independent_csa_indexer_and_compressor_compute",
        "reason": "WINDOW_KV 可与 CSA 自身的 Indexer projection/Compressor 分支并行。",
    },
    ("csa", "forward", "OVERLAP_X"): {
        "candidate": True,
        "reason_code": "independent_csa_indexer_query_projection",
        "reason": "OVERLAP_X 可与不依赖 support-x 的 CSA Indexer Query projection 并行。",
    },
    ("csa", "forward", "COMPRESSED_KI"): {
        "candidate": True,
        "reason_code": "csa_main_compressor_cover",
        "reason": "COMPRESSED_KI 在 Main Compressor 前异步发起，由 CSA Main Compressor 遮挡。",
    },
    ("csa", "forward", "COMPRESSED_KV"): {
        "candidate": True,
        "reason_code": "csa_grouped_indexer_cover",
        "reason": "COMPRESSED_KV 可与 CSA grouped Indexer score/Top-K 并行。",
    },
    ("csa", "backward", "COMPRESSED_KI"): {
        "candidate": True,
        "reason_code": "csa_sparse_backward_cover",
        "reason": (
            "COMPRESSED_KI reverse 在 CSA sparse-attention backward 前异步发起，"
            "由该 backward 计算遮挡。"
        ),
    },
    ("csa", "backward", "COMPRESSED_KV"): {
        "candidate": True,
        "reason_code": "independent_csa_backward_branch",
        "reason": "COMPRESSED_KV reverse 可与 CSA 的独立 Indexer/backward 分支并行。",
    },
    ("csa", "backward", "OVERLAP_X"): {
        "candidate": True,
        "reason_code": "csa_late_join_backward_compute",
        "reason": "OVERLAP_X reverse 采用 late join，可与 CSA 尚未完成的 backward 分支并行。",
    },
    ("csa", "backward", "WINDOW_KV"): {
        "candidate": True,
        "reason_code": "csa_late_join_backward_compute",
        "reason": "WINDOW_KV reverse 采用 late join，可与 CSA 尚未完成的 backward 分支并行。",
    },
    ("hca", "forward", "OVERLAP_X"): {
        "candidate": False,
        "reason_code": "hca_main_compressor_depends_on_overlap_x",
        "reason": (
            "HCA Main Compressor 需要 OVERLAP_X 返回的 support-x；HCA 又没有 "
            "Indexer 分支，因此 OX 完成前没有独立 HCA 计算。"
        ),
    },
    ("hca", "forward", "WINDOW_KV"): {
        "candidate": True,
        "reason_code": "hca_main_compressor_cover",
        "reason": "WINDOW_KV 与 HCA Main Compressor 独立，可由该 Compressor 计算遮挡。",
    },
    ("hca", "forward", "COMPRESSED_KV"): {
        "candidate": False,
        "reason_code": "hca_sparse_attention_depends_on_compressed_kv",
        "reason": (
            "COMPRESSED_KV 由 HCA Main Compressor 产生，随后 sparse attention 必须等待它；"
            "HCA 没有 grouped Indexer 可并行。"
        ),
    },
    ("hca", "backward", "COMPRESSED_KV"): {
        "candidate": False,
        "reason_code": "hca_main_compressor_backward_depends_on_reverse_ckv",
        "reason": (
            "HCA Main Compressor backward 需要 reverse COMPRESSED_KV 返回的梯度，"
            "因此不能用该 backward 计算遮挡这段通信。"
        ),
    },
    ("hca", "backward", "WINDOW_KV"): {
        "candidate": True,
        "reason_code": "hca_main_compressor_backward_cover",
        "reason": (
            "WINDOW_KV reverse 与 HCA Main Compressor backward 分属独立梯度分支，" "可彼此并行。"
        ),
    },
    ("hca", "backward", "OVERLAP_X"): {
        "candidate": False,
        "reason_code": "terminal_hca_support_reverse_route",
        "reason": (
            "OVERLAP_X reverse 必须等待 HCA Compressor backward 产生 support 梯度，"
            "且它是 HCA backward 的末端 route。"
        ),
    },
}
_FLASHMLA_FORWARD_KERNEL = "sparse_attn_fwd_for_small_topk_kernel"
_ATTENTION_BACKWARD_CORE_PATTERNS = {
    "main": ("kernel_cutlass_bwd_", "sparse_attention_backward"),
    "convert": ("kernel_cutlass_convert_", "sparse_attention_backward"),
    "sum_odo": ("kernel_cutlass_sum_odo_", "sparse_attention_backward"),
    "sum_dsink": ("kernel_cutlass_sum_dsink_", "sparse_attention_backward"),
}
_CSA_FORWARD_SCOPE = "magi_dsa::attention_suite::csa::forward"
_CSA_BACKWARD_SCOPE = "magi_dsa::attention_suite::csa::backward"
_CSA_COMPRESSED_KI_FORWARD_SCOPE = (
    "magi_dsa::phase::collective_all2all_v::" "attention::csa::COMPRESSED_KI.forward"
)
_CSA_COMPRESSED_KI_BACKWARD_SCOPE = (
    "magi_dsa::phase::collective_all2all_v::" "attention::csa::COMPRESSED_KI.backward"
)
_CSA_MAIN_COMPRESSOR_SCOPE = "magi_dsa::module::compressor::main"
_CSA_SPARSE_BACKWARD_SCOPE = (
    "magi_dsa::module::attention::csa::sparse_attention::cudnn_backward"
)
_STEP_PATTERN = re.compile(r"^balanced/rank_(?P<rank>\d+)/training_step_(?P<step>\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Magi-DSA W+CSA+HCA merged-step profile"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=8)
    return parser.parse_args()


def validate_attention_suite_records(
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
            raise ValueError(f"duplicate attention-suite profile record: {key}")
        by_key[key] = record
        step, rank, phase = key
        if record.get("plan") != "balanced" or phase not in _PHASES:
            raise ValueError(f"invalid attention-suite profile record: {key}")
        if record.get("nvtx_name") != _PHASE_NVTX_NAMES[phase]:
            raise ValueError(f"attention-suite NVTX name mismatch: {key}")
        if int(record.get("logical_call_count", 0)) != 1:
            raise ValueError(f"attention-suite logical call count is not one: {key}")
        gpu_time_ms = float(record.get("gpu_time_ms", float("nan")))
        if not math.isfinite(gpu_time_ms) or gpu_time_ms <= 0:
            raise ValueError(
                f"attention-suite GPU time is not finite and positive: {key}"
            )
        if int(record.get("kernel_launch_count", 0)) <= 0:
            raise ValueError(f"attention-suite phase has no kernels: {key}")
        if not (0 <= step < steps and 0 <= rank < world_size):
            raise ValueError(f"attention-suite grid index is invalid: {key}")
    if set(by_key) != expected:
        raise ValueError(
            "attention-suite profile grid mismatch: "
            f"missing={sorted(expected - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected)}"
        )
    return [by_key[key] for key in sorted(expected)]


def compute_attention_suite_rank_ranges(
    records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> list[dict[str, Any]]:
    validated = validate_attention_suite_records(records, world_size, steps)
    ranges: list[dict[str, Any]] = []
    for step in range(steps):
        for phase in _PHASES:
            values = [
                float(record["gpu_time_ms"])
                for record in validated
                if int(record["step"]) == step and record["phase"] == phase
            ]
            mean_ms = statistics.fmean(values)
            minimum = min(values)
            maximum = max(values)
            ranges.append(
                {
                    "max_rank_time_ms": maximum,
                    "mean_rank_time_ms": mean_ms,
                    "min_rank_time_ms": minimum,
                    "phase": phase,
                    "plan": "balanced",
                    "rank_range_ms": maximum - minimum,
                    "relative_rank_range": (maximum - minimum) / mean_ms,
                    "step": step,
                    "threshold": None,
                }
            )
    return ranges


def validate_attention_suite_sendrecv(
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        phase = str(record["phase"])
        if phase not in _EXPECTED_SENDRECV:
            continue
        count = int(
            record.get("kernel_name_counts", {}).get(
                "ncclDevKernel_SendRecv",
                0,
            )
        )
        expected_count = _EXPECTED_SENDRECV[phase]
        if count != expected_count:
            raise ValueError(
                f"attention-suite {phase} SendRecv count differs: "
                f"rank={record['rank']}, step={record['step']}, "
                f"count={count}, expected={expected_count}"
            )


def _matching_kernel_names(
    counts: dict[str, Any],
    patterns: tuple[str, ...],
) -> dict[str, int]:
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    return {
        str(name): int(count)
        for name, count in counts.items()
        if all(pattern in str(name).lower() for pattern in lowered_patterns)
    }


def validate_attention_suite_core_kernels(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate shared sparse-attention kernels and ratio-specific extras."""

    records_by_phase: dict[str, list[dict[str, Any]]] = {
        phase: [] for phases in _MODE_PHASES.values() for phase in phases
    }
    for record in records:
        phase = str(record.get("phase"))
        if phase in records_by_phase:
            records_by_phase[phase].append(record)
    if any(not phase_records for phase_records in records_by_phase.values()):
        missing = [
            phase
            for phase, phase_records in records_by_phase.items()
            if not phase_records
        ]
        raise ValueError(f"attention-suite core-kernel phases are missing: {missing}")

    core_names: dict[str, set[str]] = {
        "flashmla_forward": set(),
        **{name: set() for name in _ATTENTION_BACKWARD_CORE_PATTERNS},
    }
    phase_name_sets: dict[str, set[str]] = {}
    mode_summary: dict[str, Any] = {}
    for mode, (forward_phase, backward_phase) in _MODE_PHASES.items():
        forward_records = records_by_phase[forward_phase]
        backward_records = records_by_phase[backward_phase]
        forward_names: set[str] = set()
        backward_names: set[str] = set()
        for record in forward_records:
            counts = {
                str(name): int(count)
                for name, count in record.get("kernel_name_counts", {}).items()
            }
            forward_names.update(counts)
            if counts.get(_FLASHMLA_FORWARD_KERNEL, 0) != 1:
                raise ValueError(
                    "attention-suite FlashMLA forward core count differs: "
                    f"mode={mode}, rank={record['rank']}, step={record['step']}"
                )
            core_names["flashmla_forward"].add(_FLASHMLA_FORWARD_KERNEL)
            dense_indexer_names = [
                name for name in counts if "dense_indexer_backward" in name.lower()
            ]
            if dense_indexer_names:
                raise ValueError(
                    "attention-suite still contains dense Indexer backward kernels: "
                    f"mode={mode}, names={dense_indexer_names}"
                )
            sparse_indexer_names = [
                name
                for name in counts
                if "indexer_backward" in name.lower()
                and "dense_indexer_backward" not in name.lower()
            ]
            if mode == "csa" and not sparse_indexer_names:
                raise ValueError(
                    "CSA forward is missing selected-only sparse Indexer backward kernels"
                )
            if mode != "csa" and sparse_indexer_names:
                raise ValueError(
                    f"{mode} forward unexpectedly contains Indexer backward kernels"
                )

        for record in backward_records:
            counts = {
                str(name): int(count)
                for name, count in record.get("kernel_name_counts", {}).items()
            }
            backward_names.update(counts)
            for core_name, patterns in _ATTENTION_BACKWARD_CORE_PATTERNS.items():
                matches = _matching_kernel_names(counts, patterns)
                if sum(matches.values()) != 1:
                    raise ValueError(
                        "attention-suite cuDNN attention backward core count differs: "
                        f"mode={mode}, core={core_name}, rank={record['rank']}, "
                        f"step={record['step']}, matches={matches}"
                    )
                core_names[core_name].update(matches)

        phase_name_sets[forward_phase] = forward_names
        phase_name_sets[backward_phase] = backward_names
        mode_summary[mode] = {
            "backward_kernel_launch_count": {
                "max": max(
                    int(record["kernel_launch_count"]) for record in backward_records
                ),
                "min": min(
                    int(record["kernel_launch_count"]) for record in backward_records
                ),
            },
            "backward_unique_kernel_names": len(backward_names),
            "forward_kernel_launch_count": {
                "max": max(
                    int(record["kernel_launch_count"]) for record in forward_records
                ),
                "min": min(
                    int(record["kernel_launch_count"]) for record in forward_records
                ),
            },
            "forward_unique_kernel_names": len(forward_names),
        }

    for core_name, names in core_names.items():
        if len(names) != 1:
            raise ValueError(
                "attention-suite sparse-attention core kernel names differ across modes: "
                f"core={core_name}, names={sorted(names)}"
            )

    pairwise: dict[str, Any] = {}
    for direction in ("forward", "backward"):
        for left, right in (("w", "csa"), ("w", "hca"), ("csa", "hca")):
            left_names = phase_name_sets[f"{left}_{direction}"]
            right_names = phase_name_sets[f"{right}_{direction}"]
            intersection = left_names & right_names
            pairwise[f"{left}_{right}_{direction}"] = {
                "intersection": len(intersection),
                "left_only": len(left_names - intersection),
                "right_only": len(right_names - intersection),
            }

    return {
        "core_kernel_names": {
            name: next(iter(names)) for name, names in sorted(core_names.items())
        },
        "mode_summary": mode_summary,
        "pairwise_exact_kernel_name_sets": pairwise,
    }


def _validate_attention_suite_resource_signatures(
    resource_signatures: dict[str, dict[str, set[tuple[Any, ...]]]],
) -> dict[str, Any]:
    expected_modes = set(_MODE_PHASES)
    for core_kind, modes in resource_signatures.items():
        if set(modes) != expected_modes:
            raise ValueError(
                "attention-suite core resource signatures are missing modes: "
                f"core={core_kind}, modes={sorted(modes)}"
            )
        for mode, signatures in modes.items():
            if len(signatures) != 1:
                raise ValueError(
                    "attention-suite core has multiple resource signatures in one mode: "
                    f"core={core_kind}, mode={mode}, signatures={sorted(signatures)}"
                )

    expected_cores = {"flashmla_forward", *_ATTENTION_BACKWARD_CORE_PATTERNS}
    if set(resource_signatures) != expected_cores:
        raise ValueError(
            "attention-suite core resource signature set differs: "
            f"actual={sorted(resource_signatures)}, expected={sorted(expected_cores)}"
        )

    backward_exact: dict[str, bool] = {}
    for core_kind in _ATTENTION_BACKWARD_CORE_PATTERNS:
        modes = resource_signatures[core_kind]
        union = set().union(*modes.values())
        if len(union) != 1:
            raise ValueError(
                "attention-suite cuDNN backward core resources differ across modes: "
                f"core={core_kind}, signatures={sorted(union)}"
            )
        backward_exact[core_kind] = True

    flashmla = {
        mode: next(iter(signatures))
        for mode, signatures in resource_signatures["flashmla_forward"].items()
    }
    if flashmla["w"] != flashmla["hca"]:
        raise ValueError(
            "attention-suite W/HCA three-output FlashMLA resources differ: "
            f"w={flashmla['w']}, hca={flashmla['hca']}"
        )

    # Signature fields are:
    # name, registers/thread, block xyz, static shared memory, dynamic shared memory.
    def structural_signature(signature: tuple[Any, ...]) -> tuple[Any, ...]:
        return (signature[0], *signature[2:])

    structural = {
        mode: structural_signature(signature) for mode, signature in flashmla.items()
    }
    if len(set(structural.values())) != 1:
        raise ValueError(
            "attention-suite FlashMLA name/block/shared-memory resources differ: "
            f"signatures={structural}"
        )

    register_counts = {mode: int(signature[1]) for mode, signature in flashmla.items()}
    exact_flashmla = len(set(flashmla.values())) == 1
    return {
        "csa_dual_lse_specialization": {
            "allowed_differing_fields": ["registers_per_thread"],
            "enabled": True,
            "registers_per_thread": register_counts,
        },
        "cudnn_backward_exact_resource_signature_across_modes": all(
            backward_exact.values()
        ),
        "cudnn_backward_exact_resource_signature_by_core": backward_exact,
        "exact_resource_signature_across_modes": exact_flashmla,
        "flashmla_forward_exact_resource_signature_across_modes": exact_flashmla,
        "flashmla_forward_name_block_shared_memory_across_modes": True,
        "flashmla_forward_w_hca_exact_resource_signature": True,
    }


def _attention_suite_kernel_launch_audit(
    sqlite_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    row_modes: dict[int, tuple[str, str]] = {}
    for record in records:
        phase = str(record["phase"])
        if phase not in {
            phase_name for phases in _MODE_PHASES.values() for phase_name in phases
        }:
            continue
        mode, direction = phase.split("_", 1)
        for kernel in record.get("kernels", []):
            name = str(kernel["name"])
            lower_name = name.lower()
            core_kind: str | None = None
            if direction == "forward" and name == _FLASHMLA_FORWARD_KERNEL:
                core_kind = "flashmla_forward"
            elif direction == "backward":
                for candidate, patterns in _ATTENTION_BACKWARD_CORE_PATTERNS.items():
                    if all(pattern.lower() in lower_name for pattern in patterns):
                        core_kind = candidate
                        break
            if core_kind is None:
                continue
            rowid = int(kernel["rowid"])
            previous = row_modes.get(rowid)
            current = (mode, core_kind)
            if previous is not None and previous != current:
                raise ValueError(
                    f"attention core kernel row {rowid} has conflicting modes"
                )
            row_modes[rowid] = current
    if not row_modes:
        raise ValueError("attention-suite core kernel launch rows are missing")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows: list[tuple[Any, ...]] = []
        for rowid, (mode, core_kind) in row_modes.items():
            row = connection.execute(
                """
                SELECT COALESCE(strings.value, CAST(kernels.shortName AS TEXT)),
                       kernels.registersPerThread,
                       kernels.gridX,
                       kernels.gridY,
                       kernels.gridZ,
                       kernels.blockX,
                       kernels.blockY,
                       kernels.blockZ,
                       kernels.staticSharedMemory,
                       kernels.dynamicSharedMemory
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                LEFT JOIN StringIds AS strings ON strings.id = kernels.shortName
                WHERE kernels.rowid = ?
                """,
                (rowid,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"attention-suite core kernel row is absent from SQLite: {rowid}"
                )
            rows.append((mode, core_kind, *row))
    finally:
        connection.close()

    resource_signatures: dict[str, dict[str, set[tuple[Any, ...]]]] = {}
    grid_signatures: dict[str, dict[str, set[tuple[int, int, int]]]] = {}
    launch_counts: Counter[tuple[str, str]] = Counter()
    for (
        mode,
        resolved_core_kind,
        name,
        registers,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        static_shared,
        dynamic_shared,
    ) in rows:
        resource_signatures.setdefault(resolved_core_kind, {}).setdefault(
            mode, set()
        ).add(
            (
                str(name),
                int(registers),
                int(block_x),
                int(block_y),
                int(block_z),
                int(static_shared),
                int(dynamic_shared),
            )
        )
        grid_signatures.setdefault(resolved_core_kind, {}).setdefault(mode, set()).add(
            (int(grid_x), int(grid_y), int(grid_z))
        )
        launch_counts[(mode, resolved_core_kind)] += 1

    resource_contract = _validate_attention_suite_resource_signatures(
        resource_signatures
    )

    return {
        "launch_counts": {
            f"{mode}::{core_kind}": count
            for (mode, core_kind), count in sorted(launch_counts.items())
        },
        "resource_signatures": {
            core_kind: {
                mode: [list(signature) for signature in sorted(signatures)]
                for mode, signatures in sorted(modes.items())
            }
            for core_kind, modes in sorted(resource_signatures.items())
        },
        "grid_signatures": {
            core_kind: {
                mode: [list(signature) for signature in sorted(signatures)]
                for mode, signatures in sorted(modes.items())
            }
            for core_kind, modes in sorted(grid_signatures.items())
        },
        **resource_contract,
    }


def _validate_workload(workload: dict[str, Any]) -> None:
    expected = {
        "attention_order": ["w", "csa", "hca"],
        "backward_order": ["hca", "csa", "w"],
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "cp_size": 8,
        "cu_seqlens": [0, 131072],
        "dout_scale": 1.0 / 4_294_967_296,
        "dtype": "BF16",
        "gradient_accumulation": False,
        "independent_attention_graphs": True,
        "loss_capture": "none",
        "mode_backward_completion_join": {
            "w": [],
            "csa": [
                "sparse_backward_stream",
                "csa_main_stream",
                "csa_indexer_stream",
                "csa_route_stream",
            ],
            "hca": ["hca_main_stream", "hca_route_stream"],
        },
        "mode_serialization": "cuda_event_happens_before",
        "overlap_accounting": "same_mode_same_direction_non_route_compute",
        "plans": ["balanced"],
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "projection_capture": "pre_capture_once_per_attention",
        "rank_size": 8,
        "ratios": [0, 4, 128],
        "seed": 0,
        "step_mode": "attention-suite",
        "steps": 5,
        "token_layout_capture": "pre_capture_once_per_attention",
        "world_size": 8,
    }
    for name, expected_value in expected.items():
        if workload.get(name) != expected_value:
            raise ValueError(
                f"attention-suite workload mismatch: {name}={workload.get(name)!r}"
            )
    if workload.get("layout_policy", "legacy") not in ("legacy", "shared-greedy"):
        raise ValueError("attention-suite layout policy is invalid")
    local_improvement_passes = int(workload.get("local_improvement_passes", 4))
    if local_improvement_passes not in (0, 1, 4, 8):
        raise ValueError("attention-suite local improvement passes are invalid")
    profiler_attach_warmup_steps = int(workload.get("profiler_attach_warmup_steps", 0))
    if not 0 <= profiler_attach_warmup_steps <= 8:
        raise ValueError("attention-suite profiler-attach warmup count is invalid")
    gpu_clock_lock_mhz = workload.get("gpu_clock_lock_mhz")
    if gpu_clock_lock_mhz is not None and int(gpu_clock_lock_mhz) <= 0:
        raise ValueError("attention-suite GPU clock lock is invalid")
    if workload.get("layout_policy", "legacy") != "shared-greedy" and (
        local_improvement_passes != 4
        or profiler_attach_warmup_steps != 0
        or gpu_clock_lock_mhz is not None
    ):
        raise ValueError("dispatch-ablation controls require shared-greedy")


def _validate_clock_control(
    artifact_dir: Path,
    workload: dict[str, Any],
) -> dict[str, Any]:
    requested = workload.get("gpu_clock_lock_mhz")
    control_path = artifact_dir / "CLOCK_CONTROL.txt"
    if requested is None:
        if not control_path.is_file():
            return {"mode": "dynamic", "requested_graphics_clock_mhz": None}
        values = dict(
            line.split("=", 1)
            for line in control_path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        if values.get("mode") != "dynamic":
            raise ValueError("dynamic profile has inconsistent clock metadata")
        return {"mode": "dynamic", "requested_graphics_clock_mhz": None}

    requested_mhz = int(requested)
    if not control_path.is_file():
        raise FileNotFoundError("fixed-clock profile is missing CLOCK_CONTROL.txt")
    values = dict(
        line.split("=", 1)
        for line in control_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    expected = {
        "gpu_count": "8",
        "mode": "nvml_lock_gpu_clocks",
        "requested_graphics_clock_mhz": str(requested_mhz),
        "reset_required": "true",
        "reset_verified": "true",
        "verified": "true",
    }
    if any(values.get(name) != value for name, value in expected.items()):
        raise ValueError(f"fixed-clock control metadata differs: {values}")

    before_path = artifact_dir / "CLOCKS_BEFORE_LOCK.csv"
    locked_path = artifact_dir / "CLOCKS_LOCKED.csv"
    reset_path = artifact_dir / "CLOCKS_AFTER_RESET.csv"
    if (
        not before_path.is_file()
        or not locked_path.is_file()
        or not reset_path.is_file()
    ):
        raise FileNotFoundError("fixed-clock profile is missing clock audit records")

    def read_clock_rows(path: Path) -> dict[int, tuple[str, int, int]]:
        result: dict[int, tuple[str, int, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"fixed-clock record has invalid fields: {path}")
            index = int(fields[0])
            if index in result:
                raise ValueError(f"fixed-clock record repeats GPU {index}: {path}")
            result[index] = (fields[1], int(fields[2]), int(fields[3]))
        if set(result) != set(range(8)):
            raise ValueError(f"fixed-clock GPU set differs: {path}")
        return result

    before_rows = read_clock_rows(before_path)
    locked_rows = read_clock_rows(locked_path)
    reset_rows = read_clock_rows(reset_path)
    for index in range(8):
        before_uuid, _, before_max = before_rows[index]
        locked_uuid, locked_current, locked_max = locked_rows[index]
        reset_uuid, reset_current, reset_max = reset_rows[index]
        if (
            locked_uuid != before_uuid
            or reset_uuid != before_uuid
            or locked_max != before_max
            or reset_max != before_max
            or locked_current != requested_mhz
            or reset_current == requested_mhz
        ):
            raise ValueError(f"fixed-clock verification records differ for GPU {index}")
    return {
        "mode": "nvml_lock_gpu_clocks",
        "requested_graphics_clock_mhz": requested_mhz,
        "verified_gpu_count": len(locked_rows),
        "reset_verified": True,
    }


def _validate_shared_layout_metadata(
    plan_dir: Path,
    world_size: int,
    workload: dict[str, Any],
) -> dict[str, Any]:
    layout_policy = str(workload.get("layout_policy", "legacy"))
    if layout_policy == "legacy":
        return {"layout_policy": "legacy"}

    expected_config = {
        "ki_memory_budget_bytes": 1 << 30,
        "ki_workspace_reserve_bytes": 256 << 20,
        "local_improvement_passes": int(workload.get("local_improvement_passes", 4)),
    }
    modes = ("w", "csa", "hca")
    layout_hashes: set[str] = set()
    layout_keys: set[tuple[int, int, int, int]] = set()
    layout_solvers: set[str] = set()
    query_token_counts: list[int] = []
    rank_costs: list[dict[str, Any]] = []
    token_layout_timings: dict[str, list[float]] = {mode: [] for mode in modes}
    prepare_timings: dict[str, list[float]] = {mode: [] for mode in modes}

    for rank in range(world_size):
        metadata = _read_json(plan_dir / f"metadata_rank{rank}.json")
        if metadata.get("layout_policy") != "shared-greedy":
            raise ValueError(f"rank {rank} shared layout policy is missing")
        if metadata.get("shared_layout_config") != expected_config:
            raise ValueError(f"rank {rank} shared layout config differs")
        attention_modes = metadata.get("attention_modes")
        if not isinstance(attention_modes, dict):
            raise ValueError(f"rank {rank} attention mode metadata is missing")

        mode_records: list[dict[str, Any]] = []
        for mode in modes:
            mode_record = attention_modes.get(mode)
            if not isinstance(mode_record, dict):
                raise ValueError(f"rank {rank} {mode} layout metadata is missing")
            mode_records.append(mode_record)
            if mode_record.get("policy") != "shared_greedy":
                raise ValueError(f"rank {rank} {mode} did not use shared_greedy")
            timing_ms = float(mode_record.get("token_layout_forward_ms", float("nan")))
            if not math.isfinite(timing_ms) or timing_ms < 0.0:
                raise ValueError(f"rank {rank} {mode} TOKEN_LAYOUT timing is invalid")
            token_layout_timings[mode].append(timing_ms)
            prepare_seconds = float(mode_record.get("prepare_seconds", float("nan")))
            if not math.isfinite(prepare_seconds) or prepare_seconds <= 0.0:
                raise ValueError(f"rank {rank} {mode} prepare timing is invalid")
            prepare_timings[mode].append(prepare_seconds)

        rank_hashes = {
            str(record.get("query_layout_hash", "")) for record in mode_records
        }
        if len(rank_hashes) != 1 or not next(iter(rank_hashes)):
            raise ValueError(f"rank {rank} W/CSA/HCA Query layout hash differs")
        layout_hashes.update(rank_hashes)

        try:
            rank_keys = {
                tuple(int(value) for value in record["layout_key"])
                for record in mode_records
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"rank {rank} shared layout key is invalid") from error
        if len(rank_keys) != 1 or len(next(iter(rank_keys))) != 4:
            raise ValueError(f"rank {rank} W/CSA/HCA layout key differs")
        layout_keys.add(cast(tuple[int, int, int, int], next(iter(rank_keys))))

        solver_records = [record.get("layout_solver") for record in mode_records]
        if not all(isinstance(record, dict) for record in solver_records):
            raise ValueError(f"rank {rank} shared solver metadata is missing")
        serialized_solvers = {
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in solver_records
        }
        if len(serialized_solvers) != 1:
            raise ValueError(f"rank {rank} W/CSA/HCA solver metadata differs")
        layout_solvers.update(serialized_solvers)

        mode_rank_costs = [record.get("layout_rank_cost") for record in mode_records]
        if not all(isinstance(record, dict) for record in mode_rank_costs):
            raise ValueError(f"rank {rank} shared rank-cost metadata is missing")
        if any(record != mode_rank_costs[0] for record in mode_rank_costs[1:]):
            raise ValueError(f"rank {rank} W/CSA/HCA rank costs differ")
        rank_cost = dict(cast(dict[str, Any], mode_rank_costs[0]))
        if int(rank_cost.get("rank", -1)) != rank:
            raise ValueError(f"rank {rank} shared rank-cost owner differs")
        modeled_bytes = int(rank_cost.get("modeled_ki_bytes", -1))
        if (
            modeled_bytes < 0
            or modeled_bytes > expected_config["ki_memory_budget_bytes"]
        ):
            raise ValueError(f"rank {rank} shared KI modeled bytes exceed budget")
        rank_costs.append(rank_cost)

        mode_query_counts = {
            int(record.get("final_query_tokens", -1)) for record in mode_records
        }
        if len(mode_query_counts) != 1 or next(iter(mode_query_counts)) < 0:
            raise ValueError(f"rank {rank} W/CSA/HCA Query counts differ")
        query_token_counts.append(next(iter(mode_query_counts)))

    if len(layout_hashes) != 1 or len(layout_keys) != 1 or len(layout_solvers) != 1:
        raise ValueError("shared layout identity differs across ranks")
    if sum(query_token_counts) != 131072:
        raise ValueError("shared Query counts do not cover the 128K pack")

    layout_key = next(iter(layout_keys))
    recomputed_key = (
        max(int(record["indexer_cost"]) for record in rank_costs),
        max(int(record["hca_cost"]) for record in rank_costs),
        sum(int(record["token_layout_remote_rows"]) for record in rank_costs),
        sum(int(record["fragment_count"]) for record in rank_costs),
    )
    if layout_key != recomputed_key:
        raise ValueError("shared layout key does not match its rank-cost ledger")

    timing_summary = {
        mode: {
            "max": max(values),
            "mean": statistics.fmean(values),
            "min": min(values),
        }
        for mode, values in token_layout_timings.items()
    }
    prepare_summary = {
        mode: {
            "max": max(values),
            "mean": statistics.fmean(values),
            "min": min(values),
        }
        for mode, values in prepare_timings.items()
    }
    return {
        "layout_key": list(layout_key),
        "layout_policy": layout_policy,
        "layout_solver": json.loads(next(iter(layout_solvers))),
        "modeled_ki_bytes": {
            "max": max(int(record["modeled_ki_bytes"]) for record in rank_costs),
            "min": min(int(record["modeled_ki_bytes"]) for record in rank_costs),
        },
        "query_layout_hash": next(iter(layout_hashes)),
        "query_token_counts": query_token_counts,
        "rank_costs": rank_costs,
        "prepare_seconds": prepare_summary,
        "shared_layout_config": expected_config,
        "token_layout_forward_ms": timing_summary,
        "token_layout_remote_rows": recomputed_key[2],
    }


def _validate_rank_results(
    plan_dir: Path,
    world_size: int,
    workload: dict[str, Any],
) -> list[dict[str, Any]]:
    layout_policy = str(workload.get("layout_policy", "legacy"))
    local_improvement_passes = int(workload.get("local_improvement_passes", 4))
    profiler_attach_warmup_steps = int(workload.get("profiler_attach_warmup_steps", 0))
    expected_delta = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": 5,
    }
    expected_backward_completion_join = {
        "w": [],
        "csa": [
            "sparse_backward_stream",
            "csa_main_stream",
            "csa_indexer_stream",
            "csa_route_stream",
        ],
        "hca": ["hca_main_stream", "hca_route_stream"],
    }
    results: list[dict[str, Any]] = []
    for rank in range(world_size):
        metadata = _read_json(plan_dir / f"metadata_rank{rank}.json")
        result = _read_json(plan_dir / f"result_rank{rank}.json")
        if (
            metadata.get("step_mode") != "attention-suite"
            or metadata.get("attention_order") != ["w", "csa", "hca"]
            or metadata.get("backward_order") != ["hca", "csa", "w"]
            or metadata.get("ratios") != [0, 4, 128]
            or metadata.get("independent_attention_graphs") is not True
            or metadata.get("gradient_accumulation") is not False
            or metadata.get("loss_capture") != "none"
            or metadata.get("mode_serialization") != "cuda_event_happens_before"
            or metadata.get("mode_backward_completion_join")
            != expected_backward_completion_join
            or metadata.get("overlap_accounting")
            != "same_mode_same_direction_non_route_compute"
            or metadata.get("projection_capture") != "pre_capture_once_per_attention"
            or metadata.get("token_layout_capture") != "pre_capture_once_per_attention"
            or metadata.get("profile_gradient_boundary")
            != "post_projection_magi_dsa_input"
            or float(metadata.get("dout_scale", float("nan"))) != 1.0 / 4_294_967_296
            or metadata.get("layout_policy", "legacy") != layout_policy
            or int(metadata.get("local_improvement_passes", 4))
            != local_improvement_passes
            or int(metadata.get("profiler_attach_warmup_steps", 0))
            != profiler_attach_warmup_steps
        ):
            raise ValueError(f"rank {rank} attention-suite metadata differs")
        if (
            result.get("result") != "PASS"
            or result.get("plan") != "balanced"
            or result.get("step_mode") != "attention-suite"
            or result.get("attention_order") != ["w", "csa", "hca"]
            or result.get("backward_order") != ["hca", "csa", "w"]
            or result.get("ratios") != [0, 4, 128]
            or result.get("independent_attention_graphs") is not True
            or result.get("gradient_accumulation") is not False
            or result.get("loss_capture") != "none"
            or result.get("mode_serialization") != "cuda_event_happens_before"
            or result.get("mode_backward_completion_join")
            != expected_backward_completion_join
            or result.get("overlap_accounting")
            != "same_mode_same_direction_non_route_compute"
            or result.get("parameter_gradient_allreduce")
            != "one_unified_after_three_backwards"
            or result.get("layout_policy", "legacy") != layout_policy
            or int(result.get("local_improvement_passes", 4))
            != local_improvement_passes
            or int(result.get("profiler_attach_warmup_steps", 0))
            != profiler_attach_warmup_steps
        ):
            raise ValueError(f"rank {rank} attention-suite result did not pass")
        deltas = result.get("attention_counter_delta")
        if not isinstance(deltas, dict) or any(
            deltas.get(mode) != expected_delta for mode in ("w", "csa", "hca")
        ):
            raise ValueError(f"rank {rank} attention-suite counter delta differs")
        attach_delta = result.get("profiler_attach_warmup_counter_delta")
        expected_attach_delta = dict(expected_delta)
        expected_attach_delta["warm_invocations"] = profiler_attach_warmup_steps
        if not isinstance(attach_delta, dict) or any(
            attach_delta.get(mode) != expected_attach_delta
            for mode in ("w", "csa", "hca")
        ):
            raise ValueError(
                f"rank {rank} profiler-attach warmup counter delta differs"
            )
        mode_metrics = result.get("mode_metrics")
        if not isinstance(mode_metrics, dict):
            raise ValueError(f"rank {rank} is missing mode metrics")
        for mode in ("w", "csa", "hca"):
            metrics = mode_metrics.get(mode)
            if not isinstance(metrics, dict) or not all(
                metrics.get(name) is True
                for name in (
                    "output_finite",
                    "sparse_lse_finite",
                    "topk_backend_native_valid",
                )
            ):
                raise ValueError(f"rank {rank} {mode} output validation failed")
        csa_metrics = result.get("csa_shadow_metrics")
        if (
            not isinstance(csa_metrics, dict)
            or csa_metrics.get("topk_backend_native_valid") is not True
            or csa_metrics.get("output_finite") is not True
            or csa_metrics.get("topk_length_exact") is not True
            or csa_metrics.get("topk_unique") is not True
        ):
            raise ValueError(f"rank {rank} CSA shadow output validation failed")
        compared_rows = int(csa_metrics.get("output_compared_rows", -1))
        exempt_rows = int(csa_metrics.get("output_tie_exempt_rows", -1))
        if (
            compared_rows < 0
            or exempt_rows < 0
            or compared_rows + exempt_rows
            != int(metadata.get("local_source_tokens", -1))
        ):
            raise ValueError(f"rank {rank} CSA tie-aware row accounting differs")
        gradient_metrics = result.get("csa_gradient_metrics")
        if (
            not isinstance(gradient_metrics, dict)
            or gradient_metrics.get("all_close") is not True
        ):
            raise ValueError(f"rank {rank} CSA gradient shadow failed")
        gradient_finite = result.get("gradient_finite")
        if not isinstance(gradient_finite, dict):
            raise ValueError(f"rank {rank} gradient diagnostics are missing")
        mode_gradients = gradient_finite.get("modes")
        if not isinstance(mode_gradients, dict):
            raise ValueError(f"rank {rank} mode gradient diagnostics are missing")
        for mode in ("w", "csa", "hca"):
            mode_payload = mode_gradients.get(mode)
            if (
                not isinstance(mode_payload, dict)
                or int(mode_payload.get("gradient_tensors", 0)) <= 0
            ):
                raise ValueError(f"rank {rank} {mode} gradients are missing")
            tensors = mode_payload.get("tensors")
            if not isinstance(tensors, dict) or any(
                int(value.get("nonfinite", -1)) != 0
                for value in tensors.values()
                if isinstance(value, dict)
            ):
                raise ValueError(f"rank {rank} {mode} gradients are non-finite")
        results.append(result)
    return results


def _attribution_scope_names(record: dict[str, Any]) -> set[str]:
    return {
        str(scope.get("name", ""))
        for scope in record.get("attribution_path", [])
        if isinstance(scope, dict)
    }


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError(f"invalid kernel interval: start={start}, end={end}")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_overlap_ns(
    interval: tuple[int, int],
    others: list[tuple[int, int]],
) -> int:
    start, end = interval
    return sum(
        max(0, min(end, other_end) - max(start, other_start))
        for other_start, other_end in others
    )


def _overlap_stats(values: list[float]) -> dict[str, float]:
    return {
        "max": max(values),
        "mean": statistics.fmean(values),
        "min": min(values),
    }


def compute_csa_communication_overlap(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Audit CSA launch order and measure COMPRESSED_KI GPU overlap."""

    expected = {(rank, step) for rank in range(world_size) for step in range(steps)}
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {
        key: {
            "forward_ki": [],
            "forward_compute": [],
            "backward_ki": [],
            "backward_compute": [],
        }
        for key in expected
    }
    for record in attribution_records:
        key = (int(record.get("rank", -1)), int(record.get("step", -1)))
        if key not in grouped:
            continue
        scopes = _attribution_scope_names(record)
        kernel_name = str(record.get("kernel_name", ""))
        if (
            _CSA_FORWARD_SCOPE in scopes
            and _CSA_COMPRESSED_KI_FORWARD_SCOPE in scopes
            and kernel_name == "ncclDevKernel_SendRecv"
        ):
            grouped[key]["forward_ki"].append(record)
        if _CSA_FORWARD_SCOPE in scopes and any(
            scope == _CSA_MAIN_COMPRESSOR_SCOPE
            or scope.startswith(f"{_CSA_MAIN_COMPRESSOR_SCOPE}::")
            for scope in scopes
        ):
            grouped[key]["forward_compute"].append(record)
        if (
            _CSA_BACKWARD_SCOPE in scopes
            and _CSA_COMPRESSED_KI_BACKWARD_SCOPE in scopes
            and kernel_name == "ncclDevKernel_SendRecv"
        ):
            grouped[key]["backward_ki"].append(record)
        if _CSA_BACKWARD_SCOPE in scopes and _CSA_SPARSE_BACKWARD_SCOPE in scopes:
            grouped[key]["backward_compute"].append(record)

    details: dict[str, list[dict[str, Any]]] = {
        "forward": [],
        "backward": [],
    }
    direction_fields = {
        "forward": ("forward_ki", "forward_compute", "main_compressor"),
        "backward": ("backward_ki", "backward_compute", "sparse_backward"),
    }
    for key in sorted(expected):
        rank, step = key
        for direction, (
            ki_field,
            compute_field,
            compute_name,
        ) in direction_fields.items():
            ki_records = grouped[key][ki_field]
            compute_records = grouped[key][compute_field]
            if len(ki_records) != 1:
                raise ValueError(
                    "CSA COMPRESSED_KI SendRecv count differs: "
                    f"direction={direction}, rank={rank}, step={step}, "
                    f"count={len(ki_records)}"
                )
            if not compute_records:
                raise ValueError(
                    "CSA overlap compute kernels are missing: "
                    f"direction={direction}, rank={rank}, step={step}"
                )

            ki_record = ki_records[0]
            ki_runtime_start = int(ki_record["runtime_start_ns"])
            compute_runtime_start = min(
                int(record["runtime_start_ns"]) for record in compute_records
            )
            if ki_runtime_start >= compute_runtime_start:
                raise ValueError(
                    "CSA COMPRESSED_KI was not launched before compute: "
                    f"direction={direction}, rank={rank}, step={step}, "
                    f"ki_runtime_start_ns={ki_runtime_start}, "
                    f"compute_runtime_start_ns={compute_runtime_start}"
                )

            ki_interval = (
                int(ki_record["kernel_start_ns"]),
                int(ki_record["kernel_end_ns"]),
            )
            compute_intervals = _merge_intervals(
                [
                    (
                        int(record["kernel_start_ns"]),
                        int(record["kernel_end_ns"]),
                    )
                    for record in compute_records
                ]
            )
            ki_duration_ns = ki_interval[1] - ki_interval[0]
            if ki_duration_ns <= 0:
                raise ValueError(
                    "CSA COMPRESSED_KI has an invalid GPU interval: "
                    f"direction={direction}, rank={rank}, step={step}"
                )
            overlap_ns = _interval_overlap_ns(ki_interval, compute_intervals)
            details[direction].append(
                {
                    "compute": compute_name,
                    "compute_kernel_duration_us": (
                        sum(end - start for start, end in compute_intervals) / 1_000.0
                    ),
                    "compute_kernel_end_ns": max(end for _, end in compute_intervals),
                    "compute_kernel_start_ns": min(
                        start for start, _ in compute_intervals
                    ),
                    "compute_runtime_start_ns": compute_runtime_start,
                    "ki_kernel_duration_us": ki_duration_ns / 1_000.0,
                    "ki_kernel_end_ns": ki_interval[1],
                    "ki_kernel_start_ns": ki_interval[0],
                    "ki_overlap_fraction": overlap_ns / ki_duration_ns,
                    "ki_runtime_start_ns": ki_runtime_start,
                    "launch_lead_us": (compute_runtime_start - ki_runtime_start)
                    / 1_000.0,
                    "overlap_us": overlap_ns / 1_000.0,
                    "rank": rank,
                    "step": step,
                }
            )

    result: dict[str, Any] = {
        "actual_gpu_overlap_is_hard_gate": False,
        "result": "PASS",
    }
    for direction, records in details.items():
        overlap_values = [float(record["overlap_us"]) for record in records]
        fraction_values = [float(record["ki_overlap_fraction"]) for record in records]
        lead_values = [float(record["launch_lead_us"]) for record in records]
        result[direction] = {
            "ki_before_compute_launch": True,
            "ki_overlap_fraction": _overlap_stats(fraction_values),
            "launch_lead_us": _overlap_stats(lead_values),
            "overlap_us": _overlap_stats(overlap_values),
            "positive_gpu_overlap_records": sum(
                value > 0.0 for value in overlap_values
            ),
            "records": records,
            "total_records": len(records),
        }
    return result


def _is_non_route_gpu_compute(record: dict[str, Any]) -> bool:
    kernel_name = str(record.get("kernel_name", ""))
    return (
        not kernel_name.startswith("ncclDevKernel_")
        and "DsaRowCopy" not in kernel_name
        and "DsaRowCsrReduce" not in kernel_name
    )


def _record_mode_direction(
    record: dict[str, Any],
) -> tuple[str, str] | None:
    scopes = _attribution_scope_names(record)
    attribution_name = str(record.get("attribution_name", ""))
    matches = [
        (mode, direction)
        for mode, direction in _MODE_SERIAL_ORDER
        if (
            f"magi_dsa::attention_suite::{mode}::{direction}" in scopes
            or attribution_name == f"magi_dsa::attention_suite::{mode}::{direction}"
        )
    ]
    if len(matches) > 1:
        raise ValueError(
            "attention-suite kernel has conflicting mode ownership: "
            f"kernel={record.get('kernel_name')!r}, matches={matches}"
        )
    return matches[0] if matches else None


def compute_attention_suite_communication_overlap(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Validate mode serialization and measure only mode-local GPU overlap."""

    expected = {(rank, step) for rank in range(world_size) for step in range(steps)}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in expected}
    for record in attribution_records:
        key = (int(record.get("rank", -1)), int(record.get("step", -1)))
        if key in grouped:
            grouped[key].append(record)

    details: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        mode: {
            direction: {route: [] for route in routes}
            for direction, routes in directions.items()
        }
        for mode, directions in _MODE_ROUTE_ORDER.items()
    }
    route_orders: list[dict[str, Any]] = []
    mode_serial_spans: list[dict[str, Any]] = []
    for rank, step in sorted(expected):
        records = grouped[(rank, step)]
        owned_records: dict[tuple[str, str], list[dict[str, Any]]] = {
            owner: [] for owner in _MODE_SERIAL_ORDER
        }
        compute_intervals: dict[tuple[str, str], list[tuple[int, int]]] = {
            owner: [] for owner in _MODE_SERIAL_ORDER
        }
        for record in records:
            owner = _record_mode_direction(record)
            if owner is None:
                continue
            owned_records[owner].append(record)
            if _is_non_route_gpu_compute(record):
                compute_intervals[owner].append(
                    (
                        int(record["kernel_start_ns"]),
                        int(record["kernel_end_ns"]),
                    )
                )
        for owner in _MODE_SERIAL_ORDER:
            if not owned_records[owner]:
                raise ValueError(
                    "attention-suite mode GPU kernels are missing: "
                    f"mode={owner[0]}, direction={owner[1]}, "
                    f"rank={rank}, step={step}"
                )
            compute_intervals[owner] = _merge_intervals(compute_intervals[owner])

        previous_end: int | None = None
        previous_owner: tuple[str, str] | None = None
        for owner in _MODE_SERIAL_ORDER:
            start = min(
                int(record["kernel_start_ns"]) for record in owned_records[owner]
            )
            end = max(int(record["kernel_end_ns"]) for record in owned_records[owner])
            if previous_end is not None and start < previous_end:
                assert previous_owner is not None
                raise ValueError(
                    "attention-suite cross-mode GPU execution overlap exists: "
                    f"previous={previous_owner}, current={owner}, "
                    f"rank={rank}, step={step}, overlap_ns={previous_end - start}"
                )
            mode_serial_spans.append(
                {
                    "direction": owner[1],
                    "end_ns": end,
                    "mode": owner[0],
                    "rank": rank,
                    "start_ns": start,
                    "step": step,
                }
            )
            previous_end = end
            previous_owner = owner

        for mode, directions in _MODE_ROUTE_ORDER.items():
            for direction, expected_order in directions.items():
                owner = (mode, direction)
                route_records: list[tuple[str, dict[str, Any]]] = []
                for route in expected_order:
                    scope = (
                        "magi_dsa::phase::collective_all2all_v::"
                        f"attention::{mode}::{route}.{direction}"
                    )
                    matches = [
                        record
                        for record in records
                        if str(record.get("kernel_name", ""))
                        == "ncclDevKernel_SendRecv"
                        and str(record.get("attribution_name", "")) == scope
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "attention-suite route SendRecv count differs: "
                            f"mode={mode}, direction={direction}, route={route}, "
                            f"rank={rank}, step={step}, count={len(matches)}"
                        )
                    if _record_mode_direction(matches[0]) != owner:
                        raise ValueError(
                            "attention-suite route ownership differs: "
                            f"mode={mode}, direction={direction}, route={route}, "
                            f"rank={rank}, step={step}"
                        )
                    route_records.append((route, matches[0]))

                actual_order = tuple(
                    route
                    for route, _ in sorted(
                        route_records,
                        key=lambda item: int(item[1]["runtime_start_ns"]),
                    )
                )
                if actual_order != expected_order:
                    raise ValueError(
                        "attention-suite route launch order differs: "
                        f"mode={mode}, direction={direction}, rank={rank}, "
                        f"step={step}, actual={actual_order}, "
                        f"expected={expected_order}"
                    )
                route_orders.append(
                    {
                        "direction": direction,
                        "mode": mode,
                        "rank": rank,
                        "routes": list(actual_order),
                        "step": step,
                    }
                )

                for route, record in route_records:
                    interval = (
                        int(record["kernel_start_ns"]),
                        int(record["kernel_end_ns"]),
                    )
                    duration_ns = interval[1] - interval[0]
                    if duration_ns <= 0:
                        raise ValueError(
                            "attention-suite route has an invalid GPU interval: "
                            f"mode={mode}, direction={direction}, route={route}, "
                            f"rank={rank}, step={step}"
                        )
                    own_compute_intervals = compute_intervals[owner]
                    overlap_ns = _interval_overlap_ns(
                        interval,
                        own_compute_intervals,
                    )
                    foreign_compute_intervals = _merge_intervals(
                        [
                            candidate
                            for foreign_owner, candidates in compute_intervals.items()
                            if foreign_owner[0] != mode
                            for candidate in candidates
                        ]
                    )
                    foreign_overlap_ns = _interval_overlap_ns(
                        interval,
                        foreign_compute_intervals,
                    )
                    if foreign_overlap_ns != 0:
                        raise ValueError(
                            "attention-suite route overlaps foreign-mode compute: "
                            f"mode={mode}, direction={direction}, route={route}, "
                            f"rank={rank}, step={step}, "
                            f"foreign_overlap_ns={foreign_overlap_ns}"
                        )
                    details[mode][direction][route].append(
                        {
                            "foreign_compute_overlap_us": (
                                foreign_overlap_ns / 1_000.0
                            ),
                            "kernel_duration_us": duration_ns / 1_000.0,
                            "kernel_end_ns": interval[1],
                            "kernel_name": "ncclDevKernel_SendRecv",
                            "kernel_start_ns": interval[0],
                            "nvtx_path": (
                                "magi_dsa::phase::collective_all2all_v::"
                                f"attention::{mode}::{route}.{direction}"
                            ),
                            "overlap_accounting": (
                                "same_mode_same_direction_non_route_compute"
                            ),
                            "overlap_fraction": overlap_ns / duration_ns,
                            "overlap_us": overlap_ns / 1_000.0,
                            "rank": rank,
                            "runtime_start_ns": int(record["runtime_start_ns"]),
                            "step": step,
                        }
                    )

    route_summary: dict[str, Any] = {}
    for mode, mode_details in details.items():
        route_summary[mode] = {}
        for direction, direction_details in mode_details.items():
            route_summary[mode][direction] = {}
            for route, records in direction_details.items():
                duration_values = [
                    float(record["kernel_duration_us"]) for record in records
                ]
                overlap_values = [float(record["overlap_us"]) for record in records]
                fraction_values = [
                    float(record["overlap_fraction"]) for record in records
                ]
                route_summary[mode][direction][route] = {
                    "dependency": _MODE_LOCAL_OVERLAP_EXPLANATIONS[
                        (mode, direction, route)
                    ],
                    "foreign_compute_overlap_us": _overlap_stats(
                        [
                            float(record["foreign_compute_overlap_us"])
                            for record in records
                        ]
                    ),
                    "kernel_duration_us": _overlap_stats(duration_values),
                    "kernel_name": "ncclDevKernel_SendRecv",
                    "nvtx_path": (
                        "magi_dsa::phase::collective_all2all_v::"
                        f"attention::{mode}::{route}.{direction}"
                    ),
                    "overlap_fraction": _overlap_stats(fraction_values),
                    "overlap_us": _overlap_stats(overlap_values),
                    "positive_gpu_overlap_records": sum(
                        value > 0.0 for value in overlap_values
                    ),
                    "records": records,
                    "total_records": len(records),
                }

    return {
        "actual_gpu_overlap_is_hard_gate": False,
        "compute_definition": (
            "same-rank/step, same-mode, same-direction non-route GPU kernels "
            "excluding NCCL, DsaRowCopy, and DsaRowCsrReduce"
        ),
        "cross_mode_gpu_overlap_is_hard_gate": True,
        "foreign_compute_overlap_records": 0,
        "mode_serial_order": [
            {"direction": direction, "mode": mode}
            for mode, direction in _MODE_SERIAL_ORDER
        ],
        "mode_serial_spans": mode_serial_spans,
        "mode_serialization_is_hard_gate": True,
        "route_launch_order_is_hard_gate": True,
        "route_orders": route_orders,
        "routes": route_summary,
        "result": "PASS",
    }


def compute_attention_suite_step_kernel_spans(
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Measure first-to-last attributed GPU kernel span for every rank/step."""

    expected = {(rank, step) for rank in range(world_size) for step in range(steps)}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in expected}
    for record in attribution_records:
        key = (int(record.get("rank", -1)), int(record.get("step", -1)))
        if key in grouped:
            grouped[key].append(record)

    records: list[dict[str, Any]] = []
    for rank, step in sorted(expected):
        kernels = grouped[(rank, step)]
        if not kernels:
            raise ValueError(
                f"attention-suite step has no attributed kernels: rank={rank}, step={step}"
            )
        start_ns = min(int(record["kernel_start_ns"]) for record in kernels)
        end_ns = max(int(record["kernel_end_ns"]) for record in kernels)
        records.append(
            {
                "end_ns": end_ns,
                "rank": rank,
                "span_ms": (end_ns - start_ns) / 1_000_000.0,
                "start_ns": start_ns,
                "step": step,
            }
        )

    span_values = [float(record["span_ms"]) for record in records]
    five_step_values = [
        sum(
            float(record["span_ms"])
            for record in records
            if int(record["rank"]) == rank
        )
        for rank in range(world_size)
    ]
    return {
        "five_step_span_ms": _overlap_stats(five_step_values),
        "records": records,
        "result": "PASS",
        "step_span_ms": _overlap_stats(span_values),
    }


def _compute_step_gaps(
    sqlite_path: Path,
    attribution_records: list[dict[str, Any]],
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT ranges.start,
                   ranges.end,
                   COALESCE(ranges.text, strings.value)
            FROM NVTX_EVENTS AS ranges
            LEFT JOIN StringIds AS strings ON strings.id = ranges.textId
            WHERE ranges.end IS NOT NULL
            ORDER BY ranges.start
            """
        ).fetchall()
    finally:
        connection.close()
    step_ranges: dict[tuple[int, int], tuple[int, int]] = {}
    for start, end, text in rows:
        if text is None:
            continue
        match = _STEP_PATTERN.fullmatch(str(text))
        if match is None:
            continue
        key = (int(match.group("rank")), int(match.group("step")))
        if key in step_ranges:
            raise ValueError(f"duplicate attention-suite step range: {key}")
        step_ranges[key] = (int(start), int(end))
    expected = {(rank, step) for rank in range(world_size) for step in range(steps)}
    if set(step_ranges) != expected:
        raise ValueError("attention-suite step ranges are incomplete")

    kernels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for record in attribution_records:
        key = (int(record["rank"]), int(record["step"]))
        kernels.setdefault(key, []).append(
            (int(record["kernel_start_ns"]), int(record["kernel_end_ns"]))
        )
    if set(kernels) != expected:
        raise ValueError("attention-suite step kernel ranges are incomplete")

    records: list[dict[str, Any]] = []
    for rank in range(world_size):
        for step in range(steps - 1):
            current = step_ranges[(rank, step)]
            following = step_ranges[(rank, step + 1)]
            current_kernels = kernels[(rank, step)]
            following_kernels = kernels[(rank, step + 1)]
            records.append(
                {
                    "cpu_nvtx_gap_us": (following[0] - current[1]) / 1_000.0,
                    "gpu_kernel_gap_us": (
                        min(start for start, _ in following_kernels)
                        - max(end for _, end in current_kernels)
                    )
                    / 1_000.0,
                    "next_step": step + 1,
                    "rank": rank,
                    "step": step,
                }
            )
    cpu_values = [float(record["cpu_nvtx_gap_us"]) for record in records]
    gpu_values = [float(record["gpu_kernel_gap_us"]) for record in records]
    return {
        "cpu_nvtx_gap_us": {
            "max": max(cpu_values),
            "mean": statistics.fmean(cpu_values),
            "min": min(cpu_values),
        },
        "gpu_kernel_gap_us": {
            "max": max(gpu_values),
            "mean": statistics.fmean(gpu_values),
            "min": min(gpu_values),
        },
        "records": records,
    }


def main() -> None:
    args = _parse_args()
    if args.world_size != 8 or args.steps != 5:
        raise ValueError("attention-suite summary requires 8 ranks and five steps")
    workload = _read_json(args.artifact_dir / "WORKLOAD.json")
    _validate_workload(workload)
    layout_policy = str(workload.get("layout_policy", "legacy"))
    profiler_attach_warmup_steps = int(workload.get("profiler_attach_warmup_steps", 0))
    total_profiled_invocations = args.steps + profiler_attach_warmup_steps
    clock_control = _validate_clock_control(args.artifact_dir, workload)
    plan_dir = args.artifact_dir / "balanced"
    report = plan_dir / "balanced_5steps_attention_suite.nsys-rep"
    sqlite_path = plan_dir / "balanced_5steps_attention_suite.sqlite"
    if not report.is_file() or report.stat().st_size == 0:
        raise FileNotFoundError(f"missing attention-suite report: {report}")
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing attention-suite SQLite: {sqlite_path}")

    nvtx_range_counts = read_nvtx_range_counts(sqlite_path)
    token_layout_count = sum(
        count for name, count in nvtx_range_counts.items() if "TOKEN_LAYOUT" in name
    )
    projection_count = sum(
        count
        for name, count in nvtx_range_counts.items()
        if "magi_dsa::module::model_projection::" in name
    )
    loss_count = int(nvtx_range_counts.get("magi_dsa::loss", 0))
    if token_layout_count or projection_count or loss_count:
        raise ValueError(
            "attention-suite capture contains excluded work: "
            f"TOKEN_LAYOUT={token_layout_count}, projection={projection_count}, "
            f"loss={loss_count}"
        )
    expected_nvtx_counts = {
        "$Magi_DSA/capture_five_attention_suite_steps": args.world_size,
        **{
            name: args.world_size * total_profiled_invocations
            for name in _PHASE_NVTX_NAMES.values()
        },
        "magi_dsa::CUDNN_CALL::indexer_backward": (
            args.world_size * total_profiled_invocations
        ),
        "magi_dsa::CUDNN_CALL::sparse_attention_backward": (
            args.world_size * total_profiled_invocations * len(_MODE_PHASES)
        ),
    }
    if profiler_attach_warmup_steps:
        expected_nvtx_counts.update(
            {
                "$Magi_DSA/ablation_profiler_attach_warmup": args.world_size,
                "magi_dsa::ablation::profiler_attach_warmup_step": (
                    args.world_size * profiler_attach_warmup_steps
                ),
            }
        )
    for operation in (
        "unit_gradient_ready_event_wait",
        "unit_gradient_backward_lifetime",
        "scale_indexer_gradients",
        "compressed_ki_reverse_start",
        "sparse_backward_launch",
        "compressed_ki_reverse_finish",
        "sparse_backward_finish",
    ):
        expected_nvtx_counts[
            "magi_dsa::module::attention::csa::backward_overlap::" f"{operation}"
        ] = (args.world_size * total_profiled_invocations)
    expected_nvtx_counts[
        "magi_dsa::module::attention_suite::stream_overlap::gradient_join"
    ] = (args.world_size * total_profiled_invocations)
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
            "magi_dsa::module::attention_suite::stream_overlap::"
            f"backward_completion_join::{mode}"
        )
        expected_nvtx_counts[parent] = args.world_size * total_profiled_invocations
        for field in stream_fields:
            expected_nvtx_counts[f"{parent}::{field}"] = (
                args.world_size * total_profiled_invocations
            )
    for boundary in (
        "w_forward_to_csa_forward",
        "csa_forward_to_hca_forward",
        "hca_backward_to_csa_backward",
        "csa_backward_to_w_backward",
    ):
        expected_nvtx_counts[
            f"magi_dsa::module::attention_suite::mode_serial::{boundary}"
        ] = (args.world_size * total_profiled_invocations)
    for operation in ("support_routes_launch", "main_compressor_launch"):
        expected_nvtx_counts[
            "magi_dsa::module::attention::hca::stream_overlap::" f"{operation}"
        ] = (args.world_size * total_profiled_invocations)
    for mode, routes in _MODE_ROUTES.items():
        expected_nvtx_counts[
            f"magi_dsa::module::attention::{mode}::"
            "sparse_attention::flashmla_forward"
        ] = (args.world_size * total_profiled_invocations)
        expected_nvtx_counts[
            f"magi_dsa::module::attention::{mode}::" "sparse_attention::cudnn_backward"
        ] = (args.world_size * total_profiled_invocations)
        expected_nvtx_counts[f"magi_dsa::phase::attention::{mode}::sparse_backward"] = (
            args.world_size * total_profiled_invocations
        )
        for route in routes:
            for direction in ("forward", "backward"):
                expected_nvtx_counts[
                    "magi_dsa::phase::collective_all2all_v::"
                    f"attention::{mode}::{route}.{direction}"
                ] = (args.world_size * total_profiled_invocations)
    for name, expected_count in expected_nvtx_counts.items():
        if int(nvtx_range_counts.get(name, 0)) != expected_count:
            raise ValueError(
                f"attention-suite NVTX count mismatch: {name}="
                f"{nvtx_range_counts.get(name, 0)}, expected={expected_count}"
            )
    (plan_dir / "NVTX_RANGE_COUNTS.json").write_text(
        json.dumps(
            {
                "counts": nvtx_range_counts,
                "range_names": len(nvtx_range_counts),
                "ranges": sum(nvtx_range_counts.values()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _read_json(plan_dir / "NSYS_AUDIT.json")
    if (
        audit.get("result") != "PASS"
        or audit.get("step_mode") != "attention-suite"
        or int(audit.get("logical_phase_records", 0))
        != args.world_size * args.steps * len(_PHASES)
    ):
        raise ValueError("attention-suite Nsight phase audit did not pass")
    attribution_audit = _read_json(plan_dir / "NSYS_ATTRIBUTION.json")
    if (
        attribution_audit.get("result") != "PASS"
        or int(attribution_audit.get("unattributed_kernel_count", -1)) != 0
        or float(attribution_audit.get("attribution_coverage", 0.0)) != 1.0
        or int(attribution_audit.get("kernel_records", 0)) <= 0
    ):
        raise ValueError("attention-suite kernel attribution did not pass")
    attribution_records = _read_jsonl(plan_dir / "nsys_kernel_attribution.jsonl")
    if len(attribution_records) != int(attribution_audit["kernel_records"]):
        raise ValueError("attention-suite attribution record count differs")
    excluded_attribution = [
        record
        for record in attribution_records
        if any(
            "TOKEN_LAYOUT" in str(scope.get("name", ""))
            or "magi_dsa::module::model_projection::" in str(scope.get("name", ""))
            for scope in record.get("attribution_path", [])
        )
    ]
    if excluded_attribution:
        raise ValueError(
            "attention-suite attributed excluded layout/projection kernels"
        )

    records = validate_attention_suite_records(
        _read_jsonl(plan_dir / "nsys_phase_records.jsonl"),
        args.world_size,
        args.steps,
    )
    validate_attention_suite_sendrecv(records)
    core_kernel_audit = validate_attention_suite_core_kernels(records)
    launch_audit = _attention_suite_kernel_launch_audit(sqlite_path, records)
    csa_communication_overlap = compute_csa_communication_overlap(
        attribution_records,
        args.world_size,
        args.steps,
    )
    attention_communication_overlap = compute_attention_suite_communication_overlap(
        attribution_records,
        args.world_size,
        args.steps,
    )
    step_kernel_spans = compute_attention_suite_step_kernel_spans(
        attribution_records,
        args.world_size,
        args.steps,
    )
    kernel_overlap_audit = {
        **core_kernel_audit,
        "launch_metadata": launch_audit,
        "result": "PASS",
    }
    (args.artifact_dir / "KERNEL_OVERLAP_ATTENTION_SUITE.json").write_text(
        json.dumps(kernel_overlap_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "CSA_COMMUNICATION_OVERLAP.json").write_text(
        json.dumps(csa_communication_overlap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "ATTENTION_SUITE_COMMUNICATION_OVERLAP.json").write_text(
        json.dumps(attention_communication_overlap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "STEP_KERNEL_SPANS_ATTENTION_SUITE.json").write_text(
        json.dumps(step_kernel_spans, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for rank in range(args.world_size):
        raw = _read_jsonl(plan_dir / f"rank{rank}_nsys_raw.jsonl")
        if len(raw) != args.steps * len(_PHASES):
            raise ValueError(f"rank {rank} attention-suite raw grid differs")
        rank_attribution = _read_jsonl(
            plan_dir / f"rank{rank}_nsys_kernel_attribution.jsonl"
        )
        if not rank_attribution or any(
            int(record.get("rank", -1)) != rank for record in rank_attribution
        ):
            raise ValueError(f"rank {rank} attribution records are missing")

    results = _validate_rank_results(plan_dir, args.world_size, workload)
    layout_summary = _validate_shared_layout_metadata(
        plan_dir,
        args.world_size,
        workload,
    )
    ranges = compute_attention_suite_rank_ranges(
        records,
        args.world_size,
        args.steps,
    )
    step_gaps = _compute_step_gaps(
        sqlite_path,
        attribution_records,
        args.world_size,
        args.steps,
    )
    (args.artifact_dir / "profile_attention_suite.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.artifact_dir / "rank_ranges_attention_suite.json").write_text(
        json.dumps(ranges, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "STEP_GAPS_ATTENTION_SUITE.json").write_text(
        json.dumps(step_gaps, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csa_metrics = [result["csa_shadow_metrics"] for result in results]
    gradient_metrics = [result["csa_gradient_metrics"] for result in results]
    phase_mean_ms = {
        phase: statistics.fmean(
            float(record["gpu_time_ms"])
            for record in records
            if record["phase"] == phase
        )
        for phase in _PHASES
    }
    process_temporal_launches = sum(
        int(record.get("process_temporal_kernel_launch_count", 0)) for record in records
    )
    summary = {
        "attention_order": ["w", "csa", "hca"],
        "backward_order": ["hca", "csa", "w"],
        "csa_gradient_max_abs": max(
            float(metrics["max_abs"]) for metrics in gradient_metrics
        ),
        "csa_latent_kv_gradient_mismatch_ratio": max(
            float(metrics["latent_kv_mismatch_ratio"]) for metrics in gradient_metrics
        ),
        "csa_output_compared_rows": sum(
            int(metrics["output_compared_rows"]) for metrics in csa_metrics
        ),
        "csa_output_non_tie_max_abs": max(
            float(metrics["output_non_tie_max_abs"]) for metrics in csa_metrics
        ),
        "csa_output_tie_exempt_rows": sum(
            int(metrics["output_tie_exempt_rows"]) for metrics in csa_metrics
        ),
        "csa_communication_overlap": {
            direction: {
                key: value
                for key, value in csa_communication_overlap[direction].items()
                if key != "records"
            }
            for direction in ("forward", "backward")
        },
        "dispatch_ablation": {
            "gpu_clock_control": clock_control,
            "local_improvement_passes": int(
                workload.get("local_improvement_passes", 4)
            ),
            "profiler_attach_warmup_steps": profiler_attach_warmup_steps,
        },
        "attention_suite_communication_overlap": {
            mode: {
                direction: {
                    route: {
                        key: value for key, value in metrics.items() if key != "records"
                    }
                    for route, metrics in routes.items()
                }
                for direction, routes in directions.items()
            }
            for mode, directions in attention_communication_overlap["routes"].items()
        },
        "expected_sendrecv": _EXPECTED_SENDRECV,
        "gradient_accumulation": False,
        "independent_attention_graphs": True,
        "kernel_attribution_coverage": 1.0,
        "kernel_attribution_records": len(attribution_records),
        "kernel_overlap_audit": "PASS",
        "layout": layout_summary,
        "kernel_exact_resource_signature_shared_across_modes": launch_audit[
            "exact_resource_signature_across_modes"
        ],
        "kernel_flashmla_name_block_shared_memory_across_modes": launch_audit[
            "flashmla_forward_name_block_shared_memory_across_modes"
        ],
        "kernel_flashmla_w_hca_exact_resource_signature": launch_audit[
            "flashmla_forward_w_hca_exact_resource_signature"
        ],
        "kernel_cudnn_backward_exact_resource_signature_across_modes": launch_audit[
            "cudnn_backward_exact_resource_signature_across_modes"
        ],
        "logical_phase_records": len(records),
        "loss_capture": "none",
        "loss_nvtx_ranges": loss_count,
        "mode_backward_completion_join": workload["mode_backward_completion_join"],
        "mode_serialization": "cuda_event_happens_before",
        "model_projection_nvtx_ranges": projection_count,
        "nvtx_range_names": len(nvtx_range_counts),
        "nvtx_ranges": sum(nvtx_range_counts.values()),
        "phase_mean_ms": phase_mean_ms,
        "process_temporal_kernel_launches": process_temporal_launches,
        "overlap_accounting": "same_mode_same_direction_non_route_compute",
        "rank_results": len(results),
        "ratios": [0, 4, 128],
        "result": "PASS",
        "step_gaps": {
            "cpu_nvtx_gap_us": step_gaps["cpu_nvtx_gap_us"],
            "gpu_kernel_gap_us": step_gaps["gpu_kernel_gap_us"],
        },
        "step_kernel_spans": {
            key: value for key, value in step_kernel_spans.items() if key != "records"
        },
        "step_mode": "attention-suite",
        "steps": args.steps,
        "token_layout_nvtx_ranges": token_layout_count,
        "world_size": args.world_size,
    }
    (args.artifact_dir / "SUMMARY_ATTENTION_SUITE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ranges_by_key = {
        (int(record["step"]), str(record["phase"])): record for record in ranges
    }
    report_lines = [
        "# Magi-DSA W + CSA + HCA merged-step profile",
        "",
        "- 结果：PASS",
        "- Workload：8×B300，BF16，单条 128K，5 steps",
        f"- Query layout policy：`{layout_policy}`",
        "- Dispatch ablation：local improvement passes="
        f"{workload.get('local_improvement_passes', 4)}，profiler-start 后等价 warmup="
        f"{profiler_attach_warmup_steps} steps，GPU clock="
        f"{clock_control['requested_graphics_clock_mhz'] or 'dynamic'} MHz",
        "- 每 step：三张独立 autograd graph；forward=W→CSA→HCA，backward=HCA→CSA→W",
        "- 模式隔离：四个跨 mode 边界使用 CUDA event 建立 happens-before；W/CSA/HCA "
        "在 GPU 上严格串行，不允许跨 mode kernel overlap",
        "- Backward completion：CSA/HCA mode stream 在 mode-done event 前 join "
        "handle 的全部内部 streams；无全局 synchronize",
        "- 通信硬校验：W=1F+1B，CSA=4F+4B，HCA=3F+3B；总计=8F+8B",
        "- Capture boundary：每种 Attention 在 capture 前准备 post-projection "
        "leaves 和固定 dout；无 TOKEN_LAYOUT、projection、scalar loss、optimizer "
        "或跨 step 梯度累积",
        "- 梯度归约：三种 backward 完成后统一执行一次模型侧 parameter/sink bucket AllReduce",
        f"- CSA shadow：tie-exempt rows={summary['csa_output_tie_exempt_rows']}，"
        "非 tie output max-abs="
        f"{summary['csa_output_non_tie_max_abs']:.9g}，gradient max-abs="
        f"{summary['csa_gradient_max_abs']:.9g}",
        f"- Kernel 归因：{len(attribution_records)} 个 kernels，coverage=100%，"
        f"unattributed=0，process-temporal launches={process_temporal_launches}",
        "- CSA 前向调度：`COMPRESSED_KI` 在 Main Compressor 前发起；"
        "实际 GPU overlap min/mean/max="
        f"{csa_communication_overlap['forward']['overlap_us']['min']:.3f}/"
        f"{csa_communication_overlap['forward']['overlap_us']['mean']:.3f}/"
        f"{csa_communication_overlap['forward']['overlap_us']['max']:.3f} us，"
        "positive records="
        f"{csa_communication_overlap['forward']['positive_gpu_overlap_records']}/"
        f"{csa_communication_overlap['forward']['total_records']}",
        "- CSA 反向调度：`COMPRESSED_KI` reverse 在 sparse backward 前发起；"
        "实际 GPU overlap min/mean/max="
        f"{csa_communication_overlap['backward']['overlap_us']['min']:.3f}/"
        f"{csa_communication_overlap['backward']['overlap_us']['mean']:.3f}/"
        f"{csa_communication_overlap['backward']['overlap_us']['max']:.3f} us，"
        "positive records="
        f"{csa_communication_overlap['backward']['positive_gpu_overlap_records']}/"
        f"{csa_communication_overlap['backward']['total_records']}。"
        "实际 overlap 只报告、不设硬门槛；发起顺序和单次 reverse 是硬校验",
        "- Overlap 口径：只计算相同 mode、相同 forward/backward 阶段的非 route GPU kernel；"
        "跨 mode GPU overlap 为零是硬校验。所有 route 的完整 "
        "`ncclDevKernel_SendRecv`/NVTX 与依赖理由见 "
        "`ATTENTION_SUITE_COMMUNICATION_OVERLAP.json`",
        "- 实际 attributed GPU kernel span：逐 step mean="
        f"{step_kernel_spans['step_span_ms']['mean']:.6f} ms，"
        "逐 rank 五步 mean="
        f"{step_kernel_spans['five_step_span_ms']['mean']:.6f} ms",
        "- Attention core：W/CSA/HCA 均各调用 1 次 "
        "`sparse_attn_fwd_for_small_topk_kernel`。W/HCA "
        "三输出变体的完整资源签名一致；CSA dual-LSE 变体与它们具有相同 kernel "
        "名称、block 和 shared memory，允许且仅观察到 registers/thread 的编译专用化："
        f"{launch_audit['csa_dual_lse_specialization']['registers_per_thread']}。"
        "三种 backward 均各调用同一组 4 个 cuDNN DSA core kernels，且完整资源签名一致。"
        "三条完整路径并不相同，ratio-specific 额外 kernel 见 "
        "`KERNEL_OVERLAP_ATTENTION_SUITE.json`",
        "- Backward NVTX：正式模式 range 位于调用线程；autograd kernel launch 位于 "
        "worker thread，并由 `magi_dsa::module::attention::<mode>::...` 与 "
        "`magi_dsa::phase::collective_all2all_v::attention::<mode>::...` 明确标记",
        "- Step gap：CPU NVTX min/max="
        f"{step_gaps['cpu_nvtx_gap_us']['min']:.3f}/"
        f"{step_gaps['cpu_nvtx_gap_us']['max']:.3f} us；GPU kernel min/max="
        f"{step_gaps['gpu_kernel_gap_us']['min']:.3f}/"
        f"{step_gaps['gpu_kernel_gap_us']['max']:.3f} us",
    ]
    if layout_policy == "shared-greedy":
        query_counts = layout_summary["query_token_counts"]
        modeled_bytes = layout_summary["modeled_ki_bytes"]
        report_lines.extend(
            [
                "- Shared layout：hash=`"
                f"{layout_summary['query_layout_hash']}`，key="
                f"`{tuple(layout_summary['layout_key'])}`，Query/rank="
                f"{min(query_counts)}--{max(query_counts)}，TOKEN_LAYOUT remote rows="
                f"{layout_summary['token_layout_remote_rows']}，modeled KI="
                f"{modeled_bytes['min'] / (1 << 20):.3f}--"
                f"{modeled_bytes['max'] / (1 << 20):.3f} MiB/rank",
                "- TOKEN_LAYOUT forward：在 capture 外对三张独立 Attention 输入各执行一次；"
                "下表是 CUDA event 的逐 rank min/mean/max，不计入 DSA-core phase。",
                "",
                "| mode | TOKEN_LAYOUT forward min/mean/max (ms) |",
                "| --- | ---: |",
            ]
        )
        for mode in ("w", "csa", "hca"):
            timing = layout_summary["token_layout_forward_ms"][mode]
            report_lines.append(
                f"| {mode.upper()} | {timing['min']:.6f}/"
                f"{timing['mean']:.6f}/{timing['max']:.6f} |"
            )
    report_lines.extend(
        [
            "",
            "| step | W F/B mean (ms) | CSA F/B mean (ms) | HCA F/B mean (ms) "
            "| total F/B mean (ms) | grad AR mean (ms) | score/topk mean (ms) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for step in range(args.steps):

        def mean(phase: str) -> float:
            return float(ranges_by_key[(step, phase)]["mean_rank_time_ms"])

        report_lines.append(
            f"| {step} | {mean('w_forward'):.6f}/{mean('w_backward'):.6f} "
            f"| {mean('csa_forward'):.6f}/{mean('csa_backward'):.6f} "
            f"| {mean('hca_forward'):.6f}/{mean('hca_backward'):.6f} "
            f"| {mean('forward'):.6f}/{mean('backward'):.6f} "
            f"| {mean('parameter_gradient_allreduce'):.6f} "
            f"| {mean('indexer_score'):.6f}/{mean('indexer_topk'):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "| mode | forward kernels/call min-max | backward kernels/call min-max "
            "| forward/backward unique names | routes F+B |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("w", "csa", "hca"):
        metrics = core_kernel_audit["mode_summary"][mode]
        forward_count = metrics["forward_kernel_launch_count"]
        backward_count = metrics["backward_kernel_launch_count"]
        route_count = len(_MODE_ROUTES[mode])
        report_lines.append(
            f"| {mode.upper()} | {forward_count['min']}-{forward_count['max']} "
            f"| {backward_count['min']}-{backward_count['max']} "
            f"| {metrics['forward_unique_kernel_names']}/"
            f"{metrics['backward_unique_kernel_names']} "
            f"| {route_count}+{route_count} |"
        )
    report_lines.extend(
        [
            "",
            "| mode | direction | route | NCCL kernel / NVTX path "
            "| self overlap | mean duration/overlap (us) | mean fraction | 依赖理由 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for mode in ("w", "csa", "hca"):
        for direction in ("forward", "backward"):
            for route in _MODE_ROUTE_ORDER[mode][direction]:
                metrics = attention_communication_overlap["routes"][mode][direction][
                    route
                ]
                report_lines.append(
                    f"| {mode.upper()} | {direction} | {route} "
                    "| `ncclDevKernel_SendRecv` / "
                    f"`{metrics['nvtx_path']}` "
                    f"| {metrics['positive_gpu_overlap_records']}/"
                    f"{metrics['total_records']} "
                    f"| {metrics['kernel_duration_us']['mean']:.3f}/"
                    f"{metrics['overlap_us']['mean']:.3f} "
                    f"| {metrics['overlap_fraction']['mean']:.3f} "
                    f"| {metrics['dependency']['reason']} |"
                )
    report_lines.append("")
    (args.artifact_dir / "REPORT_ATTENTION_SUITE.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
