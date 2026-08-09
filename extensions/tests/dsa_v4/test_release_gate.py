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

import hashlib
import json
import statistics
from pathlib import Path
from typing import cast

import pytest

from scripts.image.finalize_release import (
    _PRO_FLASHMLA_FORWARD_KERNEL,
    _PRO_PAIR_D2D_GROUPS,
    _PRO_PAIR_D2D_SCOPES,
    _PRO_PAIR_HARD_GATE_GROUPS,
    _PRO_PAIR_MAJOR_KERNEL_GROUPS,
    _PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS,
    _PRO_PAIR_ROUTE_GROUPS,
    _PRO_PAIR_STEPS,
    _PRO_PAIR_SUPPORT_GROUPS,
    _PRO_PAIR_WORLD_SIZE,
    _expected_release_image_labels,
    _validate_communication_overlap,
    _validate_correctness,
    _validate_cp1,
    _validate_cp2,
    _validate_cp2_summary,
    _validate_indexer_d2d,
    _validate_major_kernel_balance,
    _validate_manifest,
    _validate_pro_pair_layout,
    _validate_pro_pair_summary,
    _validate_profile,
    _validate_release_image_labels,
    _validate_support_overhead,
    _write_release_summary_and_manifest,
)

from .conftest import find_repo_root

_REVISION = "1" * 40


def _record_list(
    payload: dict[str, object], key: str = "records"
) -> list[dict[str, object]]:
    records = payload[key]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return cast(list[dict[str, object]], records)


def _rank_ranges(
    records: list[dict[str, object]], groups: tuple[str, ...]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for step in range(_PRO_PAIR_STEPS):
        for group in groups:
            values = []
            for record in records:
                if record["group"] != group or record["step"] != step:
                    continue
                gpu_time_ms = record["gpu_time_ms"]
                assert isinstance(gpu_time_ms, (int, float))
                values.append(float(gpu_time_ms))
            mean = statistics.fmean(values)
            rank_range = max(values) - min(values)
            result.append(
                {
                    "group": group,
                    "max": max(values),
                    "mean": mean,
                    "min": min(values),
                    "rank_range": rank_range,
                    "relative_rank_range": rank_range / mean if mean else 0.0,
                    "step": step,
                    "threshold": None,
                    "unit": "ms",
                }
            )
    return result


def _timing_records(
    groups: tuple[str, ...], *, gpu_time_ms: float = 1.0
) -> list[dict[str, object]]:
    return [
        {
            "gpu_time_ms": gpu_time_ms,
            "group": group,
            "rank": rank,
            "step": step,
        }
        for group in groups
        for step in range(_PRO_PAIR_STEPS)
        for rank in range(_PRO_PAIR_WORLD_SIZE)
    ]


def _layout() -> dict[str, object]:
    costs: list[dict[str, object]] = []
    packing: list[dict[str, object]] = []
    for rank in range(_PRO_PAIR_WORLD_SIZE):
        costs.append(
            {
                "chunk_count": 32,
                "csa_duplicate_indexer_k_rows": 2,
                "csa_packed_indexer_k_rows": 6,
                "csa_unique_indexer_k_rows": 4,
                "fragment_count": 1,
                "native_causal_area": 123,
                "query_tokens": 16_384,
                "rank": rank,
            }
        )
        packing.append(
            {
                "duplicate_indexer_k_bytes": 512,
                "duplicate_indexer_k_rows": 2,
                "indexer_k_row_bytes": 256,
                "packed_indexer_k_bytes": 1_536,
                "packed_indexer_k_rows": 6,
                "packing_amplification": 1.5,
                "rank": rank,
                "unique_indexer_k_bytes": 1_024,
                "unique_indexer_k_rows": 4,
            }
        )
    return {
        "indexer_k_packing": packing,
        "query_layout_hash": "a" * 64,
        "query_token_counts": [16_384] * _PRO_PAIR_WORLD_SIZE,
        "rank_costs": costs,
        "rank_results": _PRO_PAIR_WORLD_SIZE,
        "result": "PASS",
    }


def _major_report() -> dict[str, object]:
    records = _timing_records(_PRO_PAIR_MAJOR_KERNEL_GROUPS)
    for record in records:
        name = (
            _PRO_FLASHMLA_FORWARD_KERNEL
            if record["group"] in ("csa_flashmla_forward", "hca_flashmla_forward")
            else "kernel"
        )
        record.update(kernel_launch_count=1, kernel_names=[name])
    ranges = _rank_ranges(records, _PRO_PAIR_MAJOR_KERNEL_GROUPS)
    for record in ranges:
        if record["group"] in _PRO_PAIR_HARD_GATE_GROUPS:
            record.update(passed=True, threshold=0.05)
    return {
        "balance_gate": "indexer_score_topk_0.05_others_report_only",
        "flashmla_forward_kernel": _PRO_FLASHMLA_FORWARD_KERNEL,
        "flashmla_forward_same_exact_variant": True,
        "hard_gate_groups": list(_PRO_PAIR_HARD_GATE_GROUPS),
        "kernel_names": {
            group: [
                (
                    _PRO_FLASHMLA_FORWARD_KERNEL
                    if group in ("csa_flashmla_forward", "hca_flashmla_forward")
                    else "kernel"
                )
            ]
            for group in _PRO_PAIR_MAJOR_KERNEL_GROUPS
        },
        "rank_ranges": ranges,
        "records": records,
        "result": "PASS",
    }


def _d2d_report() -> dict[str, object]:
    records = _timing_records(_PRO_PAIR_D2D_GROUPS, gpu_time_ms=0.0)
    for rowid, record in enumerate(records):
        record.update(
            bytes=0,
            copy_count=0,
            memcpy_activity_count=0,
            rows=[],
            same_mode_external_compute_overlap_fraction=0.0,
            same_mode_external_compute_overlap_ms=0.0,
            same_wrapper_kernel_overlap_fraction=0.0,
            same_wrapper_kernel_overlap_ms=0.0,
            wrapper_kernel_launch_count=1,
            wrapper_nvtx_name=_PRO_PAIR_D2D_SCOPES[str(record["group"])],
            wrapper_nvtx_rowid=rowid,
        )
    return {
        "accounting": "separate_from_kernel_gpu_time",
        "gpu_time_rank_ranges": _rank_ranges(records, _PRO_PAIR_D2D_GROUPS),
        "outside_known_scope": {
            "bytes": 0,
            "copy_count": 0,
            "gpu_time_ms": 0.0,
            "memcpy_activity_count": 0,
            "rows": [],
        },
        "records": records,
        "result": "PASS",
        "total_bytes": 0,
        "total_copy_count": 0,
        "total_gpu_time_ms": 0.0,
    }


def _support_report(layout: dict[str, object]) -> dict[str, object]:
    records = _timing_records(_PRO_PAIR_SUPPORT_GROUPS)
    packing = layout["indexer_k_packing"]
    assert isinstance(packing, list)
    for record in records:
        record["kernel_launch_count"] = 1
        if record["group"] != "csa_grouped_k_pack_forward":
            continue
        rank = record["rank"]
        assert isinstance(rank, int)
        source = packing[rank]
        assert isinstance(source, dict)
        record.update(
            duplicate_bytes=source["duplicate_indexer_k_bytes"],
            duplicate_rows=source["duplicate_indexer_k_rows"],
            external_compute_overlap_fraction=0.5,
            external_compute_overlap_ms=0.5,
            module_nvtx_rowid=rank + 1,
            packed_bytes=source["packed_indexer_k_bytes"],
            packed_rows=source["packed_indexer_k_rows"],
            read_bytes=source["packed_indexer_k_bytes"],
            traffic_bytes=int(source["packed_indexer_k_bytes"]) * 2,
            unique_bytes=source["unique_indexer_k_bytes"],
            unique_rows=source["unique_indexer_k_rows"],
            write_bytes=source["packed_indexer_k_bytes"],
        )
    return {
        "balance_gate": "report_only",
        "grouped_k_pack_backward_csr_reduce_launches": 0,
        "rank_ranges": _rank_ranges(records, _PRO_PAIR_SUPPORT_GROUPS),
        "records": records,
        "result": "PASS",
    }


def _communication_report() -> dict[str, object]:
    records = _timing_records(_PRO_PAIR_ROUTE_GROUPS)
    for index, record in enumerate(records):
        mode, direction, route = str(record["group"]).split(".")
        classification = (
            "overlap_capable"
            if record["group"] in _PRO_PAIR_OVERLAP_CAPABLE_ROUTE_GROUPS
            else "dependency_bound"
        )
        overlap_ns = 500_000 if classification == "overlap_capable" else 0
        start = 10_000 + index * 2_000
        record.update(
            direction=direction,
            kernel_end_ns=start + 1_000_000,
            kernel_start_ns=start,
            mode=mode,
            nvtx_path=[
                {
                    "name": (
                        "magi_dsa::phase::collective_all2all_v::"
                        f"attention::{mode}::{route}.{direction}"
                    )
                }
            ],
            observed_positive_same_mode_compute_overlap=overlap_ns > 0,
            overlap_classification=classification,
            overlap_contract_reason="frozen dependency contract",
            other_mode_compute_overlap_fraction=0.0,
            other_mode_compute_overlap_ms=0.0,
            route=route,
            runtime_start_ns=start - 100,
            same_mode_compute_overlap_fraction=(
                0.5 if classification == "overlap_capable" else 0.0
            ),
            same_mode_compute_overlap_ms=overlap_ns / 1_000_000.0,
            same_mode_compute_overlap_ns=overlap_ns,
        )
    return {
        "cross_mode_compute_overlap_is_hard_gate": True,
        "expected_sendrecv": "CSA=4F+4B,HCA=3F+3B,total=7F+7B",
        "overlap_contract": {
            "classification_counts": {
                "dependency_bound": 160,
                "overlap_capable": 400,
            },
            "dependency_bound_requires_positive_overlap": False,
            "fraction_threshold": None,
            "overlap_capable_requires_positive_overlap": False,
            "positive_time_threshold_ns": None,
        },
        "overlap_gate": "report_only",
        "rank_ranges": _rank_ranges(records, _PRO_PAIR_ROUTE_GROUPS),
        "records": records,
        "result": "PASS",
    }


def _summary(
    layout: dict[str, object],
    major: dict[str, object],
    d2d: dict[str, object],
    support: dict[str, object],
    communication: dict[str, object],
) -> dict[str, object]:
    communication_records = _record_list(communication)
    return {
        "attention_order": ["csa", "hca"],
        "backward_order": ["hca", "csa"],
        "excluded_capture_ranges": {
            "loss": 0,
            "projection": 0,
            "token_layout": 0,
            "w_mode": 0,
        },
        "expected_sendrecv": {
            "backward": 7,
            "csa_backward": 4,
            "csa_forward": 4,
            "forward": 7,
            "hca_backward": 3,
            "hca_forward": 3,
        },
        "flashmla_forward_kernel": major["flashmla_forward_kernel"],
        "flashmla_forward_same_exact_variant": major[
            "flashmla_forward_same_exact_variant"
        ],
        "grouped_k_pack_backward_csr_reduce_launches": 0,
        "independent_attention_graphs": True,
        "indexer_d2d": {
            "outside_known_scope": d2d["outside_known_scope"],
            "total_bytes": d2d["total_bytes"],
            "total_copy_count": d2d["total_copy_count"],
            "total_gpu_time_ms": d2d["total_gpu_time_ms"],
        },
        "kernel_attribution_coverage": 1.0,
        "kernel_attribution_records": 1,
        "layout": layout,
        "major_kernel_balance_gate": ("indexer_score_topk_0.05_others_report_only"),
        "major_kernel_groups": list(_PRO_PAIR_MAJOR_KERNEL_GROUPS),
        "memcpy_attribution_coverage": 1.0,
        "memcpy_attribution_records": 0,
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
        "route_timing_records": len(communication_records),
        "runtime_parameter_gradient_communication": False,
        "shared_bundle_query_layout_hash": layout["query_layout_hash"],
        "shared_source_packed_meta": True,
        "shared_source_x": True,
        "step_mode": "pro-pair",
        "steps": _PRO_PAIR_STEPS,
        "support_overhead_groups": list(_PRO_PAIR_SUPPORT_GROUPS),
        "token_layout_invocations": 1,
        "world_size": _PRO_PAIR_WORLD_SIZE,
    }


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _phase_audit(world_size: int) -> dict[str, object]:
    return {
        "all_ranks_execute_ended": True,
        "all_ranks_execute_started": True,
        "collective_stall_confirmed": False,
        "error_count": 0,
        "failure_reasons": [],
        "latest_open_phases": [None] * world_size,
        "open_phase_count": 0,
        "ranks": [
            {
                "errors": [],
                "execute_ended": True,
                "execute_started": True,
                "open_phases": [],
                "rank": rank,
            }
            for rank in range(world_size)
        ],
        "result": "PASS",
        "timed_out": False,
        "world_size": world_size,
    }


def _minimal_provenance_artifact(root: Path, *, dirty_status: str) -> None:
    root.mkdir()
    (root / "DIRTY_STATUS.txt").write_text(dirty_status, encoding="utf-8")
    (root / "SOURCE_REVISION.txt").write_text(_REVISION + "\n", encoding="utf-8")
    _write_manifest(root)


def _installed_cp8_artifact(
    root: Path,
    *,
    phase_audit: dict[str, object],
) -> None:
    root.mkdir(exist_ok=True)
    (root / "DIRTY_STATUS.txt").write_text("\n", encoding="utf-8")
    (root / "SOURCE_REVISION.txt").write_text(_REVISION + "\n", encoding="utf-8")
    (root / "PHASE_AUDIT.json").write_text(
        json.dumps(phase_audit) + "\n", encoding="utf-8"
    )
    (root / "SUMMARY.json").write_text("{}\n", encoding="utf-8")
    for rank in range(8):
        report = {
            "case": "cp8-natural-backward",
            "execution_seconds": 1.0,
            "installed_wheel": {
                "package_path": (
                    "/usr/local/lib/python3.12/site-packages/"
                    "magi_attention/__init__.py"
                ),
                "package_version": f"1.1.1+g{_REVISION}",
                "extension_path": (
                    "/usr/local/lib/python3.12/site-packages/"
                    "magi_attn_extensions/DSA/__init__.py"
                ),
                "extension_version": "1.1.0",
                "source_revision": _REVISION,
            },
            "rank": rank,
        }
        (root / f"result_rank{rank}.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
    _write_manifest(root)


def _cp1_artifact(root: Path) -> None:
    installed = {
        "package_path": "/usr/local/lib/python3.12/site-packages/magi_attention/__init__.py",
        "package_version": f"1.1.1+g{_REVISION}",
    }
    command = {
        "artifact_dir": str(root),
        "case": "cp1-kernel",
        "image": "magi-dsa:test",
        "package_import": "installed-wheel",
        "pytest": "extensions/tests/dsa_v4/test_cp1_kernel.py",
        "pytest_basetemp": "/magi-cache/pytest-tmp/run",
        "pytest_import_mode": "importlib",
        "pytest_rootdir": "/magi-cache/pytest-root",
        "source_mount": "read-only",
        "source_revision": _REVISION,
        "timeout_seconds": "1800",
    }
    image_contract = {
        "flashmla_base_revision": "9241ae3ef9bac614dd25e45e507e089f888280e0",
        "flashmla_dual_lse_patch_revision": (
            "13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece"
        ),
        "flashmla_dual_lse_patch_sha256": (
            "6957dbde516c73066c5911108761325edc1bdcd8f62e15dc0a84f4f290118d4b"
        ),
        "flashmla_pro_h128_patch_revision": (
            "b7643bd54521f563b839b98289b5cd048c062ba2"
        ),
        "flashmla_pro_h128_patch_sha256": (
            "c534e13ff432ac1c694cb24981826c11be26a2d9743d7175ddb05f887279461f"
        ),
        "cudnn_backend_version": "9.24.0.43",
        "cudnn_frontend_version": "1.26.0",
        "cudnn_frontend_revision": ("35fd7b0d0e1d4952b904c79341c5e84e3af0a328"),
        "cudnn_frontend_source": "official-unmodified",
        "cudnn_frontend_local_patches": "none",
        "cutlass_dsl_version": "4.5.0",
        "quack_version": "0.4.1",
        "tvm_ffi_version": "0.1.8.post0",
        "magi_source_revision": _REVISION,
        "install_mode": "python-wheel",
        "validation": "all_required_image_labels_exact",
    }
    (root / "COMMAND.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in command.items()),
        encoding="utf-8",
    )
    (root / "DIRTY_STATUS.txt").write_text("\n", encoding="utf-8")
    (root / "IMAGE.json").write_text("{}\n", encoding="utf-8")
    (root / "IMAGE_CONTRACT.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in image_contract.items()),
        encoding="utf-8",
    )
    (root / "INSTALLED_PACKAGE.json").write_text(
        json.dumps(installed) + "\n", encoding="utf-8"
    )
    xml = '<testsuites tests="6" failures="0" errors="0" skipped="0" />\n'
    (root / "PYTEST.xml").write_text(xml, encoding="utf-8")
    counts = {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}
    raw = {
        "case": "cp1-kernel",
        "junit_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "pytest_exit_status": 0,
        **counts,
    }
    summary = {
        "case": "cp1-kernel",
        "image": "magi-dsa:test",
        "image_contract": "PASS",
        "image_id": "sha256:test",
        "installed_package": installed,
        "pytest_exit_status": 0,
        "result": "PASS",
        "source_dirty": False,
        "source_revision": _REVISION,
        **counts,
    }
    for name, payload in (("RAW.json", raw), ("SUMMARY.json", summary)):
        (root / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "SOURCE_REVISION.txt").write_text(_REVISION + "\n", encoding="utf-8")
    (root / "STDERR.txt").write_text("", encoding="utf-8")
    (root / "STDOUT.txt").write_text("6 passed\n", encoding="utf-8")
    _write_manifest(root)


def test_release_gate_rejects_legacy_base_profile(tmp_path: Path) -> None:
    (tmp_path / "DIRTY_STATUS.txt").write_text("\n", encoding="utf-8")
    (tmp_path / "SOURCE_REVISION.txt").write_text(_REVISION + "\n", encoding="utf-8")
    (tmp_path / "SUMMARY.json").write_text(
        json.dumps(
            {
                "balanced_5pct_gate": True,
                "correctness": "PASS",
                "profile_records": 160,
                "result": "PASS",
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="SUMMARY_PRO_PAIR"):
        _validate_profile(tmp_path, _REVISION)


def test_formal_artifact_gates_reject_dirty_worktrees(tmp_path: Path) -> None:
    validators = (
        (_validate_cp2, "cp2"),
        (_validate_profile, "profile"),
        (_validate_correctness, "installed_cp8"),
    )
    for validator, directory_name in validators:
        artifact = tmp_path / directory_name
        _minimal_provenance_artifact(
            artifact,
            dirty_status=" M extensions/magi_attn_extensions/DSA/runtime.py\n",
        )
        with pytest.raises(ValueError, match="dirty worktree"):
            validator(artifact, _REVISION)


def test_installed_cp8_gate_validates_phase_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural = {"result": "PASS"}
    monkeypatch.setattr(
        "scripts.image.finalize_release._validate_correctness_summary",
        lambda _: {"result": "PASS", "structural_contract": structural},
    )
    monkeypatch.setattr(
        "scripts.image.finalize_release._validate_cp8_structural_reports",
        lambda _: structural,
    )

    invalid_phase_audit = _phase_audit(8)
    invalid_phase_audit["result"] = "FAIL"
    _installed_cp8_artifact(tmp_path, phase_audit=invalid_phase_audit)
    with pytest.raises(ValueError, match="installed-wheel CP8 phase audit"):
        _validate_correctness(tmp_path, _REVISION)

    (tmp_path / "PHASE_AUDIT.json").write_text(
        json.dumps(_phase_audit(8)) + "\n", encoding="utf-8"
    )
    _write_manifest(tmp_path)
    assert _validate_correctness(tmp_path, _REVISION)["result"] == "PASS"


def test_release_gate_rejects_forged_major_kernel_pass() -> None:
    report = _major_report()
    records = _record_list(report)
    target = next(
        record
        for record in records
        if record["group"] == "csa_indexer_score"
        and record["step"] == 0
        and record["rank"] == 7
    )
    target["gpu_time_ms"] = 1.2
    report["rank_ranges"] = _rank_ranges(records, _PRO_PAIR_MAJOR_KERNEL_GROUPS)
    for record in _record_list(report, "rank_ranges"):
        if record["group"] in _PRO_PAIR_HARD_GATE_GROUPS:
            record.update(passed=True, threshold=0.05)
    with pytest.raises(ValueError, match="does not pass the 5% gate"):
        _validate_major_kernel_balance(report)


def test_release_gate_rejects_flashmla_forward_variant_spoofs() -> None:
    report = _major_report()
    report["flashmla_forward_kernel"] = "flash_mla_with_kvcache"
    with pytest.raises(ValueError, match="major-kernel balance summary did not pass"):
        _validate_major_kernel_balance(report)

    report = _major_report()
    record = next(
        item for item in _record_list(report) if item["group"] == "csa_flashmla_forward"
    )
    record["kernel_names"] = ["sparse_attn_fwd_for_small_topk_kernel_v2"]
    with pytest.raises(ValueError, match="FlashMLA forward variant differs"):
        _validate_major_kernel_balance(report)


def test_release_gate_rejects_indexer_d2d_spoofs() -> None:
    report = _d2d_report()
    _record_list(report)[0]["wrapper_nvtx_name"] = "forged_scope"
    with pytest.raises(ValueError, match="wrapper scope differs"):
        _validate_indexer_d2d(report)

    report = _d2d_report()
    report["total_bytes"] = 1
    with pytest.raises(ValueError, match="aggregate totals differ"):
        _validate_indexer_d2d(report)

    report = _d2d_report()
    report["outside_known_scope"] = {
        "bytes": 256,
        "copy_count": 1,
        "gpu_time_ms": 0.1,
        "memcpy_activity_count": 1,
        "rows": [{}],
    }
    with pytest.raises(ValueError, match="outside known scopes"):
        _validate_indexer_d2d(report)


def test_release_gate_rejects_support_and_route_spoofs() -> None:
    layout = _layout()
    support = _support_report(layout)
    support["grouped_k_pack_backward_csr_reduce_launches"] = 1
    with pytest.raises(ValueError, match="support-overhead summary did not pass"):
        _validate_support_overhead(support, _validate_pro_pair_layout(layout))

    communication = _communication_report()
    _record_list(communication)[0]["other_mode_compute_overlap_ms"] = 0.25
    with pytest.raises(ValueError, match="overlaps other-mode compute"):
        _validate_communication_overlap(communication)

    communication = _communication_report()
    csa_window = next(
        record
        for record in _record_list(communication)
        if record["group"] == "csa.forward.WINDOW_KV"
        and record["rank"] == 0
        and record["step"] == 0
    )
    csa_overlap_x = next(
        record
        for record in _record_list(communication)
        if record["group"] == "csa.forward.OVERLAP_X"
        and record["rank"] == 0
        and record["step"] == 0
    )
    csa_window["runtime_start_ns"] = cast(int, csa_overlap_x["runtime_start_ns"]) + 1
    with pytest.raises(ValueError, match="route launch order differs"):
        _validate_communication_overlap(communication)

    communication = _communication_report()
    overlap_capable = next(
        record
        for record in _record_list(communication)
        if record["overlap_classification"] == "overlap_capable"
    )
    overlap_capable.update(
        observed_positive_same_mode_compute_overlap=False,
        same_mode_compute_overlap_fraction=0.0,
        same_mode_compute_overlap_ms=0.0,
        same_mode_compute_overlap_ns=0,
    )
    assert _validate_communication_overlap(communication)["result"] == "PASS"


def test_release_gate_rejects_stale_structural_layout_hash() -> None:
    layout = _layout()
    major = _major_report()
    d2d = _d2d_report()
    support = _support_report(layout)
    communication = _communication_report()
    validated_layout = _validate_pro_pair_layout(layout)
    _validate_major_kernel_balance(major)
    _validate_indexer_d2d(d2d)
    _validate_support_overhead(support, validated_layout)
    _validate_communication_overlap(communication)
    summary = _summary(layout, major, d2d, support, communication)
    summary["shared_bundle_query_layout_hash"] = "b" * 64
    with pytest.raises(ValueError, match="shared Query layout hash differs"):
        _validate_pro_pair_summary(
            summary,
            major=major,
            d2d=d2d,
            support=support,
            communication=communication,
        )


def test_cp2_gate_rejects_missing_artifact_and_wrong_case(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CP2 artifact directory"):
        _validate_cp2(tmp_path / "missing", _REVISION)

    summary = {
        "case": "smoke",
        "result_count": 2,
        "results": [],
        "world_size": 2,
    }
    with pytest.raises(ValueError, match="case or rank grid differs"):
        _validate_cp2_summary(summary)


def test_cp1_gate_validates_real_artifact_and_rejects_case_spoof(
    tmp_path: Path,
) -> None:
    _cp1_artifact(tmp_path)
    assert _validate_cp1(tmp_path, _REVISION)["result"] == "PASS"
    summary_path = tmp_path / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["case"] = "smoke"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="cp1-kernel contract"):
        _validate_cp1(tmp_path, _REVISION)


def test_cp1_gate_rejects_a_partial_manifest(tmp_path: Path) -> None:
    _cp1_artifact(tmp_path)
    manifest = tmp_path / "SHA256SUMS"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text(
        "\n".join(line for line in lines if not line.endswith("  COMMAND.txt")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest file grid differs"):
        _validate_cp1(tmp_path, _REVISION)


def test_release_gate_requires_exact_official_image_labels() -> None:
    labels = _expected_release_image_labels(_REVISION)
    _validate_release_image_labels(labels, _REVISION)
    labels["org.magi-dsa.cudnn-frontend-local-patches"] = "forged-patch"
    with pytest.raises(ValueError, match="release image labels differ"):
        _validate_release_image_labels(labels, _REVISION)


def test_cp1_has_a_separate_artifact_entrypoint() -> None:
    repo_root = find_repo_root()
    cp1 = (repo_root / "scripts/test/run_cp1.sh").read_text(encoding="utf-8")
    multigpu = (repo_root / "scripts/test/run_multigpu.sh").read_text(encoding="utf-8")
    for marker in (
        "case=cp1-kernel",
        "extensions/tests/dsa_v4/test_cp1_kernel.py",
        "pytest_import_mode=importlib",
        "--import-mode=importlib",
        "PYTEST.xml",
        "RAW.json",
        "SUMMARY.json",
        "SOURCE_REVISION.txt",
        "IMAGE_CONTRACT.txt",
        "SHA256SUMS",
    ):
        assert marker in cp1
    chmod_block = cp1.split("chmod 0777 \\\n", maxsplit=1)[1].split(
        "\n\nfailed=1", maxsplit=1
    )[0]
    for marker in ('"$artifact_dir"', '"$cache_dir"', '"$cache_dir/pytest-tmp"'):
        assert marker in chmod_block
    assert "cp1-kernel" not in multigpu


def test_release_orchestrator_builds_once_and_reuses_one_image_id() -> None:
    repo_root = find_repo_root()
    source = (repo_root / "scripts/image/run_release.sh").read_text(encoding="utf-8")

    assert source.count('bash "$repo_root/scripts/image/build.sh"') == 1
    assert "provided_artifacts != 0 && provided_artifacts != 3" in source
    assert "generate_artifacts=1" in source
    for command in (
        'bash "$repo_root/scripts/test/run_cp1.sh"',
        "--world-size 2",
        "--case csa-natural-backward",
        "--world-size 8",
        "--case cp8-natural-backward",
        'bash "$repo_root/scripts/profile/run_5step.sh"',
        "--case dsv4-pro-128k",
        "--step-mode pro-pair",
        "--layout-policy structural-balanced",
    ):
        assert command in source
    assert source.count("--installed-wheel") >= 4
    assert "release_image_id=" in source
    assert 'current_image_id="$(docker image inspect' in source
    assert 'current_image_id" != "$release_image_id' in source
    for kind in ("cp1", "cp2", "cp8", "profile"):
        assert f"require_artifact_image_id {kind}" in source
    assert 'tee "$artifact_dir/FINALIZE.stdout"' not in source
    assert source.index("release stage=cp1") < source.index(
        "release stage=installed-wheel-cp2"
    )
    assert source.index("release stage=installed-wheel-cp2") < source.index(
        "release stage=installed-wheel-cp8"
    )
    assert source.index("release stage=installed-wheel-cp8") < source.index(
        "release stage=pro-pair-profile"
    )
    assert source.index("release stage=pro-pair-profile") < source.index(
        "release stage=finalize"
    )


def test_release_summary_stdout_is_covered_by_final_manifest(tmp_path: Path) -> None:
    summary: dict[str, object] = {"result": "PASS", "revision": _REVISION}
    payload = _write_release_summary_and_manifest(tmp_path, summary)

    assert (tmp_path / "SUMMARY.json").read_text(encoding="utf-8") == payload
    assert (tmp_path / "FINALIZE.stdout").read_text(encoding="utf-8") == payload
    assert json.loads(payload) == summary
    manifest = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "  SUMMARY.json\n" in manifest
    assert "  FINALIZE.stdout\n" in manifest
    _validate_manifest(tmp_path)

    (tmp_path / "FINALIZE.stdout").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="command failed"):
        _validate_manifest(tmp_path)


def test_release_report_uses_q16_backend_native_contract() -> None:
    repo_root = find_repo_root()
    source = (repo_root / "scripts/image/finalize_release.py").read_text(
        encoding="utf-8"
    )
    report_source = source[
        source.index("def _report(") : source.index("def _write_manifest(")
    ]
    assert "Q16" in report_source
    assert "backend-native Top-K IDs" in report_source
    assert "canonical global compressed-block ID \u5347\u5e8f" not in report_source
    assert "Q15 exact raw-score tie" not in report_source
