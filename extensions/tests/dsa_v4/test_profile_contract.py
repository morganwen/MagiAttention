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
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Protocol, cast

import magi_attn_extensions.DSA.dist as dist_dsa_module
import pytest
import torch
from magi_attn_extensions.DSA import modeling as dsa_layer_module
from magi_attn_extensions.DSA.config import DsaRatio, MagiDSAConfig
from magi_attn_extensions.DSA.modeling import MagiDSALayer
from magi_attn_extensions.DSA.nvtx import (
    DSA_CUDNN_CALL_NVTX_PREFIX,
    DSA_MODULE_NVTX_PREFIX,
    dsa_cudnn_call_range,
    dsa_nvtx_range,
)
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr
from magi_attn_extensions.DSA.types import (
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)

from benchmarks.dsa_v4 import profile_5step as profile_5step_module
from benchmarks.dsa_v4 import profile_attention_suite as attention_suite_module
from benchmarks.dsa_v4.profile_5step import (
    _all_reduce_training_gradients,
    _clear_training_gradients,
    _compare_results,
    _compare_training_gradients,
    _make_plan_input,
    _prepare_profile_dsa_input_boundary,
    _ProfileDSAInputBoundary,
    _ProfileSource,
    _run_forward_backward_step,
    _source_order_tensor,
    _topk_diagnostics,
    _training_gradient_finite_diagnostics,
)
from benchmarks.dsa_v4.profile_attention_suite import (
    _all_reduce_attention_suite_gradients,
    _attention_gradient_snapshot,
    _AttentionProfileCase,
    _calc_attention_case,
    _prepare_pro_pair_boundaries,
    _run_attention_suite_step,
    _validate_mode_result,
)
from scripts.image.finalize_release import (
    _validate_correctness_summary,
    _validate_cp8_structural_reports,
)
from scripts.profile import extract_nsys as extract_nsys_module
from scripts.profile.extract_nsys import (
    extract_kernel_attribution_records,
    extract_memcpy_attribution_records,
    extract_profile_records,
)
from scripts.profile.summarize_5step import compute_rank_ranges, validate_phase_records
from scripts.profile.summarize_pro_pair import (
    _validate_pro_runtime_bundle_payload,
    compute_compressor_timings,
    compute_major_kernel_timings,
    compute_pro_pair_phase_rank_ranges,
    compute_route_timings,
    compute_support_overhead_timings,
    extract_indexer_d2d,
    validate_mode_serialization,
    validate_pro_pair_records,
    validate_pro_pair_sendrecv,
)

from .conftest import find_repo_root


def _small_config(ratio: DsaRatio) -> MagiDSAConfig:
    return MagiDSAConfig(
        ratio=ratio,
        hidden_size=8,
        q_lora_rank=6,
        num_query_heads=2,
        head_dim=8,
        rope_dim=4,
        indexer_heads=2,
        indexer_head_dim=4,
        indexer_topk=3,
        window_size=3,
        original_seq_len=0,
        compress_rope_theta=10000.0,
        indexer_atom_size=4,
    )


class _StringEventSink(Protocol):
    def append(self, value: str, /) -> None:
        ...


def test_dsa_module_nvtx_range_uses_stable_prefix(monkeypatch) -> None:
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda name: events.append(("push", name)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: events.append(("pop", None)),
    )

    with dsa_nvtx_range("compressor::main"):
        pass
    with dsa_cudnn_call_range("indexer_score"):
        pass
    with dsa_cudnn_call_range("indexer_topk"):
        pass
    with dsa_nvtx_range("disabled", enabled=False):
        pass

    assert events == [
        ("push", f"{DSA_MODULE_NVTX_PREFIX}compressor::main"),
        ("pop", None),
        ("push", f"{DSA_CUDNN_CALL_NVTX_PREFIX}indexer_score"),
        ("pop", None),
        ("push", f"{DSA_CUDNN_CALL_NVTX_PREFIX}indexer_topk"),
        ("pop", None),
    ]


def test_source_order_tensor_pads_scalar_rows_for_vectorized_copy(
    monkeypatch,
) -> None:
    seen_shapes: list[tuple[int, ...]] = []

    def fake_unlayout(value, route, group):
        del route, group
        seen_shapes.append(tuple(value.shape))
        return value.flip(0)

    monkeypatch.setattr(
        "benchmarks.dsa_v4.profile_5step.unlayout_dsa_query_tensor",
        fake_unlayout,
    )
    handle = SimpleNamespace(device_plan=SimpleNamespace(token_layout_route=object()))
    value = torch.tensor([3, 5, 7], dtype=torch.int32)
    actual = _source_order_tensor(value, handle)
    torch.testing.assert_close(actual, torch.tensor([7, 5, 3], dtype=torch.int32))
    assert seen_shapes == [(3, 4)]


def test_compressor_and_indexer_nvtx_names_are_nested(monkeypatch) -> None:
    entered: list[str] = []
    parents: dict[str, tuple[str, ...]] = {}
    stack: list[str] = []

    @contextmanager
    def record_range(name: str, *, enabled: bool = True) -> Iterator[None]:
        del enabled
        parents[name] = tuple(stack)
        entered.append(name)
        stack.append(name)
        try:
            yield
        finally:
            assert stack.pop() == name

    monkeypatch.setattr(dsa_layer_module, "dsa_nvtx_range", record_range)
    config = _small_config(4)
    layer = MagiDSALayer(config)
    assert layer.compressor is not None and layer.indexer is not None
    packed_x = torch.randn(2, config.compressor_support, config.hidden_size)
    valid_rows = torch.ones(2, config.compressor_support, dtype=torch.bool)
    positions = torch.tensor([0, 4], dtype=torch.int32)

    layer.compressor(packed_x, valid_rows, positions)
    layer.indexer.compressor(packed_x, valid_rows, positions)
    layer.indexer.project_queries(
        torch.randn(2, config.hidden_size),
        torch.randn(2, config.q_lora_rank),
        positions,
        detach_trunk=False,
    )

    compressor_operations = {
        "projection_input_cast",
        "value_projection",
        "gate_projection",
        "overlap_assembly",
        "validity_mask",
        "gate_softmax",
        "compression_hadamard",
        "support_reduction",
        "rms_norm",
        "rms_norm::variance_reduction",
        "rms_norm::inverse_root",
        "rms_norm::normalize_hadamard",
        "rms_norm::scale_hadamard",
        "rope",
        "rope::angle_generation",
        "rope::sincos",
        "rope::rotation_hadamard",
        "rope::output_assembly",
    }
    for branch in ("main", "indexer"):
        parent = f"compressor::{branch}"
        assert parent in entered
        for operation in compressor_operations:
            name = f"{parent}::{operation}"
            assert name in entered
            assert parent in parents[name]

    indexer_operations = {
        "query_projection",
        "query_cast",
        "query::rope",
        "query::rope::angle_generation",
        "query::rope::sincos",
        "query::rope::rotation_hadamard",
        "query::rope::output_assembly",
        "weight_projection",
        "weight_scaling",
    }
    assert "indexer_projection" in entered
    for operation in indexer_operations:
        name = f"indexer_projection::{operation}"
        assert name in entered
        assert "indexer_projection" in parents[name]
    assert not stack


def test_hca_compressor_marks_nonoverlap_assembly(monkeypatch) -> None:
    entered: list[str] = []

    @contextmanager
    def record_range(name: str, *, enabled: bool = True) -> Iterator[None]:
        del enabled
        entered.append(name)
        yield

    monkeypatch.setattr(dsa_layer_module, "dsa_nvtx_range", record_range)
    config = _small_config(128)
    layer = MagiDSALayer(config)
    assert layer.compressor is not None
    layer.compressor(
        torch.randn(1, config.compressor_support, config.hidden_size),
        torch.ones(1, config.compressor_support, dtype=torch.bool),
        torch.tensor([0], dtype=torch.int32),
    )
    assert "compressor::main::nonoverlap_assembly" in entered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hca_backward_branch_order_reconnects_source_gradient(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, object, object, str]] = []
    transfer = object()
    route = object()
    group = object()

    @contextmanager
    def record_range(name: str, *, enabled: bool = True) -> Iterator[None]:
        del enabled
        assert name == "attention::hca::backward_overlap::window_reverse_priority"
        yield

    def fake_start(grad, actual_route, actual_group, *, attention_mode):
        calls.append((grad.clone(), actual_route, actual_group, attention_mode))
        return transfer

    monkeypatch.setattr(dist_dsa_module, "dsa_nvtx_range", record_range)
    monkeypatch.setattr(dist_dsa_module, "start_dsa_reverse_route", fake_start)
    monkeypatch.setattr(
        dist_dsa_module,
        "finish_dsa_reverse_route",
        lambda actual_transfer: (
            torch.full((3, 4), 7.0, device="cuda")
            if actual_transfer is transfer
            else None
        ),
    )

    compressed_local = torch.randn(2, 4, device="cuda", requires_grad=True)
    source = torch.randn(3, 4, device="cuda", requires_grad=True)
    old_consumer = torch.randn(5, 4, device="cuda", requires_grad=True)
    compressed, consumer = dist_dsa_module._HcaBackwardBranchOrderFunction.apply(
        compressed_local,
        source,
        old_consumer,
        route,
        group,
        torch.cuda.Stream(),
        torch.cuda.current_stream(),
    )
    torch.autograd.backward(
        (compressed, consumer),
        (torch.ones_like(compressed), torch.ones_like(consumer)),
    )
    torch.cuda.synchronize()

    assert len(calls) == 1
    grad, actual_route, actual_group, attention_mode = calls[0]
    torch.testing.assert_close(grad, torch.ones_like(old_consumer))
    assert actual_route is route
    assert actual_group is group
    assert attention_mode == "hca"
    torch.testing.assert_close(compressed_local.grad, torch.ones_like(compressed_local))
    torch.testing.assert_close(source.grad, torch.full_like(source, 7.0))
    assert old_consumer.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_csa_backward_route_order_is_explicit_and_reconnects_sources(
    monkeypatch,
) -> None:
    events: list[tuple[str, object, str]] = []
    overlap_route = object()
    window_route = object()
    group = object()

    @contextmanager
    def record_range(name: str, *, enabled: bool = True) -> Iterator[None]:
        del enabled
        assert name == "attention::csa::backward_overlap::overlap_then_window_reverse"
        yield

    def fake_start(grad, route, actual_group, *, attention_mode):
        del grad
        assert actual_group is group
        events.append(("start", route, attention_mode))
        return route

    def fake_finish(transfer):
        events.append(("finish", transfer, "csa"))
        if transfer is overlap_route:
            return torch.full((3, 4), 3.0, device="cuda")
        if transfer is window_route:
            return torch.full((2, 4), 5.0, device="cuda")
        raise AssertionError("unexpected reverse transfer")

    monkeypatch.setattr(dist_dsa_module, "dsa_nvtx_range", record_range)
    monkeypatch.setattr(dist_dsa_module, "start_dsa_reverse_route", fake_start)
    monkeypatch.setattr(dist_dsa_module, "finish_dsa_reverse_route", fake_finish)

    overlap_source = torch.randn(3, 4, device="cuda", requires_grad=True)
    window_source = torch.randn(2, 4, device="cuda", requires_grad=True)
    old_overlap_consumer = torch.randn(5, 4, device="cuda", requires_grad=True)
    old_window_consumer = torch.randn(6, 4, device="cuda", requires_grad=True)
    (
        overlap_consumer,
        window_consumer,
    ) = dist_dsa_module._CsaBackwardRouteOrderFunction.apply(
        overlap_source,
        window_source,
        old_overlap_consumer,
        old_window_consumer,
        overlap_route,
        window_route,
        group,
        torch.cuda.Stream(),
        torch.cuda.current_stream(),
    )
    torch.autograd.backward(
        (overlap_consumer, window_consumer),
        (torch.ones_like(overlap_consumer), torch.ones_like(window_consumer)),
    )
    torch.cuda.synchronize()

    assert events == [
        ("start", overlap_route, "csa"),
        ("start", window_route, "csa"),
        ("finish", overlap_route, "csa"),
        ("finish", window_route, "csa"),
    ]
    torch.testing.assert_close(
        overlap_source.grad, torch.full_like(overlap_source, 3.0)
    )
    torch.testing.assert_close(window_source.grad, torch.full_like(window_source, 5.0))
    assert old_overlap_consumer.grad is None
    assert old_window_consumer.grad is None


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


def _release_cp8_plan_evidence(label: str, rank: int) -> dict[str, object]:
    policies = {
        "csa_balanced": "indexer_balanced",
        "csa_sequential": "sequential",
        "csa_structural": "structural_balanced",
        "hca": "sequential",
        "hca_structural": "structural_balanced",
        "window": "sequential",
    }
    policy = policies[label]
    ratio = 4 if label.startswith("csa_") else 128 if label.startswith("hca") else 0
    result: dict[str, object] = {
        "declared_local_token_capacity": 32,
        "fragment_count": 1,
        "local_query_tokens": 32,
        "local_source_tokens": 32,
        "plan_hash": format(list(policies).index(label) + 1, "064x"),
        "policy": policy,
        "query_layout_hash": "a" * 64,
        "query_token_counts": [32] * 8,
        "rank_query_layout_signature": format(rank, "064x"),
        "ratio": ratio,
        "source_token_counts": [32] * 8,
        "structural_layout_config": None,
        "structural_layout_metrics": None,
        "structural_rank_cost": None,
    }
    if policy == "structural_balanced":
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
            structural_rank_cost={"query_tokens": 32, "rank": rank},
        )
    return result


def _release_cp8_result(rank: int) -> dict[str, object]:
    return {
        "plan_evidence": {
            label: _release_cp8_plan_evidence(label, rank)
            for label in (
                "csa_balanced",
                "csa_sequential",
                "csa_structural",
                "hca",
                "hca_structural",
                "window",
            )
        },
        "rank": rank,
        "structural_layout_shared": True,
        "structural_query_layout_hash": "a" * 64,
    }


def test_release_finalizer_derives_pass_from_distributed_summary_contract() -> None:
    results = [_release_cp8_result(rank) for rank in range(8)]
    summary = {
        "case": "cp8-natural-backward",
        "execution_seconds": {"max": 0.08, "min": 0.07},
        "model_parameter_value_check_ranks": [0],
        "result_count": 8,
        "results": results,
        "structural_contract": _validate_cp8_structural_reports(results),
        "world_size": 8,
    }
    assert _validate_correctness_summary(summary)["result"] == "PASS"
    summary["execution_seconds"] = {"max": 60.0, "min": 0.07}
    with pytest.raises(ValueError, match="deadline"):
        _validate_correctness_summary(summary)


def test_release_finalizer_rejects_structural_policy_spoof() -> None:
    results = [_release_cp8_result(rank) for rank in range(8)]
    plans = results[3]["plan_evidence"]
    assert isinstance(plans, dict)
    csa = plans["csa_structural"]
    assert isinstance(csa, dict)
    csa["policy"] = "indexer_balanced"
    summary = {
        "case": "cp8-natural-backward",
        "execution_seconds": {"max": 0.08, "min": 0.07},
        "model_parameter_value_check_ranks": [0],
        "result_count": 8,
        "results": results,
        "structural_contract": {"result": "PASS"},
        "world_size": 8,
    }
    with pytest.raises(ValueError, match="plan policy differs"):
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


def test_hca_mode_result_returns_a_valid_non_indexer_report() -> None:
    result = MagiDSAForwardResult(
        output=torch.zeros(2, 1, 1),
        kl=torch.zeros(()),
        sparse_lse=torch.zeros(2, 1),
        topk_ids=torch.empty(2, 0, dtype=torch.int32),
        topk_length=torch.zeros(2, dtype=torch.int32),
        indexer_lse=torch.full((2,), float("-inf")),
    )

    assert _validate_mode_result("hca", result) == {
        "indexer": False,
        "output_finite": True,
        "sparse_lse_finite": True,
        "topk_backend_native_valid": True,
    }


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


def test_backend_native_shadow_exempts_only_set_changed_output_rows() -> None:
    target = _topk_result([[7, 3, -1], [8, 2, -1]], [2, 2])
    changed = _topk_result([[3, 6, -1], [2, 8, -1]], [2, 2])
    metrics = _compare_results(target, changed)
    assert metrics["topk_backend_native_valid"] is True
    assert metrics["topk_length_exact"] is True
    assert metrics["ordered_topk_exact"] is False
    assert metrics["canonical_topk_exact"] is False

    changed.output[0, 0, 0] = 1.0
    metrics = _compare_results(target, changed)
    assert metrics["output_tie_exempt_rows"] == 1
    assert metrics["output_compared_rows"] == 1
    assert metrics["output_max_abs"] == 1.0
    assert metrics["output_non_tie_max_abs"] == 0.0

    changed.output[1, 0, 0] = 1.0
    with pytest.raises(AssertionError):
        _compare_results(target, changed)


def test_backend_native_shadow_requires_finite_output_on_set_changed_rows() -> None:
    target = _topk_result([[7, 3, -1]], [2])
    changed = _topk_result([[7, 6, -1]], [2])
    changed.output[0, 0, 0] = torch.nan
    with pytest.raises(AssertionError, match="non-finite"):
        _compare_results(target, changed)


def test_forward_backward_gradient_shadow_contract(tmp_path: Path) -> None:
    target = {
        "input::latent_kv": torch.ones(16, dtype=torch.bfloat16),
        "input::q": torch.arange(16, dtype=torch.float32),
        "parameter::weight": torch.arange(8, dtype=torch.float32),
    }
    shadow = {name: value.clone() for name, value in target.items()}
    diagnostic_path = tmp_path / "gradient_comparison.json"
    metrics = _compare_training_gradients(
        target,
        shadow,
        diagnostic_path=diagnostic_path,
    )
    assert metrics["all_close"] is True
    assert metrics["latent_kv_mismatch_ratio"] == 0.0
    assert json.loads(diagnostic_path.read_text())["all_close"] is True
    shadow["input::q"][0] = 1.0
    with pytest.raises(AssertionError, match="input::q"):
        _compare_training_gradients(
            target,
            shadow,
            diagnostic_path=diagnostic_path,
        )
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["all_close"] is False
    assert diagnostic["tensors"]["input::q"]["gate_pass"] is False


def test_forward_backward_gradient_finite_diagnostics_labels_each_plan() -> None:
    target = {
        "input::x": torch.tensor([1.0, math.nan, math.inf, -math.inf]),
    }
    shadow = {
        "input::x": torch.tensor([1.0, 2.0, math.inf, math.nan]),
    }
    diagnostics = _training_gradient_finite_diagnostics(target, shadow)
    assert diagnostics["labels"] == {
        "target": "balanced",
        "shadow": "sequential",
    }
    tensors = diagnostics["tensors"]
    assert isinstance(tensors, dict)
    tensor = tensors["input::x"]
    assert tensor["balanced_target"] == {
        "elements": 4,
        "finite": 1,
        "nan": 1,
        "negative_inf": 1,
        "nonfinite": 3,
        "positive_inf": 1,
    }
    assert tensor["sequential_shadow"] == {
        "elements": 4,
        "finite": 2,
        "nan": 1,
        "negative_inf": 0,
        "nonfinite": 2,
        "positive_inf": 1,
    }
    assert tensor["finite_mask_mismatch_count"] == 1


def test_model_gradient_allreduce_validates_before_any_collective(
    monkeypatch,
) -> None:
    config = _small_config(4)
    layer = MagiDSALayer(config)
    sink = torch.zeros(config.num_query_heads, dtype=torch.float32, requires_grad=True)
    sink.grad = torch.ones_like(sink)
    source = _ProfileSource(
        x=torch.empty(0, config.hidden_size, dtype=torch.bfloat16),
        sink=sink,
        packed_meta=MagiDSAPackedMeta((0, 0), (0,)),
    )
    for parameter in layer.parameters():
        parameter.grad = torch.ones_like(parameter)

    calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor) -> None:
        calls.append(tensor)
        tensor.mul_(3.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    _all_reduce_training_gradients(layer, source)
    assert len(calls) == 1
    assert calls[0].dtype == torch.float32
    expected_elements = source.sink.numel() + sum(
        parameter.numel() for parameter in layer.parameters()
    )
    assert calls[0].shape == (expected_elements,)
    assert torch.equal(source.sink.grad, torch.full_like(source.sink, 3.0))
    for parameter in layer.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, 3.0))

    calls.clear()
    first_name, first_parameter = next(iter(layer.named_parameters()))
    first_parameter.grad = None
    with pytest.raises(AssertionError, match=first_name):
        _all_reduce_training_gradients(layer, source)
    assert calls == []


def test_forward_backward_profile_reuses_post_projection_dsa_input_leaves(
    monkeypatch,
) -> None:
    config = _small_config(4)
    tokens = 3
    source = _ProfileSource(
        x=torch.randn(tokens, config.hidden_size, dtype=torch.bfloat16),
        sink=torch.zeros(
            config.num_query_heads,
            dtype=torch.float32,
            requires_grad=True,
        ),
        packed_meta=MagiDSAPackedMeta((0, tokens), (tokens,)),
    )
    tensors = {
        "x": torch.randn(
            tokens,
            config.hidden_size,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "qr": torch.randn(
            tokens,
            config.q_lora_rank,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "q": torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "latent_kv": torch.randn(
            tokens,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    }
    boundary = _ProfileDSAInputBoundary(
        value=MagiDSAInput(
            **tensors,
            sink=source.sink,
            packed_meta=source.packed_meta,
        ),
        dout=torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            dtype=torch.bfloat16,
        ),
        dkl=torch.ones((), dtype=torch.float32),
    )
    runtime = SimpleNamespace(config=config)
    handle = SimpleNamespace(
        device_plan=SimpleNamespace(local_token_count=tokens),
    )

    def fail_prepare(*args, **kwargs):
        raise AssertionError("layout/projection must not run inside a DSA-core step")

    monkeypatch.setattr(
        profile_5step_module,
        "layout_and_project_dsa_input",
        fail_prepare,
    )
    monkeypatch.setattr(profile_5step_module, "_profile_projector", fail_prepare)
    layer = MagiDSALayer(config)
    for _ in range(2):
        dsa_input = _make_plan_input(
            source,
            cast(MagiDSARuntimeMgr, runtime),
            handle,
            retain_input_gradients=True,
            input_boundary=boundary,
        )
        assert dsa_input is boundary.value
        loss = sum(
            tensor.float().sum()
            for tensor in (
                dsa_input.x,
                dsa_input.qr,
                dsa_input.q,
                dsa_input.latent_kv,
                dsa_input.sink,
            )
        )
        loss.backward()
        assert all(tensor.grad is not None for tensor in tensors.values())
        assert source.sink.grad is not None
        _clear_training_gradients(layer, source, boundary)
        assert all(tensor.grad is None for tensor in tensors.values())
        assert source.sink.grad is None


def test_profile_dsa_input_boundary_detaches_layout_and_projection_graphs(
    monkeypatch,
) -> None:
    config = _small_config(4)
    tokens = 3
    source = _ProfileSource(
        x=torch.randn(
            tokens,
            config.hidden_size,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        sink=torch.zeros(
            config.num_query_heads,
            dtype=torch.float32,
            requires_grad=True,
        ),
        packed_meta=MagiDSAPackedMeta((0, tokens), (tokens,)),
    )
    layout_calls = 0

    def layout_hidden(value, handle):
        nonlocal layout_calls
        del handle
        layout_calls += 1
        return value * 2

    runtime = SimpleNamespace(
        config=config,
        get_position_ids=lambda handle: torch.arange(tokens, dtype=torch.int32),
        layout_hidden=layout_hidden,
    )
    handle = SimpleNamespace(
        plan=SimpleNamespace(
            rank_plans=(SimpleNamespace(local_query_global_rows=tuple(range(tokens))),)
        ),
        rank=0,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    global_dout = torch.randn(
        tokens,
        config.num_query_heads,
        config.head_dim,
        dtype=torch.bfloat16,
    )
    boundary = _prepare_profile_dsa_input_boundary(
        source,
        cast(MagiDSARuntimeMgr, runtime),
        handle,
        global_dout,
    )

    assert layout_calls == 1
    assert boundary.value.sink is source.sink
    for tensor in (
        boundary.value.x,
        boundary.value.qr,
        boundary.value.q,
        boundary.value.latent_kv,
    ):
        assert tensor.is_leaf
        assert tensor.requires_grad
        assert tensor.grad_fn is None
    assert torch.equal(boundary.dout, global_dout)
    assert boundary.dout.requires_grad is False
    assert torch.equal(boundary.dkl, torch.ones((), dtype=torch.float32))


def test_forward_backward_profile_uses_precomputed_dout_and_dkl(monkeypatch) -> None:
    config = _small_config(4)
    tokens = 3
    source = _ProfileSource(
        x=torch.randn(tokens, config.hidden_size, dtype=torch.bfloat16),
        sink=torch.zeros(
            config.num_query_heads,
            dtype=torch.float32,
            requires_grad=True,
        ),
        packed_meta=MagiDSAPackedMeta((0, tokens), (tokens,)),
    )
    tensors = {
        "x": torch.randn(
            tokens,
            config.hidden_size,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "qr": torch.randn(
            tokens,
            config.q_lora_rank,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "q": torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "latent_kv": torch.randn(
            tokens,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    }
    dout = torch.randn_like(tensors["q"])
    boundary = _ProfileDSAInputBoundary(
        value=MagiDSAInput(
            **tensors,
            sink=source.sink,
            packed_meta=source.packed_meta,
        ),
        dout=dout,
        dkl=torch.ones((), dtype=torch.float32),
    )
    handle = SimpleNamespace(
        device_plan=SimpleNamespace(local_token_count=tokens),
    )

    def calc_dsa(layer, dsa_input, execution_handle):
        del layer, execution_handle
        output = dsa_input.q * 2
        kl = (
            dsa_input.x.float().sum()
            + dsa_input.qr.float().sum()
            + dsa_input.latent_kv.float().sum()
            + dsa_input.sink.sum()
        )
        return MagiDSAForwardResult(
            output=output,
            kl=kl,
            sparse_lse=torch.zeros(tokens),
            topk_ids=torch.zeros(tokens, 1, dtype=torch.int32),
            topk_length=torch.ones(tokens, dtype=torch.int32),
            indexer_lse=torch.zeros(tokens),
        )

    runtime = SimpleNamespace(config=config, calc_dsa=calc_dsa)
    allreduce_calls = 0

    def fake_allreduce(layer, profile_source):
        nonlocal allreduce_calls
        del layer
        assert profile_source is source
        allreduce_calls += 1

    monkeypatch.setattr(
        profile_5step_module,
        "_all_reduce_training_gradients",
        fake_allreduce,
    )
    layer = MagiDSALayer(config)
    result, dsa_input = _run_forward_backward_step(
        layer,
        source,
        cast(MagiDSARuntimeMgr, runtime),
        handle,
        "balanced",
        0,
        boundary,
    )

    assert result.output.shape == dout.shape
    assert dsa_input is boundary.value
    assert allreduce_calls == 1
    torch.testing.assert_close(tensors["q"].grad, dout * 2)
    for name in ("x", "qr", "latent_kv"):
        assert tensors[name].grad is not None
    assert source.sink.grad is not None


def _fake_attention_suite_case(
    mode: str,
    ratio: DsaRatio,
    events: _StringEventSink,
) -> _AttentionProfileCase:
    config = _small_config(ratio)
    tokens = 2
    source = _ProfileSource(
        x=torch.randn(tokens, config.hidden_size, dtype=torch.bfloat16),
        sink=torch.zeros(
            config.num_query_heads,
            dtype=torch.float32,
            requires_grad=True,
        ),
        packed_meta=MagiDSAPackedMeta((0, tokens), (tokens,)),
    )
    tensors = {
        "x": torch.randn(
            tokens,
            config.hidden_size,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "qr": torch.randn(
            tokens,
            config.q_lora_rank,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "q": torch.randn(
            tokens,
            config.num_query_heads,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        "latent_kv": torch.randn(
            tokens,
            config.head_dim,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    }

    def record_backward(gradient: torch.Tensor) -> torch.Tensor:
        events.append(f"backward:{mode}")
        return gradient

    tensors["q"].register_hook(record_backward)
    boundary = _ProfileDSAInputBoundary(
        value=MagiDSAInput(
            **tensors,
            sink=source.sink,
            packed_meta=source.packed_meta,
        ),
        dout=torch.randn_like(tensors["q"]),
        dkl=torch.ones((), dtype=torch.float32),
    )

    def calc_dsa(layer, dsa_input, handle):
        del layer, handle
        events.append(f"forward:{mode}")
        output = dsa_input.q * (1 + ratio / 128)
        if ratio == 4:
            kl = (
                dsa_input.x.float().sum()
                + dsa_input.qr.float().sum()
                + dsa_input.latent_kv.float().sum()
                + dsa_input.sink.sum()
            )
        else:
            kl = torch.zeros((), dtype=torch.float32)
        return MagiDSAForwardResult(
            output=output,
            kl=kl,
            sparse_lse=torch.zeros(tokens, config.num_query_heads),
            topk_ids=torch.empty(tokens, 0, dtype=torch.int32),
            topk_length=torch.zeros(tokens, dtype=torch.int32),
            indexer_lse=torch.full((tokens,), float("-inf")),
        )

    layer = torch.nn.Module()
    layer.config = config
    runtime = SimpleNamespace(config=config, calc_dsa=calc_dsa)
    handle = SimpleNamespace(
        device_plan=SimpleNamespace(
            local_token_count=tokens,
            source_token_count=tokens,
            token_layout_route=None,
        )
    )
    return _AttentionProfileCase(
        name=mode,
        layer=layer,
        source=source,
        runtime=cast(MagiDSARuntimeMgr, runtime),
        handle=handle,
        boundary=boundary,
        prepare_seconds=0.1,
    )


def test_pro_pair_prepares_one_hidden_layout_and_independent_boundaries(
    monkeypatch,
) -> None:
    class FakeEvent:
        def record(self) -> None:
            return None

        def elapsed_time(self, other) -> float:
            del other
            return 1.25

    ratio_cases: tuple[tuple[str, DsaRatio], ...] = (("csa", 4), ("hca", 128))
    configs = {mode: _small_config(ratio) for mode, ratio in ratio_cases}
    handles = {
        mode: SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                rank_plans=(SimpleNamespace(local_query_global_rows=(0, 2)),)
            ),
        )
        for mode in ("csa", "hca")
    }

    class FakeRatioRuntime:
        def __init__(self, config: MagiDSAConfig) -> None:
            self.config = config

        def get_position_ids(self, handle) -> torch.Tensor:
            del handle
            return torch.tensor((0, 2), dtype=torch.int64)

    class FakeProRuntime:
        def __init__(self) -> None:
            self.layout_invocations = 0
            self.runtimes = {
                mode: FakeRatioRuntime(config) for mode, config in configs.items()
            }

        def layout_source_hidden_once(self, source_x, bundle) -> torch.Tensor:
            del bundle
            self.layout_invocations += 1
            return source_x.index_select(0, torch.tensor((0, 1))).contiguous()

        def runtime_for_ratio(self, ratio: int) -> FakeRatioRuntime:
            return self.runtimes["csa" if ratio == 4 else "hca"]

        def handle_for_layer(self, layer_id: int, bundle):
            del bundle
            return handles["csa" if layer_id == 2 else "hca"]

    packed_meta = MagiDSAPackedMeta((0, 4), (2, 2))
    source_x = torch.randn(2, 8, dtype=torch.bfloat16)
    sources = {
        mode: _ProfileSource(
            x=source_x,
            sink=torch.randn(2, dtype=torch.float32, requires_grad=True),
            packed_meta=packed_meta,
        )
        for mode in ("csa", "hca")
    }
    bundle = SimpleNamespace(csa=handles["csa"], hca=handles["hca"])
    global_dout = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    runtime = FakeProRuntime()
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: FakeEvent())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    boundaries = _prepare_pro_pair_boundaries(
        sources,
        runtime,  # type: ignore[arg-type]
        bundle,  # type: ignore[arg-type]
        global_dout,
    )

    assert runtime.layout_invocations == 1
    assert set(boundaries) == {"csa", "hca"}
    assert boundaries["csa"].value.x is not boundaries["hca"].value.x
    assert boundaries["csa"].value.x.data_ptr() != boundaries["hca"].value.x.data_ptr()
    torch.testing.assert_close(
        boundaries["csa"].value.x,
        boundaries["hca"].value.x,
    )
    torch.testing.assert_close(
        boundaries["csa"].dout,
        global_dout.index_select(0, torch.tensor((0, 2))),
    )
    assert boundaries["csa"].token_layout_forward_ms == pytest.approx(1.25)
    assert boundaries["hca"].token_layout_forward_ms == pytest.approx(1.25)
    for mode in ("csa", "hca"):
        value = boundaries[mode].value
        assert value.x.is_leaf and value.x.requires_grad
        assert value.qr.is_leaf and value.qr.requires_grad
        assert value.q.is_leaf and value.q.requires_grad
        assert value.latent_kv.is_leaf and value.latent_kv.requires_grad


def test_pro_pair_calculation_uses_pro_runtime_layer_entrypoint() -> None:
    calls: list[tuple[int, object, object]] = []
    result = object()

    class FakeProRuntime:
        def calc_layer(self, layer_id, layer, dsa_input, bundle):
            del layer
            calls.append((layer_id, dsa_input, bundle))
            return result

    layer = SimpleNamespace(layer_id=2)
    dsa_input = SimpleNamespace()
    bundle = SimpleNamespace()
    case = SimpleNamespace(
        layer=layer,
        runtime=SimpleNamespace(
            calc_dsa=lambda *args: (_ for _ in ()).throw(
                AssertionError("ordinary runtime entrypoint must not run")
            )
        ),
        handle=SimpleNamespace(),
        pro_runtime=FakeProRuntime(),
        pro_bundle=bundle,
    )

    observed = _calc_attention_case(
        case,  # type: ignore[arg-type]
        dsa_input,  # type: ignore[arg-type]
    )
    assert observed is result
    assert calls == [(2, dsa_input, bundle)]


def test_pro_pair_model_side_gradient_reducer_uses_fp32_bucket(monkeypatch) -> None:
    cases = {}
    owners: list[tuple[torch.Tensor, torch.Tensor]] = []
    for mode in ("csa", "hca"):
        sink = torch.ones(2, dtype=torch.float32, requires_grad=True)
        sink.grad = torch.ones_like(sink)
        parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        parameter.grad = torch.ones_like(parameter)
        layer = SimpleNamespace(
            named_parameters=lambda parameter=parameter: iter((("weight", parameter),))
        )
        cases[mode] = SimpleNamespace(
            layer=layer,
            source=SimpleNamespace(sink=sink),
        )
        owners.append((sink, parameter))

    reduced_dtypes: list[torch.dtype] = []

    def fake_all_reduce(tensor: torch.Tensor) -> None:
        reduced_dtypes.append(tensor.dtype)
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    attention_suite_module._configure_capture("pro-pair")
    try:
        _all_reduce_attention_suite_gradients(cases)  # type: ignore[arg-type]
    finally:
        attention_suite_module._configure_capture("pro-pair")

    assert reduced_dtypes == [torch.float32]
    for sink, parameter in owners:
        assert sink.grad is not None and sink.grad.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.bfloat16
        torch.testing.assert_close(sink.grad, torch.full_like(sink, 2))
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 2))


def test_attention_suite_runs_three_independent_graphs_in_frozen_order(
    monkeypatch,
) -> None:
    events: list[str] = []
    ratio_cases: tuple[tuple[str, DsaRatio], ...] = (
        ("csa", 4),
        ("hca", 128),
    )
    cases = {
        mode: _fake_attention_suite_case(mode, ratio, events)
        for mode, ratio in ratio_cases
    }

    @contextmanager
    def no_cuda_nvtx(name: str) -> Iterator[None]:
        del name
        yield

    def fake_allreduce(profile_cases) -> None:
        assert profile_cases is cases
        events.append("allreduce")

    monkeypatch.setattr(attention_suite_module, "_nvtx_range", no_cuda_nvtx)
    monkeypatch.setattr(
        attention_suite_module,
        "_all_reduce_attention_suite_gradients",
        fake_allreduce,
    )
    results, inputs = _run_attention_suite_step(cases, rank=0)
    assert list(results) == ["csa", "hca"]
    assert list(inputs) == ["csa", "hca"]
    assert events == [
        "forward:csa",
        "forward:hca",
        "backward:hca",
        "backward:csa",
        "allreduce",
    ]


def test_attention_suite_backward_completion_joins_before_mode_done(
    monkeypatch,
) -> None:
    events: list[object] = []

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            events.append(("wait_event", self.name, event.event_id))

        def wait_stream(self, stream) -> None:
            events.append(("wait_stream", self.name, stream.name))

    next_event_id = 0

    class FakeEvent:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal next_event_id
            del args, kwargs
            self.event_id = next_event_id
            next_event_id += 1

        def record(self, stream=None) -> None:
            stream_name = None if stream is None else stream.name
            events.append(("event_record", stream_name, self.event_id))

    @contextmanager
    def fake_cuda_stream(stream) -> Iterator[None]:
        events.append(("stream_enter", stream.name))
        try:
            yield
        finally:
            events.append(("stream_exit", stream.name))

    @contextmanager
    def no_cuda_nvtx(name: str) -> Iterator[None]:
        del name
        yield

    @contextmanager
    def record_dsa_nvtx(name: str, *, enabled: bool = True) -> Iterator[None]:
        if enabled:
            events.append(("nvtx_push", name))
        try:
            yield
        finally:
            if enabled:
                events.append(("nvtx_pop", name))

    ratio_cases: tuple[tuple[str, DsaRatio], ...] = (
        ("csa", 4),
        ("hca", 128),
    )
    base_cases = {
        mode: _fake_attention_suite_case(mode, ratio, events)
        for mode, ratio in ratio_cases
    }
    execution_streams = {mode: FakeStream(f"{mode}_execution") for mode in base_cases}
    internal_streams = {
        "sparse_backward_stream": FakeStream("csa_sparse"),
        "csa_main_stream": FakeStream("csa_main"),
        "csa_indexer_stream": FakeStream("csa_indexer"),
        "csa_route_stream": FakeStream("csa_route"),
        "hca_main_stream": FakeStream("hca_main"),
        "hca_route_stream": FakeStream("hca_route"),
    }
    cases: dict[str, _AttentionProfileCase] = {}
    for mode, base in base_cases.items():
        handle_fields = {
            name: (
                stream
                if name.startswith(mode)
                or (mode == "csa" and name.startswith("sparse"))
                else None
            )
            for name, stream in internal_streams.items()
        }
        handle = SimpleNamespace(
            device_plan=base.handle.device_plan,
            **handle_fields,
        )
        cases[mode] = _AttentionProfileCase(
            name=base.name,
            layer=base.layer,
            source=base.source,
            runtime=base.runtime,
            handle=handle,
            boundary=base.boundary,
            prepare_seconds=base.prepare_seconds,
            execution_stream=execution_streams[mode],
        )

    caller_stream = FakeStream("caller")
    monkeypatch.setattr(attention_suite_module, "_nvtx_range", no_cuda_nvtx)
    monkeypatch.setattr(
        attention_suite_module,
        "dsa_nvtx_range",
        record_dsa_nvtx,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller_stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", fake_cuda_stream)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mode completion join must not globally synchronize")
        ),
    )
    monkeypatch.setattr(
        attention_suite_module,
        "_all_reduce_attention_suite_gradients",
        lambda profile_cases: events.append(("allreduce", profile_cases is cases)),
    )

    _run_attention_suite_step(cases, rank=0)

    done_event_ids: dict[str, int] = {}
    for mode, internal_names in {
        "hca": ("hca_main", "hca_route"),
        "csa": ("csa_sparse", "csa_main", "csa_indexer", "csa_route"),
    }.items():
        backward_index = events.index(f"backward:{mode}")
        wait_events = [
            (index, cast(tuple[str, str, str], event))
            for index, event in enumerate(events)
            if isinstance(event, tuple)
            and event[:2] == ("wait_stream", f"{mode}_execution")
        ]
        wait_indices = [index for index, _ in wait_events]
        assert [event[2] for _, event in wait_events] == list(internal_names)
        done_index, done_event = next(
            (index, cast(tuple[str, str, int], event))
            for index, event in enumerate(events)
            if index > max(wait_indices)
            and isinstance(event, tuple)
            and event[:2] == ("event_record", f"{mode}_execution")
        )
        assert backward_index < min(wait_indices) <= max(wait_indices) < done_index
        done_event_ids[mode] = done_event[2]

    # The Pro pair serializes HCA before CSA in backward, so CSA's stream must
    # observe HCA's completion event before its own backward is submitted.
    csa_waits_for_hca = events.index(
        ("wait_event", "csa_execution", done_event_ids["hca"])
    )
    assert csa_waits_for_hca < events.index("backward:csa")
    allreduce_index = events.index(("allreduce", True))
    for event_id in done_event_ids.values():
        caller_wait = events.index(("wait_event", "caller", event_id))
        assert caller_wait < allreduce_index

    pushed = [
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "nvtx_push"
    ]
    for mode, fields in {
        "csa": (
            "sparse_backward_stream",
            "csa_main_stream",
            "csa_indexer_stream",
            "csa_route_stream",
        ),
        "hca": ("hca_main_stream", "hca_route_stream"),
    }.items():
        parent = "pro_pair::stream_overlap::" f"backward_completion_join::{mode}"
        assert parent in pushed
        for field in fields:
            assert f"{parent}::{field}" in pushed


@pytest.mark.parametrize("step_mode", ("pro-pair",))
def test_backward_completion_join_nvtx_is_capture_mode_specific(
    monkeypatch,
    step_mode: str,
) -> None:
    events: list[tuple[str, str]] = []

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_stream(self, stream) -> None:
            events.append((self.name, stream.name))

    @contextmanager
    def record_dsa_nvtx(name: str, *, enabled: bool = True) -> Iterator[None]:
        if enabled:
            events.append(("push", name))
        yield

    execution = FakeStream("execution")
    internal = FakeStream("sparse")
    case = SimpleNamespace(
        execution_stream=execution,
        handle=SimpleNamespace(
            sparse_backward_stream=internal,
            csa_main_stream=None,
            csa_indexer_stream=None,
            csa_route_stream=None,
        ),
    )
    monkeypatch.setattr(
        attention_suite_module,
        "dsa_nvtx_range",
        record_dsa_nvtx,
    )
    attention_suite_module._configure_capture(step_mode)
    try:
        joined = attention_suite_module._join_attention_case_backward_streams(
            "csa",
            cast(_AttentionProfileCase, case),
        )
    finally:
        attention_suite_module._configure_capture("pro-pair")

    module_scope = "attention_suite" if step_mode == "attention-suite" else "pro_pair"
    parent = f"{module_scope}::stream_overlap::backward_completion_join::csa"
    assert joined == ("sparse_backward_stream",)
    assert ("push", parent) in events
    assert ("push", f"{parent}::sparse_backward_stream") in events
    assert ("execution", "sparse") in events


def test_attention_suite_profiler_attach_warmup_is_counter_audited(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cases = {
        mode: SimpleNamespace(runtime=SimpleNamespace(calls=0))
        for mode in ("csa", "hca")
    }
    nvtx_events: list[tuple[str, str | None]] = []

    def fake_counter(runtime) -> dict[str, int]:
        return {
            "device_materializations": 0,
            "health_checks": 0,
            "object_collective_invocations": 0,
            "solver_invocations": 0,
            "warm_invocations": runtime.calls,
        }

    def fake_step(profile_cases, rank):
        assert profile_cases is cases
        assert rank == 0
        for case in profile_cases.values():
            case.runtime.calls += 1
        return {}, {}

    monkeypatch.setattr(attention_suite_module, "_counter_dict", fake_counter)
    monkeypatch.setattr(
        attention_suite_module,
        "_clear_attention_suite_gradients",
        lambda profile_cases: None,
    )
    monkeypatch.setattr(attention_suite_module, "_run_attention_suite_step", fake_step)
    monkeypatch.setattr(attention_suite_module, "_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda name: nvtx_events.append(("push", name)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: nvtx_events.append(("pop", None)),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)

    delta = attention_suite_module._profiler_attach_warmup(
        cast(dict[str, _AttentionProfileCase], cases),
        steps=1,
        artifact_dir=tmp_path,
        control_group=SimpleNamespace(),
        rank=0,
    )
    assert all(record["warm_invocations"] == 1 for record in delta.values())
    assert nvtx_events == [
        ("push", "$Magi_DSA/ablation_profiler_attach_warmup"),
        ("push", "magi_dsa::ablation::profiler_attach_warmup_step"),
        ("pop", None),
        ("pop", None),
    ]


def test_attention_suite_gradient_schema_is_ratio_specific() -> None:
    """HCA feeds x into its Compressor but has no Indexer, so qr stays dry."""

    events: list[str] = []
    case = _fake_attention_suite_case("hca", 128, events)
    dsa_input = case.boundary.value
    dsa_input.x.grad = torch.ones_like(dsa_input.x)
    dsa_input.q.grad = torch.ones_like(dsa_input.q)
    dsa_input.latent_kv.grad = torch.ones_like(dsa_input.latent_kv)
    case.source.sink.grad = torch.ones_like(case.source.sink)
    snapshot = _attention_gradient_snapshot(case, dsa_input)
    assert set(snapshot) == {
        "input::x",
        "input::q",
        "input::latent_kv",
        "input::sink",
    }

    dsa_input.qr.grad = torch.ones_like(dsa_input.qr)
    with pytest.raises(AssertionError, match="unexpectedly produced gradient: qr"):
        _attention_gradient_snapshot(case, dsa_input)


def _create_nsys_fixture(
    path: Path,
    plan: str,
    world_size: int,
    steps: int,
    *,
    step_mode: str = "forward",
) -> None:
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
    phase_nvtx_names = {
        "forward": {
            "indexer_score": "magi_dsa::indexer_score",
            "indexer_topk": "magi_dsa::indexer_topk",
        },
        "forward-backward": {
            phase: f"magi_dsa::{phase}"
            for phase in (
                "backward",
                "forward",
                "indexer_score",
                "indexer_topk",
                "parameter_gradient_allreduce",
            )
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
            "parameter_gradient_allreduce": ("magi_dsa::parameter_gradient_allreduce"),
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
    }[step_mode]
    correlation = 1
    for rank in range(world_size):
        global_pid = (1000 + rank) << 24
        global_tid = global_pid + 123
        for step in range(steps):
            step_start = (rank * steps + step) * 30_000 + 1_000
            step_end = step_start + (len(phase_nvtx_names) + 1) * 2_000
            connection.execute(
                "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) VALUES(?, ?, ?, ?, NULL)",
                (
                    step_start,
                    step_end,
                    f"{plan}/rank_{rank}/training_step_{step}",
                    global_tid,
                ),
            )
            for phase_index, (phase, nvtx_name) in enumerate(phase_nvtx_names.items()):
                phase_start = step_start + 500 + phase_index * 2_000
                phase_end = phase_start + 1_500
                connection.execute(
                    "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) "
                    "VALUES(?, ?, ?, ?, NULL)",
                    (phase_start, phase_end, nvtx_name, global_tid),
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


def test_nsys_sqlite_extracts_pro_pair_grid(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "pro_pair.sqlite"
    report_path = tmp_path / "pro_pair.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=2,
        steps=2,
        step_mode="pro-pair",
    )
    records, summary = extract_profile_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=2,
        steps=2,
        step_mode="pro-pair",
    )
    assert len(validate_pro_pair_records(records, 2, 2)) == 36
    assert len(compute_pro_pair_phase_rank_ranges(records, 2, 2)) == 18
    assert summary["logical_phase_records"] == 36
    assert summary["step_mode"] == "pro-pair"
    names = {record["phase"]: record["nvtx_name"] for record in records}
    assert names["csa_forward"] == "magi_dsa::pro_pair::csa::forward"
    assert names["hca_backward"] == "magi_dsa::pro_pair::hca::backward"


def test_pro_pair_sendrecv_reporting() -> None:
    expected_sendrecv = {
        "backward": 7,
        "csa_backward": 4,
        "csa_forward": 4,
        "forward": 7,
        "hca_backward": 3,
        "hca_forward": 3,
    }
    sendrecv_records = [
        {
            "kernel_name_counts": {"ncclDevKernel_SendRecv": count},
            "phase": phase,
            "rank": 0,
            "step": 0,
        }
        for phase, count in expected_sendrecv.items()
    ]
    validate_pro_pair_sendrecv(sendrecv_records)
    sendrecv_records[0]["kernel_name_counts"] = {"ncclDevKernel_SendRecv": 8}
    with pytest.raises(ValueError, match="SendRecv count differs"):
        validate_pro_pair_sendrecv(sendrecv_records)


def test_pro_pair_summarizer_requires_one_runtime_bundle_and_one_layout() -> None:
    payload = {
        "parameter_gradient_allreduce_in_7f7b": False,
        "parameter_gradient_reducer_precision": (
            "fp32_cp_bucket_model_side_diagnostic"
        ),
        "pro_bundle_handles": {
            "csa": {
                "declared_local_token_capacity": 16384,
                "layer_id": 2,
                "plan_hash": "csa-plan",
                "policy": "structural_balanced",
                "query_layout_hash": "shared-query-layout",
                "ratio": 4,
            },
            "hca": {
                "declared_local_token_capacity": 16384,
                "layer_id": 3,
                "plan_hash": "hca-plan",
                "policy": "structural_balanced",
                "query_layout_hash": "shared-query-layout",
                "ratio": 128,
            },
        },
        "pro_runtime_bundle": True,
        "runtime_parameter_gradient_communication": False,
        "shared_bundle_query_layout_hash": "shared-query-layout",
        "shared_source_packed_meta": True,
        "shared_source_x": True,
        "token_layout_invocations": 1,
    }
    assert (
        _validate_pro_runtime_bundle_payload(payload, rank=0, label="fixture")
        == "shared-query-layout"
    )

    broken = json.loads(json.dumps(payload))
    broken["token_layout_invocations"] = 2
    with pytest.raises(ValueError, match="runtime bundle contract differs"):
        _validate_pro_runtime_bundle_payload(broken, rank=0, label="fixture")

    broken = json.loads(json.dumps(payload))
    broken["pro_bundle_handles"]["hca"]["declared_local_token_capacity"] = 32768
    with pytest.raises(ValueError, match="hca bundle handle differs"):
        _validate_pro_runtime_bundle_payload(broken, rank=0, label="fixture")


def test_pro_pair_major_kernel_reporting() -> None:
    kernels_by_phase = {
        "indexer_score": [
            {"duration_ns": 1_000, "name": "cudnn_indexer_forward_sm100"}
        ],
        "indexer_topk": [
            {"duration_ns": 2_000, "name": "cudnn_indexer_topk_kernel_sm100"}
        ],
        "csa_forward": [
            {
                "duration_ns": 3_000,
                "name": "sparse_attn_fwd_for_small_topk_kernel",
            },
            {"duration_ns": 4_000, "name": "cudnn_indexer_backward_gemm"},
            {"duration_ns": 5_000, "name": "cudnn_indexer_backward_score_grad"},
        ],
        "hca_forward": [
            {
                "duration_ns": 6_000,
                "name": "sparse_attn_fwd_for_small_topk_kernel",
            }
        ],
        "csa_backward": [
            {
                "duration_ns": 7_000,
                "name": "kernel_cutlass_bwd_cudnn_sparse_attention_backward",
            }
        ],
        "hca_backward": [
            {
                "duration_ns": 8_000,
                "name": "kernel_cutlass_bwd_cudnn_sparse_attention_backward",
            }
        ],
    }
    records = [
        {"kernels": kernels, "phase": phase, "rank": 0, "step": 0}
        for phase, kernels in kernels_by_phase.items()
    ]
    audit = compute_major_kernel_timings(records, world_size=1, steps=1)
    assert audit["result"] == "PASS"
    assert audit["flashmla_forward_kernel"] == "sparse_attn_fwd_for_small_topk_kernel"
    assert audit["flashmla_forward_same_exact_variant"] is True
    assert audit["kernel_names"]["csa_flashmla_forward"] == [
        "sparse_attn_fwd_for_small_topk_kernel"
    ]
    assert audit["kernel_names"]["hca_flashmla_forward"] == [
        "sparse_attn_fwd_for_small_topk_kernel"
    ]
    assert len(audit["records"]) == 7
    assert audit["hard_gate_groups"] == ["csa_indexer_score", "csa_indexer_topk"]
    assert all(
        (
            item["threshold"] == 0.05
            if item["group"] in audit["hard_gate_groups"]
            else item["threshold"] is None
        )
        for item in audit["rank_ranges"]
    )
    grouped = {item["group"]: item["gpu_time_ms"] for item in audit["records"]}
    assert grouped["csa_selected_indexer_backward"] == pytest.approx(0.009)

    kernels_by_phase["hca_forward"][0]["name"] = "sparse_attn_fwd_kernel"
    with pytest.raises(ValueError, match="major kernel group is empty"):
        compute_major_kernel_timings(records, world_size=1, steps=1)


def test_pro_pair_reports_every_indexer_d2d_wrapper_separately() -> None:
    wrapper_scopes = {
        "indexer_score": "magi_dsa::CUDNN_CALL::indexer_score",
        "indexer_topk": "magi_dsa::CUDNN_CALL::indexer_topk",
        "selected_attention_recompute": (
            "magi_dsa::CUDNN_CALL::selected_attention_recompute"
        ),
        "selected_indexer_backward": "magi_dsa::CUDNN_CALL::indexer_backward",
        "selected_indexer_recompute": (
            "magi_dsa::CUDNN_CALL::selected_indexer_recompute"
        ),
    }
    attribution_records: list[dict[str, object]] = []
    wrapper_rowids: dict[tuple[int, int, str], int] = {}
    for rank in range(2):
        for step in range(2):
            grid_base = (rank * 2 + step) * 100_000
            for group_index, (group, scope) in enumerate(wrapper_scopes.items()):
                rowid = 1_000 + rank * 100 + step * 10 + group_index
                wrapper_rowids[(rank, step, group)] = rowid
                start_ns = grid_base + group_index * 1_000 + 100
                attribution_records.append(
                    {
                        "attribution_path": [
                            {"name": "magi_dsa::pro_pair::csa::forward", "rowid": 1},
                            {"name": scope, "rowid": rowid},
                        ],
                        "kernel_end_ns": start_ns + 300,
                        "kernel_name": f"{group}_kernel",
                        "kernel_start_ns": start_ns,
                        "rank": rank,
                        "step": step,
                    }
                )
            attribution_records.append(
                {
                    "attribution_path": [
                        {"name": "magi_dsa::pro_pair::csa::forward", "rowid": 1}
                    ],
                    "kernel_end_ns": grid_base + 380,
                    "kernel_name": "external_main_compressor_kernel",
                    "kernel_start_ns": grid_base + 180,
                    "rank": rank,
                    "step": step,
                }
            )

    target_scope = wrapper_scopes["indexer_score"]
    memcpy_records: list[dict[str, object]] = [
        {
            "attribution_path": [
                {"name": "magi_dsa::pro_pair::csa::forward", "rowid": 1},
                {
                    "name": target_scope,
                    "rowid": wrapper_rowids[(0, 0, "indexer_score")],
                },
            ],
            "bytes": 4_096,
            "copy_count": 3,
            "copy_kind": "CUDA_MEMCPY_KIND_DTOD",
            "memcpy_end_ns": 350,
            "memcpy_rowid": 7,
            "memcpy_start_ns": 200,
            "rank": 0,
            "record_type": "magi_dsa_memcpy_attribution",
            "runtime_rowid": 8,
            "step": 0,
        },
        {
            "attribution_path": [
                {
                    "name": "magi_dsa::module::selected_kl::workspace_copy",
                    "rowid": 9,
                }
            ],
            "bytes": 1_024,
            "copy_count": 2,
            "copy_kind": "CUDA_MEMCPY_KIND_DTOD",
            "memcpy_end_ns": 550,
            "memcpy_rowid": 9,
            "memcpy_start_ns": 500,
            "rank": 0,
            "record_type": "magi_dsa_memcpy_attribution",
            "runtime_rowid": 10,
            "step": 0,
        },
    ]
    audit = extract_indexer_d2d(
        memcpy_records,
        attribution_records,
        world_size=2,
        steps=2,
    )
    assert audit["result"] == "PASS"
    assert audit["accounting"] == "separate_from_kernel_gpu_time"
    assert audit["total_copy_count"] == 3
    assert audit["total_bytes"] == 4096
    assert len(audit["records"]) == 20
    score = next(
        record
        for record in audit["records"]
        if record["rank"] == 0
        and record["step"] == 0
        and record["group"] == "indexer_score"
    )
    assert score["gpu_time_ms"] == pytest.approx(0.00015)
    assert score["same_wrapper_kernel_overlap_ms"] == pytest.approx(0.00015)
    assert score["same_mode_external_compute_overlap_ms"] == pytest.approx(0.00015)
    assert audit["outside_known_scope"]["memcpy_activity_count"] == 1
    assert audit["outside_known_scope"]["bytes"] == 1_024

    duplicate_wrapper = [
        *attribution_records,
        {
            "attribution_path": [
                {"name": "magi_dsa::pro_pair::csa::forward", "rowid": 1},
                {"name": target_scope, "rowid": 999_999},
            ],
            "kernel_end_ns": 700,
            "kernel_name": "duplicate_indexer_score_kernel",
            "kernel_start_ns": 600,
            "rank": 0,
            "step": 0,
        },
    ]
    with pytest.raises(ValueError, match="wrapper NVTX range count differs"):
        extract_indexer_d2d(
            memcpy_records,
            duplicate_wrapper,
            world_size=2,
            steps=2,
        )


def test_pro_pair_summary_hard_counts_selected_recompute_cudnn_calls() -> None:
    source = (find_repo_root() / "scripts/profile/summarize_pro_pair.py").read_text(
        encoding="utf-8"
    )
    assert '"magi_dsa::CUDNN_CALL::selected_attention_recompute": invocations' in source
    assert '"magi_dsa::CUDNN_CALL::selected_indexer_recompute": invocations' in source


def test_pro_pair_reports_routes_serialization_and_support_overheads() -> None:
    route_orders = {
        ("csa", "forward"): (
            "WINDOW_KV",
            "OVERLAP_X",
            "COMPRESSED_KI",
            "COMPRESSED_KV",
        ),
        ("hca", "forward"): ("OVERLAP_X", "WINDOW_KV", "COMPRESSED_KV"),
        ("hca", "backward"): ("COMPRESSED_KV", "WINDOW_KV", "OVERLAP_X"),
        ("csa", "backward"): (
            "COMPRESSED_KI",
            "COMPRESSED_KV",
            "OVERLAP_X",
            "WINDOW_KV",
        ),
    }
    serial_order = tuple(route_orders)
    compute_intervals = {
        ("csa", "forward"): (100, 1_000),
        ("hca", "forward"): (290, 360),
        ("hca", "backward"): (290, 360),
        ("csa", "backward"): (100, 1_000),
    }
    records: list[dict[str, object]] = []
    runtime_start = 1
    for phase_index, (mode, direction) in enumerate(serial_order):
        base = phase_index * 10_000
        phase_scope = f"magi_dsa::pro_pair::{mode}::{direction}"
        compute_start, compute_end = compute_intervals[(mode, direction)]
        records.append(
            {
                "attribution_path": [{"name": phase_scope}],
                "gpu_time_ms": 0.001,
                "kernel_end_ns": base + compute_end,
                "kernel_name": f"{mode}_{direction}_compute",
                "kernel_start_ns": base + compute_start,
                "rank": 0,
                "runtime_start_ns": runtime_start,
                "step": 0,
            }
        )
        runtime_start += 1
        for route_index, route in enumerate(route_orders[(mode, direction)]):
            route_scope = (
                "magi_dsa::phase::collective_all2all_v::"
                f"attention::{mode}::{route}.{direction}"
            )
            start = base + 200 + route_index * 100
            records.append(
                {
                    "attribution_path": [
                        {"name": phase_scope},
                        {"name": route_scope},
                    ],
                    "gpu_time_ms": 0.00005,
                    "kernel_end_ns": start + 50,
                    "kernel_name": "ncclDevKernel_SendRecv",
                    "kernel_start_ns": start,
                    "rank": 0,
                    "runtime_start_ns": runtime_start,
                    "step": 0,
                }
            )
            runtime_start += 1

    support_specs = (
        (
            "csa",
            "forward",
            "range_gather_per_range_kernel",
            "magi_dsa::module::packing::csa::indexer_key_support::forward_copy",
        ),
        (
            "csa",
            "forward",
            "range_gather_per_range_kernel",
            "magi_dsa::module::route::attention::csa::WINDOW_KV::forward::send_pack",
        ),
        (
            "csa",
            "backward",
            "range_sum_reduce_deter_kernel",
            "magi_dsa::module::route::attention::csa::WINDOW_KV::backward::owner_reduce",
        ),
        (
            "hca",
            "forward",
            "range_gather_per_range_kernel",
            "magi_dsa::module::route::attention::hca::WINDOW_KV::forward::send_pack",
        ),
        (
            "hca",
            "backward",
            "range_sum_reduce_deter_kernel",
            "magi_dsa::module::route::attention::hca::WINDOW_KV::backward::owner_reduce",
        ),
        (
            "csa",
            "forward",
            "CatArrayBatchedCopy_vectorized",
            "magi_dsa::module::attention::csa::kv_bank_assembly",
        ),
        (
            "hca",
            "forward",
            "CatArrayBatchedCopy_vectorized",
            "magi_dsa::module::attention::hca::kv_bank_assembly",
        ),
    )
    phase_offsets = {pair: index * 10_000 for index, pair in enumerate(serial_order)}
    for item_index, (mode, direction, kernel_name, module_scope) in enumerate(
        support_specs
    ):
        base = phase_offsets[(mode, direction)]
        module_range: dict[str, object] = {"name": module_scope}
        if module_scope.endswith("indexer_key_support::forward_copy"):
            module_range["rowid"] = 4242
            kernel_start_ns = base + 300
            kernel_end_ns = base + 400
        else:
            kernel_start_ns = base + 1_200 + item_index
            kernel_end_ns = base + 1_300 + item_index
        records.append(
            {
                "attribution_path": [
                    {"name": f"magi_dsa::pro_pair::{mode}::{direction}"},
                    module_range,
                ],
                "gpu_time_ms": 0.0001,
                "kernel_end_ns": kernel_end_ns,
                "kernel_name": kernel_name,
                "kernel_start_ns": kernel_start_ns,
                "rank": 0,
                "runtime_start_ns": runtime_start,
                "step": 0,
            }
        )
        runtime_start += 1

    compressor_specs = (
        (
            "csa",
            "forward",
            "csa_indexer_compressor_kernel",
            "magi_dsa::module::compressor::indexer",
        ),
        (
            "csa",
            "forward",
            "csa_main_compressor_kernel",
            "magi_dsa::module::compressor::main",
        ),
        (
            "hca",
            "forward",
            "hca_main_compressor_kernel",
            "magi_dsa::module::compressor::main",
        ),
        (
            "csa",
            "backward",
            "_csa_compressor_backward_kernel",
            None,
        ),
        (
            "csa",
            "backward",
            "_csa_compressor_backward_kernel",
            None,
        ),
        (
            "csa",
            "backward",
            "_csa_compressor_ape_backward_kernel",
            None,
        ),
        (
            "csa",
            "backward",
            "_csa_compressor_ape_backward_kernel",
            None,
        ),
    )
    for item_index, (mode, direction, kernel_name, module_scope) in enumerate(
        compressor_specs
    ):
        base = phase_offsets[(mode, direction)]
        attribution_path = [{"name": f"magi_dsa::pro_pair::{mode}::{direction}"}]
        if module_scope is not None:
            attribution_path.append({"name": module_scope})
        records.append(
            {
                "attribution_path": attribution_path,
                "gpu_time_ms": 0.0002,
                "kernel_end_ns": base + 1_500 + item_index,
                "kernel_name": kernel_name,
                "kernel_start_ns": base + 1_400 + item_index,
                "rank": 0,
                "runtime_start_ns": runtime_start,
                "step": 0,
            }
        )
        runtime_start += 1

    route_audit = compute_route_timings(records, world_size=1, steps=1)
    assert route_audit["result"] == "PASS"
    assert route_audit["overlap_gate"] == "report_only"
    assert route_audit["overlap_contract"] == {
        "classification_counts": {
            "dependency_bound": 4,
            "overlap_capable": 10,
        },
        "dependency_bound_requires_positive_overlap": False,
        "fraction_threshold": None,
        "overlap_capable_requires_positive_overlap": False,
        "positive_time_threshold_ns": None,
    }
    assert len(route_audit["records"]) == 14
    overlap_capable = [
        record
        for record in route_audit["records"]
        if record["overlap_classification"] == "overlap_capable"
    ]
    dependency_bound = [
        record
        for record in route_audit["records"]
        if record["overlap_classification"] == "dependency_bound"
    ]
    expected_csa_overlap_capable = {
        ("csa", direction, route)
        for direction in ("forward", "backward")
        for route in route_orders[("csa", direction)]
    }
    assert {
        (record["mode"], record["direction"], record["route"])
        for record in overlap_capable
    } == expected_csa_overlap_capable | {
        ("hca", "forward", "WINDOW_KV"),
        ("hca", "backward", "WINDOW_KV"),
    }
    assert {
        (record["mode"], record["direction"], record["route"])
        for record in dependency_bound
    } == {
        ("hca", "forward", "OVERLAP_X"),
        ("hca", "forward", "COMPRESSED_KV"),
        ("hca", "backward", "COMPRESSED_KV"),
        ("hca", "backward", "OVERLAP_X"),
    }
    assert all(
        record["same_mode_compute_overlap_ns"] > 0
        and record["same_mode_compute_overlap_fraction"] > 0
        and record["observed_positive_same_mode_compute_overlap"] is True
        for record in overlap_capable
    )
    assert all(
        record["same_mode_compute_overlap_ns"] == 0
        and record["same_mode_compute_overlap_fraction"] == 0
        and record["overlap_contract_reason"]
        and record["observed_positive_same_mode_compute_overlap"] is False
        for record in dependency_bound
    )
    assert all(
        record["other_mode_compute_overlap_fraction"] == 0 and record["nvtx_path"]
        for record in route_audit["records"]
    )
    missing_required_overlap = [
        record for record in records if record["kernel_name"] != "csa_forward_compute"
    ]
    zero_overlap_audit = compute_route_timings(
        missing_required_overlap,
        world_size=1,
        steps=1,
    )
    zero_overlap_csa_forward = [
        record
        for record in zero_overlap_audit["records"]
        if record["mode"] == "csa" and record["direction"] == "forward"
    ]
    assert zero_overlap_audit["result"] == "PASS"
    assert all(
        record["overlap_classification"] == "overlap_capable"
        and record["observed_positive_same_mode_compute_overlap"] is False
        for record in zero_overlap_csa_forward
    )
    cross_mode_overlap = [
        *records,
        {
            "attribution_path": [{"name": "magi_dsa::pro_pair::hca::forward"}],
            "gpu_time_ms": 0.00005,
            "kernel_end_ns": 275,
            "kernel_name": "hca_foreign_compute_kernel",
            "kernel_start_ns": 225,
            "rank": 0,
            "runtime_start_ns": runtime_start + 1,
            "step": 0,
        },
    ]
    with pytest.raises(ValueError, match="overlaps other-mode compute"):
        compute_route_timings(cross_mode_overlap, world_size=1, steps=1)
    serial_audit = validate_mode_serialization(records, world_size=1, steps=1)
    assert serial_audit["result"] == "PASS"
    support_audit = compute_support_overhead_timings(
        records,
        [
            {
                "duplicate_indexer_k_bytes": 512,
                "duplicate_indexer_k_rows": 2,
                "indexer_k_row_bytes": 256,
                "packed_indexer_k_bytes": 1_536,
                "packed_indexer_k_rows": 6,
                "rank": 0,
                "unique_indexer_k_bytes": 1_024,
                "unique_indexer_k_rows": 4,
            }
        ],
        world_size=1,
        steps=1,
    )
    assert support_audit["result"] == "PASS"
    assert support_audit["grouped_k_pack_backward_csr_reduce_launches"] == 0
    assert len(support_audit["records"]) == 7
    grouped_k_pack = next(
        record
        for record in support_audit["records"]
        if record["group"] == "csa_grouped_k_pack_forward"
    )
    assert grouped_k_pack["kernel_launch_count"] == 1
    assert grouped_k_pack["module_nvtx_rowid"] == 4242
    assert grouped_k_pack["unique_bytes"] == 1_024
    assert grouped_k_pack["packed_bytes"] == 1_536
    assert grouped_k_pack["duplicate_bytes"] == 512
    assert grouped_k_pack["read_bytes"] == 1_536
    assert grouped_k_pack["write_bytes"] == 1_536
    assert grouped_k_pack["traffic_bytes"] == 3_072
    assert grouped_k_pack["external_compute_overlap_fraction"] == 1.0
    duplicate_grouped_k_pack = [
        *records,
        {
            **next(
                record
                for record in records
                if any(
                    "indexer_key_support::forward_copy" in scope["name"]
                    for scope in cast(list[dict[str, str]], record["attribution_path"])
                    if isinstance(scope, dict)
                )
            ),
            "runtime_start_ns": runtime_start + 2,
        },
    ]
    with pytest.raises(ValueError, match="range_gather count differs"):
        compute_support_overhead_timings(
            duplicate_grouped_k_pack,
            [
                {
                    "duplicate_indexer_k_bytes": 512,
                    "duplicate_indexer_k_rows": 2,
                    "indexer_k_row_bytes": 256,
                    "packed_indexer_k_bytes": 1_536,
                    "packed_indexer_k_rows": 6,
                    "rank": 0,
                    "unique_indexer_k_bytes": 1_024,
                    "unique_indexer_k_rows": 4,
                }
            ],
            world_size=1,
            steps=1,
        )
    compressor_audit = compute_compressor_timings(
        records,
        world_size=1,
        steps=1,
    )
    assert compressor_audit["result"] == "PASS_WITH_DOCUMENTED_COVERAGE_GAP"
    assert compressor_audit["attribution_complete"] is False
    assert {record["group"] for record in compressor_audit["records"]} == {
        "csa_compressor_backward_explicit",
        "csa_compressor_forward",
        "csa_indexer_compressor_forward",
        "csa_main_compressor_forward",
        "hca_compressor_forward",
    }

    unexpected = [
        *records,
        {
            "attribution_path": [
                {"name": "magi_dsa::pro_pair::csa::backward"},
                {
                    "name": "magi_dsa::module::packing::csa::"
                    "indexer_key_support::backward_reduce"
                },
            ],
            "gpu_time_ms": 0.0001,
            "kernel_end_ns": 31_100,
            "kernel_name": "range_sum_reduce_deter_kernel",
            "kernel_start_ns": 31_000,
            "rank": 0,
            "runtime_start_ns": runtime_start,
            "step": 0,
        },
    ]
    with pytest.raises(ValueError, match="unexpectedly ran a backward reduction"):
        compute_support_overhead_timings(
            unexpected,
            [
                {
                    "duplicate_indexer_k_bytes": 512,
                    "duplicate_indexer_k_rows": 2,
                    "indexer_k_row_bytes": 256,
                    "packed_indexer_k_bytes": 1_536,
                    "packed_indexer_k_rows": 6,
                    "rank": 0,
                    "unique_indexer_k_bytes": 1_024,
                    "unique_indexer_k_rows": 4,
                }
            ],
            world_size=1,
            steps=1,
        )


def test_nsys_sqlite_attributes_all_step_kernels_and_cross_thread_fallback(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "attribution.sqlite"
    report_path = tmp_path / "attribution.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=1,
        steps=1,
        step_mode="forward-backward",
    )
    connection = sqlite3.connect(sqlite_path)
    global_pid = 1000 << 24
    global_tid = global_pid + 123
    worker_tid = global_pid + 456
    step_start = 1_000
    backward_start = step_start + 500
    forward_start = step_start + 2_500
    connection.executemany(
        "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) "
        "VALUES(?, ?, ?, ?, NULL)",
        (
            (
                backward_start + 50,
                backward_start + 900,
                "magi_dsa::module::fixture::backward",
                global_tid,
            ),
            (
                backward_start + 90,
                backward_start + 250,
                "magi_dsa::CUDNN_CALL::fixture",
                global_tid,
            ),
            (
                forward_start + 50,
                forward_start + 900,
                "magi_dsa::module::fixture::forward",
                global_tid,
            ),
        ),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start, end, globalTid, correlationId) "
        "VALUES(?, ?, ?, ?)",
        (
            (forward_start + 300, forward_start + 400, worker_tid, 100),
            (step_start + 100, step_start + 200, global_tid, 101),
            (
                backward_start + 100,
                backward_start + 200,
                (2000 << 24) + 123,
                1,
            ),
        ),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, end, correlationId, globalPid, shortName) "
        "VALUES(?, ?, ?, ?, 1)",
        (
            (forward_start + 500, forward_start + 800, 100, global_pid),
            (step_start + 250, step_start + 450, 101, global_pid),
        ),
    )
    connection.commit()
    connection.close()

    records, summary = extract_kernel_attribution_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=1,
        steps=1,
    )
    assert len(records) == 7
    assert summary["result"] == "FAIL"
    assert summary["attribution_coverage"] == 6 / 7
    assert summary["attribution_kind_counts"] == {
        "cudnn_call": 1,
        "logical": 4,
        "module": 1,
        "unattributed": 1,
    }
    cudnn = next(record for record in records if record["correlation_id"] == 1)
    assert cudnn["attribution_kind"] == "cudnn_call"
    assert cudnn["attribution_name"] == "magi_dsa::CUDNN_CALL::fixture"
    module = next(record for record in records if record["correlation_id"] == 2)
    assert module["attribution_kind"] == "module"
    cross_thread = next(record for record in records if record["correlation_id"] == 100)
    assert cross_thread["attribution_kind"] == "logical"
    assert cross_thread["attribution_name"] == "magi_dsa::forward"
    assert cross_thread["attribution_thread_scope"] == "process_temporal"
    unattributed = next(record for record in records if record["correlation_id"] == 101)
    assert unattributed["attribution_kind"] == "unattributed"
    assert unattributed["attribution_path"] == []


def _add_nsys_memcpy_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(
            id INTEGER PRIMARY KEY,
            name TEXT,
            label TEXT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            correlationId INTEGER,
            globalPid INTEGER,
            bytes INTEGER NOT NULL,
            copyKind INTEGER NOT NULL,
            copyCount INTEGER
        );
        INSERT INTO ENUM_CUDA_MEMCPY_OPER(id, name, label)
        VALUES(8, 'CUDA_MEMCPY_KIND_DTOD', 'Device-to-Device');
        """
    )


def test_nsys_sqlite_attributes_d2d_by_process_runtime_and_nvtx_path(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "memcpy_attribution.sqlite"
    report_path = tmp_path / "memcpy_attribution.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=2,
        steps=1,
        step_mode="forward-backward",
    )
    connection = sqlite3.connect(sqlite_path)
    _add_nsys_memcpy_tables(connection)
    runtime_rows: list[tuple[int, int, int, int, int]] = []
    for rank in range(2):
        global_pid = (1000 + rank) << 24
        caller_tid = global_pid + 123
        phase_start = connection.execute(
            "SELECT start FROM NVTX_EVENTS "
            "WHERE text = 'magi_dsa::indexer_score' AND globalTid = ?",
            (caller_tid,),
        ).fetchone()[0]
        runtime_row = connection.execute(
            "SELECT rowid, start, end, globalTid, correlationId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE start >= ? AND end <= ?",
            (phase_start, phase_start + 1_500),
        ).fetchone()
        assert runtime_row is not None
        rowid, start, end, global_tid, correlation_id = runtime_row
        runtime_rows.append(
            (
                int(rowid),
                int(start),
                int(end),
                int(global_tid),
                int(correlation_id),
            )
        )

    shared_correlation_id = 777
    rank0_runtime = runtime_rows[0]
    rank0_worker_tid = (1000 << 24) + 456
    connection.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid = ?, correlationId = ? "
        "WHERE rowid = ?",
        (rank0_worker_tid, shared_correlation_id, rank0_runtime[0]),
    )
    rank1_runtime = runtime_rows[1]
    connection.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId = ? WHERE rowid = ?",
        (shared_correlation_id, rank1_runtime[0]),
    )
    connection.execute(
        "INSERT INTO NVTX_EVENTS(start, end, text, globalTid, textId) "
        "VALUES(?, ?, ?, ?, NULL)",
        (
            rank1_runtime[1] - 10,
            rank1_runtime[2] + 10,
            "magi_dsa::CUDNN_CALL::indexer_score",
            rank1_runtime[3],
        ),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY"
        "(start, end, correlationId, globalPid, bytes, copyKind, copyCount) "
        "VALUES(?, ?, ?, ?, ?, 8, ?)",
        (
            (
                rank0_runtime[2] + 100,
                rank0_runtime[2] + 300,
                shared_correlation_id,
                1000 << 24,
                4_096,
                3,
            ),
            (
                rank1_runtime[2] + 100,
                rank1_runtime[2] + 400,
                shared_correlation_id,
                1001 << 24,
                8_192,
                2,
            ),
        ),
    )
    connection.commit()
    connection.close()

    records, summary = extract_memcpy_attribution_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=2,
        steps=1,
    )
    assert len(records) == 2
    assert summary["result"] == "PASS"
    assert summary["attribution_coverage"] == 1.0
    assert summary["attribution_byte_coverage"] == 1.0
    assert summary["attribution_copy_count_coverage"] == 1.0
    assert summary["bytes"] == 12_288
    assert summary["copy_count"] == 5
    assert summary["attribution_kind_activity_counts"] == {
        "cudnn_call": 1,
        "logical": 1,
    }
    assert len(summary["step_records"]) == 2
    assert all(record["result"] == "PASS" for record in summary["step_records"])
    assert all(
        record["attribution_coverage"] == 1.0
        and record["attribution_byte_coverage"] == 1.0
        and record["attribution_copy_count_coverage"] == 1.0
        for record in summary["step_records"]
    )

    rank0 = next(record for record in records if record["rank"] == 0)
    assert rank0["correlation_id"] == shared_correlation_id
    assert rank0["bytes"] == 4_096
    assert rank0["copy_count"] == 3
    assert rank0["attribution_kind"] == "logical"
    assert rank0["attribution_name"] == "magi_dsa::indexer_score"
    assert rank0["attribution_thread_scope"] == "process_temporal"

    rank1 = next(record for record in records if record["rank"] == 1)
    assert rank1["correlation_id"] == shared_correlation_id
    assert rank1["bytes"] == 8_192
    assert rank1["copy_count"] == 2
    assert rank1["attribution_kind"] == "cudnn_call"
    assert rank1["attribution_name"] == "magi_dsa::CUDNN_CALL::indexer_score"
    assert rank1["attribution_thread_scope"] == "same_thread"


def test_nsys_d2d_attribution_reports_empty_and_unowned_activity(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "memcpy_unowned.sqlite"
    report_path = tmp_path / "memcpy_unowned.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=1,
        steps=1,
        step_mode="forward-backward",
    )
    connection = sqlite3.connect(sqlite_path)
    _add_nsys_memcpy_tables(connection)
    connection.commit()
    connection.close()

    records, summary = extract_memcpy_attribution_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=1,
        steps=1,
    )
    assert records == []
    assert summary["result"] == "PASS"
    assert summary["attribution_coverage"] == 1.0
    assert summary["attribution_byte_coverage"] == 1.0
    assert summary["attribution_copy_count_coverage"] == 1.0
    assert len(summary["step_records"]) == 1
    assert summary["step_records"][0]["result"] == "PASS"
    assert summary["step_records"][0]["attribution_coverage"] == 1.0
    assert summary["step_records"][0]["attribution_byte_coverage"] == 1.0
    assert summary["step_records"][0]["attribution_copy_count_coverage"] == 1.0

    connection = sqlite3.connect(sqlite_path)
    global_pid = 1000 << 24
    global_tid = global_pid + 123
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start, end, globalTid, correlationId) "
        "VALUES(?, ?, ?, 900)",
        (1_100, 1_200, global_tid),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY"
        "(start, end, correlationId, globalPid, bytes, copyKind, copyCount) "
        "VALUES(?, ?, 900, ?, 1024, 8, 1)",
        (1_250, 1_450, global_pid),
    )
    connection.commit()
    connection.close()

    records, summary = extract_memcpy_attribution_records(
        sqlite_path,
        report_path,
        "balanced",
        world_size=1,
        steps=1,
    )
    assert len(records) == 1
    assert records[0]["attribution_kind"] == "unattributed"
    assert records[0]["attribution_path"] == []
    assert summary["result"] == "FAIL"
    assert summary["attribution_coverage"] == 0.0
    assert summary["unattributed_bytes"] == 1_024
    assert summary["unattributed_copy_count"] == 1
    assert summary["step_records"][0]["result"] == "FAIL"
    assert summary["step_records"][0]["attribution_coverage"] == 0.0
    assert summary["step_records"][0]["attribution_byte_coverage"] == 0.0
    assert summary["step_records"][0]["attribution_copy_count_coverage"] == 0.0


def test_nsys_d2d_attribution_rejects_ambiguous_runtime_correlation(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "memcpy_ambiguous_runtime.sqlite"
    report_path = tmp_path / "memcpy_ambiguous_runtime.nsys-rep"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=1,
        steps=1,
        step_mode="forward",
    )
    connection = sqlite3.connect(sqlite_path)
    _add_nsys_memcpy_tables(connection)
    runtimes = connection.execute(
        "SELECT rowid, end, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "ORDER BY start LIMIT 2"
    ).fetchall()
    assert len(runtimes) == 2
    shared_correlation_id = 700
    connection.executemany(
        "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId = ? WHERE rowid = ?",
        ((shared_correlation_id, int(runtime[0])) for runtime in runtimes),
    )
    global_pid = int(runtimes[0][2]) & ~((1 << 24) - 1)
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY"
        "(start, end, correlationId, globalPid, bytes, copyKind, copyCount) "
        "VALUES(?, ?, ?, ?, 1024, 8, 1)",
        (
            int(runtimes[0][1]) + 100,
            int(runtimes[0][1]) + 300,
            shared_correlation_id,
            global_pid,
        ),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="matched multiple runtime launches"):
        extract_memcpy_attribution_records(
            sqlite_path,
            report_path,
            "balanced",
            world_size=1,
            steps=1,
        )


def test_nsys_main_writes_global_and_per_rank_memcpy_attribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sqlite_path = tmp_path / "memcpy_output.sqlite"
    report_path = tmp_path / "memcpy_output.nsys-rep"
    output_dir = tmp_path / "output"
    report_path.write_bytes(b"fixture")
    _create_nsys_fixture(
        sqlite_path,
        "balanced",
        world_size=2,
        steps=1,
        step_mode="forward",
    )
    connection = sqlite3.connect(sqlite_path)
    _add_nsys_memcpy_tables(connection)
    runtime = connection.execute(
        "SELECT rowid, start, end, globalTid, correlationId "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start LIMIT 1"
    ).fetchone()
    assert runtime is not None
    global_pid = int(runtime[3]) & ~((1 << 24) - 1)
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY"
        "(start, end, correlationId, globalPid, bytes, copyKind, copyCount) "
        "VALUES(?, ?, ?, ?, 2048, 8, 2)",
        (int(runtime[2]) + 100, int(runtime[2]) + 300, int(runtime[4]), global_pid),
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr(
        extract_nsys_module,
        "_parse_args",
        lambda: SimpleNamespace(
            output_dir=output_dir,
            plan="balanced",
            report=report_path,
            sqlite=sqlite_path,
            step_mode="forward",
            steps=1,
            world_size=2,
        ),
    )
    extract_nsys_module.main()

    global_records = [
        json.loads(line)
        for line in (output_dir / "nsys_memcpy_attribution.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rank_records = [
        json.loads(line)
        for line in (output_dir / "rank0_nsys_memcpy_attribution.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    empty_rank_records = (output_dir / "rank1_nsys_memcpy_attribution.jsonl").read_text(
        encoding="utf-8"
    )
    summary = json.loads(
        (output_dir / "NSYS_MEMCPY_ATTRIBUTION.json").read_text(encoding="utf-8")
    )
    assert global_records == rank_records
    assert empty_rank_records == ""
    assert len(global_records) == 1
    assert global_records[0]["bytes"] == 2_048
    assert global_records[0]["copy_count"] == 2
    assert summary["result"] == "PASS"
    assert summary["attribution_coverage"] == 1.0
    assert summary["bytes"] == 2_048
    assert summary["copy_count"] == 2
    assert summary["attributed_bytes"] == 2_048
    assert summary["attributed_copy_count"] == 2
    assert summary["step_records"][0]["result"] == "PASS"
    assert summary["step_records"][0]["bytes"] == 2_048
    assert summary["step_records"][0]["copy_count"] == 2


def test_pro_cudnn_preflight_freezes_full_indexer_attention_and_sentinel_abi() -> None:
    repo_root = find_repo_root()
    source = (repo_root / "scripts/test/preflight_cudnn_dsa.py").read_text(
        encoding="utf-8"
    )
    required_constants = (
        "_PRO_INDEXER_HEADS = 64",
        "_PRO_INDEXER_DIM = 128",
        "_PRO_ATTENTION_HEADS = 128",
        "_PRO_ATTENTION_DIM = 512",
        "_PRO_INDEXER_TOPK = 1024",
        "_PRO_WINDOW_TOPK = 128",
        "_PRO_TOTAL_TOPK = _PRO_INDEXER_TOPK + _PRO_WINDOW_TOPK",
    )
    for required in required_constants:
        assert required in source
    assert "topk_length=None" in source
    assert source.count(".masked_fill(~valid, -1)") >= 2
    assert "effective_lengths = compressed_lengths + window_lengths" in source
    assert "qhead_per_kv_head=indexer_heads" in source
    assert "qhead_per_kv_head=attention_heads" in source
    assert "reference_predict" in source
    assert "reference_target" in source
    assert 'backward["d_index_k"][0, ~selected_rows]' in source


def test_correctness_and_profile_entrypoints_require_the_full_image_contract() -> None:
    repo_root = find_repo_root()
    obsolete_cudnn_patch_markers = (
        "cudnn_frontend_dense_indexer_" "no_host_sync.patch",
        "9428e144c5c44d09268f6f6281e5f8291" "b0a837d2c11d200ff32728b067b9b9a",
    )
    required_contract = {
        "org.magi-dsa.flashmla-revision": ("9241ae3ef9bac614dd25e45e507e089f888280e0"),
        "org.magi-dsa.flashmla-dual-lse-patch-revision": (
            "13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece"
        ),
        "org.magi-dsa.flashmla-dual-lse-patch-sha256": (
            "6957dbde516c73066c5911108761325edc1bdcd8f62e15dc0a84f4f290118d4b"
        ),
        "org.magi-dsa.flashmla-pro-h128-patch-revision": (
            "b7643bd54521f563b839b98289b5cd048c062ba2"
        ),
        "org.magi-dsa.flashmla-pro-h128-patch-sha256": (
            "c534e13ff432ac1c694cb24981826c11be26a2d9743d7175ddb05f887279461f"
        ),
        "org.magi-dsa.cudnn-backend": "9.24.0.43",
        "org.magi-dsa.cudnn-frontend": "1.26.0",
        "org.magi-dsa.cudnn-frontend-revision": (
            "35fd7b0d0e1d4952b904c79341c5e84e3af0a328"
        ),
        "org.magi-dsa.cudnn-frontend-local-patches": "none",
        "org.magi-dsa.cudnn-frontend-source": "official-unmodified",
        "org.magi-dsa.cutlass-dsl": "4.5.0",
        "org.magi-dsa.quack-kernels": "0.4.1",
        "org.magi-dsa.tvm-ffi": "0.1.8.post0",
        "org.magi-dsa.magi-attention-revision": "magi_source_revision",
        "org.magi-dsa.install-mode": "python-wheel",
    }
    for relative_path in (
        "scripts/test/run_multigpu.sh",
        "scripts/profile/run_5step.sh",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "require_image_label()" in source
        assert 'magi_source_revision="$(git -C "$repo_root" rev-parse HEAD)"' in source
        assert '"$artifact_dir/IMAGE_CONTRACT.txt"' in source
        for label, expected in required_contract.items():
            assert label in source, (relative_path, label)
            assert expected in source, (relative_path, expected)
        for obsolete_marker in obsolete_cudnn_patch_markers:
            assert obsolete_marker not in source

    for relative_path in (
        "docker/Dockerfile.dsa-v4",
        "scripts/image/finalize_release.py",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "org.magi-dsa.cudnn-frontend-source" in source
        assert "official-unmodified" in source
        for obsolete_marker in obsolete_cudnn_patch_markers:
            assert obsolete_marker not in source


def test_flashmla_patch_chain_checks_source_provenance_and_pro_parent() -> None:
    repo_root = find_repo_root()
    source = (repo_root / "docker/Dockerfile.dsa-v4").read_text(encoding="utf-8")
    compact = " ".join(source.replace("\\\n", " ").split())
    assert (
        "git fetch --no-tags --filter=blob:none origin "
        '"${FLASHMLA_DUAL_LSE_PATCH_REVISION}" '
        '"${FLASHMLA_PRO_H128_PATCH_REVISION}"'
    ) in compact
    assert (
        'test "$(git rev-parse "${FLASHMLA_DUAL_LSE_PATCH_REVISION}^")" '
        '= "${FLASHMLA_DUAL_LSE_PATCH_PARENT_REVISION}"'
    ) in compact
    assert "4c3d231ff3c85db9e27d87fe3356f93d0281fc33" in source
    assert (
        'test "$(git rev-parse "${FLASHMLA_PRO_H128_PATCH_REVISION}^")" '
        '= "${FLASHMLA_DUAL_LSE_PATCH_REVISION}"'
    ) in compact


def test_pro_pair_entrypoint_requires_same_capture_memcpy_and_summary_artifacts() -> (
    None
):
    repo_root = find_repo_root()
    source = (repo_root / "scripts/profile/run_5step.sh").read_text(encoding="utf-8")

    required_memcpy_artifacts = (
        "nsys_memcpy_attribution.jsonl",
        "rank{rank}_nsys_memcpy_attribution.jsonl",
        "NSYS_MEMCPY_ATTRIBUTION.json",
    )
    for artifact in required_memcpy_artifacts:
        assert artifact in source
    for field in (
        "attribution_coverage",
        "attribution_byte_coverage",
        "attribution_copy_count_coverage",
        "unattributed_memcpy_activity_count",
        "unattributed_bytes",
        "unattributed_copy_count",
        "unattributed_gpu_time_ms",
    ):
        assert field in source
    for artifact in (
        "INDEXER_D2D_PRO_PAIR.json",
        "SUPPORT_OVERHEAD_PRO_PAIR.json",
        "PRO_PAIR_COMMUNICATION_OVERLAP.json",
    ):
        assert artifact in source

    preflight = source.index("NSYS_MEMCPY_PREFLIGHT.stdout")
    summarize = source.index('"$summary_script"')
    final_contract = source.index("PRO_PAIR_ARTIFACT_CONTRACT.stdout")
    assert preflight < summarize < final_contract
    assert "same_balanced_aggregate_capture_no_replay" in source
    assert "capture_replay=none" in source
    assert source.count("/scripts/profile/run_plan.sh") == 1


def test_pro_pair_records_selected_recompute_cudnn_memcpy_scopes_without_capture_io() -> (
    None
):
    repo_root = find_repo_root()
    entrypoint_source = (repo_root / "scripts/profile/run_5step.sh").read_text(
        encoding="utf-8"
    )
    backend_source = (
        repo_root / "extensions/magi_attn_extensions/DSA/backend.py"
    ).read_text(encoding="utf-8")
    worker_source = (
        repo_root / "benchmarks/dsa_v4/profile_attention_suite.py"
    ).read_text(encoding="utf-8")

    scopes = (
        "magi_dsa::CUDNN_CALL::selected_indexer_recompute",
        "magi_dsa::CUDNN_CALL::selected_attention_recompute",
    )
    for scope in scopes:
        assert scope in entrypoint_source
    assert 'dsa_cudnn_call_range("selected_indexer_recompute")' in backend_source
    assert 'dsa_cudnn_call_range("selected_attention_recompute")' in backend_source

    capture_begin = worker_source.index(
        "torch.cuda.nvtx.range_push(_CAPTURE_SPEC.outer_nvtx)"
    )
    capture_end = worker_source.index(
        "for step in submitted_steps:",
        capture_begin,
    )
    capture_body = worker_source[capture_begin:capture_end]
    assert capture_body.count("torch.cuda.synchronize()") == 1
    assert "_record(" not in capture_body
    assert "_atomic_json(" not in capture_body
