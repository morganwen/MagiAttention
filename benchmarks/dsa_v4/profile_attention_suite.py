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
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
import torch.distributed as dist

# Add only the benchmark root so magi_attention still resolves from the installed wheel.
_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from dsa_v4.profile_5step import (  # noqa: E402
    _assert_input_identity,
    _assert_unique_topk,
    _atomic_json,
    _clear_training_gradients,
    _compare_results,
    _compare_training_gradients,
    _counter_delta,
    _counter_dict,
    _cudnn_cache_inventory,
    _derived_seed,
    _gradient_nonfinite_counts,
    _input_digest,
    _make_inputs,
    _make_plan_input,
    _parameter_digest,
    _prepare_profile_dsa_input_boundary,
    _profile_projector,
    _ProfileDSAInputBoundary,
    _ProfileSource,
    _record,
    _run_forward_backward_step,
    _source_order_result,
    _source_order_tensor,
    _training_gradient_finite_diagnostics,
    _training_gradient_snapshot,
    _wait_for_file,
)
from magi_attn_extensions.DSA.config import (  # noqa: E402
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAProModelSpec,
)
from magi_attn_extensions.DSA.modeling import MagiDSALayer  # noqa: E402
from magi_attn_extensions.DSA.nvtx import dsa_nvtx_range  # noqa: E402
from magi_attn_extensions.DSA.pro_runtime import (  # noqa: E402
    MagiDSAProExecutionBundle,
    MagiDSAProRuntimeMgr,
)
from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr  # noqa: E402
from magi_attn_extensions.DSA.types import (  # noqa: E402
    MagiDSAForwardResult,
    MagiDSAInput,
)


@dataclass(frozen=True)
class _CaptureSpec:
    """Static contract for one merged Attention capture mode."""

    step_mode: str
    forward_order: tuple[str, ...]
    outer_nvtx: str
    module_scope: str
    merged_label: str
    parameter_gradient_allreduce: str
    representative_layer_ids: dict[str, int] | None = None


_PRO_PAIR_SPEC = _CaptureSpec(
    step_mode="pro-pair",
    forward_order=("csa", "hca"),
    outer_nvtx="$Magi_DSA/capture_five_pro_pair_steps",
    module_scope="pro_pair",
    merged_label="representative_csa_hca_pair",
    parameter_gradient_allreduce="one_unified_after_two_backwards",
    representative_layer_ids={"csa": 2, "hca": 3},
)
_CAPTURE_SPEC = _PRO_PAIR_SPEC
_FORWARD_ORDER = _CAPTURE_SPEC.forward_order
_BACKWARD_ORDER = tuple(reversed(_FORWARD_ORDER))
_RATIO_BY_MODE: dict[str, Literal[4, 128]] = {
    "csa": 4,
    "hca": 128,
}
_EXPECTED_ACTIVATION_GRADIENTS = {
    4: frozenset(("x", "qr", "q", "latent_kv", "sink")),
    128: frozenset(("x", "q", "latent_kv", "sink")),
}
_EXPECTED_SENDRECV = {
    "csa": {"forward": 4, "backward": 4},
    "hca": {"forward": 3, "backward": 3},
}
_BACKWARD_INTERNAL_STREAM_FIELDS = {
    "csa": (
        "sparse_backward_stream",
        "csa_main_stream",
        "csa_indexer_stream",
        "csa_route_stream",
    ),
    "hca": (
        "hca_main_stream",
        "hca_route_stream",
    ),
}


def _configure_capture(step_mode: str) -> None:
    global _BACKWARD_ORDER, _CAPTURE_SPEC, _FORWARD_ORDER

    specs = {
        _PRO_PAIR_SPEC.step_mode: _PRO_PAIR_SPEC,
    }
    try:
        _CAPTURE_SPEC = specs[step_mode]
    except KeyError as error:
        raise ValueError(f"unsupported Attention capture mode: {step_mode}") from error
    _FORWARD_ORDER = _CAPTURE_SPEC.forward_order
    _BACKWARD_ORDER = tuple(reversed(_FORWARD_ORDER))


def _active_expected_sendrecv() -> dict[str, dict[str, int]]:
    return {mode: _EXPECTED_SENDRECV[mode] for mode in _FORWARD_ORDER}


def _module_range(operation: str) -> str:
    return f"{_CAPTURE_SPEC.module_scope}::{operation}"


@dataclass(frozen=True)
class _AttentionProfileCase:
    """One independent Attention graph captured inside the merged step."""

    name: str
    layer: MagiDSALayer
    source: _ProfileSource
    runtime: MagiDSARuntimeMgr
    handle: Any
    boundary: _ProfileDSAInputBoundary
    prepare_seconds: float
    execution_stream: torch.cuda.Stream | None = None
    pro_runtime: MagiDSAProRuntimeMgr | None = None
    pro_bundle: MagiDSAProExecutionBundle | None = None


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Magi-DSA CP8 merged Attention profile worker"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile",), required=True)
    parser.add_argument("--plan", choices=("balanced",), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--step-mode",
        choices=("pro-pair",),
        required=True,
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--layout-policy",
        choices=("structural-balanced",),
        default="structural-balanced",
    )
    parser.add_argument("--local-improvement-passes", type=int, default=4)
    parser.add_argument("--profiler-attach-warmup-steps", type=int, default=0)
    return parser.parse_args()


def _prepare_runtime(
    config: MagiDSAConfig,
    source: _ProfileSource,
    artifact_dir: Path,
    rank: int,
    label: str,
    structural_layout_config: DsaStructuralLayoutConfig | None = None,
) -> tuple[MagiDSARuntimeMgr, Any, float]:
    _record(
        artifact_dir,
        "prepare_begin",
        rank,
        attention_mode=label,
        policy="structural_balanced",
    )
    started = time.perf_counter()
    runtime = MagiDSARuntimeMgr(
        config,
        dist.group.WORLD,
        structural_layout_config=structural_layout_config,
    )
    local_token_capacity = max(
        source.packed_meta.local_token_count(rank),
        (source.packed_meta.cu_seqlens[-1] + dist.get_world_size() - 1)
        // dist.get_world_size(),
    )
    handle = runtime.prepare_execution(
        source.packed_meta,
        source.x.device,
        local_token_capacity=local_token_capacity,
        health_check=False,
    )
    elapsed_seconds = time.perf_counter() - started
    _record(
        artifact_dir,
        "prepare_end",
        rank,
        attention_mode=label,
        counters=_counter_dict(runtime),
        plan_hash=handle.plan_hash,
        policy="structural_balanced",
        elapsed_seconds=elapsed_seconds,
    )
    return runtime, handle, elapsed_seconds


def _prepare_pro_runtime_bundle(
    model_spec: MagiDSAProModelSpec,
    source: _ProfileSource,
    artifact_dir: Path,
    rank: int,
    structural_layout_config: DsaStructuralLayoutConfig,
) -> tuple[MagiDSAProRuntimeMgr, MagiDSAProExecutionBundle, float]:
    """Prepare the representative pair from one Pro runtime and one plan bundle."""

    _record(
        artifact_dir,
        "prepare_begin",
        rank,
        attention_mode="pro_runtime_bundle",
        policy="structural_balanced",
    )
    started = time.perf_counter()
    pro_runtime = MagiDSAProRuntimeMgr(
        dist.group.WORLD,
        model_spec=model_spec,
        structural_layout_config=structural_layout_config,
    )
    local_token_capacity = max(
        source.packed_meta.local_token_count(rank),
        (source.packed_meta.cu_seqlens[-1] + dist.get_world_size() - 1)
        // dist.get_world_size(),
    )
    bundle = pro_runtime.prepare_execution(
        source.packed_meta,
        source.x.device,
        local_token_capacity=local_token_capacity,
        health_check=False,
    )
    elapsed_seconds = time.perf_counter() - started
    _record(
        artifact_dir,
        "prepare_end",
        rank,
        attention_mode="pro_runtime_bundle",
        counters={
            "csa": _counter_dict(pro_runtime.csa_runtime),
            "hca": _counter_dict(pro_runtime.hca_runtime),
        },
        plan_hash={"csa": bundle.csa.plan_hash, "hca": bundle.hca.plan_hash},
        policy="structural_balanced",
        query_layout_hash=bundle.query_layout_hash,
        elapsed_seconds=elapsed_seconds,
    )
    return pro_runtime, bundle, elapsed_seconds


def _prepare_pro_pair_boundaries(
    sources: dict[str, _ProfileSource],
    pro_runtime: MagiDSAProRuntimeMgr,
    bundle: MagiDSAProExecutionBundle,
    global_dout: torch.Tensor,
) -> dict[str, _ProfileDSAInputBoundary]:
    """Layout one source hidden tensor once, then project two independent leaves."""

    if set(sources) != {"csa", "hca"}:
        raise ValueError("the Pro pair requires exactly CSA and HCA sources")
    csa_source = sources["csa"]
    hca_source = sources["hca"]
    if (
        csa_source.x is not hca_source.x
        or csa_source.packed_meta is not hca_source.packed_meta
    ):
        raise ValueError("the Pro pair must share one source x and packed metadata")

    layout_start = torch.cuda.Event(enable_timing=True)
    layout_end = torch.cuda.Event(enable_timing=True)
    projected: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        layout_start.record()
        local_x = pro_runtime.layout_source_hidden_once(
            csa_source.x.detach(),
            bundle,
        )
        layout_end.record()
        for mode, layer_id in (("csa", 2), ("hca", 3)):
            runtime = pro_runtime.runtime_for_ratio(_RATIO_BY_MODE[mode])
            handle = pro_runtime.handle_for_layer(layer_id, bundle)
            projected[mode] = _profile_projector(
                local_x,
                runtime.get_position_ids(handle),
                runtime.config,
            )
        rank_plan = bundle.csa.plan.rank_plans[bundle.csa.rank]
        query_rows = torch.tensor(
            rank_plan.local_query_global_rows,
            dtype=torch.int64,
            device=global_dout.device,
        )
        local_dout = global_dout.index_select(0, query_rows).contiguous()
    torch.cuda.synchronize(local_x.device)
    layout_ms = float(layout_start.elapsed_time(layout_end))

    boundaries: dict[str, _ProfileDSAInputBoundary] = {}
    for mode in ("csa", "hca"):
        qr, q, latent_kv = projected[mode]
        source = sources[mode]
        boundaries[mode] = _ProfileDSAInputBoundary(
            value=MagiDSAInput(
                x=local_x.detach().clone().requires_grad_(True),
                qr=qr.detach().requires_grad_(True),
                q=q.detach().requires_grad_(True),
                latent_kv=latent_kv.detach().requires_grad_(True),
                sink=source.sink,
                packed_meta=source.packed_meta,
            ),
            dout=local_dout,
            dkl=torch.ones((), dtype=torch.float32, device=local_x.device),
            token_layout_forward_ms=layout_ms,
        )
    return boundaries


def _calc_attention_case(
    case: _AttentionProfileCase,
    dsa_input: MagiDSAInput,
) -> MagiDSAForwardResult:
    if case.pro_runtime is None:
        if case.pro_bundle is not None:
            raise ValueError("a Pro bundle requires its owning Pro runtime")
        return case.runtime.calc_dsa(case.layer.projections(), dsa_input, case.handle)
    if case.pro_bundle is None or case.layer.layer_id is None:
        raise ValueError("a Pro profile case requires a bound layer and bundle")
    return case.pro_runtime.calc_layer(
        case.layer.layer_id,
        case.layer,
        dsa_input,
        case.pro_bundle,
    )


def _clear_attention_suite_gradients(
    cases: dict[str, _AttentionProfileCase],
) -> None:
    for name in _FORWARD_ORDER:
        case = cases[name]
        _clear_training_gradients(case.layer, case.source, case.boundary)


def _all_reduce_attention_suite_gradients(
    cases: dict[str, _AttentionProfileCase],
) -> None:
    named_gradients: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    missing: list[str] = []
    for mode in _FORWARD_ORDER:
        case = cases[mode]
        sink_gradient = case.source.sink.grad
        if sink_gradient is None:
            missing.append(f"{mode}::sink")
        else:
            named_gradients.append((f"{mode}::sink", case.source.sink, sink_gradient))
        for name, parameter in case.layer.named_parameters():
            if parameter.grad is None:
                missing.append(f"{mode}::{name}")
            else:
                named_gradients.append((f"{mode}::{name}", parameter, parameter.grad))
    if missing:
        raise AssertionError(
            f"{_CAPTURE_SPEC.step_mode} profile is missing model gradients: {missing}"
        )

    devices = {gradient.device for _, _, gradient in named_gradients}
    if len(devices) != 1:
        raise AssertionError(
            f"{_CAPTURE_SPEC.step_mode} gradients must share one device before "
            "the FP32 main-grad reducer"
        )
    device = next(iter(devices))
    device_name = str(device).replace(":", "_")
    scope = (
        f"{_CAPTURE_SPEC.module_scope}::gradient_allreduce::bucket::"
        f"0::{device_name}::fp32_main_grad"
    )
    with dsa_nvtx_range(f"{scope}::pack", enabled=device.type == "cuda"):
        flat_gradient = torch.cat(
            [
                gradient.detach().reshape(-1).float()
                for _, _, gradient in named_gradients
            ]
        )
    with dsa_nvtx_range(f"{scope}::collective", enabled=device.type == "cuda"):
        dist.all_reduce(flat_gradient)
    with dsa_nvtx_range(f"{scope}::bind_views", enabled=device.type == "cuda"):
        offset = 0
        for _, owner, gradient in named_gradients:
            next_offset = offset + gradient.numel()
            reduced = flat_gradient[offset:next_offset].view_as(gradient)
            owner.grad = reduced.to(dtype=gradient.dtype)
            offset = next_offset
        if offset != flat_gradient.numel():
            raise AssertionError(
                f"{_CAPTURE_SPEC.step_mode} FP32 main-grad bucket reconstruction "
                "is incomplete"
            )


def _backward_attention_case(
    case: _AttentionProfileCase,
    result: MagiDSAForwardResult,
) -> None:
    if result.output.shape != case.boundary.dout.shape:
        raise ValueError(f"{case.name} dout shape does not match its output")
    if case.layer.config.ratio == 4:
        if result.kl.ndim != 0 or result.kl.dtype != case.boundary.dkl.dtype:
            raise ValueError("CSA dkl does not match the selected KL scalar")
        torch.autograd.backward(
            (result.output, result.kl),
            (case.boundary.dout, case.boundary.dkl),
        )
        return
    if result.kl.requires_grad or result.kl.ndim != 0:
        raise ValueError(f"{case.name} must expose a detached scalar zero KL")
    torch.autograd.backward((result.output,), (case.boundary.dout,))


def _join_attention_case_backward_streams(
    mode: str,
    case: _AttentionProfileCase,
) -> tuple[str, ...]:
    """Make the mode stream observe every internal backward stream."""

    execution_stream = case.execution_stream
    if execution_stream is None:
        return ()
    try:
        stream_fields = _BACKWARD_INTERNAL_STREAM_FIELDS[mode]
    except KeyError as error:
        raise ValueError(
            f"unsupported Attention mode for backward join: {mode}"
        ) from error
    streams: list[tuple[str, torch.cuda.Stream]] = []
    for field in stream_fields:
        internal_stream = getattr(case.handle, field)
        if internal_stream is not None:
            streams.append((field, internal_stream))
    if not streams:
        return ()

    parent_scope = _module_range(f"stream_overlap::backward_completion_join::{mode}")
    with dsa_nvtx_range(parent_scope):
        for field, internal_stream in streams:
            with dsa_nvtx_range(f"{parent_scope}::{field}"):
                execution_stream.wait_stream(internal_stream)
    return tuple(field for field, _ in streams)


def _run_attention_suite_step(
    cases: dict[str, _AttentionProfileCase],
    rank: int,
) -> tuple[dict[str, MagiDSAForwardResult], dict[str, MagiDSAInput]]:
    results: dict[str, MagiDSAForwardResult] = {}
    inputs: dict[str, MagiDSAInput] = {}
    stream_enabled = all(
        cases[mode].execution_stream is not None for mode in _FORWARD_ORDER
    )
    caller_stream = torch.cuda.current_stream() if stream_enabled else None
    suite_ready = torch.cuda.Event() if stream_enabled else None
    if suite_ready is not None:
        assert caller_stream is not None
        suite_ready.record(caller_stream)
    forward_done: dict[str, torch.cuda.Event] = {}
    previous_mode: str | None = None
    with _nvtx_range("magi_dsa::forward"):
        for mode in _FORWARD_ORDER:
            case = cases[mode]
            with _nvtx_range(
                f"magi_dsa::{_CAPTURE_SPEC.module_scope}::{mode}::forward"
            ):
                if case.execution_stream is None:
                    dsa_input = _make_plan_input(
                        case.source,
                        case.runtime,
                        case.handle,
                        retain_input_gradients=True,
                        input_boundary=case.boundary,
                    )
                    with _nvtx_range(f"balanced/rank_{rank}/{mode.upper()}/O"):
                        result = _calc_attention_case(case, dsa_input)
                else:
                    assert suite_ready is not None
                    ready = (
                        suite_ready
                        if previous_mode is None
                        else forward_done[previous_mode]
                    )
                    if previous_mode is None:
                        case.execution_stream.wait_event(ready)
                    else:
                        with dsa_nvtx_range(
                            f"{_CAPTURE_SPEC.module_scope}::mode_serial::"
                            f"{previous_mode}_forward_to_{mode}_forward"
                        ):
                            case.execution_stream.wait_event(ready)
                    with torch.cuda.stream(case.execution_stream):
                        dsa_input = _make_plan_input(
                            case.source,
                            case.runtime,
                            case.handle,
                            retain_input_gradients=True,
                            input_boundary=case.boundary,
                        )
                        with _nvtx_range(f"balanced/rank_{rank}/{mode.upper()}/O"):
                            result = _calc_attention_case(case, dsa_input)
                if result.output.shape != case.boundary.dout.shape:
                    raise ValueError(f"{mode} output has the wrong profile shape")
                inputs[mode] = dsa_input
                results[mode] = result
                if case.execution_stream is not None:
                    done = torch.cuda.Event()
                    done.record(case.execution_stream)
                    forward_done[mode] = done
                previous_mode = mode

    backward_done: dict[str, torch.cuda.Event] = {}
    previous_mode = None
    with _nvtx_range("magi_dsa::backward"):
        for mode in _BACKWARD_ORDER:
            case = cases[mode]
            with _nvtx_range(
                f"magi_dsa::{_CAPTURE_SPEC.module_scope}::{mode}::backward"
            ):
                if case.execution_stream is None:
                    _backward_attention_case(case, results[mode])
                else:
                    if previous_mode is not None:
                        with dsa_nvtx_range(
                            f"{_CAPTURE_SPEC.module_scope}::mode_serial::"
                            f"{previous_mode}_backward_to_{mode}_backward"
                        ):
                            case.execution_stream.wait_event(
                                backward_done[previous_mode]
                            )
                    with torch.cuda.stream(case.execution_stream):
                        _backward_attention_case(case, results[mode])
                        _join_attention_case_backward_streams(mode, case)
                if case.execution_stream is not None:
                    done = torch.cuda.Event()
                    done.record(case.execution_stream)
                    backward_done[mode] = done
                previous_mode = mode

    with _nvtx_range("magi_dsa::parameter_gradient_allreduce"):
        if stream_enabled:
            assert caller_stream is not None
            with dsa_nvtx_range(_module_range("stream_overlap::gradient_join")):
                for mode in _BACKWARD_ORDER:
                    caller_stream.wait_event(backward_done[mode])
        _all_reduce_attention_suite_gradients(cases)
    return results, inputs


def _make_global_dout(
    config: MagiDSAConfig,
    tokens: int,
    seed: int,
    device: torch.device,
    artifact_dir: Path,
    rank: int,
) -> tuple[torch.Tensor, float]:
    dout_seed = _derived_seed(seed, "dout", -1)
    global_output_elements = tokens * config.num_query_heads * config.head_dim
    if global_output_elements <= 0:
        raise ValueError(f"{_CAPTURE_SPEC.step_mode} dout requires a non-empty output")
    scale = 1.0 / global_output_elements
    _record(
        artifact_dir,
        "profile_global_dout_begin",
        rank,
        global_output_elements=global_output_elements,
        scale=scale,
        seed=dout_seed,
        shape=(tokens, config.num_query_heads, config.head_dim),
    )
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(dout_seed)
        global_dout = torch.randn(
            (tokens, config.num_query_heads, config.head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        global_dout.mul_(scale)
    return global_dout, scale


def _prewarm(
    cases: dict[str, _AttentionProfileCase],
    csa_shadow: _AttentionProfileCase,
    warmup: int,
    artifact_dir: Path,
    rank: int,
) -> None:
    if warmup <= 0:
        raise ValueError(f"{_CAPTURE_SPEC.step_mode} warmup count must be positive")

    for iteration in range(warmup):
        _record(
            artifact_dir,
            "prewarm_begin",
            rank,
            attention_mode="csa_shadow",
            iteration=iteration,
            step_mode=_CAPTURE_SPEC.step_mode,
        )
        _clear_training_gradients(
            csa_shadow.layer,
            csa_shadow.source,
            csa_shadow.boundary,
        )
        result, dsa_input = _run_forward_backward_step(
            csa_shadow.layer,
            csa_shadow.source,
            csa_shadow.runtime,
            csa_shadow.handle,
            "sequential",
            rank,
            csa_shadow.boundary,
        )
        torch.cuda.synchronize()
        del result, dsa_input
        _record(
            artifact_dir,
            "prewarm_end",
            rank,
            attention_mode="csa_shadow",
            iteration=iteration,
            step_mode=_CAPTURE_SPEC.step_mode,
        )

    for iteration in range(warmup):
        _record(
            artifact_dir,
            "prewarm_begin",
            rank,
            attention_mode=_CAPTURE_SPEC.merged_label,
            iteration=iteration,
            step_mode=_CAPTURE_SPEC.step_mode,
        )
        _clear_attention_suite_gradients(cases)
        results, inputs = _run_attention_suite_step(cases, rank)
        torch.cuda.synchronize()
        del results, inputs
        _record(
            artifact_dir,
            "prewarm_end",
            rank,
            attention_mode=_CAPTURE_SPEC.merged_label,
            iteration=iteration,
            step_mode=_CAPTURE_SPEC.step_mode,
        )

    _clear_attention_suite_gradients(cases)
    _clear_training_gradients(
        csa_shadow.layer,
        csa_shadow.source,
        csa_shadow.boundary,
    )


def _profiler_attach_warmup(
    cases: dict[str, _AttentionProfileCase],
    steps: int,
    artifact_dir: Path,
    control_group: dist.ProcessGroup,
    rank: int,
) -> dict[str, dict[str, int]]:
    if steps < 0:
        raise ValueError("profiler-attach warmup steps must be non-negative")

    before = {mode: _counter_dict(cases[mode].runtime) for mode in _FORWARD_ORDER}
    if steps == 0:
        return {
            mode: _counter_delta(before[mode], before[mode]) for mode in _FORWARD_ORDER
        }

    _record(
        artifact_dir,
        "profiler_attach_warmup_begin",
        rank,
        steps=steps,
        step_mode=_CAPTURE_SPEC.step_mode,
    )
    torch.cuda.nvtx.range_push("$Magi_DSA/ablation_profiler_attach_warmup")
    try:
        with torch.enable_grad():
            for iteration in range(steps):
                torch.cuda.nvtx.range_push(
                    "magi_dsa::ablation::profiler_attach_warmup_step"
                )
                try:
                    _clear_attention_suite_gradients(cases)
                    results, inputs = _run_attention_suite_step(cases, rank)
                finally:
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del results, inputs
    finally:
        torch.cuda.nvtx.range_pop()

    _clear_attention_suite_gradients(cases)
    dist.barrier(group=control_group)
    after = {mode: _counter_dict(cases[mode].runtime) for mode in _FORWARD_ORDER}
    deltas = {
        mode: _counter_delta(before[mode], after[mode]) for mode in _FORWARD_ORDER
    }
    expected = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": steps,
    }
    if any(delta != expected for delta in deltas.values()):
        raise AssertionError(
            "profiler-attach warmup counter delta mismatch: " f"{deltas}"
        )
    _record(
        artifact_dir,
        "profiler_attach_warmup_end",
        rank,
        counter_delta=deltas,
        steps=steps,
        step_mode=_CAPTURE_SPEC.step_mode,
    )
    return deltas


def _attention_gradient_snapshot(
    case: _AttentionProfileCase,
    dsa_input: MagiDSAInput,
) -> dict[str, torch.Tensor]:
    expected = _EXPECTED_ACTIVATION_GRADIENTS[case.layer.config.ratio]
    tensors = {
        "x": dsa_input.x,
        "qr": dsa_input.qr,
        "q": dsa_input.q,
        "latent_kv": dsa_input.latent_kv,
        "sink": case.source.sink,
    }
    snapshot: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        gradient = tensor.grad
        if name in expected and gradient is None:
            raise AssertionError(
                f"{case.name} profile is missing activation gradient: {name}"
            )
        if name not in expected and gradient is not None:
            raise AssertionError(
                f"{case.name} profile unexpectedly produced gradient: {name}"
            )
        if gradient is None:
            continue
        value = gradient.detach()
        if name != "sink":
            value = _source_order_tensor(value, case.handle)
        else:
            value = value.clone()
        snapshot[f"input::{name}"] = value
    for name, parameter in case.layer.named_parameters():
        if parameter.grad is None:
            raise AssertionError(
                f"{case.name} profile is missing parameter gradient: {name}"
            )
        snapshot[f"parameter::{name}"] = parameter.grad.detach().clone()
    return snapshot


def _finite_gradient_diagnostics(
    snapshots: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    modes: dict[str, object] = {}
    for mode in _FORWARD_ORDER:
        tensors: dict[str, object] = {}
        for name, value in sorted(snapshots[mode].items()):
            counts = _gradient_nonfinite_counts(value)
            if int(counts["nonfinite"]) != 0:
                raise AssertionError(
                    f"{mode} profile gradient contains non-finite values: {name}"
                )
            tensors[name] = counts
        modes[mode] = {
            "gradient_tensors": len(tensors),
            "tensors": tensors,
        }
    return {"modes": modes}


def _validate_mode_result(
    mode: str,
    result: MagiDSAForwardResult,
) -> dict[str, object]:
    output_finite = bool(torch.all(torch.isfinite(result.output)).item())
    sparse_lse_finite = bool(torch.all(torch.isfinite(result.sparse_lse)).item())
    if not output_finite or not sparse_lse_finite:
        raise AssertionError(f"{mode} output or sparse LSE contains non-finite values")
    if mode == "csa":
        _assert_unique_topk(result, f"{_CAPTURE_SPEC.step_mode} CSA")
        return {
            "indexer": True,
            "output_finite": output_finite,
            "sparse_lse_finite": sparse_lse_finite,
            "topk_backend_native_valid": True,
        }
    if (
        result.topk_ids.shape != (result.output.shape[0], 0)
        or result.topk_length.shape != (result.output.shape[0],)
        or bool(torch.any(result.topk_length != 0).item())
        or bool(torch.any(~torch.isneginf(result.indexer_lse)).item())
        or float(result.kl.item()) != 0.0
    ):
        raise AssertionError(f"{mode} must expose the non-Indexer result schema")
    report: dict[str, object] = {
        "indexer": False,
        "output_finite": output_finite,
        "sparse_lse_finite": sparse_lse_finite,
        "topk_backend_native_valid": True,
    }
    return report


def _runtime_metadata(case: _AttentionProfileCase, rank: int) -> dict[str, object]:
    rank_plan = case.handle.plan.rank_plans[rank]
    layout_metrics = case.handle.plan.layout_metrics
    rank_cost = (
        None if layout_metrics is None else asdict(layout_metrics.rank_costs[rank])
    )
    layout_solver: dict[str, object] | None = None
    if layout_metrics is not None:
        layout_solver = asdict(layout_metrics)
        layout_solver.pop("rank_costs", None)
        layout_solver.pop("key", None)
    indexer_packing: dict[str, object] | None = None
    if case.layer.config.ratio == 4:
        packed_rows = rank_plan.packed_indexer_k_count
        route = case.handle.plan.compressed_ki_route
        if route is None:
            raise AssertionError("CSA profile plan is missing COMPRESSED_KI metadata")
        unique_rows = route.consumer_row_count(case.handle.rank)
        duplicate_rows = packed_rows - unique_rows
        if duplicate_rows < 0:
            raise AssertionError("CSA packed Indexer rows are smaller than unique rows")
        row_bytes = case.layer.config.indexer_head_dim * 2
        indexer_packing = {
            "duplicate_indexer_k_bytes": duplicate_rows * row_bytes,
            "duplicate_indexer_k_rows": duplicate_rows,
            "indexer_k_row_bytes": row_bytes,
            "packed_indexer_k_bytes": packed_rows * row_bytes,
            "packed_indexer_k_rows": packed_rows,
            "packing_amplification": (
                packed_rows / unique_rows if unique_rows else 1.0
            ),
            "unique_indexer_k_bytes": unique_rows * row_bytes,
            "unique_indexer_k_rows": unique_rows,
        }
    return {
        "attention_mode": case.name,
        "counter_snapshot": _counter_dict(case.runtime),
        "declared_local_token_capacity": case.handle.local_token_capacity,
        "final_query_tokens": rank_plan.local_token_count,
        "layer_id": case.layer.layer_id,
        "plan_hash": case.handle.plan_hash,
        "policy": "structural_balanced",
        "query_layout_hash": case.handle.plan.query_layout_hash,
        "layout_key": (
            None
            if layout_metrics is None or not hasattr(layout_metrics, "key")
            else list(layout_metrics.key)
        ),
        "layout_solver": layout_solver,
        "layout_rank_cost": rank_cost,
        "indexer_k_packing": indexer_packing,
        "prepare_seconds": case.prepare_seconds,
        "query_fragments": len(rank_plan.query_fragments),
        "ratio": case.layer.config.ratio,
        "source_tokens": rank_plan.source_token_count,
        "token_layout_forward_ms": case.boundary.token_layout_forward_ms,
    }


def _pro_bundle_contract(
    cases: dict[str, _AttentionProfileCase],
) -> dict[str, object]:
    """Validate and serialize the single-bundle Pro-pair execution contract."""

    if _CAPTURE_SPEC is not _PRO_PAIR_SPEC or set(cases) != {"csa", "hca"}:
        raise ValueError("the Pro bundle contract is only valid for the Pro pair")
    csa = cases["csa"]
    hca = cases["hca"]
    if (
        csa.pro_runtime is None
        or csa.pro_runtime is not hca.pro_runtime
        or csa.pro_bundle is None
        or csa.pro_bundle is not hca.pro_bundle
    ):
        raise ValueError("the Pro pair does not share one runtime execution bundle")
    pro_runtime = csa.pro_runtime
    bundle = csa.pro_bundle
    if (
        csa.runtime is not pro_runtime.csa_runtime
        or hca.runtime is not pro_runtime.hca_runtime
        or csa.handle is not bundle.csa
        or hca.handle is not bundle.hca
        or csa.layer.layer_id != 2
        or hca.layer.layer_id != 3
    ):
        raise ValueError("the Pro pair cases are not bound to bundle layers 2 and 3")
    if (
        csa.source.x is not hca.source.x
        or csa.source.packed_meta is not hca.source.packed_meta
    ):
        raise ValueError("the Pro pair does not share one source hidden layout")

    handles: dict[str, dict[str, object]] = {}
    for mode, case in (("csa", csa), ("hca", hca)):
        handles[mode] = {
            "declared_local_token_capacity": case.handle.local_token_capacity,
            "layer_id": case.layer.layer_id,
            "plan_hash": case.handle.plan_hash,
            "policy": "structural_balanced",
            "query_layout_hash": case.handle.plan.query_layout_hash,
            "ratio": case.handle.plan.ratio,
        }
    return {
        "pro_bundle_handles": handles,
        "pro_runtime_bundle": True,
        "shared_bundle_query_layout_hash": bundle.query_layout_hash,
        "shared_source_packed_meta": True,
        "shared_source_x": True,
        "token_layout_invocations": 1,
    }


def _run_profile(
    args: argparse.Namespace,
    cases: dict[str, _AttentionProfileCase],
    csa_shadow: _AttentionProfileCase,
    identities: dict[str, dict[str, tuple[int, int]]],
    control_group: dist.ProcessGroup,
    rank: int,
    dout_scale: float,
) -> dict[str, object]:
    ready_counters = {
        mode: _counter_dict(cases[mode].runtime) for mode in _FORWARD_ORDER
    }
    cache_before = _cudnn_cache_inventory()
    _atomic_json(
        args.artifact_dir / f"ready_rank{rank}.json",
        {
            "attention_order": list(_FORWARD_ORDER),
            "backward_order": list(_BACKWARD_ORDER),
            "cache": cache_before,
            "counters": ready_counters,
            "plan": "balanced",
            "profiler_attach_warmup_steps": args.profiler_attach_warmup_steps,
            "rank": rank,
            "step_mode": _CAPTURE_SPEC.step_mode,
        },
    )
    _record(
        args.artifact_dir,
        "profile_ready",
        rank,
        attention_order=list(_FORWARD_ORDER),
        plan="balanced",
        step_mode=_CAPTURE_SPEC.step_mode,
    )
    _wait_for_file(
        args.artifact_dir / "control" / "start",
        900.0,
        args.artifact_dir,
        rank,
    )
    dist.barrier(group=control_group)
    attach_warmup_delta = _profiler_attach_warmup(
        cases,
        args.profiler_attach_warmup_steps,
        args.artifact_dir,
        control_group,
        rank,
    )
    cache_after_attach_warmup = _cudnn_cache_inventory()
    if cache_after_attach_warmup != cache_before:
        raise AssertionError(
            "cuDNN cache changed during profiler-attach warmup: "
            f"before={cache_before}, after={cache_after_attach_warmup}"
        )
    before = {mode: _counter_dict(cases[mode].runtime) for mode in _FORWARD_ORDER}
    shadow_before = _counter_dict(csa_shadow.runtime)
    torch.cuda.reset_peak_memory_stats()

    last_results: dict[str, MagiDSAForwardResult] | None = None
    last_inputs: dict[str, MagiDSAInput] | None = None
    submitted_steps: list[int] = []
    torch.cuda.nvtx.range_push(_CAPTURE_SPEC.outer_nvtx)
    try:
        with torch.enable_grad():
            for step in range(args.steps):
                torch.cuda.nvtx.range_push(f"balanced/rank_{rank}/training_step_{step}")
                try:
                    with dsa_nvtx_range(
                        _module_range("gradient_clear"),
                        enabled=True,
                    ):
                        _clear_attention_suite_gradients(cases)
                    last_results, last_inputs = _run_attention_suite_step(cases, rank)
                finally:
                    torch.cuda.nvtx.range_pop()
                submitted_steps.append(step)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()

    for step in submitted_steps:
        _record(
            args.artifact_dir,
            "profile_step_submitted",
            rank,
            deferred_until_capture_sync=True,
            plan="balanced",
            step=step,
            step_mode=_CAPTURE_SPEC.step_mode,
        )
    if last_results is None or last_inputs is None:
        raise AssertionError(
            f"{_CAPTURE_SPEC.step_mode} profile did not execute a step"
        )

    after = {mode: _counter_dict(cases[mode].runtime) for mode in _FORWARD_ORDER}
    deltas = {
        mode: _counter_delta(before[mode], after[mode]) for mode in _FORWARD_ORDER
    }
    expected_delta = {
        "device_materializations": 0,
        "health_checks": 0,
        "object_collective_invocations": 0,
        "solver_invocations": 0,
        "warm_invocations": args.steps,
    }
    for mode, delta in deltas.items():
        if delta != expected_delta:
            raise AssertionError(
                f"{mode} {_CAPTURE_SPEC.step_mode} warm counter delta mismatch: {delta}"
            )
    shadow_capture_delta = _counter_delta(
        shadow_before,
        _counter_dict(csa_shadow.runtime),
    )
    expected_shadow_delta = dict(expected_delta)
    expected_shadow_delta["warm_invocations"] = 0
    if shadow_capture_delta != expected_shadow_delta:
        raise AssertionError(
            "CSA sequential shadow ran inside capture: " f"{shadow_capture_delta}"
        )

    dist.barrier(group=control_group)
    _atomic_json(
        args.artifact_dir / f"capture_done_rank{rank}.json",
        {
            "attention_counter_delta": deltas,
            "plan": "balanced",
            "rank": rank,
            "step_mode": _CAPTURE_SPEC.step_mode,
        },
    )
    _record(
        args.artifact_dir,
        "profile_capture_done",
        rank,
        plan="balanced",
        step_mode=_CAPTURE_SPEC.step_mode,
    )
    _wait_for_file(
        args.artifact_dir / "control" / "capture_stopped",
        600.0,
        args.artifact_dir,
        rank,
    )

    target_snapshots = {
        mode: _attention_gradient_snapshot(cases[mode], last_inputs[mode])
        for mode in _FORWARD_ORDER
    }
    gradient_finite = _finite_gradient_diagnostics(target_snapshots)
    mode_metrics = {
        mode: _validate_mode_result(mode, last_results[mode]) for mode in _FORWARD_ORDER
    }

    csa_case = cases["csa"]
    _clear_training_gradients(
        csa_case.layer,
        csa_case.source,
        csa_case.boundary,
    )
    _clear_training_gradients(
        csa_shadow.layer,
        csa_shadow.source,
        csa_shadow.boundary,
    )
    _record(
        args.artifact_dir,
        "shadow_begin",
        rank,
        plan="sequential",
        step_mode=_CAPTURE_SPEC.step_mode,
    )
    shadow_result_local, shadow_input = _run_forward_backward_step(
        csa_shadow.layer,
        csa_shadow.source,
        csa_shadow.runtime,
        csa_shadow.handle,
        "sequential",
        rank,
        csa_shadow.boundary,
    )
    torch.cuda.synchronize()
    shadow_gradients = _training_gradient_snapshot(
        csa_shadow.layer,
        csa_shadow.source,
        shadow_input,
        csa_shadow.handle,
        csa_shadow.boundary,
    )
    finite_comparison = _training_gradient_finite_diagnostics(
        target_snapshots["csa"],
        shadow_gradients,
    )
    finite_comparison.update(
        {
            "rank": rank,
            "shadow_plan": "structural_balanced",
            "target_plan": "balanced",
        }
    )
    _atomic_json(
        args.artifact_dir / f"gradient_finite_diagnostic_rank{rank}.json",
        finite_comparison,
    )
    gradient_metrics = _compare_training_gradients(
        target_snapshots["csa"],
        shadow_gradients,
        diagnostic_path=(args.artifact_dir / f"gradient_comparison_rank{rank}.json"),
    )
    target_result = _source_order_result(last_results["csa"], csa_case.handle)
    shadow_result = _source_order_result(
        shadow_result_local,
        csa_shadow.handle,
    )
    csa_metrics = _compare_results(
        target_result,
        shadow_result,
        diagnostic_path=args.artifact_dir / f"topk_diagnostic_rank{rank}.json",
    )
    _record(
        args.artifact_dir,
        "shadow_end",
        rank,
        plan="sequential",
        step_mode=_CAPTURE_SPEC.step_mode,
    )

    for mode in _FORWARD_ORDER:
        _assert_input_identity(cases[mode].source, identities[mode])
    cache_after = _cudnn_cache_inventory()
    if cache_after != cache_before:
        raise AssertionError(
            f"cuDNN cache changed after {_CAPTURE_SPEC.step_mode} capture/prewarm: "
            f"before={cache_before}, after={cache_after}"
        )
    dist.barrier(group=control_group)
    report = {
        "attention_counter_delta": deltas,
        "attention_order": list(_FORWARD_ORDER),
        "backward_order": list(_BACKWARD_ORDER),
        "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
        "cache_after": cache_after,
        "cache_before": cache_before,
        "csa_gradient_metrics": gradient_metrics,
        "csa_shadow_metrics": csa_metrics,
        "dout_scale": dout_scale,
        "expected_sendrecv": _active_expected_sendrecv(),
        "gradient_accumulation": False,
        "gradient_finite": gradient_finite,
        "independent_attention_graphs": True,
        "loss_capture": "none",
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "mode_serialization": "cuda_event_happens_before",
        "mode_backward_completion_join": {
            mode: list(_BACKWARD_INTERNAL_STREAM_FIELDS[mode])
            for mode in _FORWARD_ORDER
        },
        "mode_metrics": mode_metrics,
        "overlap_accounting": "same_mode_same_direction_non_route_compute",
        "parameter_gradient_allreduce": (_CAPTURE_SPEC.parameter_gradient_allreduce),
        "plan": "balanced",
        "profiler_attach_warmup_counter_delta": attach_warmup_delta,
        "profiler_attach_warmup_steps": args.profiler_attach_warmup_steps,
        "profile_gradient_boundary": "post_projection_magi_dsa_input",
        "projection_capture": "pre_capture_once_per_attention",
        "rank": rank,
        "ratios": [_RATIO_BY_MODE[mode] for mode in _FORWARD_ORDER],
        "representative_layer_ids": _CAPTURE_SPEC.representative_layer_ids,
        "representative_pair_semantics": (
            "independent_post_projection_graphs_serialized_in_layer_order"
            if _CAPTURE_SPEC is _PRO_PAIR_SPEC
            else None
        ),
        "result": "PASS",
        "shadow_plan": "csa_shadow",
        "step_mode": _CAPTURE_SPEC.step_mode,
        "token_layout_capture": "pre_capture_once_per_attention",
        "layout_policy": args.layout_policy,
    }
    if _CAPTURE_SPEC is _PRO_PAIR_SPEC:
        report.update(_pro_bundle_contract(cases))
        report.update(
            {
                "parameter_gradient_allreduce_in_7f7b": False,
                "parameter_gradient_reducer_precision": (
                    "fp32_cp_bucket_model_side_diagnostic"
                ),
                "runtime_parameter_gradient_communication": False,
            }
        )
    return report


def main() -> None:
    args = _parse_args()
    _configure_capture(args.step_mode)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "control").mkdir(exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError(f"the {_CAPTURE_SPEC.step_mode} profile requires eight ranks")
    if args.seed != 0:
        raise ValueError(f"the {_CAPTURE_SPEC.step_mode} seed is frozen to zero")
    if args.tokens != 131072 or args.steps != 5 or args.warmup != 3:
        raise ValueError(
            f"the {_CAPTURE_SPEC.step_mode} profile requires 128K, five steps, "
            "and three warmups"
        )
    if not 0 <= args.profiler_attach_warmup_steps <= 8:
        raise ValueError("profiler-attach warmup steps must be between zero and eight")
    if args.layout_policy != "structural-balanced":
        raise ValueError("the Pro pair requires the structural-balanced layout")
    if args.profiler_attach_warmup_steps != 0:
        raise ValueError("the formal Pro pair does not enable the attach warmup")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(minutes=10),
    )
    try:
        device = torch.device("cuda", local_rank)
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise RuntimeError(
                f"the {_CAPTURE_SPEC.step_mode} profile requires B300 SM103"
            )

        model_spec = MagiDSAProModelSpec() if _CAPTURE_SPEC is _PRO_PAIR_SPEC else None
        if model_spec is None:
            configs = {
                mode: MagiDSAConfig(ratio=_RATIO_BY_MODE[mode])
                for mode in _FORWARD_ORDER
            }
        else:
            assert _CAPTURE_SPEC.representative_layer_ids is not None
            configs = {
                mode: model_spec.make_layer_config(
                    _CAPTURE_SPEC.representative_layer_ids[mode]
                )
                for mode in _FORWARD_ORDER
            }
            if any(
                configs[mode].ratio != _RATIO_BY_MODE[mode] for mode in _FORWARD_ORDER
            ):
                raise AssertionError(
                    "the representative Pro layer pair is not CSA->HCA"
                )
        for config in configs.values():
            config.validate_release_contract()
        layers: dict[str, MagiDSALayer] = {}
        sources: dict[str, _ProfileSource] = {}
        identities: dict[str, dict[str, tuple[int, int]]] = {}
        tensor_seeds: dict[str, dict[str, int]] = {}
        input_sha256: dict[str, str] = {}
        input_tensor_sha256: dict[str, dict[str, str]] = {}
        parameter_sha256: dict[str, str] = {}
        runtimes: dict[str, tuple[MagiDSARuntimeMgr, Any, float]] = {}
        structural_layout_config = DsaStructuralLayoutConfig()

        pro_runtime: MagiDSAProRuntimeMgr | None = None
        pro_bundle: MagiDSAProExecutionBundle | None = None
        pro_sources: dict[str, _ProfileSource] | None = None
        pro_tensor_seeds: dict[str, int] | None = None
        if model_spec is not None:
            shared_source, pro_tensor_seeds, _ = _make_inputs(
                configs["csa"],
                args.tokens,
                rank,
                world_size,
                args.seed,
                device,
                requires_grad=True,
            )
            hca_sink = (
                shared_source.sink.detach().clone().contiguous().requires_grad_(True)
            )
            pro_sources = {
                "csa": shared_source,
                "hca": _ProfileSource(
                    x=shared_source.x,
                    sink=hca_sink,
                    packed_meta=shared_source.packed_meta,
                ),
            }

        for mode in _FORWARD_ORDER:
            layer_id = (
                None
                if _CAPTURE_SPEC.representative_layer_ids is None
                else _CAPTURE_SPEC.representative_layer_ids[mode]
            )
            parameter_seed_name = (
                f"{mode}_parameters"
                if layer_id is None
                else f"layer_{layer_id}_{mode}_parameters"
            )
            layer_seed = _derived_seed(args.seed, parameter_seed_name, -1)
            torch.manual_seed(layer_seed)
            torch.cuda.manual_seed_all(layer_seed)
            layer = MagiDSALayer(configs[mode], layer_id=layer_id).to(device)
            if pro_sources is None:
                source, mode_tensor_seeds, mode_identities = _make_inputs(
                    configs[mode],
                    args.tokens,
                    rank,
                    world_size,
                    args.seed,
                    device,
                    requires_grad=True,
                )
            else:
                assert pro_tensor_seeds is not None
                source = pro_sources[mode]
                mode_tensor_seeds = dict(pro_tensor_seeds)
                mode_identities = {
                    name: (tensor.data_ptr(), tensor._version)
                    for name, tensor in (("x", source.x), ("sink", source.sink))
                }
            layers[mode] = layer
            sources[mode] = source
            identities[mode] = mode_identities
            tensor_seeds[mode] = {
                **mode_tensor_seeds,
                "parameters": layer_seed,
            }
            mode_input_digest, mode_tensor_digests = _input_digest(
                source,
                args.artifact_dir,
                rank,
            )
            input_sha256[mode] = mode_input_digest
            input_tensor_sha256[mode] = mode_tensor_digests
            parameter_sha256[mode] = _parameter_digest(
                layer,
                args.artifact_dir,
                rank,
            )
            if model_spec is None:
                runtime, handle, prepare_seconds = _prepare_runtime(
                    configs[mode],
                    source,
                    args.artifact_dir,
                    rank,
                    mode,
                    structural_layout_config=structural_layout_config,
                )
                runtimes[mode] = (runtime, handle, prepare_seconds)

        if model_spec is not None:
            pro_runtime, pro_bundle, prepare_seconds = _prepare_pro_runtime_bundle(
                model_spec,
                sources["csa"],
                args.artifact_dir,
                rank,
                structural_layout_config,
            )
            runtimes = {
                "csa": (pro_runtime.csa_runtime, pro_bundle.csa, prepare_seconds),
                "hca": (pro_runtime.hca_runtime, pro_bundle.hca, prepare_seconds),
            }

        if True:
            layout_hashes = {
                runtimes[mode][1].plan.query_layout_hash for mode in _FORWARD_ORDER
            }
            query_counts = {
                runtimes[mode][1].plan.query_token_counts for mode in _FORWARD_ORDER
            }
            if len(layout_hashes) != 1 or len(query_counts) != 1:
                raise AssertionError(
                    f"{args.layout_policy} Attention plans do not share one Query layout"
                )

        shadow_runtime, shadow_handle, shadow_prepare_seconds = _prepare_runtime(
            configs["csa"],
            sources["csa"],
            args.artifact_dir,
            rank,
            "csa_shadow",
            structural_layout_config=structural_layout_config,
        )
        global_dout, dout_scale = _make_global_dout(
            configs["csa"],
            args.tokens,
            args.seed,
            device,
            args.artifact_dir,
            rank,
        )
        if pro_runtime is None or pro_bundle is None:
            boundaries = {
                mode: _prepare_profile_dsa_input_boundary(
                    sources[mode],
                    runtimes[mode][0],
                    runtimes[mode][1],
                    global_dout,
                )
                for mode in _FORWARD_ORDER
            }
        else:
            boundaries = _prepare_pro_pair_boundaries(
                sources,
                pro_runtime,
                pro_bundle,
                global_dout,
            )
        shadow_boundary = _prepare_profile_dsa_input_boundary(
            sources["csa"],
            shadow_runtime,
            shadow_handle,
            global_dout,
        )
        del global_dout
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        _record(
            args.artifact_dir,
            "profile_global_dout_end",
            rank,
            seed=_derived_seed(args.seed, "dout", -1),
        )

        cases = {
            mode: _AttentionProfileCase(
                name=mode,
                layer=layers[mode],
                source=sources[mode],
                runtime=runtimes[mode][0],
                handle=runtimes[mode][1],
                boundary=boundaries[mode],
                prepare_seconds=runtimes[mode][2],
                execution_stream=torch.cuda.Stream(
                    device=device,
                    priority=-1 if mode == "hca" else 0,
                ),
                pro_runtime=pro_runtime,
                pro_bundle=pro_bundle,
            )
            for mode in _FORWARD_ORDER
        }
        pro_contract = (
            _pro_bundle_contract(cases) if _CAPTURE_SPEC is _PRO_PAIR_SPEC else {}
        )
        csa_shadow = _AttentionProfileCase(
            name="csa_shadow",
            layer=layers["csa"],
            source=sources["csa"],
            runtime=shadow_runtime,
            handle=shadow_handle,
            boundary=shadow_boundary,
            prepare_seconds=shadow_prepare_seconds,
            execution_stream=torch.cuda.Stream(device=device),
        )
        _prewarm(
            cases,
            csa_shadow,
            args.warmup,
            args.artifact_dir,
            rank,
        )
        dist.barrier(group=control_group)

        config_payload = {mode: asdict(configs[mode]) for mode in _FORWARD_ORDER}
        metadata = {
            "attention_order": list(_FORWARD_ORDER),
            "attention_modes": {
                mode: _runtime_metadata(cases[mode], rank) for mode in _FORWARD_ORDER
            },
            "backward_order": list(_BACKWARD_ORDER),
            "backward_seed": "precomputed_global_mean_scaled_dout_and_unit_dkl",
            "config": config_payload,
            "config_sha256": hashlib.sha256(
                json.dumps(
                    config_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "dout_global_output_elements": (
                args.tokens * configs["csa"].num_query_heads * configs["csa"].head_dim
            ),
            "dout_global_shape": [
                args.tokens,
                configs["csa"].num_query_heads,
                configs["csa"].head_dim,
            ],
            "dout_recipe": (
                "one global BF16 randn with shared seed, scaled by "
                "1/global_output_elements, then indexed independently by each "
                "Attention plan local_query_global_rows"
            ),
            "dout_scale": dout_scale,
            "dtype": "torch.bfloat16",
            "expected_sendrecv": _active_expected_sendrecv(),
            "gradient_accumulation": False,
            "independent_attention_graphs": True,
            "input_sha256": input_sha256,
            "input_tensor_sha256": input_tensor_sha256,
            "local_source_tokens": sources["csa"].packed_meta.local_token_count(rank),
            "loss_capture": "none",
            "layout_policy": args.layout_policy,
            "profiler_attach_warmup_steps": args.profiler_attach_warmup_steps,
            "structural_layout_config": asdict(structural_layout_config),
            "mode": args.mode,
            "mode_serialization": "cuda_event_happens_before",
            "mode_backward_completion_join": {
                mode: list(_BACKWARD_INTERNAL_STREAM_FIELDS[mode])
                for mode in _FORWARD_ORDER
            },
            "overlap_accounting": "same_mode_same_direction_non_route_compute",
            "parameter_gradient_allreduce": (
                _CAPTURE_SPEC.parameter_gradient_allreduce
            ),
            "parameter_sha256": parameter_sha256,
            "plan": "balanced",
            "profile_gradient_boundary": "post_projection_magi_dsa_input",
            "projection_capture": "pre_capture_once_per_attention",
            "rank": rank,
            "ratios": [_RATIO_BY_MODE[mode] for mode in _FORWARD_ORDER],
            "representative_layer_ids": _CAPTURE_SPEC.representative_layer_ids,
            "representative_pair_semantics": (
                "independent_post_projection_graphs_serialized_in_layer_order"
                if _CAPTURE_SPEC is _PRO_PAIR_SPEC
                else None
            ),
            "pro_model_spec": None if model_spec is None else asdict(model_spec),
            "seed": args.seed,
            "seed_recipe": (
                "sha256('magi-dsa-v4-profile:{seed}:{tensor}:{rank-or--1}')[:8]"
            ),
            "shadow": _runtime_metadata(csa_shadow, rank),
            "step_mode": args.step_mode,
            "tensor_seeds": tensor_seeds,
            "token_layout_capture": "pre_capture_once_per_attention",
            "tokens": args.tokens,
            "warmup": args.warmup,
            "world_size": world_size,
        }
        if _CAPTURE_SPEC is _PRO_PAIR_SPEC:
            metadata.update(pro_contract)
            metadata.update(
                {
                    "parameter_gradient_allreduce_in_7f7b": False,
                    "parameter_gradient_reducer_precision": (
                        "fp32_cp_bucket_model_side_diagnostic"
                    ),
                    "runtime_parameter_gradient_communication": False,
                }
            )
        _atomic_json(args.artifact_dir / f"metadata_rank{rank}.json", metadata)
        report = _run_profile(
            args,
            cases,
            csa_shadow,
            identities,
            control_group,
            rank,
            dout_scale,
        )
        _atomic_json(args.artifact_dir / f"result_rank{rank}.json", report)
        _record(args.artifact_dir, "worker_complete", rank, mode=args.mode)
    except BaseException as error:
        failure = {
            "error": str(error),
            "error_type": type(error).__name__,
            "rank": rank,
            "traceback": traceback.format_exc(),
        }
        try:
            _atomic_json(args.artifact_dir / f"failure_rank{rank}.json", failure)
            _record(args.artifact_dir, "worker_failed", rank, error=repr(error))
        finally:
            raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
