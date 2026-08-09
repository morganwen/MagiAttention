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

"""Typed payload routes on top of the Core group collective.

A DSA route is a group-cast from the rows a rank produces to the rows every
rank needs, and its adjoint is the symmetric group-reduce. Both directions come
straight from MagiAttention Core, so this module only owns the launch/wait
split that lets a collective overlap independent compute, and the single
autograd node that ties the consumer tensor back to its producer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.common.enum import GrpCollBufferName
from magi_attention.common.range_op import range_gather, range_reduce

from .nvtx import dsa_nvtx_range
from .packing import DsaDeviceRoutePlan
from .phase import dsa_phase


def _validate_route_tensor(source: torch.Tensor, expected_rows: int, name: str) -> None:
    if not source.is_cuda or not source.is_contiguous() or source.ndim < 2:
        raise ValueError(
            f"{name} must be a contiguous CUDA tensor with at least two dimensions"
        )
    if source.shape[0] != expected_rows:
        raise ValueError(f"{name} has {source.shape[0]} rows, expected {expected_rows}")


def _validate_group(
    route: DsaDeviceRoutePlan, group: dist.ProcessGroup | None
) -> None:
    if group is None:
        if route.cp_size != 1 or route.rank != 0:
            raise ValueError("a null CP group is valid only for a one-rank route")
        return
    if dist.get_world_size(group) != route.cp_size:
        raise ValueError("CP group size does not match the route")
    if dist.get_rank(group) != route.rank:
        raise ValueError("CP group rank does not match the route")


def _route_name(route: DsaDeviceRoutePlan, attention_mode: str | None) -> str:
    if attention_mode is None:
        return route.name
    return f"attention::{attention_mode}::{route.name}"


@dataclass
class DsaRouteTransfer:
    """One in-flight group-cast and the producer tensor that owns its gradient."""

    source: torch.Tensor
    output: torch.Tensor
    work: WorkWithPostProcessFn | None
    route: DsaDeviceRoutePlan
    group: dist.ProcessGroup | None
    attention_mode: str | None
    _finished: bool = False

    def wait(self) -> torch.Tensor:
        """Drain an in-flight transfer without building an autograd edge."""

        if self.work is None:
            return self.output
        return self.work.wait_post_process(self.output)


@dataclass
class DsaReverseRouteTransfer:
    """One in-flight group-reduce awaiting its owner-side rows."""

    output: torch.Tensor
    work: WorkWithPostProcessFn | None
    route: DsaDeviceRoutePlan
    attention_mode: str | None
    _finished: bool = False

    def wait(self) -> torch.Tensor:
        if self.work is None:
            return self.output
        return self.work.wait_post_process(self.output)


def _local_gather(source: torch.Tensor, route: DsaDeviceRoutePlan) -> torch.Tensor:
    """One-rank forward route: select the consumed ranges out of the producer."""

    ranges = route.local_gather_ranges
    if ranges is None:
        raise RuntimeError(f"{route.name}: one-rank route has no local range plan")
    if route.producer_row_count == route.consumer_row_count and ranges.shape[0] == 1:
        return source
    return range_gather(source, ranges, total_size=route.consumer_row_count)


def _local_scatter_add(
    consumer_grad: torch.Tensor, route: DsaDeviceRoutePlan
) -> torch.Tensor:
    """One-rank reverse route: accumulate consumer rows back onto the producer."""

    ranges = route.local_gather_ranges
    if ranges is None:
        raise RuntimeError(f"{route.name}: one-rank route has no local range plan")
    if route.producer_row_count == route.consumer_row_count and ranges.shape[0] == 1:
        return consumer_grad
    owner_grad = torch.zeros(
        (route.producer_row_count, *consumer_grad.shape[1:]),
        dtype=consumer_grad.dtype,
        device=consumer_grad.device,
    )
    lengths = (ranges[:, 1] - ranges[:, 0]).to(torch.int32)
    input_ends = torch.cumsum(lengths, dim=0)
    input_ranges = torch.stack((input_ends - lengths, input_ends), dim=1).to(
        torch.int32
    )
    range_reduce(
        input=consumer_grad,
        output=owner_grad,
        input_ranges=input_ranges,
        output_ranges=ranges,
        deterministic=True,
    )
    return owner_grad


def _launch_group_cast(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    phase_name: str,
) -> tuple[torch.Tensor, WorkWithPostProcessFn | None]:
    if route.cp_size == 1:
        return _local_gather(source, route), None
    arg = route.group_collective_arg
    if arg is None:
        raise RuntimeError(f"{route.name}: distributed route has no collective plan")
    output = torch.empty(
        (route.consumer_row_count, *source.shape[1:]),
        dtype=source.dtype,
        device=source.device,
    )
    with dsa_phase(f"collective_group_cast::{phase_name}"):
        work = group_cast(
            input=source,
            output=output,
            **arg.to_group_cast_args(),
            group=group,
            async_op=True,
            buffer_name=GrpCollBufferName.GroupCastDefault,
        )
    return output, work


def _launch_group_reduce(
    consumer_grad: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    phase_name: str,
) -> tuple[torch.Tensor, WorkWithPostProcessFn | None]:
    if route.cp_size == 1:
        return _local_scatter_add(consumer_grad, route), None
    arg = route.group_collective_arg
    if arg is None:
        raise RuntimeError(f"{route.name}: distributed route has no collective plan")
    owner_grad = torch.zeros(
        (route.producer_row_count, *consumer_grad.shape[1:]),
        dtype=consumer_grad.dtype,
        device=consumer_grad.device,
    )
    with dsa_phase(f"collective_group_reduce::{phase_name}"):
        work = group_reduce(
            input=consumer_grad,
            output=owner_grad,
            **arg.to_group_reduce_args(),
            group=group,
            async_op=True,
            acc_reduce=True,
            buffer_name=GrpCollBufferName.GroupCastDefault,
        )
    return owner_grad, work


class _DsaRouteFinishFunction(torch.autograd.Function):
    """Tie a consumer bank to its producer with one symmetric collective pair.

    The forward collective is already in flight when this node runs, so forward
    only waits. Backward is the group-reduce that is the exact adjoint of that
    group-cast, which is why no consumer permutation has to be inverted here.
    """

    @staticmethod
    def forward(ctx, source: torch.Tensor, transfer: DsaRouteTransfer):
        ctx.route = transfer.route
        ctx.group = transfer.group
        ctx.attention_mode = transfer.attention_mode
        ctx.set_materialize_grads(False)
        route_name = _route_name(transfer.route, transfer.attention_mode)
        with dsa_nvtx_range(
            f"route::{route_name}::forward::collective_wait",
            enabled=source.is_cuda,
        ):
            return transfer.wait()

    @staticmethod
    def backward(ctx, consumer_grad: torch.Tensor | None):
        if consumer_grad is None:
            return None, None
        route: DsaDeviceRoutePlan = ctx.route
        route_name = _route_name(route, ctx.attention_mode)
        _validate_route_tensor(
            consumer_grad.contiguous(),
            route.consumer_row_count,
            f"{route.name} reverse",
        )
        owner_grad, work = _launch_group_reduce(
            consumer_grad.contiguous(),
            route,
            ctx.group,
            f"{route_name}.backward",
        )
        with dsa_nvtx_range(
            f"route::{route_name}::backward::collective_wait",
            enabled=owner_grad.is_cuda,
        ):
            if work is not None:
                owner_grad = work.wait_post_process(owner_grad)
        return owner_grad, None


def start_dsa_tensor_route(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> DsaRouteTransfer:
    """Launch one typed route without waiting for its consumer rows."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA routes support BF16 and FP32")
    _validate_route_tensor(source, route.producer_row_count, route.name)
    _validate_group(route, group)
    route_name = _route_name(route, attention_mode)
    output, work = _launch_group_cast(
        source.detach(),
        route,
        group,
        f"{route_name}.forward",
    )
    return DsaRouteTransfer(
        source=source,
        output=output,
        work=work,
        route=route,
        group=group,
        attention_mode=attention_mode,
    )


def finish_dsa_tensor_route(transfer: DsaRouteTransfer) -> torch.Tensor:
    """Wait for a typed route and attach its symmetric backward collective."""

    if transfer._finished:
        raise RuntimeError("DSA route transfer has already been finished")
    transfer._finished = True
    return _DsaRouteFinishFunction.apply(transfer.source, transfer)


def route_dsa_tensor(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> torch.Tensor:
    """Autograd-aware group-cast with the symmetric group-reduce backward."""

    return finish_dsa_tensor_route(
        start_dsa_tensor_route(source, route, group, attention_mode=attention_mode)
    )


def start_dsa_reverse_route(
    consumer_grad: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> DsaReverseRouteTransfer:
    """Launch one reverse route so it can overlap independent backward work."""

    _validate_route_tensor(
        consumer_grad, route.consumer_row_count, f"{route.name} reverse"
    )
    _validate_group(route, group)
    route_name = _route_name(route, attention_mode)
    output, work = _launch_group_reduce(
        consumer_grad, route, group, f"{route_name}.backward"
    )
    return DsaReverseRouteTransfer(output, work, route, attention_mode)


def finish_dsa_reverse_route(transfer: DsaReverseRouteTransfer) -> torch.Tensor:
    """Wait for a reverse route and return the owner-local gradient rows."""

    if transfer._finished:
        raise RuntimeError("DSA reverse route transfer has already been finished")
    transfer._finished = True
    route_name = _route_name(transfer.route, transfer.attention_mode)
    with dsa_nvtx_range(
        f"route::{route_name}::backward::collective_wait",
        enabled=transfer.output.is_cuda,
    ):
        return transfer.wait()


class _DsaTokenLayoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source: torch.Tensor, route: DsaDeviceRoutePlan, group):
        ctx.route = route
        ctx.group = group
        with dsa_nvtx_range("layout::TOKEN_LAYOUT", enabled=source.is_cuda):
            output, work = _launch_group_cast(
                source.detach(), route, group, "TOKEN_LAYOUT.forward"
            )
            return output if work is None else work.wait_post_process(output)

    @staticmethod
    def backward(ctx, consumer_grad: torch.Tensor):
        with dsa_nvtx_range(
            "layout::TOKEN_LAYOUT::backward", enabled=consumer_grad.is_cuda
        ):
            owner_grad, work = _launch_group_reduce(
                consumer_grad.contiguous(),
                ctx.route,
                ctx.group,
                "TOKEN_LAYOUT.backward",
            )
            if work is not None:
                owner_grad = work.wait_post_process(owner_grad)
        return owner_grad, None, None


def layout_dsa_hidden(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Route pre-projection hidden states to their final Query owners."""

    if route.name != "TOKEN_LAYOUT":
        raise ValueError("layout_dsa_hidden requires the TOKEN_LAYOUT route")
    if source.dtype != torch.bfloat16:
        raise TypeError("TOKEN_LAYOUT supports BF16 hidden states")
    _validate_route_tensor(source, route.producer_row_count, route.name)
    _validate_group(route, group)
    return _DsaTokenLayoutFunction.apply(source, route, group)


@torch.no_grad()
def unlayout_dsa_query_tensor(
    consumer: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Return a final-Query tensor to source-owner order for diagnostics."""

    if consumer.dtype not in (torch.bfloat16, torch.float32, torch.int32):
        raise TypeError("TOKEN_LAYOUT diagnostics support BF16, FP32, and int32")
    if route.name != "TOKEN_LAYOUT":
        raise ValueError("unlayout requires the TOKEN_LAYOUT boundary route")
    _validate_route_tensor(
        consumer, route.consumer_row_count, f"{route.name} reverse"
    )
    _validate_group(route, group)
    with dsa_nvtx_range(
        "layout::TOKEN_LAYOUT::diagnostic_unlayout", enabled=consumer.is_cuda
    ):
        # TOKEN_LAYOUT is a global bijection, so the reverse group-reduce sums
        # exactly one contribution per owner row and is a pure permutation.
        reduce_dtype = (
            torch.float32 if consumer.dtype == torch.int32 else consumer.dtype
        )
        output, work = _launch_group_reduce(
            consumer.to(reduce_dtype).contiguous(),
            route,
            group,
            "TOKEN_LAYOUT.diagnostic",
        )
        if work is not None:
            output = work.wait_post_process(output)
        return output.to(consumer.dtype)


__all__ = [
    "DsaReverseRouteTransfer",
    "DsaRouteTransfer",
    "finish_dsa_reverse_route",
    "finish_dsa_tensor_route",
    "layout_dsa_hidden",
    "route_dsa_tensor",
    "start_dsa_reverse_route",
    "start_dsa_tensor_route",
    "unlayout_dsa_query_tensor",
]
