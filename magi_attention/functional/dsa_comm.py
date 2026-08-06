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

from dataclasses import dataclass

import torch
import torch.distributed as dist

from magi_attention.comm.primitive import all2all_v
from magi_attention.comm.work import GeneralWork
from magi_attention.dsa_nvtx import dsa_nvtx_range
from magi_attention.utils import nvtx

from .dsa_packing import (
    DsaDeviceCopyMap,
    DsaDeviceReduceMap,
    DsaDeviceRoutePlan,
    copy_dsa_device_map,
    reduce_dsa_device_map,
)
from .dsa_phase import dsa_phase


def _validate_route_tensor(source: torch.Tensor, expected_rows: int, name: str) -> None:
    if not source.is_cuda or not source.is_contiguous() or source.ndim < 2:
        raise ValueError(
            f"{name} must be a contiguous CUDA tensor with at least two dimensions"
        )
    if source.shape[0] != expected_rows:
        raise ValueError(f"{name} has {source.shape[0]} rows, expected {expected_rows}")


def _validate_group(route: DsaDeviceRoutePlan, group: dist.ProcessGroup | None) -> None:
    world_size = len(route.send_counts)
    if group is None:
        if world_size != 1 or route.rank != 0:
            raise ValueError("a null CP group is valid only for a one-rank route")
        return
    if dist.get_world_size(group) != world_size:
        raise ValueError("CP group size does not match the route")
    if dist.get_rank(group) != route.rank:
        raise ValueError("CP group rank does not match the route")


def _route_name(route: DsaDeviceRoutePlan, attention_mode: str | None) -> str:
    if attention_mode is None:
        return route.name
    return f"attention::{attention_mode}::{route.name}"


def _exchange(
    source: torch.Tensor,
    input_counts: tuple[int, ...],
    output_counts: tuple[int, ...],
    group: dist.ProcessGroup | None,
    phase_name: str,
) -> torch.Tensor:
    transfer = _start_exchange(
        source,
        input_counts,
        output_counts,
        group,
        phase_name,
    )
    with dsa_nvtx_range(
        f"route::{phase_name}::collective_wait", enabled=source.is_cuda
    ):
        return transfer.wait()


@dataclass
class _DsaExchangeWork:
    """Keep one private All2AllV send/output pair alive until consumption."""

    output: torch.Tensor
    send_buffer: torch.Tensor
    work: GeneralWork | None
    _waited: bool = False

    def wait(self) -> torch.Tensor:
        if not self._waited:
            if self.work is not None:
                self.work.wait()
            self._waited = True
        return self.output


@dataclass
class DsaReverseRouteTransfer:
    """One private reverse-route exchange awaiting owner-side reduction."""

    exchange: _DsaExchangeWork
    route: DsaDeviceRoutePlan
    attention_mode: str | None
    _finished: bool = False

    def wait(self) -> None:
        """Drain an in-flight reverse exchange after sibling work fails."""

        self.exchange.wait()


def _start_exchange(
    source: torch.Tensor,
    input_counts: tuple[int, ...],
    output_counts: tuple[int, ...],
    group: dist.ProcessGroup | None,
    phase_name: str,
) -> _DsaExchangeWork:
    output = torch.empty(
        (sum(output_counts), *source.shape[1:]),
        dtype=source.dtype,
        device=source.device,
    )
    if source.stride() != output.stride():
        with dsa_nvtx_range(
            f"route::{phase_name}::stride_normalization", enabled=source.is_cuda
        ):
            normalized = torch.empty(
                source.shape,
                dtype=source.dtype,
                device=source.device,
            )
            normalized.copy_(source)
            source = normalized
    if group is None:
        if input_counts != output_counts or len(input_counts) != 1:
            raise ValueError("one-rank All2AllV splits must be identical")
        with dsa_nvtx_range(
            f"route::{phase_name}::single_rank_copy", enabled=source.is_cuda
        ):
            output.copy_(source)
        return _DsaExchangeWork(output, source, None)
    with dsa_phase(f"collective_all2all_v::{phase_name}"):
        work = all2all_v(
            input=source,
            output=output,
            input_split_size_list=list(input_counts),
            output_split_size_list=list(output_counts),
            group=group,
            async_op=True,
        )
    return _DsaExchangeWork(output, source, work)


def _forward_route(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> torch.Tensor:
    _validate_route_tensor(source, route.producer_row_count, route.name)
    _validate_group(route, group)
    route_name = _route_name(route, attention_mode)
    scope = f"route::{route_name}::forward"
    with dsa_nvtx_range(f"{scope}::send_pack", enabled=source.is_cuda):
        packed_send = copy_dsa_device_map(source, route.send_pack)
    received = _exchange(
        packed_send,
        route.send_counts,
        route.recv_counts,
        group,
        f"{route_name}.forward",
    )
    with dsa_nvtx_range(f"{scope}::consumer_pack", enabled=source.is_cuda):
        return copy_dsa_device_map(received, route.consumer_pack)


def _reverse_route(
    consumer: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> torch.Tensor:
    return finish_dsa_reverse_route(
        start_dsa_reverse_route(
            consumer,
            route,
            group,
            attention_mode=attention_mode,
        )
    )


def start_dsa_reverse_route(
    consumer: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> DsaReverseRouteTransfer:
    """Pack and launch one reverse typed route without waiting for owner rows."""

    _validate_route_tensor(consumer, route.consumer_row_count, f"{route.name} reverse")
    _validate_group(route, group)
    route_name = _route_name(route, attention_mode)
    scope = f"route::{route_name}::backward"
    with dsa_nvtx_range(f"{scope}::received_pack", enabled=consumer.is_cuda):
        received_order = copy_dsa_device_map(consumer, route.received_pack)
    return _start_dsa_reverse_received_route(
        received_order,
        route,
        group,
        attention_mode=attention_mode,
    )


def _start_dsa_reverse_received_route(
    received_order: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> DsaReverseRouteTransfer:
    _validate_route_tensor(
        received_order,
        route.received_row_count,
        f"{route.name} reverse received",
    )
    _validate_group(route, group)
    route_name = _route_name(route, attention_mode)
    exchange = _start_exchange(
        received_order,
        route.recv_counts,
        route.send_counts,
        group,
        f"{route_name}.backward",
    )
    return DsaReverseRouteTransfer(exchange, route, attention_mode)


def finish_dsa_reverse_route(transfer: DsaReverseRouteTransfer) -> torch.Tensor:
    """Wait for a reverse typed route and reduce rows on their producer owner."""

    if transfer._finished:
        raise RuntimeError("DSA reverse route transfer has already been finished")
    route = transfer.route
    route_name = _route_name(route, transfer.attention_mode)
    scope = f"route::{route_name}::backward"
    with dsa_nvtx_range(
        f"{scope}::collective_wait",
        enabled=transfer.exchange.output.is_cuda,
    ):
        reverse_received = transfer.exchange.wait()
    transfer._finished = True
    if route.owner_restore is not None:
        with dsa_nvtx_range(
            f"{scope}::owner_restore",
            enabled=reverse_received.is_cuda,
        ):
            return copy_dsa_device_map(reverse_received, route.owner_restore)
    with dsa_nvtx_range(
        f"{scope}::owner_csr_reduce",
        enabled=reverse_received.is_cuda,
    ):
        return reduce_dsa_device_map(reverse_received, route.owner_reduce)


def _reverse_received_route(
    received_order: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> torch.Tensor:
    return finish_dsa_reverse_route(
        _start_dsa_reverse_received_route(
            received_order,
            route,
            group,
            attention_mode=attention_mode,
        )
    )


@dataclass
class _DsaRouteLaunch:
    exchange: _DsaExchangeWork | None = None


@dataclass
class DsaRouteTransfer:
    """One disposable asynchronous typed-route transfer."""

    received: torch.Tensor
    route: DsaDeviceRoutePlan
    exchange: _DsaExchangeWork
    attention_mode: str | None
    _finished: bool = False

    def wait(self) -> None:
        """Drain an in-flight transfer after a sibling launch fails."""

        self.exchange.wait()


class _DsaRouteStartFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        route: DsaDeviceRoutePlan,
        group: dist.ProcessGroup | None,
        launch: _DsaRouteLaunch,
        attention_mode: str | None,
    ) -> torch.Tensor:
        _validate_route_tensor(source, route.producer_row_count, route.name)
        _validate_group(route, group)
        ctx.route = route
        ctx.group = group
        ctx.attention_mode = attention_mode
        ctx.set_materialize_grads(False)
        route_name = _route_name(route, attention_mode)
        scope = f"route::{route_name}::forward"
        with dsa_nvtx_range(f"{scope}::send_pack", enabled=source.is_cuda):
            packed_send = copy_dsa_device_map(source, route.send_pack)
        launch.exchange = _start_exchange(
            packed_send,
            route.send_counts,
            route.recv_counts,
            group,
            f"{route_name}.forward",
        )
        return launch.exchange.output

    @staticmethod
    def backward(ctx, received_grad: torch.Tensor | None):
        if received_grad is None:
            return None, None, None, None, None
        source_grad = _reverse_received_route(
            received_grad.contiguous(),
            ctx.route,
            ctx.group,
            attention_mode=ctx.attention_mode,
        )
        return source_grad, None, None, None, None


class _DsaRouteFinishFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        received: torch.Tensor,
        route: DsaDeviceRoutePlan,
        exchange: _DsaExchangeWork,
        attention_mode: str | None,
    ) -> torch.Tensor:
        _validate_route_tensor(received, route.received_row_count, route.name)
        ctx.route = route
        ctx.attention_mode = attention_mode
        ctx.set_materialize_grads(False)
        route_name = _route_name(route, attention_mode)
        with dsa_nvtx_range(
            f"route::{route_name}::forward::collective_wait",
            enabled=received.is_cuda,
        ):
            exchange.wait()
        with dsa_nvtx_range(
            f"route::{route_name}::forward::consumer_pack",
            enabled=received.is_cuda,
        ):
            return copy_dsa_device_map(received, route.consumer_pack)

    @staticmethod
    def backward(ctx, consumer_grad: torch.Tensor | None):
        if consumer_grad is None:
            return None, None, None, None
        route: DsaDeviceRoutePlan = ctx.route
        route_name = _route_name(route, ctx.attention_mode)
        _validate_route_tensor(
            consumer_grad,
            route.consumer_row_count,
            f"{route.name} reverse",
        )
        with dsa_nvtx_range(
            f"route::{route_name}::backward::received_pack",
            enabled=consumer_grad.is_cuda,
        ):
            received_grad = copy_dsa_device_map(
                consumer_grad.contiguous(), route.received_pack
            )
        return received_grad, None, None, None


class _DsaTokenLayoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        route: DsaDeviceRoutePlan,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.route = route
        ctx.group = group
        with nvtx.add_nvtx_event("magi_dsa::layout::TOKEN_LAYOUT"):
            return _forward_route(source, route, group)

    @staticmethod
    def backward(ctx, consumer_grad: torch.Tensor):
        with nvtx.add_nvtx_event("magi_dsa::layout::TOKEN_LAYOUT::backward"):
            source_grad = _reverse_route(
                consumer_grad.contiguous(), ctx.route, ctx.group
            )
        return source_grad, None, None


class _DsaCopyReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        copy_map: DsaDeviceCopyMap,
        reduce_map: DsaDeviceReduceMap,
        nvtx_scope: str,
    ) -> torch.Tensor:
        ctx.reduce_map = reduce_map
        ctx.nvtx_scope = nvtx_scope
        with dsa_nvtx_range(
            f"packing::{nvtx_scope}::forward_copy", enabled=source.is_cuda
        ):
            return copy_dsa_device_map(source, copy_map)

    @staticmethod
    def backward(ctx, packed_grad: torch.Tensor):
        with dsa_nvtx_range(
            f"packing::{ctx.nvtx_scope}::backward_csr_reduce",
            enabled=packed_grad.is_cuda,
        ):
            source_grad = reduce_dsa_device_map(
                packed_grad.contiguous(), ctx.reduce_map
            )
        return source_grad, None, None, None


def route_dsa_tensor(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> torch.Tensor:
    """Autograd-aware unique-row All2AllV route with reverse owner CSR."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA routes support BF16 and FP32")
    return finish_dsa_tensor_route(
        start_dsa_tensor_route(
            source,
            route,
            group,
            attention_mode=attention_mode,
        )
    )


def start_dsa_tensor_route(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
    *,
    attention_mode: str | None = None,
) -> DsaRouteTransfer:
    """Launch one typed route without waiting for its consumer-order output."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA routes support BF16 and FP32")
    launch = _DsaRouteLaunch()
    received = _DsaRouteStartFunction.apply(
        source,
        route,
        group,
        launch,
        attention_mode,
    )
    if launch.exchange is None:
        raise RuntimeError("typed route did not publish its exchange work")
    return DsaRouteTransfer(received, route, launch.exchange, attention_mode)


def finish_dsa_tensor_route(transfer: DsaRouteTransfer) -> torch.Tensor:
    """Wait for an asynchronous typed route and return consumer row order."""

    if transfer._finished:
        raise RuntimeError("DSA route transfer has already been finished")
    consumer = _DsaRouteFinishFunction.apply(
        transfer.received,
        transfer.route,
        transfer.exchange,
        transfer.attention_mode,
    )
    transfer._finished = True
    return consumer


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
    if route.owner_restore is None:
        raise ValueError("TOKEN_LAYOUT must be a global bijection")
    with dsa_nvtx_range(
        "layout::TOKEN_LAYOUT::diagnostic_unlayout", enabled=consumer.is_cuda
    ):
        return _reverse_route(consumer.contiguous(), route, group)


def copy_dsa_tensor_with_csr(
    source: torch.Tensor,
    copy_map: DsaDeviceCopyMap,
    reduce_map: DsaDeviceReduceMap,
    *,
    nvtx_scope: str = "support",
) -> torch.Tensor:
    """Gather duplicate rows and reduce their gradients with a static CSR."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA packing supports BF16 and FP32")
    return _DsaCopyReduceFunction.apply(source, copy_map, reduce_map, nvtx_scope)


__all__ = [
    "DsaReverseRouteTransfer",
    "DsaRouteTransfer",
    "copy_dsa_tensor_with_csr",
    "finish_dsa_reverse_route",
    "finish_dsa_tensor_route",
    "layout_dsa_hidden",
    "route_dsa_tensor",
    "start_dsa_reverse_route",
    "start_dsa_tensor_route",
    "unlayout_dsa_query_tensor",
]
