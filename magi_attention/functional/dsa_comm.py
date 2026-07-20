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

import torch
import torch.distributed as dist

from magi_attention.comm.primitive import all2all_v

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


def _exchange(
    source: torch.Tensor,
    input_counts: tuple[int, ...],
    output_counts: tuple[int, ...],
    group: dist.ProcessGroup | None,
    phase_name: str,
) -> torch.Tensor:
    output = torch.empty(
        (sum(output_counts), *source.shape[1:]),
        dtype=source.dtype,
        device=source.device,
    )
    if group is None:
        if input_counts != output_counts or len(input_counts) != 1:
            raise ValueError("one-rank All2AllV splits must be identical")
        output.copy_(source)
        return output
    with dsa_phase(f"collective_all2all_v::{phase_name}"):
        work = all2all_v(
            input=source,
            output=output,
            input_split_size_list=list(input_counts),
            output_split_size_list=list(output_counts),
            group=group,
            async_op=False,
        )
        work.wait()
    return output


def _forward_route(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    _validate_route_tensor(source, route.producer_row_count, route.name)
    _validate_group(route, group)
    packed_send = copy_dsa_device_map(source, route.send_pack)
    received = _exchange(
        packed_send,
        route.send_counts,
        route.recv_counts,
        group,
        f"{route.name}.forward",
    )
    return copy_dsa_device_map(received, route.consumer_pack)


def _reverse_route(
    consumer: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    _validate_route_tensor(consumer, route.consumer_row_count, f"{route.name} reverse")
    _validate_group(route, group)
    received_order = copy_dsa_device_map(consumer, route.received_pack)
    reverse_received = _exchange(
        received_order,
        route.recv_counts,
        route.send_counts,
        group,
        f"{route.name}.backward",
    )
    return reduce_dsa_device_map(reverse_received, route.owner_reduce)


class _DsaRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        route: DsaDeviceRoutePlan,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.route = route
        ctx.group = group
        return _forward_route(source, route, group)

    @staticmethod
    def backward(ctx, consumer_grad: torch.Tensor):
        source_grad = _reverse_route(consumer_grad.contiguous(), ctx.route, ctx.group)
        return source_grad, None, None


class _DsaCopyReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        copy_map: DsaDeviceCopyMap,
        reduce_map: DsaDeviceReduceMap,
    ) -> torch.Tensor:
        ctx.reduce_map = reduce_map
        return copy_dsa_device_map(source, copy_map)

    @staticmethod
    def backward(ctx, packed_grad: torch.Tensor):
        return (
            reduce_dsa_device_map(packed_grad.contiguous(), ctx.reduce_map),
            None,
            None,
        )


def route_dsa_tensor(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Autograd-aware unique-row All2AllV route with reverse owner CSR."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA routes support BF16 and FP32")
    return _DsaRouteFunction.apply(source, route, group)


@torch.no_grad()
def route_dsa_tensor_no_grad(
    source: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Forward-only typed route, including int32 auxiliary payloads."""

    if source.dtype not in (torch.bfloat16, torch.float32, torch.int32):
        raise TypeError("DSA routes support BF16, FP32, and int32 payloads")
    return _forward_route(source, route, group)


@torch.no_grad()
def restore_dsa_bijective_tensor(
    consumer: torch.Tensor,
    route: DsaDeviceRoutePlan,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Restore a globally bijective worker layout to producer-owner order."""

    if consumer.dtype not in (torch.bfloat16, torch.float32, torch.int32):
        raise TypeError("DSA restore supports BF16, FP32, and int32 payloads")
    if route.owner_restore is None:
        raise ValueError(f"{route.name} is not a globally bijective route")
    _validate_route_tensor(consumer, route.consumer_row_count, f"{route.name} restore")
    _validate_group(route, group)
    received_order = copy_dsa_device_map(consumer, route.received_pack)
    reverse_received = _exchange(
        received_order,
        route.recv_counts,
        route.send_counts,
        group,
        f"{route.name}.restore",
    )
    return copy_dsa_device_map(reverse_received, route.owner_restore)


def copy_dsa_tensor_with_csr(
    source: torch.Tensor,
    copy_map: DsaDeviceCopyMap,
    reduce_map: DsaDeviceReduceMap,
) -> torch.Tensor:
    """Gather duplicate rows and reduce their gradients with a static CSR."""

    if source.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("differentiable DSA packing supports BF16 and FP32")
    return _DsaCopyReduceFunction.apply(source, copy_map, reduce_map)


__all__ = [
    "copy_dsa_tensor_with_csr",
    "restore_dsa_bijective_tensor",
    "route_dsa_tensor",
    "route_dsa_tensor_no_grad",
]
