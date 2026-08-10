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

"""Explicitly driven subgraphs for a model-owned callback.

A DSA layer has to decide when each collective is launched and waited relative
to compute. Leaving that to the autograd engine means the only lever is adding
order-forcing nodes, which is how this extension once carried three autograd
Functions whose whole job was to reorder work.

Instead each callback runs on detached leaves and keeps its ordinary autograd
subgraph. The schedule then drives those subgraphs by hand, so launch and wait
order is literally program order. This is the pattern Magi-MSA uses.

The nodes run inside an enclosing ``torch.autograd.Function.forward``, where
grad mode is off, so ``forward`` re-enables it locally: without that the
callback would build no graph and there would be nothing to drive.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.autograd import Variable


class DsaScheduleNode:
    """One model callback held as a detached, explicitly driven subgraph."""

    def __init__(
        self,
        forward_func: Callable,
        *,
        input_requires_grad: Sequence[bool] | None = None,
        enabled: bool = True,
    ) -> None:
        self.forward_func: Callable | None = forward_func
        self.input_requires_grad = input_requires_grad
        # An inference call has no backward to drive, so it must not retain a
        # subgraph per callback.
        self.enabled = enabled
        self.inputs: tuple[torch.Tensor, ...] | None = None
        self.outputs: tuple[torch.Tensor, ...] | None = None

    def forward(self, *inputs):
        """Run the callback on detached leaves and retain its subgraph."""

        forward_func = self.forward_func
        if forward_func is None:
            raise RuntimeError("this DSA schedule node was already released")
        self.inputs = self._make_leaves(inputs)
        with torch.set_grad_enabled(self.enabled):
            result = forward_func(*self.inputs)
        self.outputs = result if isinstance(result, tuple) else (result,)
        return result

    def backward(
        self,
        output_grads: Sequence[torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, ...]:
        """Drive this subgraph and return the gradients of its leaves.

        Parameter gradients are accumulated by the engine exactly as they would
        be in an ordinary backward; only the ordering is ours.
        """

        if self.inputs is None or self.outputs is None:
            raise RuntimeError("this DSA schedule node is not available for backward")
        active_outputs = []
        active_grads = []
        for output, grad in zip(self.outputs, output_grads, strict=True):
            if (
                isinstance(output, torch.Tensor)
                and output.requires_grad
                and grad is not None
            ):
                active_outputs.append(output)
                active_grads.append(grad)
        if active_outputs:
            Variable._execution_engine.run_backward(
                tuple(active_outputs),
                tuple(active_grads),
                False,  # keep_graph
                False,  # create_graph
                tuple(),  # inputs
                True,  # allow_unreachable
                True,  # accumulate_grad
            )
        input_grads = tuple(
            leaf.grad if isinstance(leaf, torch.Tensor) else None
            for leaf in self.inputs
        )
        self.release()
        return input_grads

    def release(self) -> None:
        """Drop the retained subgraph once its gradients have been read."""

        self.inputs = None
        self.outputs = None
        self.forward_func = None

    def _make_leaves(self, inputs: tuple) -> tuple:
        if not self.enabled:
            return tuple(
                tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
                for tensor in inputs
            )
        requires = self.input_requires_grad
        if requires is None:
            requires = tuple(
                isinstance(t, torch.Tensor)
                and t.requires_grad
                and t.is_floating_point()
                for t in inputs
            )
        leaves = []
        for tensor, needs_grad in zip(inputs, requires, strict=True):
            if not isinstance(tensor, torch.Tensor):
                leaves.append(tensor)
                continue
            leaf = tensor.detach()
            if needs_grad:
                leaf.requires_grad_(True)
            leaves.append(leaf)
        return tuple(leaves)


__all__ = ["DsaScheduleNode"]
