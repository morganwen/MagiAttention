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

from contextlib import AbstractContextManager, nullcontext

from magi_attention.utils import nvtx

DSA_MODULE_NVTX_PREFIX = "magi_dsa::module::"
DSA_CUDNN_CALL_NVTX_PREFIX = "magi_dsa::CUDNN_CALL::"
DSA_PHASE_NVTX_PREFIX = "magi_dsa::phase::"


def _dsa_named_nvtx_range(
    prefix: str, name: str, *, enabled: bool
) -> AbstractContextManager[object]:
    if not enabled:
        return nullcontext()
    if not name or name.startswith(":") or name.endswith(":"):
        raise ValueError("DSA NVTX names must be non-empty relative paths")
    return nvtx.add_nvtx_event(f"{prefix}{name}")


def dsa_nvtx_range(
    name: str, *, enabled: bool = True
) -> AbstractContextManager[object]:
    """Return a synchronization-free NVTX range for one stable DSA module path."""

    return _dsa_named_nvtx_range(DSA_MODULE_NVTX_PREFIX, name, enabled=enabled)


def dsa_cudnn_call_range(
    name: str, *, enabled: bool = True
) -> AbstractContextManager[object]:
    """Return a prominent NVTX range for one cuDNN frontend wrapper call."""

    return _dsa_named_nvtx_range(DSA_CUDNN_CALL_NVTX_PREFIX, name, enabled=enabled)


def dsa_collective_launch_range(
    name: str, *, enabled: bool = True
) -> AbstractContextManager[object]:
    """Return the phase-scoped NVTX range used to attribute one collective.

    This deliberately does not use :class:`dsa_phase`. Under
    ``MAGI_DSA_PHASE_LOG=1`` that helper brackets a phase with a device-wide
    ``torch.cuda.synchronize()``, which is fine around a coarse phase but
    deadlocks around an asynchronous collective launch: the launching rank
    blocks inside the sync while its peers move on to the next collective.
    """

    return _dsa_named_nvtx_range(DSA_PHASE_NVTX_PREFIX, name, enabled=enabled)


__all__ = [
    "DSA_CUDNN_CALL_NVTX_PREFIX",
    "DSA_MODULE_NVTX_PREFIX",
    "dsa_cudnn_call_range",
    "dsa_nvtx_range",
]
