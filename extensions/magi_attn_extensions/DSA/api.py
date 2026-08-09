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

"""Thin aggregation layer over the Magi-DSA public API.

This module only re-exports classes and data structures that already existed
before the extension migration; it does not introduce new API surface. Internal
solvers, packing maps, backend helpers and kernels stay out of this list and are
reached through their own submodules (``.solver``, ``.packing``, ``.backend``,
``.kernels`` ...) as advanced API.
"""

from __future__ import annotations

from .config import (
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAProModelSpec,
)
from .modeling import MagiDSALayer, MagiDSAProLayerStack
from .model_adapter import (
    MagiDSAProjector,
    layout_and_project_dsa_input,
    layout_source_hidden_once,
    project_local_dsa_input,
)
from .projection import DsaProjections
from .pro_runtime import MagiDSAProExecutionBundle, MagiDSAProRuntimeMgr
from .runtime import MagiDSARuntimeMgr
from .types import MagiDSAForwardResult, MagiDSAInput, MagiDSAPackedMeta

__all__ = [
    "DsaProjections",
    "DsaRatio",
    "DsaStructuralLayoutConfig",
    "MagiDSAConfig",
    "MagiDSAForwardResult",
    "MagiDSAInput",
    "MagiDSALayer",
    "MagiDSAPackedMeta",
    "MagiDSAProExecutionBundle",
    "MagiDSAProLayerStack",
    "MagiDSAProModelSpec",
    "MagiDSAProRuntimeMgr",
    "MagiDSAProjector",
    "MagiDSARuntimeMgr",
    "layout_and_project_dsa_input",
    "layout_source_hidden_once",
    "project_local_dsa_input",
]
