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

"""Magi-DSA: the DeepSeek-style sparse attention extension built on MagiAttention.

The dependency direction is one-way: ``magi_attn_extensions.DSA`` imports
MagiAttention Core, never the other way round. The Core capabilities this
extension relies on are:

- ``magi_attention.common.AttnRanges`` and ``common.enum.AttnMaskType``
- ``magi_attention.common.range_op`` range gather and reduce
- ``magi_attention.comm.primitive.grpcoll`` group cast and group reduce
- ``magi_attention.comm.work.WorkWithPostProcessFn``
- ``magi_attention.meta.collection.comm_meta`` group-collective args
- ``magi_attention.meta.solver.dynamic_attn_solver`` range lowering
- ``magi_attention.meta.solver.dispatch_solver`` dispatch types and algorithms
- ``magi_attention.utils.general._make_device_tensor`` and ``utils.nvtx``
- ``magi_attention.meta._make_dispatch_meta`` dispatch meta and buckets
  (the only private path; guarded below)
"""

from __future__ import annotations

import inspect

try:
    from magi_attention.meta._make_dispatch_meta import (
        make_dispatch_meta_from_qk_ranges,
    )
except ImportError as error:
    raise RuntimeError(
        "magi_attn_extensions.DSA requires a magi_attention build that exposes "
        "magi_attention.meta._make_dispatch_meta.make_dispatch_meta_from_qk_ranges"
    ) from error

# ``solver.py`` calls the helper above with these five keywords. A Core that
# still exposes the symbol under an older signature would only fail at runtime,
# so the mismatch is turned into an import-time error here.
_REQUIRED_DISPATCH_META_PARAMS = frozenset(
    {
        "dispatch_config",
        "is_same_source",
        "is_q_permutable",
        "is_k_permutable",
        "uneven_shard",
    }
)
_missing = _REQUIRED_DISPATCH_META_PARAMS.difference(
    inspect.signature(make_dispatch_meta_from_qk_ranges).parameters
)
if _missing:
    raise RuntimeError(
        "magi_attn_extensions.DSA requires magi_attention's "
        f"make_dispatch_meta_from_qk_ranges to accept {sorted(_missing)}"
    )

del inspect, make_dispatch_meta_from_qk_ranges, _missing

from .config import (  # noqa: E402
    DsaRatio,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAProModelSpec,
)
from .modeling import MagiDSALayer, MagiDSAProLayerStack  # noqa: E402
from .projection import (  # noqa: E402
    DsaProjections,
    MagiDSAProjector,
    layout_and_project_dsa_input,
    layout_source_hidden_once,
    project_local_dsa_input,
)
from .pro_runtime import (  # noqa: E402
    MagiDSAProExecutionBundle,
    MagiDSAProRuntimeMgr,
)
from .runtime import MagiDSARuntimeMgr  # noqa: E402
from .types import (  # noqa: E402
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSAPackedMeta,
)

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
