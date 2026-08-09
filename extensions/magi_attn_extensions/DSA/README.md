# Magi-DSA

Magi-DSA is a DeepSeek-style sparse-attention extension built on top of
MagiAttention Core. It lives in its own Python distribution
(`magi_attn_extensions`) and owns its models, plans, runtime, communication
schedule, backends and kernels.

```text
magi_attn_extensions/DSA  ---->  magi_attention (Core)
```

The dependency is one-way. Core never imports this package.

## Boundary

Magi-DSA relies on exactly seven Core capabilities:

| Core capability | Used by |
|---|---|
| `magi_attention.common.AttnRanges` | `solver.py` |
| `magi_attention.common.enum.AttnMaskType` | `solver.py` |
| `magi_attention.comm.primitive.all2all_v` | `comm.py` |
| `magi_attention.comm.work.GeneralWork` | `comm.py` |
| `magi_attention.meta.solver.dispatch_solver` dispatch types and algorithms | `solver.py` |
| `magi_attention.utils.nvtx` | `nvtx.py`, `comm.py`, `backend.py` |
| `magi_attention.meta._make_dispatch_meta.make_dispatch_meta_from_qk_ranges` | `solver.py` |

The last entry is the only private path. `__init__.py` checks both that the
symbol exists and that its signature still accepts `dispatch_config`,
`is_same_source`, `is_q_permutable`, `is_k_permutable` and `uneven_shard`, so an
incompatible Core fails at import time rather than deep inside a plan build.

If a future capability is reusable by other extensions or by dense attention, it
belongs in Core as a minimal, DSA-free primitive — not here, and DSA plans,
configs, runtimes and kernels never move back into Core.

## Public API

```python
from magi_attn_extensions.DSA import (
    DsaPlanPolicy,
    DsaRatio,
    DsaSharedLayoutConfig,
    DsaStructuralLayoutConfig,
    MagiDSAConfig,
    MagiDSAForwardResult,
    MagiDSAInput,
    MagiDSALayer,
    MagiDSAPackedMeta,
    MagiDSAProExecutionBundle,
    MagiDSAProLayerStack,
    MagiDSAProModelSpec,
    MagiDSAProRuntimeMgr,
    MagiDSAProjector,
    MagiDSARuntimeMgr,
    layout_and_project_dsa_input,
    layout_source_hidden_once,
    project_local_dsa_input,
)
```

`DsaRatio` and `DsaPlanPolicy` are `Literal` type aliases exported for type
annotation only; `MagiDSAProjector` is a `Protocol` implemented by callers.

### Declared advanced submodules

These are not part of the top-level `__all__` but are imported by in-repo
benchmarks and scripts, so renaming them requires updating those consumers:

```text
magi_attn_extensions.DSA.nvtx                        # dsa_nvtx_range
magi_attn_extensions.DSA.comm                        # unlayout_dsa_query_tensor
magi_attn_extensions.DSA.kernels.triton.diagnostics  # nonfinite row/block stats
magi_attn_extensions.DSA.kernels.cutedsl.pack        # AOT pack/reduce kernels
```

Runtime return types and diagnostics (`DsaExecutionHandle`,
`DsaRuntimeCounters`), plan dataclasses and the solver stay reachable from
`.runtime`, `.meta` and `.solver` respectively, also as advanced API.

## Layout

```text
DSA/
├── __init__.py, api.py          # stable public exports + Core compat check
├── config.py, types.py          # configuration and tensor-facing types
├── layer.py, model_adapter.py   # model layer and projection adapters
├── runtime.py, pro_runtime.py   # runtime managers
├── meta.py, solver.py           # execution plan collection and solver
├── comm.py, packing.py          # routing/layout collectives and packing maps
├── backend.py, dist.py, phase.py, reference.py
├── kernels/cutedsl/pack.py
├── kernels/triton/*.py          # 8 Triton kernels
└── docs/                        # design notes (repo-only, not shipped in the wheel)
```

## Install

Core and extension are two distributions and both must be installed:

```bash
python3 -m pip install --no-build-isolation -e .
python3 -m pip install --no-build-isolation -e ./extensions
```

Note: this repository and the mentor MSA repository both build a distribution
named `magi_attn_extensions` that owns the same parent `__init__.py`, so their
wheels **cannot be safely installed side by side into one environment**. MSA and
DSA are "peers" in package layout and dependency direction, not in installation.

## Run

```bash
# static gates, source import and type check
PYTHONPATH=.:extensions python3 -c 'from magi_attn_extensions.DSA import MagiDSALayer'
PYTHONPATH=.:extensions pytest -q extensions/tests/dsa_v4
PYTHONPATH=.:extensions python3 -m mypy magi_attention extensions/magi_attn_extensions/DSA

# multi-GPU, backend and profile entry points stay at the repo root
scripts/test/run_cp1.sh
scripts/test/run_multigpu.sh
scripts/profile/run_5step.sh
```
