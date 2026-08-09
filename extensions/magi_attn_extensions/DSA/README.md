# Magi-DSA

Magi-DSA is a DeepSeek-style sparse-attention extension built on top of
MagiAttention Core. It lives in its own Python distribution
(`magi_attn_extensions`) and owns its plans, runtime, communication schedule,
backends and kernels.

```text
magi_attn_extensions/DSA  ---->  magi_attention (Core)
```

The dependency is one-way. Core never imports this package.

## Boundary

Magi-DSA states what it needs as ranges and lets Core do the lowering. It does
not own a collective, a row map, or a dispatch algorithm of its own.

| Core capability | Used by |
|---|---|
| `magi_attention.common.AttnRanges` | `meta.py`, `solver.py` |
| `magi_attention.common.enum.AttnMaskType` | `solver.py` |
| `magi_attention.common.range_op.range_gather` / `range_reduce` | `comm.py`, `dist.py` |
| `magi_attention.comm.primitive.grpcoll.group_cast` / `group_reduce` | `comm.py` |
| `magi_attention.comm.work.WorkWithPostProcessFn` | `comm.py` |
| `magi_attention.meta.collection.comm_meta` group-collective args | `packing.py` |
| `magi_attention.meta.solver.dynamic_attn_solver` range lowering | `packing.py` |
| `magi_attention.meta.solver.dispatch_solver` dispatch types and algorithms | `solver.py` |
| `magi_attention.utils.general._make_device_tensor` | `packing.py` |
| `magi_attention.utils.nvtx` | `nvtx.py`, `backend.py` |
| `magi_attention.meta._make_dispatch_meta` dispatch meta and buckets | `solver.py` |

The last entry is the only private path. `__init__.py` checks both that the
symbol exists and that its signature still accepts `dispatch_config`,
`is_same_source`, `is_q_permutable`, `is_k_permutable` and `uneven_shard`, so an
incompatible Core fails at import time rather than deep inside a plan build.

If a future capability is reusable by other extensions or by dense attention, it
belongs in Core as a minimal, DSA-free primitive.

## What this extension does not own

These are stated explicitly because each was once reimplemented here:

- no All2AllV data plane. A route is a group-cast, and its adjoint is the
  symmetric group-reduce, both from Core.
- no per-row send, receive, consumer or reverse index tables. Routes are ranges,
  so a plan is sized by fragments, not by tokens or compressed rows.
- no row-gather or CSR-reduce kernel. Duplicate-row gathers are `index_select`,
  whose backward already accumulates, and prefix gathers are Core range ops.
- no trainable parameter. Every weight belongs to the model and reaches the
  runtime as a `DsaProjections` callback.
- no object collective. The plan is a pure function of caller metadata, so each
  rank rebuilds a bit-identical copy instead of gathering or broadcasting one.
- no plan policy. `structural_balanced` is the only layout, and it is the native
  causal-area MinHeap over packed-global chunks.

## Public API

```python
from magi_attn_extensions.DSA import (
    DsaProjections,
    DsaRatio,
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

`DsaRatio` is a `Literal` type alias exported for annotation only, and
`MagiDSAProjector` is a `Protocol` implemented by callers.

`MagiDSAPackedMeta` carries `cu_seqlens` and the whole `source_token_counts`
split. The caller already owns that sharding, and stating it makes the execution
plan a pure function of caller metadata, which is what removes the plan
broadcast and the owner-layout gather.

### Declared advanced submodules

These are not in the top-level `__all__` but are imported by in-repo benchmarks
and scripts, so renaming them requires updating those consumers:

```text
magi_attn_extensions.DSA.nvtx                        # dsa_nvtx_range
magi_attn_extensions.DSA.comm                        # unlayout_dsa_query_tensor
magi_attn_extensions.DSA.kernels.triton.diagnostics  # nonfinite row/block stats
```

Runtime return types and diagnostics (`DsaExecutionHandle`,
`DsaRuntimeCounters`), plan dataclasses and the solver stay reachable from
`.runtime`, `.meta` and `.solver` respectively, also as advanced API.

## Layout

```text
DSA/
├── __init__.py, api.py          # stable public exports + Core compat check
├── config.py, types.py          # configuration and tensor-facing types
├── projection.py                # the model-owned callback boundary
├── modeling.py                  # reference model modules (model side)
├── model_adapter.py             # projection adapters
├── runtime.py, pro_runtime.py   # runtime managers
├── meta.py, solver.py           # range-shaped plan and its solver
├── comm.py, packing.py          # Core group-collective routes and device maps
├── backend.py, dist.py, phase.py, reference.py
├── kernels/triton/*.py          # 8 fused elementwise and index kernels
└── docs/                        # design notes (repo-only, not shipped)
```

`modeling.py` is model-side code. Nothing on the execution path (`runtime`,
`dist`, `comm`, `solver`, `packing`, `backend`) imports it; the runtime only
ever sees the callbacks from `MagiDSALayer.projections()`. A different model may
replace that file entirely as long as it supplies a `DsaProjections` bundle.

## Install

Core and extension are two distributions and both must be installed:

```bash
python3 -m pip install --no-build-isolation -e .
python3 -m pip install --no-build-isolation -e ./extensions
```

Note: this repository and the mentor MSA repository both build a distribution
named `magi_attn_extensions` that owns the same parent `__init__.py`, so their
wheels **cannot be safely installed side by side into one environment**. MSA and
DSA are peers in package layout and dependency direction, not in installation.

## Run

```bash
# static gates, source import and type check
PYTHONPATH=.:extensions python3 -c 'from magi_attn_extensions.DSA import MagiDSALayer'
PYTHONPATH=.:extensions pytest -q extensions/tests/dsa_v4
PYTHONPATH=.:extensions python3 -m mypy magi_attention extensions/magi_attn_extensions/DSA

# multi-GPU, backend and profile entry points stay at the repo root
scripts/test/run_cp1.sh
scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward
scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-pro-128k \
    --plans balanced --steps 5 --step-mode pro-pair \
    --layout-policy structural-balanced
```
