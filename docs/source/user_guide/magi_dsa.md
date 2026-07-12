# Magi DSA

Magi DSA is the packed DeepSeek V4 hybrid-attention primitive exposed by the
stable `magi_attention.api` package. Its public surface is:

```python
from magi_attention.api import (
    DsaOverlapConfig,
    DsaPackedMeta,
    MagiDSAConfig,
    MagiDSAInput,
    MagiDSARuntimeMgr,
    MagiDSAYarnConfig,
    calc_dsa,
)
```

The installed package does not contain the former
`magi_attention.experimental.dsa_v4` module; no experimental import is needed
or supported. New code should use the names above. The V4-spelled config
aliases exist only for compatibility with callers of the earlier prototype.

## Layer forms

One `MagiDSARuntimeMgr` has one immutable `compress_ratio`, which selects the
layer form:

| `compress_ratio` | Layer form | Indexer KL |
| --- | --- | --- |
| `0` | causal sliding window plus the attention sink | scalar FP32 zero |
| `4` | overlapping compressor, Lightning Indexer top-k, sliding window, and sink | local differentiable contribution |
| `128` | non-overlapping compressor, dense compressed causal prefix, sliding window, and sink | scalar FP32 zero |

The V1 contract fixes 64 query heads, head/KV dimension 512, compressed RoPE
dimension 64, window size 128, and (for ratio 4) 64 Indexer heads of dimension
128 with top-k 512. `hidden_size`, `q_lora_rank`, and `softmax_scale` come from
the model. Parameters and row tensors are BF16; the attention sink is FP32.
`backend="reference"` selects the correctness-oriented PyTorch path, while
`backend="kernel"` requires the packaged CUDA kernel dependencies and a
supported GPU.

## Minimal CP=1 example

The following is a complete forward/backward example for one packed sample on
one CUDA device. It uses deliberately small model-side projection dimensions;
the fixed attention dimensions remain unchanged.

```python
import torch

from magi_attention.api import (
    DsaPackedMeta,
    MagiDSAConfig,
    MagiDSAInput,
    MagiDSARuntimeMgr,
    calc_dsa,
)

torch.cuda.set_device(0)
torch.manual_seed(42)
device = torch.device("cuda", 0)

config = MagiDSAConfig(
    compress_ratio=4,
    hidden_size=8,
    q_lora_rank=8,
    softmax_scale=512**-0.5,
    backend="reference",
)
runtime = MagiDSARuntimeMgr(config).to(device).train()


def trainable(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)


tokens = 4
dsa_input = MagiDSAInput(
    x=trainable(tokens, config.hidden_size),
    qr=trainable(tokens, config.q_lora_rank),
    q=trainable(tokens, config.num_heads, config.kv_dim),
    latent_kv=trainable(tokens, config.kv_dim),
    sink=trainable(config.num_heads, dtype=torch.float32),
    packed_meta=DsaPackedMeta(
        torch.tensor([0, tokens], dtype=torch.int32)
    ),
)

output, indexer_kl = calc_dsa(dsa_input, runtime)
loss = output.float().square().mean() + indexer_kl
loss.backward()

assert output.shape == (tokens, 64, 512)
assert output.dtype == torch.bfloat16
assert indexer_kl.shape == torch.Size([])
assert indexer_kl.dtype == torch.float32
assert dsa_input.q.grad is not None
assert dsa_input.latent_kv.grad is not None
assert dsa_input.sink.grad is not None
```

For multiple packed samples, use cumulative global sample boundaries such as
`[0, 5, 32]` for lengths 5 and 27. `cu_seqlens` must be contiguous, one
dimensional, start at zero, be non-decreasing, and have dtype `torch.int32`.
It is host-inspected while building the static plan, so keeping it on CPU is
recommended.

## Tensor, output, and gradient contract

All row tensors for an invocation must be on the same CUDA device:

| Value | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x` | `[T_local, hidden_size]` | BF16 | trunk state consumed by compressed forms |
| `qr` | `[T_local, q_lora_rank]` | BF16 | low-rank query state for the ratio-4 Indexer |
| `q` | `[T_local, 64, 512]` | BF16 | main attention query |
| `latent_kv` | `[T_local, 512]` | BF16 | latent key/value state |
| `sink` | `[64]` | FP32 | replicated, learnable per-head attention sink |
| `packed_meta.cu_seqlens` | `[num_samples + 1]` | int32 | global logical packed-sample boundaries |

`calc_dsa` returns owner-local `output[T_local, 64, 512]` in BF16 and a scalar
FP32 Indexer KL contribution. In CP=1, the contribution is already normalized
by the global query-token count. In CP>1, each rank receives its differentiable
local contribution; the sum of detached contributions is the global value for
logging. Do not all-reduce the live KL tensor before backward.

Autograd follows the mathematical dependencies below. `None` is intentional,
not a missing reduction:

| Ratio | `dx` | `dqr` | `dq` | `dlatent_kv` | `dsink` | Runtime parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `0` | `None` | `None` | yes | yes | yes | no compressor/Indexer parameters |
| `4` | yes | `None` | yes | yes | yes | compressor and Indexer gradients |
| `128` | yes | `None` | yes | yes | yes | compressor gradients |

The ratio-4 Indexer branch deliberately detaches the trunk `x`/`qr` inputs;
`x` still receives its compressor-path gradient, while `qr` has no other DSA
path. The sink gradient and replicated runtime-parameter gradients are summed
inside DSA across the CP group and marked as CP-reduced. Training frameworks
must not reduce those tensors over the same CP group a second time. Owner-local
input gradients are returned in the same immutable local fragment order as the
inputs.

## CP=8 runtime boundary

Run CP=8 as one NCCL process per GPU. Initialize `torch.distributed`, select
the rank's CUDA device, and pass the same eight-rank CP process group to every
runtime:

```python
import os

import torch
import torch.distributed as dist

from magi_attention.api import MagiDSARuntimeMgr

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
cp_group = dist.group.WORLD  # or an explicit eight-rank group

runtime = MagiDSARuntimeMgr(
    config,
    cp_group=cp_group,
    dispatch_policy="balanced",
).cuda()
```

The snippet assumes the same `config` construction as the CP=1 example. An
application that creates an explicit group should retain and pass that group.
If the global job also has data or tensor parallel dimensions, `cp_group` must
be the explicit eight-rank CP subgroup rather than `dist.group.WORLD`.

`packed_meta` still describes the global sample layout on every rank, but the
five input tensors contain only the rows owned by that rank. The caller must
materialize them in the immutable fragment order returned by
`runtime.get_dispatch_plan(packed_meta).ranks[rank]`; `calc_dsa` does not scatter
a global tensor. A balanced plan may assign zero rows to a rank, and zero-row
local tensors are valid. Output rows preserve the same owner-local order.

For the single-node native CP=8 path, set
`MAGI_ATTENTION_NATIVE_GRPCOLL=1` before the first DSA invocation and use a
package built with the native `magi_attn_comm` extension and NVSHMEM. The
deployment bootstrap must initialize one native GrpColl buffer manager for the
CP group before use and release it on normal shutdown. Native buffer lifecycle
is intentionally not part of `magi_attention.api`; do not make application
model code depend on the internal buffer-manager modules.

The validated single-node topology uses native intra-node
`GrpCollIntraHandle` communication, not an AllToAll-v fallback. Its deployment
configuration uses `num_sms=20`, `nvl_chunk_size=8`, `nvl_buffer_size=256`,
`rdma_chunk_size=8`, and `rdma_buffer_size=256`; it reserves 1 GiB of NVLink
storage per typed buffer, sets `num_rdma_bytes=0`, and leaves
`NVSHMEM_SYMMETRIC_SIZE` unset. Production launchers should fail their
preflight if the native extension, allocation, or actual handle check fails
rather than silently changing transport.

The two switches in `DsaOverlapConfig` change scheduling only:

```python
overlap = DsaOverlapConfig(
    compressed_cast_indexer=True,
    dki_reduce_sparse_backward=True,
)
runtime = MagiDSARuntimeMgr(config, cp_group=cp_group, overlap_config=overlap)
```

They do not change parameters, outputs, gradients, or checkpoint contents.

## Checkpoint and failure boundaries

A process group is external runtime state and is not pickleable. Save only the
distributed runtime's state dict, then construct a new runtime on the target
group and load it:

```python
# Save on the source job.
torch.save(runtime.state_dict(), "magi_dsa.pt")

# Restore after creating the destination cp_group.
restored = MagiDSARuntimeMgr(config, cp_group=cp_group).cuda()
restored.load_state_dict(torch.load("magi_dsa.pt", map_location="cuda"))
```

Do not `torch.save(runtime)` for CP=8. Device communication plans and native
work are invocation-local or rematerialized; they are not checkpoint state.
CP=1 supports module deepcopy/pickle, although `state_dict` remains the most
portable checkpoint format.

Normal completion and coordinated recoverable exceptions drain work launched
by the invocation. A rank-local CUDA/kernel error or a failed collective has
fail-stop semantics: DSA requests a best-effort process-group abort, but a peer
may already be blocked in CUDA, NCCL, or NVSHMEM. The external launcher must
terminate all eight workers within a 60-second watchdog. Do not attempt
same-process recovery; recreate the workers, process group, native buffers,
and runtime from a committed checkpoint.

## Integration scope

This API is a stable packed hybrid-attention primitive, not a claimed drop-in
replacement for an arbitrary model layer. A real model integration must still
map checkpoint weights, produce `x`/`qr`/`q`/`latent_kv`, materialize CP-local
fragments, register optimizer parameters, and own native-launcher lifecycle.
That full model adapter is outside the current Magi DSA release scope.
