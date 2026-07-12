# Magi_DSA V4 tests

## B300 platform and image source

The steps 1–7 acceptance target is one `world_size=cp=8` process group over
GPU 0–7 of an 8× NVIDIA B300 SXM6 AC node (SM103). CP=1 is only the
same-weights, same-global-input numerical and gradient oracle; it is not a
distributed acceptance result. CP=2 remains a compatibility regression but is
not an exit criterion. The earlier four-pair CP=2 interpretation is superseded;
the single-group CP=8 steps 1–7 matrix has now passed.

The authoritative host worktree is:

```text
/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll
```

The B300 image recipe is in the main checkout, not in the worktree:

```text
/home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile
```

The recipe directory retains its historical step number. It pins
`nvcr.io/nvidia/pytorch:26.06-py3` by registry digest
`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`
and pins the FlashMLA, cudnn-frontend, fast-hadamard-transform, and CUTLASS DSL
sources. Build the current local test tag from that exact file with:

```bash
docker build \
  -f /home/scratch.wewen_gpu/MagiAttention/agents/docker/magi-dsa-b300-step5/Dockerfile \
  -t magi-dsa-b300-cp8:dev \
  /home/scratch.wewen_gpu/MagiAttention
```

fast-hadamard-transform is built for compute capability 10.3. FlashMLA is
built with `FLASH_MLA_DISABLE_SM90=1`, producing the SM100-family path used by
B300. `dsa_pack.py` has an architecture-neutral mapping/frontend and includes
the actual `(major, minor)` device capability in every compile-cache key, so
SM103 receives its own packing/remap/CSR cache entries.

Image tags are mutable aliases. The base image used for this steps 1–7 rerun is
`sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da`.
Record and verify the image ID from `docker image inspect` with every report.

## Native communication is an acceptance prerequisite

Compilation and execution are separate phases. The image used for final B300
acceptance must additionally contain `nvidia-nvshmem-cu13==3.6.5` and a
`magi_attention.magi_attn_comm` extension built from the current repository
revision for compute capability 100 (the repository build uses the SM100
family target on B300). NVSHMEM must not have been compiled out with
`DISABLE_NVSHMEM`.

The base Dockerfile above supplies the B300 compute dependencies; by itself it
does not prove the native communication requirement. The CP=8 native run fixes
`GrpCollConfig.num_nvl_bytes=1073741824` (1 GiB) for each
`(group, buffer_name)`. The full path keeps four typed-payload buffers plus one
replicated-gradient buffer resident. Including 32 MiB of workspace per buffer,
this requires at least `5,536,482,080` bytes (about 5.15625 GiB) per GPU; step 8
must dry-allocate that footprint before timing. On this single-node, all-NVLink
path `num_rdma_bytes=0`, so `NVSHMEM_SYMMETRIC_SIZE` must be unset and is
reported as N/A. Steps 8/9 must set
`B300_IMAGE` to the immutable ID of a derived native-enabled image and must not
use the base tag while it still lacks the extension. The current steps 1–7 rerun
will use the explicitly recorded incremental container below before steps 8/9.

For the CP=8 verification, the base image was extended in an isolated
container with NVSHMEM 3.6.5. The loaded binaries have SHA256
`0427073e7a0f16450bade229528638bfd1c9f610bf31d00f04d7f7899ffa0eaf`
(`magi_attn_ext`) and
`ca98aa8439007b647c52b8aa4e18b6b0beb631593846445bd877528346c87030`
(`magi_attn_comm`); `cuobjdump --list-elf` reports only `sm_100`, and both were
loaded on SM103. The CP=8 tests additionally asserted `GrpCollIntraHandle` and
executed native full backward without fallback. This incremental container is
still not the immutable native image that steps 8/9 must build.

```bash
NATIVE_CONTAINER=magi-dsa-b300-cp8-native
test "$(docker inspect --format '{{.Image}}' "${NATIVE_CONTAINER}")" = \
  'sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da'
docker exec -w /workspace "${NATIVE_CONTAINER}" \
  python -c 'import nvidia.nvshmem; import magi_attention.magi_attn_comm'
```

If the extension is installed only in an image-side checkout, do not shadow it
with a different bind-mounted package. Build it in the mounted worktree or run
the exact same repository revision contained in the acceptance image.

Every `test_native_grpcoll_*` case must execute in the eight-rank B300 group. A
skip because `magi_attn_comm` or NVSHMEM is unavailable is an acceptance
failure; the CP=8 steps 1–7 acceptance run must report zero skipped tests.

Native rows are padded internally to the alignment required by their dtype and
restored to the logical row shape after wait. This covers BF16 compressed-Ki
rows and FP32 GroupReduce rows such as hidden width 64. Large replicated FP32
gradients are split into aligned rows no wider than 4096 elements so native
scratch does not scale with one giant flattened row.

## Runtime and failure boundaries

CP=1 supports full-module deepcopy and `torch.save`; deserialization rebuilds
locks and clears the device-bound cache so forward plans rematerialize on
demand. A distributed `ProcessGroup`, including CP=8, is external,
non-pickleable PyTorch state, so distributed runtimes support `state_dict`
only: construct a runtime on the target group and then load the state.

Normal exit and coordinated recoverable exceptions drain launched work. A
collective-wait or rank-local compute/kernel failure requests a best-effort
`ProcessGroup.abort()` after draining the common prefix. On B300, abort does not
guarantee that a peer already blocked in a CUDA/NCCL/NVSHMEM stream wait is
released, so unrecoverable faults are fail-stop: an external launcher must
terminate every worker within a 60-second watchdog. The step-7 suite proves
coordinated drain/reuse and the abort request; step 9 must test isolated
single-rank faults under that launcher and must not claim same-process recovery.

## Validation commands

These are the commands used for the completed CP=8 steps 1–7 rerun. The
single-GPU command produces oracle evidence only. The
distributed exit uses one eight-rank group in the recorded incremental native
container with this worktree mounted at `/workspace`. Do not use an old copied
`/tmp/native-test/test_dsa_cp.py`. Steps 8/9 must replace the incremental
container with their new immutable native image.

```bash
WORKTREE=/home/scratch.wewen_gpu/MagiAttention/agents/worktrees/magi-dsa-v4-plan-grpcoll
B300_BASE_IMAGE=sha256:b4aca4fdd2ad71ba398ea9ef9a93e9530c8df9c17bfe021ec9b725a362ee49da
NATIVE_CONTAINER=magi-dsa-b300-cp8-native
GPU=0
GPU_GROUP=0,1,2,3,4,5,6,7
GRPCOLL_NUM_NVL_BYTES=1073741824  # 1 GiB; the test config must use this value

# CP=1 is only the numerical/gradient oracle, not the distributed exit.
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES="${GPU}" \
  -v "${WORKTREE}:/workspace/MagiAttention" \
  -v /home/scratch.wewen_gpu/megatron-lm:/ws/megatron-lm:ro \
  -e MAGI_DSA_MEGATRON_PATH=/ws/megatron-lm \
  -w /workspace/MagiAttention "${B300_BASE_IMAGE}" \
  timeout 1200 pytest -q -rs \
    tests/test_dsa/test_dsa_api.py \
    tests/test_dsa/test_dsa_dispatch.py \
    tests/test_dsa/test_dsa_megatron.py \
    tests/test_dsa/test_dsa_solver.py \
    tests/test_dsa/test_dsa_pack_kernel.py

# The only distributed exit for steps 3/5/6/7: one GPU0..7, world_size=cp=8 group.
# This is the single-node NVLink path (num_rdma_bytes=0); the symmetric heap is N/A.
docker exec \
  -e CUDA_VISIBLE_DEVICES="${GPU_GROUP}" \
  -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29617 \
  -w /workspace "${NATIVE_CONTAINER}" \
  bash -lc 'unset NVSHMEM_SYMMETRIC_SIZE; \
    test -z "${NVSHMEM_SYMMETRIC_SIZE+x}"; \
    timeout 5400 pytest -q -rs tests/test_dsa/test_dsa_cp.py'
```

`test_dsa_cp.py` is no longer transport-only. It currently contains:

- A2AV and native GroupCast/GroupReduce transport, including empty routes and
  payload-private state (steps 3/4);
- CP=8 reference/kernel forward, sequential/balanced plans, and CP=1 oracle
  parity (step 5);
- CP=8 reference/kernel backward, owner/replicated reductions, and saved-state
  checks (step 6); and
- the 2×2 overlap matrix, two in-flight microbatches, reentrant and retained
  backward, gradient accumulation, an empty rank, coordinated real-collective
  drain/reuse, abort-request checks, and early-exit draining (step 7).

`test_dsa_megatron.py` checks ratio-4 compressor forward parity against the
read-only checkout at exact revision
`c6449f0b23be397449f21c0967c5fc90785e55ea`. Missing source, dependency import,
or revision mismatch produces a skip, which is an acceptance failure even
though pytest itself may return zero.

`test_dsa_pack_kernel.py` contains the packing/remap/FP32-CSR execution
watchdogs. Post-compilation kernel tests use a 30-second watchdog. CP8
transport cases use 180 seconds, full/concurrency cases use 600–900 seconds,
and the first SM100 cuDNN sparse-backward cold compile uses 1200 seconds.

The current 2026-07-12 CP=8 results are `32 passed` for API, `36 passed` for
dispatch+solver, `18 passed` for packing+Megatron, and `27 passed, 0 skipped`
for the single eight-rank CP file (637.08 seconds): `113 passed, 0 skipped` in
the formal directory. Legacy regressions add `17 passed` on one GPU and
`2 passed` on CP=8. Black, isort, Ruff 0.12.5, compileall, and
`git diff --check` passed. Steps 8/9 remain unexecuted.

## Current scope boundary

Steps 1–7 now use CP=8 and the eight-rank correctness, recoverable-cleanup,
and best-effort abort-request matrix above has run. Step 8 performance
calibration, final solver-coefficient fitting, fixed 20-pack runs, and E2E
gates have not run. Step 9's immutable image, isolated launcher-failure matrix,
integration, and atomic commits have also not run. The steps 1–7 results are
not evidence that steps 8 or 9 are complete.
