# Magi DSA release validation

## Supported public surface

Magi DSA is released from the stable `magi_attention.api` namespace. New code
should import only these public symbols:

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

The installed production wheel and public API must not contain or resolve
`magi_attention.experimental`; historical prototypes in other worktrees or
backups are outside that installed boundary and need not be physically deleted
for this release. `MagiDSAV4Config` and
`MagiDSAV4YarnConfig` remain compatibility aliases, but new serialized
configurations use `MagiDSAConfig` and `MagiDSAYarnConfig`.

One `MagiDSARuntimeMgr` owns the parameters and immutable
`compress_ratio` for one layer form: ratio 0 is window-only, ratio 4 is CSA,
and ratio 128 is HCA. See the
[Magi DSA user guide](../../docs/source/user_guide/magi_dsa.md) for tensor
contracts, distributed setup, checkpointing, and the complete examples.

## Runnable CP=1 example

This example runs forward and backward through the installed stable API on one
CUDA device. CP=1 is the same-input numerical/gradient oracle; it is not the
distributed release exit.

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
    packed_meta=DsaPackedMeta(torch.tensor([0, tokens], dtype=torch.int32)),
)
output, indexer_kl = calc_dsa(dsa_input, runtime)
(output.float().square().mean() + indexer_kl).backward()

assert output.shape == (tokens, 64, 512)
assert output.dtype == torch.bfloat16
assert indexer_kl.dtype == torch.float32
assert dsa_input.q.grad is not None
assert dsa_input.latent_kv.grad is not None
assert dsa_input.sink.grad is not None
```

The release matrix also executes
`tests/test_dsa/installed_package_smoke.py --cuda` from `/tmp`, outside the
source checkout. That check covers ratios 0, 4, and 128 and rejects an import
that resolves to the archived repository instead of the installed wheel.

## Immutable production image

The production image is built by
`agents/release/magi-dsa-v4/build_image.sh` using
`agents/release/magi-dsa-v4/Dockerfile.production`. The builder resolves the
requested Git revision to a full 40-character commit, exports only committed
Git objects at that commit, expands the frozen submodule objects into a
temporary context, builds and installs a versioned wheel, builds the native
extensions, and records their hashes in `/opt/magi-dsa-build-manifest.json`.
Release commands pass the full commit explicitly even though the builder can
normalize another unambiguous Git revision. It never copies the caller's dirty
worktree into the image.

The Dockerfile pins the B300 base image by registry digest
`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`
and installs `nvidia-nvshmem-cu13==3.6.5`. Image tags are mutable aliases;
reports and commands must use the content-addressed `sha256:...` image ID.

Build the clean revision and run the ordered final matrix in one command.
`REPO` must be an initialized disposable integration worktree whose recursive
submodules are checked out at the gitlinks recorded by the release-image source
revision; do not point it at the development worktree:

```bash
REPO="<ABSOLUTE_INITIALIZED_DISPOSABLE_INTEGRATION_WORKTREE>"
RELEASE_IMAGE_SOURCE_REVISION="<FULL_40_CHAR_RELEASE_IMAGE_SOURCE_GIT_SHA>"
RELEASE_OUTPUT="<ABSOLUTE_RELEASE_ARTIFACT_DIRECTORY>"

"${REPO}/agents/release/magi-dsa-v4/build_image.sh" \
  --revision "${RELEASE_IMAGE_SOURCE_REVISION}" \
  --tag "magi-dsa-v4-b300:${RELEASE_IMAGE_SOURCE_REVISION:0:12}" \
  --output-dir "${RELEASE_OUTPUT}" \
  --run-final-matrix

B300_IMAGE=$(<"${RELEASE_OUTPUT}/image_id.txt")
test "${B300_IMAGE}" = \
  "$(docker image inspect --format '{{.Id}}' "${B300_IMAGE}")"
```

Only the frozen-pack, performance/profile-output, and final-result artifact
directories are bind-mounted for validation. Source directories and Megatron
checkouts must never be mounted; there is no editable install, image-side
development checkout, or fallback to an uninstalled package.
`installed-wheel-smoke.log`, `image_inspect.json`, the build log, matrix
results, and SHA256 manifests under `RELEASE_OUTPUT` are the authoritative
evidence.

## CP=8 native and zero-skip exit

The distributed exit is exactly one `world_size=cp_size=8` process group over
GPU 0–7 of one 8× NVIDIA B300 SXM6 AC node (SM103). CP=2 remains a
compatibility regression and cannot replace this exit.

`final_matrix.json` and `run_final_matrix.py` enforce the following ordered
cases inside the immutable image:

- Installed public-API forward/backward smoke from outside the repository.
- CP=1 API, solver, packing-kernel, and pinned Megatron oracle coverage.
- CP=8 native transport and full forward/backward coverage for all ratios and
  policies, gradients, the 2×2 overlap matrix, concurrent/reentrant calls,
  empty routes/ranks, and coordinated recoverable failures.
- An isolated single-rank failure followed by launcher cleanup and a fresh
  process-group native health check.

The time-boxed sampled closure does not sample this matrix: all four ordered
cases remained required on the rebuilt final image. Run
`b300-cp8-final-f8ad2e5d-20260713T110210Z` passed and sealed all four, so step
9 is complete under the sampled closure. This does not turn the sampled
performance evidence into formal step-8 acceptance or formal release
qualification.

The first step-9 run, `b300-cp8-final-f88ca22d-20260713T103532Z`, used the f88
candidate image
`sha256:4bd9b55dd1303043256fd6accf79a5362f630b85448a5425958c42a2467d0bea`.
Public smoke passed in `8.679 s`, CP1 passed `93/93` tests in `99.657 s`, and
CP8 reported `4 passed, 14 failed` in `7.728 s` before the ordered run stopped.
The root cause was a spawned subprocess launched from `/tmp` under pytest's
importlib mode failing with `ModuleNotFoundError: tests`, not DSA correctness,
OOM, or watchdog failure. With `--import-mode=append`, the focused native
8-rank diagnostic passed (`1 passed` in `36.61 s`) while `magi_attention`
continued to resolve from the installed wheel's `site-packages`. The failed-run
and outer manifest-file SHA256s are
`81f8f21a524f4d2dd050ca591358234b133f222de8e03523c75b083caf44757d` and
`0818fe00f144b63d7718e8d38f3a775a91da4acef824419b639491b3a59107b3`.
The fix commits are DEV `e8f50e97896fec4eee5646988e6797e9c5a8b76c` and integration
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`; the rebuilt final step-9
candidate is the sealed f8 image recorded below. The old f88 image is retained
only as the sampled-timing and failed-matrix predecessor. Since runtime,
kernel, and solver behavior did not change, the sampled timing is not rerun.

The CP=8 case sets `MAGI_ATTENTION_NATIVE_GRPCOLL=1`, sets
`MAGI_ATTENTION_HIERARCHICAL_COMM=0` and `CUDA_DEVICE_MAX_CONNECTIONS=8`, and
unsets `NVSHMEM_SYMMETRIC_SIZE`. The frozen `GrpCollConfig` uses `num_sms=20`,
`num_nvl_bytes=1073741824` (1 GiB per buffer), and `num_rdma_bytes=0`. All four
typed payload buffers plus the replicated-gradient buffer are dry-allocated
before timing, and the runtime must instantiate an actual
`GrpCollIntraHandle`. DSA communication uses native GroupCast/GroupReduce;
direct A2AV or hierarchical fallback is not accepted. Missing native
extensions, NVSHMEM, pinned dependencies, test data, or any required test is a
hard failure. Every JUnit case must report zero skipped tests, zero errors, and
zero failures. A timeout, OOM, non-finite tensor, native fallback, JIT miss in
a measured region, or revision/artifact mismatch invalidates the run.

A2AV was exercised only as compatibility coverage in the historical step 1–7
development matrix. The formal step 8/9 exit accepts only the actual
`GrpCollIntraHandle` native path; A2AV and hierarchical fallback are hard
failures.

For ratio 4, every merged dispatch fragment is subdivided into
sample-relative, 128-row canonical query tiles for Indexer projection, top-k,
forward KL, and backward KL recompute. Dispatch ownership may change
communication and load balance, but it must not change the GEMM row shape for
the same logical query or perturb the top-k boundary.

The cuDNN top-k launch retains the configured K=512 for short samples and
fills entries beyond the actual compressed-key count with `-1`. It must not
specialize launch K to an odd short-sample width such as 473, which violates
the frozen CuTe kernel's two-element vector-store divisibility requirement.

## Performance acceptance

Step 8 is driven by `agents/benchmarks/magi-dsa-v4-balance/`; its
[README](../../agents/benchmarks/magi-dsa-v4-balance/README.md) is the
executable workload and gate specification. Generate the seed-42, 20-pack file
exactly once before calibration and seal its file hash. The production build
flow first creates a clean, immutable installed-wheel calibration image; that
image generates the cost coefficients but is not the final measure/release
image. After the coefficients are reviewed, frozen, and committed, the same
build flow creates a distinct immutable final image for the formal
measure/profile run and release matrix. Formal measurement reuses the
hash-checked calibration file and never invokes `run.sh packs` again. The
wrapper mounts only pack, performance, and profile artifact directories:

```bash
REPO="<ABSOLUTE_INITIALIZED_DISPOSABLE_INTEGRATION_WORKTREE>"
DEVELOPMENT_REPO="<ABSOLUTE_DEVELOPMENT_WORKTREE>"
export PACKS_DIR="<ABSOLUTE_FROZEN_PACK_ARTIFACT_DIRECTORY>"
export PACKS_FILE="${PACKS_DIR}/packs-seed42.json"

# Generate and seal the pack file once, before calibration.
export MAGI_DSA_IMAGE="<IMMUTABLE_CALIBRATION_IMAGE_ID>"
export RUN_ID="calibration-run-id"
export PERF_DIR="<ABSOLUTE_PERFORMANCE_ARTIFACT_ROOT>/${RUN_ID}"
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" packs
(
  cd "${PACKS_DIR}"
  sha256sum "${PACKS_FILE##*/}" > "${PACKS_FILE##*/}.sha256"
)
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" calibrate

python3 "${DEVELOPMENT_REPO}/agents/benchmarks/magi-dsa-v4-balance/freeze_calibration.py" \
  --input "${PERF_DIR}/calibration.json"
# Review and commit dsa_calibration.py in DEVELOPMENT_REPO, cherry-pick that
# atomic commit into REPO, then build B300_IMAGE from the clean integration HEAD.

# Reuse the exact sealed file for the formal measure/profile run.
(
  cd "${PACKS_DIR}"
  sha256sum --check "${PACKS_FILE##*/}.sha256"
)
export MAGI_DSA_IMAGE="${B300_IMAGE}"
export RUN_ID="final-performance-run-id"
export PERF_DIR="<ABSOLUTE_PERFORMANCE_ARTIFACT_ROOT>/${RUN_ID}"
export PROFILE_DIR="<ABSOLUTE_PROFILE_ARTIFACT_ROOT>/${RUN_ID}"

"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" measure
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" profile
"${REPO}/agents/benchmarks/magi-dsa-v4-balance/run.sh" validate
```

The validator recomputes all timing and load-balance statistics from raw
per-pack, per-rank, per-iteration records; verifies candidate correctness
before timing; proves both overlap windows from Nsight CUDA intervals; and
seals the performance and profile roots in one SHA256 manifest. Skips and
partial runs are never accepted.

### Time-boxed sampled closure (non-formal)

On 2026-07-13 the user chose a roughly three-hour sampled closure. This does
not remove or relax the formal 20-pack contract above. The stopped formal
calibration attempt `b300-cp8-calibration-66d69258-20260713T054451Z` used
revision `66d69258fc9df1ee96f4c91df5d8225173b7fae2` and image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`.
Its progress is 57/120 units: all 40 ratio-0 units and
`r4-sequential-00` for packs 0 through 16. It is partial diagnostic evidence,
not resumable calibration and not formal acceptance. The container received
an external `SIGTERM` at `2026-07-13T07:45:12Z`; it was not OOM-killed, and
this stopped run was not a fail-stop test. Its surviving progress and
correctness records are diagnostic evidence only. Its
`agents/perf/magi-dsa-v4-balance/b300-cp8-calibration-66d69258-20260713T054451Z/progress.json`
file SHA256 is
`6f5520099807c290452b1a68da479e95240dbb1c9dfb19e54cc8908c025c528f`.
Under `agents/perf/magi-dsa-v4-release/calibration-logs`, the corresponding
`magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.docker-events.jsonl`
and `magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.termination.json`
file SHA256s are
`308245df4cdf2cf8db521960fb6ab5f1a3b9cab7e4823909ca20f3eb4ee81ce7` and
`927974b501330e4339a9e2793b5236697eb92d46294dc8733d0e1bd148c50870`.

The non-formal calibration runs packs `[0, 1]` across the six frozen
calibration cases: ratio 0/4/128, each with `sequential:00` and
`balanced:00`. Its 12-unit `sample_calibration.json` target is
`sampled-b300-sm103`, with `formal=false` and `scope=sampled_non_formal`; the
freeze command requires `--allow-sampled-non-formal`.

That calibration completed as run
`b300-cp8-sample-calibration-20260713T083426Z`, revision
`66d69258fc9df1ee96f4c91df5d8225173b7fae2`, image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`,
and progress 12/12. It contains 12 correctness, 96 plan, and 960 raw-timing
records. The frozen coefficient ID is
`df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d`.
In
`agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-calibration-20260713T083426Z`,
`sample_calibration.json` has SHA256
`5d42c3ea327cf866f7faed7db3334d5d8c7cf56e44d1b60624c219d608e4c0e3`,
and the `artifact_manifest.sha256` file has SHA256
`82ccd78736f7972b4160e442350ea8fee63a591bfe2597ed46d427c74e742883`.
The ratio-4 bounded least-squares fit has `R²=0.923198` and
`RMSE=83.118 ms`. It contains only two independent packs, and its window and
overlap features are exactly collinear (`window_rows = 31.75 * overlap_rows`,
design rank `6/7`), so those coefficients are not separately identifiable.
This is a sampled diagnostic fit, not evidence of formal model-fit quality.

After rebuilding the final image, the sampled performance target is nine
case-pack units:

- Pack 0: ratio 4 `sequential:00` plus balanced `00/01/10/11`, and ratio 128
  `sequential:00` plus `balanced:11`.
- Pack 16: ratio 4 `sequential:00` plus `balanced:11`.

These units retain one compile pass, two warm-ups, ten measured iterations,
and elementwise forward/all-gradient comparison. Their summaries must keep
`formal_acceptance=false` and `formal_gates_evaluated=false`. Keep the pack
sets distinct: the stopped attempt covers all 20 packs for ratio 0 and packs
0 through 16 for ratio-4 sequential; sampled calibration selects packs 0 and
1; final-image sampled measurement selects packs 0 and 16. Combining these
records does not create a new sampled pack set and must not be reported as
evidence for the 20/20 candidate imbalance and speed gates. See the benchmark
README for the exact `sample` command contract.

No sampled/diagnostic profile was run or produced. The formal ratio-4 Nsight
overlap gate therefore remains unevaluated.

## Runtime and failure boundary

CP=1 runtimes support complete-module deepcopy and serialization. A
distributed `ProcessGroup` is external non-pickleable state, so CP runtimes are
checkpointed with `state_dict`: construct a new runtime on the target group,
then load the state.

Normal exit and coordinated recoverable exceptions drain launched work. A
rank-local compute/kernel or collective-wait failure requests a best-effort
process-group abort. A peer already blocked in a CUDA/NCCL/NVSHMEM stream wait
may not return, so unrecoverable distributed faults are fail-stop. The external
supervisor must terminate all eight workers within its 60-second watchdog and
start a fresh process group; same-process recovery is not promised.

Magi DSA provides the packed attention primitive and its CP runtime. Wiring it
into a real DeepSeek V4 model, mapping model weights, choosing per-layer forms,
and running end-to-end model training or inference remain outside this release
scope.

## Final artifact record

These values were populated only from the sealed build, benchmark, and matrix
artifacts:

- Formal step-8 status: **not complete**; the frozen 20-pack gates remain
  unchanged and were not evaluated by the sampled closure.
- Stopped formal calibration attempt: revision
  `66d69258fc9df1ee96f4c91df5d8225173b7fae2`, image
  `sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`,
  run `b300-cp8-calibration-66d69258-20260713T054451Z`, progress `57/120`;
  externally stopped by `SIGTERM` at `2026-07-13T07:45:12Z`, not OOM and not
  a fail-stop test.
- Sampled calibration: run `b300-cp8-sample-calibration-20260713T083426Z`,
  revision `66d69258fc9df1ee96f4c91df5d8225173b7fae2`, image
  `sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`,
  progress `12/12`, record counts correctness/plans/raw timing `12/96/960`,
  target `sampled-b300-sm103`, scope `sampled_non_formal`, `formal=false`, and
  frozen coefficient ID
  `df625f1805024f6c7c78e2bdcb07476c8be94347c3e762a80006a87aea0e882d`.
  `sample_calibration.json` SHA256 is
  `5d42c3ea327cf866f7faed7db3334d5d8c7cf56e44d1b60624c219d608e4c0e3`;
  the `artifact_manifest.sha256` file SHA256 is
  `82ccd78736f7972b4160e442350ea8fee63a591bfe2597ed46d427c74e742883`.
- Predecessor f88 sampled timing completed all `9/9` selected correctness units and
  produced `72` plan records and `720` raw timing records. Source runs
  `b300-cp8-sample-measure-a-f88ca22d-20260713T093515Z`,
  `b300-cp8-sample-measure-b-f88ca22d-20260713T101533Z`, and
  `b300-cp8-sample-measure-c-f88ca22d-20260713T103247Z` have
  `artifact_manifest.sha256` file SHA256 values
  `8ec095b475228012a2029a49066595692ae7c431fb12fe0a33981c952a7948f5`,
  `a508503306049cc5460640b6babdb1ef69be2a942ea3bd210ba5c37e7013741e`,
  and `7423cfc56ad63ee8f2f9f6c4716eff7f9f6164ca7890e9e8a76f8c19dcf6d64c`,
  respectively. The aggregate evidence directory is
  `agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-measure-f88ca22d-20260713T103500Z-aggregate`;
  `sampled_measure_diagnostic.json` SHA256 is
  `47fb9e1ab4e852be90046c03438ac29ef8c29c552193cfe68e9ff36e262cded5`,
  and the `artifact_manifest.sha256` file SHA256 is
  `955e17348d4ab00c11a582be2c2dfb4b7bcb6abe79bd9d8eadb6d89dfc93a62b`.
  The artifact records `formal=false`, `formal_acceptance=false`,
  `formal_gates_evaluated=false`, and `all_gates_pass=null`. For the ratio-4
  two-pack diagnostic, baseline/candidate E2E is
  `28602.9863/29245.3057 ms` (`+2.245637%`, candidate slower), and Indexer is
  `474.7959/493.6558 ms` (`+3.972199%`, candidate slower); both sampled
  candidate packs remain within 5% E2E rank imbalance. For ratio 128 on pack
  0, E2E is `321.7919/320.7076 ms` (`-0.336954%`, candidate faster), with
  `0.10824%` candidate rank imbalance. Consequently,
  `sampled_performance_observation_pass=false`; this is a sampled observation,
  not a formal gate result.
- Sampled/diagnostic profile: not run and not produced; the formal ratio-4
  Nsight overlap gate remains unevaluated.
- Sampled-closure final-matrix result: run
  `b300-cp8-final-f8ad2e5d-20260713T110210Z`, `status=passed`, `error=null`;
  all four ordered cases passed. This completes step 9 only under the sampled
  closure; formal step 8 remains incomplete and sampled ratio-4 direction is
  failing.
- Sampled-closeout candidate source revision:
  `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`
- Installed candidate package version: `1.1.1+dsa.f8ad2e5d4f8b`
- Immutable sampled candidate image ID:
  `sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`
- Candidate build evidence:
  `agents/perf/magi-dsa-v4-release/sampled-final-f8ad2e5d-20260713T104133Z`;
  embedded `/opt/magi-dsa-build-manifest.json` SHA256
  `c07f71865a118786e43a54d3c980b14ffaa2db0484ec5e8094ed671ae57e7519`;
  `artifact_manifest.sha256`, `image_id.txt`, `package_version.txt`,
  `installed-wheel-smoke.log`, `image_inspect.json`, and `docker-build.log`
  file SHA256s are respectively
  `fa69dad9035c326cea00f6dc8ec7865ec3ea9f8bc568cf6a54a64fc7dfd13eff`,
  `27f7f3362179b16d6faf7541b55db5f86bce8506cab0af8164293b2a29b55985`,
  `965323aa062b7753d197ccf7b0920ad546316c6da7182a5c59224f8976768cce`,
  `eebb12ff3aa2518e74cbc3b9dad5cb04e0b3a2d75877228758eaf6bc534951c5`,
  `7417ec701e0b6f592be477df8269dbbf58ce9c7204c0c6ea8a4d778177f1c674`,
  and `15d5a5be9c7b496b930fc02b236496e840f3fed01f1dfbfc6e4efc3635fc02a8`.
  The installed-wheel smoke passed. Runtime, kernel, and solver behavior is
  unchanged from f88, so the sealed nine-unit sampled timing remains tied to
  the old f88 image and is not rerun; f88 is also retained as the failed-matrix
  predecessor. The new image is a sampled-closeout candidate, not a formal
  production/release image, and it does not satisfy PLAN step 8.
- Calibration artifacts/status: formal calibration is incomplete; run
  `b300-cp8-calibration-66d69258-20260713T054451Z` stopped at `57/120`.
  The sampled, non-formal calibration evidence is run
  `b300-cp8-sample-calibration-20260713T083426Z`, as recorded above.
- Formal performance run and gate summary: no `validation.json` was produced;
  the formal 20-pack validation was not run.
- Profile run and overlap summary: no profile/overlap JSON was produced because
  profiling was not run; the formal ratio-4 overlap gate remains unevaluated.
- Final-matrix cases: installed public-API smoke `8.629054 s`; CP1
  `100.860094 s`, JUnit `93/93`, zero failure/error/skip; CP8 native
  `550.507443 s`, JUnit `18/18`, zero failure/error/skip; isolated fail-stop
  `47.943240 s`.
- Isolated fail-stop: passed. Fault rank 3 exited rc `86`; all seven peers had
  launched native GroupCast, all workers were reclaimed in `3.064738 s`, and
  `survivors=[]`. Fresh health returned rc `0` on all 8/8 ranks, each using
  `GrpCollIntraHandle` with NVL `1073741824`, RDMA `0`, `num_rdma_ranks=1`,
  and `num_sms=20`.
- Sealed evidence root:
  `agents/perf/magi-dsa-v4-release/final-matrix-f8ad2e5d`. The inner and outer
  `artifact_manifest.sha256` file SHA256s are
  `fdc9a4cd0a96a5f69d554d6b89c898f6c83de1a26e81ab53de7e22285ae20c11`
  and `b045d7fb03db5e03f47c16c43a5dfde862e64545791bc3ad58c68eda47977d11`;
  `final_result.json`, `fault_result.json`, and the inspect file SHA256s are
  `bb83d46d7f1e0b2a2db4591b4cbe8f45dffafaf3b5adc8d5dc5a7859e5cf1687`,
  `77cfa8a1a937a8281bc590cd3c6b7fdf2098e68796adfe52c2133ef3c24783e1`,
  and `6420dcac04ca064a2bbe4d9ec239e52a9e550fbd154de2d0247273807ff59449`.
  Read-only verification passed all 80 artifacts. Container inspection records
  exit `0`, OOM false, exact image identity, one artifact-only mount, no source
  bind, and no `NVSHMEM_SYMMETRIC_SIZE`.
- Matrix-tested integration HEAD:
  `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`; corresponding DEV commit:
  `e8f50e97896fec4eee5646988e6797e9c5a8b76c`.
- Development/integration pre-documentation committed-tree equality: both are
  `38846f45dc5c10b30d0679f1f6decc252e78c1b2`. Final documentation commit IDs
  are recorded in the delivery response to avoid commit self-reference.
- Frozen recursive gitlinks:
  - `magi_attention/csrc/cutlass`:
    `81a43e6d92cdd8c20d22392f9579604ed5f710a1`
  - `magi_attention/functional/flash-attention`:
    `ee1d15159cda6f3f97bfab9e487da146a8254970`
  - `magi_attention/functional/flash-attention/csrc/composable_kernel`:
    `e8709c24f403173ad21a2da907d1347957e324fb`
  - `magi_attention/functional/flash-attention/csrc/cutlass`:
    `b1d6e2c9b334dfa811e4183dfbd02419249e4b52`
