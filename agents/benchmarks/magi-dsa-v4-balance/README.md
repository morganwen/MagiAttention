# Magi_DSA V4 B300 CP8 performance acceptance

This directory is the executable contract for PLAN step 8. It is intentionally
separate from `exps/dist_attn`: the generic benchmark reduces CUDA times to a
WORLD maximum too early and cannot preserve the per-rank evidence required by
the DSA balance gates.

## Frozen workload

- One node, one `world_size=cp_size=8` process group over physical GPU 0–7.
- Eight NVIDIA B300 SXM6 AC devices, compute capability 10.3, all-pairs NVLink.
- Global packed length 196608; target 24576 owner-local query rows per rank.
- BF16 input/kernel path and FP32 sink/KL; FlashMLA forward and cuDNN DSA
  backward/Indexer wrappers.
- Native GroupCast/GroupReduce only. The driver requires
  `MAGI_ATTENTION_NATIVE_GRPCOLL=1`, proves a `GrpCollIntraHandle`, and rejects
  A2AV or hierarchical fallback.
- `GrpCollConfig`: `num_sms=20`, `num_nvl_bytes=1073741824` per buffer,
  `num_rdma_bytes=0`; all five resident buffers are allocated before timing.
- `NVSHMEM_SYMMETRIC_SIZE` is unset and `CUDA_DEVICE_MAX_CONNECTIONS=8`.
- DeepSeek-V4-Flash reference
  `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`:
  `hidden_size=4096`, `q_lora_rank=1024`, and
  `softmax_scale=512**-0.5`. The fetched formal config SHA256 recorded in every
  run is
  `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`.
- The existing `DatasetSampler` and default real-data histogram are used with
  `seed=42`, `pack_num=20`, `chunk_ratio=0.25`. The histogram SHA256 is
  `67fe5f333fa3775ba547319e897b3781fda74ff5e19f3a38dc042a2f10414593`.
- One compile pass, two warm-up passes, then ten measured iterations per
  `(case, pack, rank)`. Input construction, barriers, correctness checks,
  synchronization and artifact writes are outside the CUDA-event window.
- `progress.json` is run-local, identity-bound evidence at case-pack
  granularity. Rank 0 creates it with zero completed units immediately after
  the fresh-directory claim, then atomically advances it only after an
  out-of-window barrier proves all eight ranks completed that case-pack.

The measure case matrix is fixed:

```text
ratio 4:   sequential:00, balanced:00, balanced:01,
           balanced:10, balanced:11
ratio 128: sequential:00, balanced:11
ratio 0:   sequential:00, balanced:11 (report only)
```

The two overlap bits are, in order, compressed GroupCast versus Indexer
projection and dKi GroupReduce versus sparse backward. `sequential:00` is the
baseline and `balanced:11` is the candidate. Frozen Indexer time is the sum of
`indexer_projection`, `indexer_topk`, `indexer_score_recompute`, and
`indexer_backward`; the validator recomputes that sum from phase records.

For each pack, the validator first takes the median of ten iterations on each
rank, then computes `max(rank medians)`, `mean(rank medians)`, and
`max/mean-1`. The final scalar is the mean of the 20 pack maxima. It never
computes an iteration-level WORLD maximum before retaining raw rank data.

## Calibration and measure revisions

Calibration and formal measurement are deliberately different revisions:

1. Run `driver.py calibrate` on a clean calibration revision. It writes the
   observations, plan features and `calibration.json`.
2. Run `freeze_calibration.py --input calibration.json`. Review and commit the
   generated `magi_attention/meta/solver/dsa_calibration.py`.
3. Build a new immutable native-enabled image from that clean commit.
4. Run `driver.py measure` with the same `packs.json`. Measure refuses a
   calibration target other than `b300-sm103`.

The frozen `CALIBRATION_ID` hashes the normalized ratio 0/4/128 coefficients.
It is recorded separately from `plan_hash`, because a logical plan hash does
not contain predictor coefficients.

## Immutable image and installed-wheel boundary

Distributed commands run only in the production image from step 9. They
require `/opt/magi-dsa-build-manifest.json`, `.magi-source-revision`, and the
exact archived submodule marker, re-hash every artifact named by the build
manifest, and reject a `magi_attention` import beneath `/opt/MagiAttention`.
The benchmark script is read from the archived source, but the library under
test must resolve from the installed wheel's `site-packages` directory. There
is no Git fallback for calibration, measure, or profile.

`run.sh` mounts only pack/performance/profile artifact directories; it never
bind-mounts the host checkout. It obtains the immutable image ID and revision
label with `docker image inspect`.

```bash
export MAGI_DSA_IMAGE=magi-dsa-v4-b300:<pinned-tag>
export RUN_ID=<unique-run-id>
./agents/benchmarks/magi-dsa-v4-balance/run.sh packs
./agents/benchmarks/magi-dsa-v4-balance/run.sh measure
./agents/benchmarks/magi-dsa-v4-balance/run.sh profile
./agents/benchmarks/magi-dsa-v4-balance/run.sh validate
```

Use a separate `RUN_ID` and a calibration image for `run.sh calibrate`, then
freeze/commit the coefficients and rebuild the measure image.

## Direct commands inside the immutable image

Generate the packs once:

```bash
python agents/benchmarks/magi-dsa-v4-balance/driver.py packs \
  --output /artifacts/packs.json
```

Every distributed command requires the exact clean Git revision and immutable
Docker image ID supplied by the host:

```bash
export MAGI_ATTENTION_NATIVE_GRPCOLL=1
export MAGI_ATTENTION_HIERARCHICAL_COMM=0
export CUDA_DEVICE_MAX_CONNECTIONS=8
unset NVSHMEM_SYMMETRIC_SIZE

torchrun --standalone --nproc_per_node=8 \
  agents/benchmarks/magi-dsa-v4-balance/driver.py calibrate \
  --run-id "$RUN_ID" --run-dir "$PERF_DIR" --packs /artifacts/packs.json \
  --expected-revision "$REVISION" --expected-image-id "$IMAGE_ID"

torchrun --standalone --nproc_per_node=8 \
  agents/benchmarks/magi-dsa-v4-balance/driver.py measure \
  --run-id "$RUN_ID" --run-dir "$PERF_DIR" --packs /artifacts/packs.json \
  --expected-revision "$REVISION" --expected-image-id "$IMAGE_ID"
```

The fixed profile replays pack 0 once for the `balanced:11` candidate:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  -o "$PROFILE_DIR/ratio4_pack0" \
  torchrun --standalone --nproc_per_node=8 \
    agents/benchmarks/magi-dsa-v4-balance/driver.py profile \
    --run-id "$RUN_ID" --run-dir "$PROFILE_DIR" --packs /artifacts/packs.json \
    --expected-revision "$REVISION" --expected-image-id "$IMAGE_ID"
```

`DsaTelemetry` reports current-stream phase durations and the two launch-to-wait
windows; those windows are not accepted as overlap proof. During final
validation, `validate.py` exports the `.nsys-rep` to SQLite, selects the
candidate's slowest rank, maps CUDA launches inside native GroupCast/Reduce and
phase NVTX ranges to kernels through CUPTI correlation IDs, and computes the
actual cross-stream GPU interval intersections. Both intersections must be
positive in `profile_overlap.json`.

Validate, recompute the summary, and freeze a cross-directory manifest:

```bash
python agents/benchmarks/magi-dsa-v4-balance/validate.py \
  --mode measure --run-dir "$PERF_DIR" --profile-dir "$PROFILE_DIR" \
  --expected-revision "$REVISION" --expected-image-id "$IMAGE_ID" \
  --write-summary --write-manifest --verify-manifest
```

Calibration must be followed by a reviewed commit and image rebuild, so the
wrapper intentionally does not combine calibration and measure into one
command.

## Preflight and failure policy

The driver fails rather than skipping when any prerequisite is absent. It
checks the archive revision and build-manifest artifact hashes, installed-wheel
import path, image ID, public API import, GPU names/capabilities/UUIDs, one-host
placement, `nvidia-smi topo -m`, native environment flags, all five
dry-allocated buffers, actual handle type, RDMA rank count, calibration ID, and
dependency/binary fingerprints. It records cuobjdump-proven FlashMLA `sm_100`
cubins. After compile it records every actual `dsa_pack` cache key and rejects
any key that does not contain architecture `(10, 3)`.

The launcher must wrap distributed commands in an external timeout. OOM,
non-finite output, missing records, a packing cache miss during measure,
correctness mismatch, native fallback, watchdog termination, or any worker
failure makes the run invalid. There is no `skip` mode.

The driver never resumes a claimed directory. `progress.json` binds the schema,
run ID, mode, source revision, immutable image ID, and calibration ID, and
contains canonical expected/completed case-pack unit lists plus both counts.
Validation of calibration, measure, and profile runs requires the completed
units to be unique and exactly equal to the mode's expected set (120, 180, and
1 units respectively). A partial progress file is useful fail-stop evidence,
but it is never accepted as a completed run; `progress.json` is also included
in fresh-run stale-artifact detection and the final SHA256 manifest.

## Correctness gate

Logical row values are generated deterministically from packed-global row IDs,
so sequential and non-contiguous balanced ownership receive identical data and
upstream gradients. Before timing each candidate pack, a fresh sequential
reference is run. Benchmark-only `all_to_all_single` redistributes each
policy's owner-local output, `dx`, `dqr`, `dQ`, and owner `dKV` into the fixed
contiguous 24576-global-row shard for each rank; chunked elementwise
`torch.testing.assert_close` then uses the formal CP tolerance `rtol=3e-2,
atol=3e-3`. The detached Indexer contract makes `dqr` absent in current formal
paths; correctness requires both policies to report `None`. Replicated
`d_sink` and every parameter gradient are compared directly and also checked
for CP replication. Moments/digests are never used for acceptance.

For ratio 4, merged dispatch fragments are subdivided into sample-relative
128-row canonical query tiles for Indexer projection, selection, forward KL and
backward KL recompute. The tiling is policy-invariant: changing ownership may
change communication and load balance, but cannot select a different GEMM
shape for the same logical query and perturb its top-k boundary.

The cuDNN top-k launch keeps the configured output width at 512 even when a
short sample has fewer compressed keys; unused entries remain `-1` and are
removed before semantic boundary handling. It must not specialize the kernel K
to an odd compressed-key count such as 473, because the frozen CuTe kernel
statically compiles a two-element vector-store path that requires an even K.

## Artifacts

`agents/perf/magi-dsa-v4-balance/<RUN_ID>` contains:

```text
environment.json
progress.json            # atomic case-pack completion evidence
packs.json
plans.jsonl
correctness.jsonl
raw_timing.jsonl
calibration.json       # calibration runs only
summary.json           # measure runs only
validation.json
report.schema.json
artifact_manifest.sha256
```

`agents/profiles/magi-dsa-v4-balance/<RUN_ID>` contains the profile-mode JSON
evidence, `profile_overlap.json`, and the `.nsys-rep`/SQLite exports. The final manifest in the
performance directory hashes both roots with stable `perf/` and `profile/`
relative names.

`validate.py` has a CUDA-free contract test:

```bash
python agents/benchmarks/magi-dsa-v4-balance/validate.py --self-test
```
