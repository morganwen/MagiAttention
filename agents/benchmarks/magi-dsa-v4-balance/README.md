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

Distributed commands run only in clean immutable images produced by the
production-image build path. Calibration uses the image built from its clean
calibration revision; after the coefficients are frozen and committed,
measure, profile, and the release matrix use the rebuilt final image. Both
images require `/opt/magi-dsa-build-manifest.json`, `.magi-source-revision`,
and the exact archived submodule marker, re-hash every artifact named by the
build manifest, and reject a `magi_attention` import beneath
`/opt/MagiAttention`. The benchmark script is read from the archived source,
but the library under test must resolve from the installed wheel's
`site-packages` directory. There is no Git fallback for calibration, measure,
or profile.

`run.sh` mounts only pack/performance/profile artifact directories; it never
bind-mounts the host checkout. It obtains the immutable image ID and revision
label with `docker image inspect`.

```bash
export MAGI_DSA_IMAGE="magi-dsa-v4-b300:<pinned-tag>"
export RUN_ID="<unique-run-id>"
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

## Time-boxed sampled closure (non-formal)

The `sample` subcommand is a diagnostic escape hatch for the explicitly
time-boxed 2026-07-13 repository closure. It does not change any case in
`CALIBRATION_CASES`, `MEASURE_CASES`, or `PROFILE_CASES`; it accepts an
explicit subset and always writes `formal=false`, `formal_acceptance=false`,
and `formal_gates_evaluated=false`. Its output is never accepted by the formal
validator and cannot satisfy PLAN step 8.

The stopped formal attempt
`b300-cp8-calibration-66d69258-20260713T054451Z` reached 57/120 synchronized
case-pack units: both ratio-0 calibration cases for all 20 packs, followed by
`r4-sequential-00` for packs 0 through 16. The container received an external
`SIGTERM` at `2026-07-13T07:45:12Z`; it was not OOM-killed, and this was not a
fail-stop test. It is preserved only as stopped-run progress and correctness
diagnostic evidence. It must not be resumed, fitted as a formal calibration,
or presented as a completed run. Its
`agents/perf/magi-dsa-v4-balance/b300-cp8-calibration-66d69258-20260713T054451Z/progress.json`
file has SHA256
`6f5520099807c290452b1a68da479e95240dbb1c9dfb19e54cc8908c025c528f`.
Under `agents/perf/magi-dsa-v4-release/calibration-logs`, the corresponding
`magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.docker-events.jsonl`
and `magi-dsa-calibration-66d69258-20260713t054451z-sampled-closeout.termination.json`
evidence files have SHA256s
`308245df4cdf2cf8db521960fb6ab5f1a3b9cab7e4823909ca20f3eb4ee81ce7` and
`927974b501330e4339a9e2793b5236697eb92d46294dc8733d0e1bd148c50870`.

The sampled calibration selects packs 0 and 1 and exactly the six frozen
calibration cases (12 case-pack units):

```bash
torchrun --standalone --nproc_per_node=8 \
  agents/benchmarks/magi-dsa-v4-balance/driver.py sample \
  --run-id "$RUN_ID" --run-dir "$PERF_DIR" --packs "$PACKS_FILE" \
  --expected-revision "$REVISION" --expected-image-id "$IMAGE_ID" \
  --case-id r0-sequential-00 --case-id r0-balanced-00 \
  --case-id r4-sequential-00 --case-id r4-balanced-00 \
  --case-id r128-sequential-00 --case-id r128-balanced-00 \
  --pack-index 0 --pack-index 1 --fit-calibration

python agents/benchmarks/magi-dsa-v4-balance/freeze_calibration.py \
  --input "$PERF_DIR/sample_calibration.json" \
  --allow-sampled-non-formal
```

`sample_calibration.json` is deliberately labeled with target
`sampled-b300-sm103`, scope `sampled_non_formal`, and the selected case/pack
identities. A final image built with those coefficients must continue using
`sample`; formal `measure` and `profile` reject a sampled target.

The completed non-formal calibration record is run
`b300-cp8-sample-calibration-20260713T083426Z`, revision
`66d69258fc9df1ee96f4c91df5d8225173b7fae2`, image
`sha256:81f635bd1bcadf0eb159d75621744ab32fa978cf7961a9a6e07021791b27aba2`,
and progress 12/12. It contains 12 correctness, 96 plan, and 960 raw-timing
records, with target `sampled-b300-sm103`, scope `sampled_non_formal`, and
`formal=false`. The frozen coefficient ID is
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
design rank 6/7), so those individual coefficients are not identifiable. The
fit is suitable only for this sampled diagnostic target, not a formal model
quality claim.

The clean matrix-only-fix revision
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9` produced the current
sampled-closeout candidate package `1.1.1+dsa.f8ad2e5d4f8b` and immutable image
`sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`.
The build evidence is in
`agents/perf/magi-dsa-v4-release/sampled-final-f8ad2e5d-20260713T104133Z`.
The embedded `/opt/magi-dsa-build-manifest.json` SHA256 is
`c07f71865a118786e43a54d3c980b14ffaa2db0484ec5e8094ed671ae57e7519`.
The SHA256s of the evidence files `artifact_manifest.sha256`, `image_id.txt`,
`package_version.txt`, `installed-wheel-smoke.log`, `image_inspect.json`, and
`docker-build.log` are, respectively,
`fa69dad9035c326cea00f6dc8ec7865ec3ea9f8bc568cf6a54a64fc7dfd13eff`,
`27f7f3362179b16d6faf7541b55db5f86bce8506cab0af8164293b2a29b55985`,
`965323aa062b7753d197ccf7b0920ad546316c6da7182a5c59224f8976768cce`,
`eebb12ff3aa2518e74cbc3b9dad5cb04e0b3a2d75877228758eaf6bc534951c5`,
`7417ec701e0b6f592be477df8269dbbf58ce9c7204c0c6ea8a4d778177f1c674`,
and `15d5a5be9c7b496b930fc02b236496e840f3fed01f1dfbfc6e4efc3635fc02a8`.
The installed-wheel smoke passed. This image is only a candidate for the
sampled-closeout final matrix; it is not a formal release image and does not
satisfy PLAN step 8. Runtime, kernel, and solver behavior is unchanged from the
f88 image, so the sealed nine-unit sampled timing remains tied to f88 and is
not rerun.

The first step-9 run, `b300-cp8-final-f88ca22d-20260713T103532Z`, used the old
f88 sampled-timing candidate image. Public smoke passed in `8.679 s`, CP1
passed `93/93` tests in `99.657 s`,
and CP8 reported `4 passed, 14 failed` in `7.728 s` before the ordered run
stopped. The cause was a spawned subprocess launched from `/tmp` under
pytest's importlib mode failing with `ModuleNotFoundError: tests`, not DSA
correctness, OOM, or watchdog failure. With `--import-mode=append`, the focused
native 8-rank diagnostic passed (`1 passed` in `36.61 s`) while
`magi_attention` continued to resolve from the installed wheel's
`site-packages`. The failed-run and outer manifest-file SHA256s are
`81f8f21a524f4d2dd050ca591358234b133f222de8e03523c75b083caf44757d` and
`0818fe00f144b63d7718e8d38f3a775a91da4acef824419b639491b3a59107b3`.
The fix commits are DEV `e8f50e97896fec4eee5646988e6797e9c5a8b76c` and integration
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`; the rebuilt final step-9
candidate is the sealed f8 image above. The f88 image is retained only as the
sampled-timing and failed-matrix predecessor.

The rebuilt f8 candidate completed the sampled-closure delivery matrix as run
`b300-cp8-final-f8ad2e5d-20260713T110210Z`, with `status=passed` and
`error=null`. It used revision
`f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`, package
`1.1.1+dsa.f8ad2e5d4f8b`, and image
`sha256:12075f8738456a417cac72f39a328378bb1bb1ee313c5fd7c34c92c6d0bf5c37`.
Installed public-API smoke took `8.629054 s`; CP1 took `100.860094 s` with
JUnit `93/93` and zero failure/error/skip; CP8 native took `550.507443 s` with
JUnit `18/18` and zero failure/error/skip; the isolated fail-stop case took
`47.943240 s`. Fault rank 3 exited with rc `86` after all seven peers launched
native GroupCast; all workers were reclaimed in `3.064738 s` with
`survivors=[]`. A fresh process group then returned rc `0` on all 8/8 ranks,
all using `GrpCollIntraHandle` with NVL `1073741824`, RDMA `0`,
`num_rdma_ranks=1`, and `num_sms=20`.

Container inspection records exit `0`, OOM false, the exact image, one
artifact-only mount, no source bind, and no `NVSHMEM_SYMMETRIC_SIZE`. Evidence
is rooted at `agents/perf/magi-dsa-v4-release/final-matrix-f8ad2e5d`. The
inner and outer `artifact_manifest.sha256` file SHA256s are
`fdc9a4cd0a96a5f69d554d6b89c898f6c83de1a26e81ab53de7e22285ae20c11` and
`b045d7fb03db5e03f47c16c43a5dfde862e64545791bc3ad58c68eda47977d11`;
`final_result.json`, `fault_result.json`, and the inspect file have SHA256s
`bb83d46d7f1e0b2a2db4591b4cbe8f45dffafaf3b5adc8d5dc5a7859e5cf1687`,
`77cfa8a1a937a8281bc590cd3c6b7fdf2098e68796adfe52c2133ef3c24783e1`,
and `6420dcac04ca064a2bbe4d9ec239e52a9e550fbd154de2d0247273807ff59449`.
Read-only manifest verification passed all 80 artifacts. The matrix-tested
integration HEAD is `f8ad2e5d4f8bfa1b8f80ffdc02411379e20930f9`, paired with DEV
commit `e8f50e97896fec4eee5646988e6797e9c5a8b76c`; both had pre-documentation
committed tree `38846f45dc5c10b30d0679f1f6decc252e78c1b2`. Final documentation
commit IDs are reported at delivery time to avoid self-reference. This
completes step 9 only under the sampled closure: formal step 8 remains
incomplete, the sampled ratio-4 direction failed, and this is not formal
release qualification.

The final-image performance diagnostic is nine case-pack units split across
fresh run directories so that the selection is not accidentally expanded as
a Cartesian product:

- Pack 0: ratio 4 `sequential:00` and balanced `00/01/10/11`, plus ratio 128
  `sequential:00` and `balanced:11` (seven units).
- Pack 16: ratio 4 `sequential:00` and `balanced:11` (two units).

Each unit retains the formal timing mechanics (one compile, two warm-up, ten
measured iterations) and the elementwise forward/all-gradient correctness
comparison. The resulting observations are diagnostics, not gates. Keep the
pack sets distinct: the stopped attempt covers all 20 packs for ratio 0 and
packs 0 through 16 for ratio-4 sequential; sampled calibration selects packs
0 and 1; final-image sampled measurement selects packs 0 and 16. Combining
those records does not create a new sampled pack set and cannot prove the
20/20 candidate imbalance or speed requirements.

The actual f88 sampled timing completed all 9/9 correctness units and produced
72 plan records and 720 raw timing records. Source runs
`b300-cp8-sample-measure-a-f88ca22d-20260713T093515Z`,
`b300-cp8-sample-measure-b-f88ca22d-20260713T101533Z`, and
`b300-cp8-sample-measure-c-f88ca22d-20260713T103247Z` have
`artifact_manifest.sha256` file SHA256s
`8ec095b475228012a2029a49066595692ae7c431fb12fe0a33981c952a7948f5`,
`a508503306049cc5460640b6babdb1ef69be2a942ea3bd210ba5c37e7013741e`,
and `7423cfc56ad63ee8f2f9f6c4716eff7f9f6164ca7890e9e8a76f8c19dcf6d64c`.
The aggregate is
`agents/perf/magi-dsa-v4-balance-sampled/b300-cp8-sample-measure-f88ca22d-20260713T103500Z-aggregate`;
its `sampled_measure_diagnostic.json` SHA256 is
`47fb9e1ab4e852be90046c03438ac29ef8c29c552193cfe68e9ff36e262cded5`,
and its `artifact_manifest.sha256` file SHA256 is
`955e17348d4ab00c11a582be2c2dfb4b7bcb6abe79bd9d8eadb6d89dfc93a62b`.
The artifact records `formal=false`, `formal_acceptance=false`,
`formal_gates_evaluated=false`, and `all_gates_pass=null`. Across ratio-4
packs 0 and 16, baseline/candidate E2E is `28602.9863/29245.3057 ms`
(`+2.245637%`, candidate slower) and Indexer is `474.7959/493.6558 ms`
(`+3.972199%`, candidate slower), so
`sampled_performance_observation_pass=false`; both candidate packs remain
within 5% E2E rank imbalance. Ratio 128 pack 0 is
`321.7919/320.7076 ms` (`-0.336954%`, candidate slightly faster), with
`0.10824%` candidate rank imbalance. These measurements remain tied to the old
f88 image; the matrix-only f8 rebuild does not replace or relabel them.

No sampled/diagnostic profile was run or produced for this closure, so the
formal ratio-4 Nsight overlap gate remains unevaluated.

The initial sampled calibration may record a read-only bind of the committed
`driver.py` because its calibration image predates the `sample` subcommand.
That exception is diagnostic-only: the tested `magi_attention` package must
still resolve from the installed wheel, and any driver bind makes the run
ineligible for formal acceptance. Formal calibration/measure/profile retain
the no-source-bind rule above.

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
