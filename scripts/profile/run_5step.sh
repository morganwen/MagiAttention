#!/usr/bin/env bash
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

set -euo pipefail

usage() {
    echo "Usage: $0 --world-size 8 --cp-size 8 --case dsv4-flash-128k --plans sequential,balanced --steps 5 [--skip-smoke] [--image IMAGE]" >&2
}

world_size=""
cp_size=""
case_name=""
plans=""
steps=""
skip_smoke=0
image="${MAGI_DSA_IMAGE:-magi-dsa-v4:preflight-68c2f15}"
while (($# > 0)); do
    case "$1" in
        --world-size)
            world_size="${2:-}"
            shift 2
            ;;
        --cp-size)
            cp_size="${2:-}"
            shift 2
            ;;
        --case)
            case_name="${2:-}"
            shift 2
            ;;
        --plans)
            plans="${2:-}"
            shift 2
            ;;
        --steps)
            steps="${2:-}"
            shift 2
            ;;
        --skip-smoke)
            skip_smoke=1
            shift
            ;;
        --image)
            image="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ "$world_size" != "8" || "$cp_size" != "8" || "$case_name" != "dsv4-flash-128k" ]]; then
    usage
    exit 2
fi
if [[ "$plans" != "sequential,balanced" || "$steps" != "5" ]]; then
    usage
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
pack_sha="$(sha256sum "$repo_root/magi_attention/kernel/cutedsl/dsa_pack.py" | cut -c1-16)"
aot_dir="${MAGI_DSA_CUTE_AOT:-$repo_root/.cache/magi-dsa-v4/aot-cutlass-4.5.0-sm103-$pack_sha}"
cudnn_cache_dir="${MAGI_DSA_CUDNN_CACHE:-$repo_root/.cache/magi-dsa-v4/cudnn-dsa-9.24.0.43-frontend-35fd7b0d-cutlass-4.5.0-sm103}"
aot_object_count="$(find "$aot_dir" -maxdepth 1 -type f -name '*.o' 2>/dev/null | wc -l)"
if ((aot_object_count != 11)); then
    echo "missing the 11 frozen CuTe AOT objects; run bash scripts/test/prewarm_cute.sh" >&2
    exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
non_b300_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | rg -v '^NVIDIA B300([[:space:]]|$)' | wc -l || true)"
if ((gpu_count != 8 || non_b300_count != 0)); then
    echo "the release profile requires exactly 8 NVIDIA B300 GPUs" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')" ]]; then
    echo "one or more GPUs already have compute processes; refusing to contaminate the CP8 profile" >&2
    exit 1
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-dsv4-flash-128k"
artifact_dir="$repo_root/artifacts/profile/$run_id"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite profile artifact directory: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir/smoke" "$artifact_dir/sequential" "$artifact_dir/balanced" \
    "$cudnn_cache_dir/cuda" "$cudnn_cache_dir/cute-dsl" "$cudnn_cache_dir/magi-workspace"
chmod 0777 "$artifact_dir" "$artifact_dir/smoke" "$artifact_dir/sequential" \
    "$artifact_dir/balanced" "$cudnn_cache_dir" "$cudnn_cache_dir/cuda" \
    "$cudnn_cache_dir/cute-dsl" "$cudnn_cache_dir/magi-workspace"

current_container=""
current_runner_pid=""
current_runner_pgid=""

terminate_current_stage() {
    if [[ -n "$current_runner_pid" ]] && kill -0 "$current_runner_pid" 2>/dev/null; then
        kill -TERM -- "-$current_runner_pgid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$current_runner_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$current_runner_pid" 2>/dev/null; then
            kill -KILL -- "-$current_runner_pgid" 2>/dev/null || true
        fi
    fi
    if [[ -n "$current_container" ]] && docker container inspect "$current_container" >/dev/null 2>&1; then
        docker container stop --time 5 "$current_container" >/dev/null 2>&1 || true
        docker container rm --force "$current_container" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    terminate_current_stage
}
trap cleanup EXIT INT TERM

{
    command_line="bash scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-flash-128k --plans sequential,balanced --steps 5"
    if ((skip_smoke == 1)); then
        command_line+=" --skip-smoke"
    fi
    command_line+=" --image $image"
    echo "$command_line"
    echo "run_id=$run_id"
    echo "artifact_dir=$artifact_dir"
    echo "aot_dir=$aot_dir"
    echo "cudnn_cache_dir=$cudnn_cache_dir"
    echo "smoke_deadline_seconds=600"
    echo "profile_deadline_seconds_per_plan=1800"
    echo "no_progress_watchdog_seconds=60"
    echo "watchdog_kill_sequence=TERM,wait-5s,KILL"
    echo "capture_order=balanced,sequential"
    echo "skip_smoke=$skip_smoke"
} >"$artifact_dir/COMMAND.txt"

cat >"$artifact_dir/MSA_REFERENCE.txt" <<'EOF'
balanced_report=/home/scratch.wewen_gpu/Magi-MSA/balanced_5steps(1).nsys-rep
balanced_report_sha256=01a429b2c6d42532a99c39ba5e7a2e08ee594df56f20a5413eac474a882a288f
reference_report_nsys_version=2026.3.1.157
frozen_runtime_nsys_version=2026.2.1.210
reference_nvtx_outer=$Magi_MSA/capture_five_training_steps
reference_nvtx_step=balanced/rank_<rank>/training_step_<step>
reference_nvtx_output=balanced/rank_<rank>/O
reference_nvtx_indexer=Magi_MSA/indexer
dsa_nvtx_outer=$Magi_DSA/capture_five_training_steps
dsa_nvtx_step=<plan>/rank_<rank>/training_step_<step>
dsa_nvtx_output=<plan>/rank_<rank>/O
dsa_nvtx_indexer=Magi_DSA/indexer
dsa_logical_phases=magi_dsa::indexer_score,magi_dsa::indexer_topk
EOF

git rev-parse HEAD >"$artifact_dir/SOURCE_REVISION.txt"
git remote get-url origin >"$artifact_dir/SOURCE_REMOTE.txt"
git submodule status --recursive >"$artifact_dir/SUBMODULES.txt"
git status --short >"$artifact_dir/DIRTY_STATUS.txt"
docker image inspect "$image" >"$artifact_dir/IMAGE.json"
nvidia-smi -q >"$artifact_dir/HARDWARE.txt"
timeout --signal=TERM --kill-after=5s 60s docker run --rm --entrypoint bash "$image" -lc \
    'python -VV; python -m pip freeze; nsys --version' \
    >"$artifact_dir/ENVIRONMENT.txt" 2>&1
smoke_mode="executed"
if ((skip_smoke == 1)); then
    smoke_mode="skipped_by_user"
fi
cat >"$artifact_dir/WORKLOAD.json" <<EOF
{
  "case": "dsv4-flash-128k",
  "cp_size": 8,
  "cu_seqlens": [0, 131072],
  "dtype": "BF16",
  "capture_order": ["balanced", "sequential"],
  "plans": ["sequential", "balanced"],
  "rank_size": 8,
  "ratio": 4,
  "seed": 0,
  "smoke": "$smoke_mode",
  "steps": 5,
  "warmup_steps_per_plan": 3,
  "world_size": 8
}
EOF

run_stage() {
    local stage="$1"
    local deadline_seconds="$2"
    local stage_dir="$3"
    shift 3
    local stage_log="$artifact_dir/${stage}_stdout_stderr.log"
    current_container="magi-dsa-profile-${run_id}-${stage}-$$"
    echo "stage=$stage container=$current_container deadline=${deadline_seconds}s" \
        | tee -a "$artifact_dir/WATCHDOG.log"
    setsid timeout --signal=TERM --kill-after=5s "${deadline_seconds}s" docker run \
        --rm \
        --name "$current_container" \
        --gpus all \
        --ipc=host \
        --stop-timeout=5 \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        "$@" \
        >"$stage_log" 2>&1 &
    current_runner_pid="$!"
    current_runner_pgid="$(ps -o pgid= -p "$current_runner_pid" | tr -d ' ')"
    local started_at
    local last_status_at
    local last_progress_at
    local last_signature
    started_at="$(date +%s)"
    last_status_at="$started_at"
    last_progress_at="$started_at"
    last_signature=""
    while kill -0 "$current_runner_pid" 2>/dev/null; do
        local now signature
        now="$(date +%s)"
        signature="$({
            find "$stage_dir" -type f -printf '%T@:%s:%p\n' 2>/dev/null
            if [[ -f "$stage_log" ]]; then
                stat -c '%Y:%s:%n' "$stage_log"
            fi
        } | sort | sha256sum | cut -d' ' -f1)"
        if [[ "$signature" != "$last_signature" ]]; then
            last_signature="$signature"
            last_progress_at="$now"
        fi
        if ((now - last_progress_at >= 60)); then
            echo "stage=$stage no progress for 60s; terminating pid=$current_runner_pid pgid=$current_runner_pgid container=$current_container" \
                | tee -a "$artifact_dir/WATCHDOG.log"
            terminate_current_stage
            return 124
        fi
        if ((now - last_status_at >= 30)); then
            echo "stage=$stage pid=$current_runner_pid pgid=$current_runner_pgid elapsed=$((now - started_at))s remaining=$((deadline_seconds - now + started_at))s last_progress_age=$((now - last_progress_at))s" \
                | tee -a "$artifact_dir/WATCHDOG.log"
            tail -n 3 "$stage_log" || true
            last_status_at="$now"
        fi
        sleep 5
    done
    set +e
    wait "$current_runner_pid"
    local status=$?
    set -e
    current_runner_pid=""
    current_runner_pgid=""
    if docker container inspect "$current_container" >/dev/null 2>&1; then
        echo "recorded task container survived stage exit: $current_container" \
            | tee -a "$artifact_dir/WATCHDOG.log"
        docker container stop --time 5 "$current_container" >/dev/null 2>&1 || true
        docker container rm --force "$current_container" >/dev/null 2>&1 || true
        status=1
    fi
    echo "stage=$stage exit_status=$status" | tee -a "$artifact_dir/WATCHDOG.log"
    current_container=""
    return "$status"
}

common_docker_args=(
    --env CUDA_CACHE_PATH=/cudnn-dsa-cache/cuda
    --env CUTE_DSL_CACHE_DIR=/cudnn-dsa-cache/cute-dsl
    --env MAGI_ATTENTION_WORKSPACE_BASE=/cudnn-dsa-cache/magi-workspace
    --env MAGI_DSA_CUTE_AOT_DIR=/dsa-pack-aot
    --env MAGI_DSA_CUTE_AOT_REQUIRED=1
    --env MAGI_DSA_PHASE_LOG=0
    --env NCCL_DEBUG=WARN
    --env PYTHONPATH=/workspace/Magi-DSA
    --env PYTHONUNBUFFERED=1
    --env TORCH_DISTRIBUTED_DEBUG=INFO
    --volume "$aot_dir:/dsa-pack-aot:ro"
    --volume "$artifact_dir:/profile-artifact"
    --volume "$cudnn_cache_dir:/cudnn-dsa-cache"
    --volume "$repo_root:/workspace/Magi-DSA:ro"
    --workdir /workspace/Magi-DSA
)

if ((skip_smoke == 0)); then
    run_stage \
        smoke \
        600 \
        "$artifact_dir/smoke" \
        "${common_docker_args[@]}" \
        --entrypoint /usr/local/bin/torchrun \
        "$image" \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=8 \
        benchmarks/dsa_v4/profile_5step.py \
        --mode smoke \
        --artifact-dir /profile-artifact/smoke \
        --seed 0 \
        --tokens 4096 \
        --steps 1 \
        --warmup 1
else
    echo "stage=smoke skipped_by_user=1" | tee -a "$artifact_dir/WATCHDOG.log"
fi

for plan in balanced sequential; do
    run_stage \
        "$plan" \
        1800 \
        "$artifact_dir/$plan" \
        "${common_docker_args[@]}" \
        --entrypoint bash \
        "$image" \
        /workspace/Magi-DSA/scripts/profile/run_plan.sh \
        --plan "$plan" \
        --artifact-dir "/profile-artifact/$plan" \
        --world-size 8 \
        --steps 5 \
        --tokens 131072 \
        --warmup 3
done

set +e
timeout --signal=TERM --kill-after=5s 600s python3 \
    "$repo_root/scripts/profile/summarize_5step.py" \
    --artifact-dir "$artifact_dir" \
    --world-size 8 \
    --steps 5 \
    >"$artifact_dir/SUMMARIZE.stdout" 2>"$artifact_dir/SUMMARIZE.stderr"
summary_status=$?
set -e

timeout --signal=TERM --kill-after=5s 600s python3 - "$artifact_dir" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest = root / "SHA256SUMS"
entries = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest:
        continue
    if path.is_symlink():
        raise SystemExit(f"refusing symlink in profile artifact: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    entries.append(f"{digest}  {path.relative_to(root)}")
manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
PY
timeout --signal=TERM --kill-after=5s 600s bash -c 'cd "$1" && sha256sum -c SHA256SUMS' \
    _ "$artifact_dir"

if ((summary_status != 0)); then
    echo "profile validation failed; preserved artifact: $artifact_dir" >&2
    exit "$summary_status"
fi
echo "profile complete: $artifact_dir"
