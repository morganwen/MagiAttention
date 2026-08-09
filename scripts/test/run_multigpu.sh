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
    echo "Usage: $0 --world-size {2|8} --case CASE [--image IMAGE] [--installed-wheel]" >&2
}

world_size=""
case_name=""
image="${MAGI_DSA_IMAGE:-magi-dsa-v4:preflight-68c2f15}"
nccl_debug="${NCCL_DEBUG:-WARN}"
torch_distributed_debug="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
installed_wheel=0
while (($# > 0)); do
    case "$1" in
        --world-size)
            world_size="${2:-}"
            shift 2
            ;;
        --case)
            case_name="${2:-}"
            shift 2
            ;;
        --image)
            image="${2:-}"
            shift 2
            ;;
        --installed-wheel)
            installed_wheel=1
            shift
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done
case "$case_name" in
    smoke|csa-natural|csa-natural-backward)
        if [[ "$world_size" != "2" ]]; then
            usage
            exit 2
        fi
        ;;
    cp8-natural-backward|cp8-topk-diagnostic)
        if [[ "$world_size" != "8" ]]; then
            usage
            exit 2
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
if [[ -z "$case_name" ]]; then
    usage
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
flashmla_base_revision="9241ae3ef9bac614dd25e45e507e089f888280e0"
flashmla_patch_revision="13d173ac48abd8ec88a4e742bcaaa59c5ccf4ece"
flashmla_patch_sha256="6957dbde516c73066c5911108761325edc1bdcd8f62e15dc0a84f4f290118d4b"
flashmla_pro_patch_revision="b7643bd54521f563b839b98289b5cd048c062ba2"
flashmla_pro_patch_sha256="c534e13ff432ac1c694cb24981826c11be26a2d9743d7175ddb05f887279461f"
cudnn_backend_version="9.24.0.43"
cudnn_frontend_version="1.26.0"
cudnn_frontend_revision="35fd7b0d0e1d4952b904c79341c5e84e3af0a328"
cutlass_dsl_version="4.5.0"
quack_version="0.4.1"
tvm_ffi_version="0.1.8.post0"
magi_source_revision="$(git -C "$repo_root" rev-parse HEAD)"

image_label() {
    docker image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"
}

require_image_label() {
    local label="$1"
    local expected="$2"
    local description="$3"
    local actual
    actual="$(image_label "$label")"
    if [[ "$actual" != "$expected" ]]; then
        echo "correctness image $description label mismatch: expected $expected, found ${actual:-<missing>}" >&2
        exit 1
    fi
}

require_image_label "org.magi-dsa.flashmla-revision" "$flashmla_base_revision" \
    "FlashMLA base revision"
require_image_label "org.magi-dsa.flashmla-dual-lse-patch-revision" \
    "$flashmla_patch_revision" "FlashMLA dual-LSE patch revision"
require_image_label "org.magi-dsa.flashmla-dual-lse-patch-sha256" \
    "$flashmla_patch_sha256" "FlashMLA dual-LSE patch SHA-256"
require_image_label "org.magi-dsa.flashmla-pro-h128-patch-revision" \
    "$flashmla_pro_patch_revision" "FlashMLA Pro H128 patch revision"
require_image_label "org.magi-dsa.flashmla-pro-h128-patch-sha256" \
    "$flashmla_pro_patch_sha256" "FlashMLA Pro H128 patch SHA-256"
require_image_label "org.magi-dsa.cudnn-backend" "$cudnn_backend_version" \
    "cuDNN backend version"
require_image_label "org.magi-dsa.cudnn-frontend" "$cudnn_frontend_version" \
    "cuDNN frontend version"
require_image_label "org.magi-dsa.cudnn-frontend-revision" \
    "$cudnn_frontend_revision" "cuDNN frontend revision"
require_image_label "org.magi-dsa.cudnn-frontend-source" \
    "official-unmodified" "cuDNN frontend source"
require_image_label "org.magi-dsa.cudnn-frontend-local-patches" \
    "none" "cuDNN frontend local patches"
require_image_label "org.magi-dsa.cutlass-dsl" "$cutlass_dsl_version" \
    "CUTLASS DSL version"
require_image_label "org.magi-dsa.quack-kernels" "$quack_version" \
    "Quack version"
require_image_label "org.magi-dsa.tvm-ffi" "$tvm_ffi_version" \
    "TVM-FFI version"
require_image_label "org.magi-dsa.magi-attention-revision" \
    "$magi_source_revision" "Magi source revision"
require_image_label "org.magi-dsa.install-mode" "python-wheel" \
    "Magi install mode"

image_flashmla_patch_revision="$(image_label \
    "org.magi-dsa.flashmla-dual-lse-patch-revision")"
image_flashmla_patch_sha256="$(image_label \
    "org.magi-dsa.flashmla-dual-lse-patch-sha256")"
image_flashmla_pro_patch_revision="$(image_label \
    "org.magi-dsa.flashmla-pro-h128-patch-revision")"
image_flashmla_pro_patch_sha256="$(image_label \
    "org.magi-dsa.flashmla-pro-h128-patch-sha256")"
cudnn_cache_dir="${MAGI_DSA_CUDNN_CACHE:-$repo_root/.cache/magi-dsa-v4/cudnn-dsa-9.24.0.43-frontend-35fd7b0d-cutlass-4.5.0-sm103}"
run_scope="cp8"
if [[ "$world_size" == "2" ]]; then
    run_scope="cp2"
fi
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$run_scope-$case_name"
if ((installed_wheel == 1)); then
    run_id="${run_id}-installed-wheel"
fi
rank_cache_dir="${MAGI_DSA_RANK_CACHE:-$repo_root/.cache/magi-dsa-v4/distributed-rank-cache/$run_id}"
artifact_dir="$repo_root/artifacts/correctness/$run_id"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite existing artifact directory: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir/workdir" "$cudnn_cache_dir/cuda" \
    "$cudnn_cache_dir/cute-dsl" "$cudnn_cache_dir/magi-workspace" \
    "$rank_cache_dir/launcher/quack" "$rank_cache_dir/launcher/tmp" \
    "$rank_cache_dir/launcher/torch-extensions" \
    "$rank_cache_dir/launcher/torch-home" \
    "$rank_cache_dir/launcher/torchinductor" \
    "$rank_cache_dir/launcher/triton" "$rank_cache_dir/launcher/xdg" \
    "$rank_cache_dir/workers"
chmod 0777 "$artifact_dir" "$cudnn_cache_dir" \
    "$cudnn_cache_dir/cuda" "$cudnn_cache_dir/cute-dsl"
chmod 0777 "$cudnn_cache_dir/magi-workspace"
chmod 0777 "$artifact_dir/workdir" "$rank_cache_dir" \
    "$rank_cache_dir/launcher" "$rank_cache_dir/launcher/quack" \
    "$rank_cache_dir/launcher/tmp" "$rank_cache_dir/launcher/torch-extensions" \
    "$rank_cache_dir/launcher/torch-home" \
    "$rank_cache_dir/launcher/torchinductor" \
    "$rank_cache_dir/launcher/triton" "$rank_cache_dir/launcher/xdg" \
    "$rank_cache_dir/workers"
container_name="magi-dsa-$run_scope-$case_name-$$"
runner_pid=""
runner_pgid=""
log_tail_pid=""

cleanup() {
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        docker container stop --time 5 "$container_name" >/dev/null 2>&1 || true
        docker container rm --force "$container_name" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "artifact_dir=$artifact_dir" | tee "$artifact_dir/COMMAND.txt"
echo "cudnn_cache_dir=$cudnn_cache_dir" | tee -a "$artifact_dir/COMMAND.txt"
echo "rank_cache_dir=$rank_cache_dir" | tee -a "$artifact_dir/COMMAND.txt"
echo "rank_cache_policy=per-run-per-rank" | tee -a "$artifact_dir/COMMAND.txt"
echo "image=$image" | tee -a "$artifact_dir/COMMAND.txt"
echo "flashmla_base_revision=$flashmla_base_revision" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "flashmla_dual_lse_patch_revision=$flashmla_patch_revision" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "flashmla_dual_lse_patch_sha256=$flashmla_patch_sha256" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "flashmla_pro_h128_patch_revision=$flashmla_pro_patch_revision" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "flashmla_pro_h128_patch_sha256=$flashmla_pro_patch_sha256" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cudnn_backend_version=$cudnn_backend_version" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cudnn_frontend_version=$cudnn_frontend_version" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cudnn_frontend_revision=$cudnn_frontend_revision" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cudnn_frontend_source=official-unmodified" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cudnn_frontend_local_patches=none" \
    | tee -a "$artifact_dir/COMMAND.txt"
echo "cutlass_dsl_version=$cutlass_dsl_version" | tee -a "$artifact_dir/COMMAND.txt"
echo "quack_version=$quack_version" | tee -a "$artifact_dir/COMMAND.txt"
echo "tvm_ffi_version=$tvm_ffi_version" | tee -a "$artifact_dir/COMMAND.txt"
echo "magi_source_revision=$magi_source_revision" | tee -a "$artifact_dir/COMMAND.txt"
echo "installed_wheel=$installed_wheel" | tee -a "$artifact_dir/COMMAND.txt"
echo "NCCL_DEBUG=$nccl_debug" | tee -a "$artifact_dir/COMMAND.txt"
echo "TORCH_DISTRIBUTED_DEBUG=$torch_distributed_debug" | tee -a "$artifact_dir/COMMAND.txt"
echo "torchrun --standalone --nproc-per-node=$world_size extensions/tests/dsa_v4/distributed_worker.py --case $case_name" \
    | tee -a "$artifact_dir/COMMAND.txt"

total_deadline_seconds=60
if [[ "$case_name" == "csa-natural-backward" || "$case_name" == "cp8-natural-backward" ]]; then
    total_deadline_seconds=1800
elif [[ "$case_name" == "cp8-topk-diagnostic" ]]; then
    total_deadline_seconds=600
fi
actual_deadline_seconds=60
verification_deadline_seconds=600
echo "total_deadline_seconds=$total_deadline_seconds" | tee -a "$artifact_dir/COMMAND.txt"
echo "actual_execution_deadline_seconds=$actual_deadline_seconds" | tee -a "$artifact_dir/COMMAND.txt"
echo "verification_deadline_seconds=$verification_deadline_seconds" | tee -a "$artifact_dir/COMMAND.txt"
git rev-parse HEAD >"$artifact_dir/SOURCE_REVISION.txt"
cat >"$artifact_dir/IMAGE_CONTRACT.txt" <<EOF
flashmla_base_revision=$flashmla_base_revision
flashmla_dual_lse_patch_revision=$flashmla_patch_revision
flashmla_dual_lse_patch_sha256=$flashmla_patch_sha256
flashmla_pro_h128_patch_revision=$flashmla_pro_patch_revision
flashmla_pro_h128_patch_sha256=$flashmla_pro_patch_sha256
cudnn_backend_version=$cudnn_backend_version
cudnn_frontend_version=$cudnn_frontend_version
cudnn_frontend_revision=$cudnn_frontend_revision
cudnn_frontend_source=official-unmodified
cudnn_frontend_local_patches=none
cutlass_dsl_version=$cutlass_dsl_version
quack_version=$quack_version
tvm_ffi_version=$tvm_ffi_version
magi_source_revision=$magi_source_revision
install_mode=python-wheel
validation=all_required_image_labels_exact
EOF
cat >"$artifact_dir/FLASHMLA_DUAL_LSE_PATCH.txt" <<EOF
base_revision=$flashmla_base_revision
patch_revision=$flashmla_patch_revision
patch_sha256=$flashmla_patch_sha256
image_label_revision=$image_flashmla_patch_revision
image_label_sha256=$image_flashmla_patch_sha256
pro_h128_patch_revision=$flashmla_pro_patch_revision
pro_h128_patch_sha256=$flashmla_pro_patch_sha256
pro_h128_image_label_revision=$image_flashmla_pro_patch_revision
pro_h128_image_label_sha256=$image_flashmla_pro_patch_sha256
behavior=CSA sparse forward returns full sparse_lse and compressed-prefix lse_indexer from the same FlashMLA kernel invocation
EOF
git remote get-url origin >"$artifact_dir/SOURCE_REMOTE.txt"
git submodule status --recursive >"$artifact_dir/SUBMODULES.txt"
git status --short >"$artifact_dir/DIRTY_STATUS.txt"
docker image inspect --format '{{json .Id}} {{json .RepoDigests}}' "$image" \
    >"$artifact_dir/IMAGE.txt"
nvidia-smi -q >"$artifact_dir/HARDWARE.txt"
if [[ "$case_name" == "cp8-natural-backward" ]]; then
    echo '{"layer_seeds":{"csa":440,"hca":442,"window":441},"input_seeds":{"csa":450,"hca":452,"window":451}}' \
        >"$artifact_dir/SEEDS.json"
elif [[ "$case_name" == "cp8-topk-diagnostic" ]]; then
    echo '{"layer_seed":440,"input_seed":450}' >"$artifact_dir/SEEDS.json"
else
    echo '{"layer_seed":410,"input_seed":411}' >"$artifact_dir/SEEDS.json"
fi
timeout --signal=TERM --kill-after=5s 60s docker run --rm --entrypoint bash "$image" -lc \
    'python -VV; python -m pip freeze; nsys --version 2>&1 || true' \
    >"$artifact_dir/ENVIRONMENT.txt" 2>&1

log_path="$artifact_dir/stdout_stderr.log"
: >"$log_path"
worker_path="extensions/tests/dsa_v4/distributed_worker.py"
container_workdir="/workspace/MagiAttention"
source_environment=(--env PYTHONPATH=/workspace/MagiAttention:/workspace/MagiAttention/extensions)
installed_environment=()
if ((installed_wheel == 1)); then
    worker_path="/workspace/MagiAttention/extensions/tests/dsa_v4/distributed_worker.py"
    container_workdir="/cp2-artifact/workdir"
    source_environment=()
    installed_environment=(
        --env MAGI_DSA_REQUIRE_INSTALLED_WHEEL=1
        --env "MAGI_DSA_EXPECTED_REVISION=$(git rev-parse HEAD)"
    )
fi
setsid timeout --signal=TERM --kill-after=5s "${total_deadline_seconds}s" docker run \
    --rm \
    --name "$container_name" \
    --entrypoint /usr/local/bin/torchrun \
    --gpus all \
    --ipc=host \
    --stop-timeout=5 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env CUDA_CACHE_PATH=/cudnn-dsa-cache/cuda \
    --env CUTE_DSL_CACHE_DIR=/cudnn-dsa-cache/cute-dsl \
    --env MAGI_DSA_RANK_CACHE_ROOT=/rank-cache/workers \
    --env MAGI_DSA_CP2_ARTIFACT_DIR=/cp2-artifact \
    --env MAGI_DSA_PHASE_LOG=1 \
    --env MAGI_ATTENTION_WORKSPACE_BASE=/cudnn-dsa-cache/magi-workspace \
    --env QUACK_CACHE_DIR=/rank-cache/launcher/quack \
    --env TEMP=/rank-cache/launcher/tmp \
    --env TMP=/rank-cache/launcher/tmp \
    --env TMPDIR=/rank-cache/launcher/tmp \
    --env TORCH_EXTENSIONS_DIR=/rank-cache/launcher/torch-extensions \
    --env TORCH_HOME=/rank-cache/launcher/torch-home \
    --env TORCHINDUCTOR_CACHE_DIR=/rank-cache/launcher/torchinductor \
    --env TRITON_CACHE_DIR=/rank-cache/launcher/triton \
    --env XDG_CACHE_HOME=/rank-cache/launcher/xdg \
    --env "NCCL_DEBUG=$nccl_debug" \
    "${source_environment[@]}" \
    "${installed_environment[@]}" \
    --env "TORCH_DISTRIBUTED_DEBUG=$torch_distributed_debug" \
    --volume "$artifact_dir:/cp2-artifact" \
    --volume "$cudnn_cache_dir:/cudnn-dsa-cache" \
    --volume "$rank_cache_dir:/rank-cache" \
    --volume "$repo_root:/workspace/MagiAttention:ro" \
    --workdir "$container_workdir" \
    "$image" \
    --standalone --nproc-per-node="$world_size" \
        "$worker_path" --case "$case_name" \
    >"$log_path" 2>&1 &
runner_pid="$!"
runner_pgid="$(ps -o pgid= -p "$runner_pid" | tr -d ' ')"
tail --pid="$runner_pid" -n +1 -f "$log_path" &
log_tail_pid="$!"

started_at="$(date +%s)"
last_status_at="$started_at"
actual_started=0
actual_completed=0
actual_deadline_at=0
actual_timed_out=0
verification_started=0
verification_completed=0
verification_deadline_at=0
verification_timed_out=0
while kill -0 "$runner_pid" 2>/dev/null; do
    now="$(date +%s)"
    execute_begin_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'control_rank*.jsonl' \
        -exec rg -l '"event": "execute_begin"' {} + 2>/dev/null | wc -l || true)"
    if ((actual_started == 0 && execute_begin_count >= 1)); then
        actual_started=1
        actual_deadline_at=$((now + actual_deadline_seconds))
        echo "$run_scope actual execution started; deadline=${actual_deadline_seconds}s runner_pid=$runner_pid" \
            | tee -a "$artifact_dir/WATCHDOG.log"
    fi
    if ((actual_started == 1 && actual_completed == 0)); then
        execute_end_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'control_rank*.jsonl' \
            -exec rg -l '"event": "execute_end"' {} + 2>/dev/null | wc -l || true)"
        if ((execute_end_count >= world_size)); then
            actual_completed=1
        fi
    fi
    verification_begin_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'control_rank*.jsonl' \
        -exec rg -l '"event": "verification_begin"' {} + 2>/dev/null | wc -l || true)"
    if ((verification_started == 0 && verification_begin_count >= 1)); then
        verification_started=1
        verification_deadline_at=$((now + verification_deadline_seconds))
    fi
    if ((verification_started == 1 && verification_completed == 0)); then
        verification_end_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'control_rank*.jsonl' \
            -exec rg -l '"event": "verification_end"' {} + 2>/dev/null | wc -l || true)"
        if ((verification_end_count >= world_size)); then
            verification_completed=1
        fi
    fi
    if ((actual_started == 1 && actual_completed == 0 && now >= actual_deadline_at)); then
        actual_timed_out=1
        echo "$run_scope actual execution exceeded ${actual_deadline_seconds}s; terminating container=$container_name pgid=$runner_pgid" \
            | tee -a "$artifact_dir/WATCHDOG.log"
        cleanup
        if [[ "$runner_pgid" =~ ^[0-9]+$ ]] && kill -0 "$runner_pid" 2>/dev/null; then
            kill -TERM -- "-$runner_pgid" 2>/dev/null || true
            for _ in 1 2 3 4 5; do
                if ! kill -0 "$runner_pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$runner_pid" 2>/dev/null; then
                kill -KILL -- "-$runner_pgid" 2>/dev/null || true
            fi
        fi
        break
    fi
    if ((verification_started == 1 && verification_completed == 0 && now >= verification_deadline_at)); then
        verification_timed_out=1
        echo "$run_scope verification exceeded ${verification_deadline_seconds}s; terminating container=$container_name pgid=$runner_pgid" \
            | tee -a "$artifact_dir/WATCHDOG.log"
        cleanup
        if [[ "$runner_pgid" =~ ^[0-9]+$ ]] && kill -0 "$runner_pid" 2>/dev/null; then
            kill -TERM -- "-$runner_pgid" 2>/dev/null || true
            for _ in 1 2 3 4 5; do
                if ! kill -0 "$runner_pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$runner_pid" 2>/dev/null; then
                kill -KILL -- "-$runner_pgid" 2>/dev/null || true
            fi
        fi
        break
    fi
    if ((now - last_status_at >= 30)); then
        recent_progress="$(rg 'MAGI_DSA_(CP2|PHASE)' "$log_path" | tail -n 1 || true)"
        current_phase="prewarm"
        if ((actual_started == 1 && actual_completed == 0)); then
            current_phase="execute"
        elif ((actual_completed == 1)); then
            current_phase="verify"
        fi
        echo "watchdog phase=$current_phase runner_pid=$runner_pid elapsed=$((now - started_at))s recent=${recent_progress:-none}" \
            | tee -a "$artifact_dir/WATCHDOG.log"
        last_status_at="$now"
    fi
    sleep 1
done

runner_status=0
if wait "$runner_pid"; then
    runner_status=0
else
    runner_status="$?"
fi
if [[ -n "$log_tail_pid" ]]; then
    wait "$log_tail_pid" 2>/dev/null || true
fi
if ((actual_timed_out == 1 || verification_timed_out == 1)); then
    runner_status=124
fi

summary_status=0
if python3 scripts/test/summarize_distributed.py \
    --artifact-dir "$artifact_dir" \
    --case "$case_name" \
    --output "$artifact_dir/SUMMARY.json" \
    --world-size "$world_size" \
    >"$artifact_dir/SUMMARY.stdout" 2>"$artifact_dir/SUMMARY.stderr"; then
    summary_status=0
else
    summary_status="$?"
fi
if ((runner_status == 0 && summary_status != 0)); then
    runner_status="$summary_status"
fi

audit_arguments=(
    --log "$log_path"
    --output "$artifact_dir/PHASE_AUDIT.json"
    --world-size "$world_size"
)
if ((actual_timed_out == 1)); then
    audit_arguments+=(--timed-out)
fi
phase_audit_status=0
if python3 scripts/test/audit_dsa_phases.py "${audit_arguments[@]}" \
    | tee "$artifact_dir/PHASE_AUDIT.stdout"; then
    phase_audit_status=0
else
    phase_audit_status="$?"
fi
if ((runner_status == 0 && phase_audit_status != 0)); then
    runner_status="$phase_audit_status"
fi

echo "path=$cudnn_cache_dir" >"$artifact_dir/CUDNN_CACHE.txt"
if timeout --signal=TERM --kill-after=5s 60s docker run --rm \
    --entrypoint bash \
    --volume "$cudnn_cache_dir:/cudnn-dsa-cache:ro" \
    "$image" \
    -lc 'find /cudnn-dsa-cache -type f -printf "%s %P\n" | sort' \
    >"$artifact_dir/CUDNN_CACHE_TREE.txt"; then
    echo "file_count=$(wc -l <"$artifact_dir/CUDNN_CACHE_TREE.txt")" >>"$artifact_dir/CUDNN_CACHE.txt"
    echo "bytes=$(awk '{total += $1} END {print total + 0}' "$artifact_dir/CUDNN_CACHE_TREE.txt")" \
        >>"$artifact_dir/CUDNN_CACHE.txt"
else
    echo "cache_tree_inspection=failed" >>"$artifact_dir/CUDNN_CACHE.txt"
fi
find "$artifact_dir" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    >"$artifact_dir/SHA256SUMS"

exit "$runner_status"
