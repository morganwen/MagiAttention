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
    echo "Usage: $0 [--image IMAGE] [--profiler-attach-warmup-steps N] [--local-improvement-passes N] [--lock-gpu-clock-mhz MHZ]" >&2
    echo "" >&2
    echo "The capture is fixed: 8 ranks, cp=8, dsv4-pro-128k, 5 steps, pro-pair," >&2
    echo "structural-balanced. Those were flags once, and every one of them had" >&2
    echo "exactly one legal value." >&2
}

world_size="8"
cp_size="8"
case_name="dsv4-pro-128k"
plans="balanced"
steps="5"
step_mode="pro-pair"
layout_policy="structural-balanced"
local_improvement_passes="4"
profiler_attach_warmup_steps="0"
lock_gpu_clock_mhz=""
image="${MAGI_DSA_IMAGE:-magi-dsa-v4:preflight-68c2f15}"
while (($# > 0)); do
    case "$1" in
        --local-improvement-passes)
            local_improvement_passes="${2:-}"
            shift 2
            ;;
        --profiler-attach-warmup-steps)
            profiler_attach_warmup_steps="${2:-}"
            shift 2
            ;;
        --lock-gpu-clock-mhz)
            lock_gpu_clock_mhz="${2:-}"
            shift 2
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

if [[ ! "$profiler_attach_warmup_steps" =~ ^[0-8]$ ]]; then
    usage
    exit 2
fi
if [[ -n "$lock_gpu_clock_mhz" && ! "$lock_gpu_clock_mhz" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi
if [[ "$profiler_attach_warmup_steps" != "0" || -n "$lock_gpu_clock_mhz" ]]; then
    echo "the formal Pro pair does not enable an attach warmup or a clock override" >&2
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
        echo "profile image $description label mismatch: expected $expected, found ${actual:-<missing>}" >&2
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
msa_reference_report="${MAGI_DSA_MSA_REFERENCE_REPORT:-not-configured}"

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
non_b300_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -cvE '^NVIDIA B300([[:space:]]|$)' || true)"
if ((gpu_count != 8 || non_b300_count != 0)); then
    echo "the release profile requires exactly 8 NVIDIA B300 GPUs" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')" ]]; then
    echo "one or more GPUs already have compute processes; refusing to contaminate the CP8 profile" >&2
    exit 1
fi

run_suffix=""
if [[ "$step_mode" == "forward-backward" ]]; then
    run_suffix="-forward-backward-dsa-core-precomputed-dout"
elif [[ "$step_mode" == "attention-suite" ]]; then
    run_suffix="-attention-suite-precomputed-dout"
    if [[ "$layout_policy" == "shared-greedy" ]]; then
        run_suffix="-attention-suite-shared-greedy-precomputed-dout"
        if [[ "$local_improvement_passes" != "4" || \
              "$profiler_attach_warmup_steps" != "0" || \
              -n "$lock_gpu_clock_mhz" ]]; then
            clock_label="dynamic"
            if [[ -n "$lock_gpu_clock_mhz" ]]; then
                clock_label="$lock_gpu_clock_mhz"
            fi
            run_suffix="-attention-suite-shared-greedy-passes${local_improvement_passes}-clock${clock_label}-attachwarm${profiler_attach_warmup_steps}-precomputed-dout"
        fi
    fi
elif [[ "$step_mode" == "pro-pair" ]]; then
    run_suffix="-pro-pair-structural-balanced-precomputed-dout"
fi
run_id="$(date -u +%Y%m%d-%H%M%S)-cp8-$(git -C "$repo_root" rev-parse --short=12 HEAD)"
artifact_dir="$repo_root/scripts/profile/result/$run_id"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite profile artifact directory: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir/balanced" \
    "$cudnn_cache_dir/cuda" "$cudnn_cache_dir/cute-dsl" \
    "$cudnn_cache_dir/magi-workspace" "$cudnn_cache_dir/tmp" \
    "$cudnn_cache_dir/xdg-cache" "$cudnn_cache_dir/torch" \
    "$cudnn_cache_dir/triton"
chmod 0777 "$artifact_dir" "$artifact_dir/balanced" \
    "$cudnn_cache_dir" "$cudnn_cache_dir/cuda" \
    "$cudnn_cache_dir/cute-dsl" "$cudnn_cache_dir/magi-workspace" \
    "$cudnn_cache_dir/tmp" "$cudnn_cache_dir/xdg-cache" \
    "$cudnn_cache_dir/torch" "$cudnn_cache_dir/triton"

current_container=""
current_runner_pid=""
current_runner_pgid=""
gpu_clocks_locked=0
profile_capture_order="balanced"
if true; then
    profile_capture_order="balanced"
fi

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

clock_control() {
    docker run \
        --rm \
        --gpus all \
        --cap-add SYS_ADMIN \
        --entrypoint nvidia-smi \
        "$image" \
        "$@"
}

reset_locked_gpu_clocks() {
    if ((gpu_clocks_locked == 0)); then
        return 0
    fi
    if ! clock_control -rgc >"$artifact_dir/CLOCK_RESET.log" 2>&1; then
        echo "failed to reset locked GPU clocks; inspect $artifact_dir/CLOCK_RESET.log" >&2
        return 1
    fi
    gpu_clocks_locked=0
    nvidia-smi \
        --query-gpu=index,uuid,clocks.current.graphics,clocks.max.graphics \
        --format=csv,noheader,nounits \
        >"$artifact_dir/CLOCKS_AFTER_RESET.csv"
}

lock_gpu_clocks() {
    local requested_mhz="$1"
    nvidia-smi \
        --query-gpu=index,uuid,clocks.current.graphics,clocks.max.graphics \
        --format=csv,noheader,nounits \
        >"$artifact_dir/CLOCKS_BEFORE_LOCK.csv"
    mapfile -t max_clocks < <(
        nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits \
            | tr -d ' '
    )
    if ((${#max_clocks[@]} != 8)); then
        echo "expected eight GPU max-clock records, found ${#max_clocks[@]}" >&2
        return 1
    fi
    for max_mhz in "${max_clocks[@]}"; do
        if ((requested_mhz > max_mhz)); then
            echo "requested GPU clock $requested_mhz exceeds device max $max_mhz" >&2
            return 1
        fi
    done

    gpu_clocks_locked=1
    if ! clock_control -lgc "$requested_mhz,$requested_mhz" \
        >"$artifact_dir/CLOCK_LOCK.log" 2>&1; then
        reset_locked_gpu_clocks || true
        echo "failed to lock all GPU clocks; inspect $artifact_dir/CLOCK_LOCK.log" >&2
        return 1
    fi
    nvidia-smi \
        --query-gpu=index,uuid,clocks.current.graphics,clocks.max.graphics \
        --format=csv,noheader,nounits \
        >"$artifact_dir/CLOCKS_LOCKED.csv"
    mapfile -t current_clocks < <(
        nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits \
            | tr -d ' '
    )
    if ((${#current_clocks[@]} != 8)); then
        reset_locked_gpu_clocks || true
        echo "expected eight locked-clock records, found ${#current_clocks[@]}" >&2
        return 1
    fi
    for current_mhz in "${current_clocks[@]}"; do
        if ((current_mhz != requested_mhz)); then
            reset_locked_gpu_clocks || true
            echo "GPU clock lock verification failed: requested=$requested_mhz current=$current_mhz" >&2
            return 1
        fi
    done
    cat >"$artifact_dir/CLOCK_CONTROL.txt" <<EOF
mode=nvml_lock_gpu_clocks
requested_graphics_clock_mhz=$requested_mhz
gpu_count=8
verified=true
reset_required=true
EOF
}

cleanup() {
    terminate_current_stage
    reset_locked_gpu_clocks || true
}

handle_signal() {
    local status="$1"
    trap - EXIT INT TERM
    cleanup
    exit "$status"
}

trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

{
    command_line="bash scripts/profile/run_5step.sh --local-improvement-passes $local_improvement_passes --profiler-attach-warmup-steps $profiler_attach_warmup_steps"
    if [[ -n "$lock_gpu_clock_mhz" ]]; then
        command_line+=" --lock-gpu-clock-mhz $lock_gpu_clock_mhz"
    fi
    command_line+=" --image $image"
    echo "$command_line"
    echo "run_id=$run_id"
    echo "artifact_dir=$artifact_dir"
    echo "cudnn_cache_dir=$cudnn_cache_dir"
    echo "flashmla_base_revision=$flashmla_base_revision"
    echo "cudnn_frontend_source=official-unmodified"
    echo "cudnn_frontend_local_patches=none"
    echo "cudnn_backend_version=$cudnn_backend_version"
    echo "cudnn_frontend_version=$cudnn_frontend_version"
    echo "cudnn_frontend_revision=$cudnn_frontend_revision"
    echo "cutlass_dsl_version=$cutlass_dsl_version"
    echo "quack_version=$quack_version"
    echo "tvm_ffi_version=$tvm_ffi_version"
    echo "magi_source_revision=$magi_source_revision"
    echo "flashmla_dual_lse_patch_revision=$flashmla_patch_revision"
    echo "flashmla_dual_lse_patch_sha256=$flashmla_patch_sha256"
    echo "flashmla_pro_h128_patch_revision=$flashmla_pro_patch_revision"
    echo "flashmla_pro_h128_patch_sha256=$flashmla_pro_patch_sha256"
    echo "profile_deadline_seconds_per_plan=1800"
    echo "no_progress_watchdog_seconds=60"
    echo "watchdog_kill_sequence=TERM,wait-5s,KILL"
    echo "capture_order=$profile_capture_order"
    echo "step_mode=$step_mode"
    echo "layout_policy=$layout_policy"
    echo "local_improvement_passes=$local_improvement_passes"
    echo "profiler_attach_warmup_steps=$profiler_attach_warmup_steps"
    echo "lock_gpu_clock_mhz=${lock_gpu_clock_mhz:-none}"
    if [[ "$step_mode" == "forward-backward" ]]; then
        echo "profile_gradient_boundary=post_projection_magi_dsa_input"
        echo "backward_seed=precomputed_global_mean_scaled_dout_and_unit_dkl"
        echo "dout_scale=1/global_output_elements=2.3283064365386963e-10"
        echo "loss_capture=none"
        echo "projection_capture=pre_capture_once"
        echo "token_layout_capture=pre_capture_once"
    elif [[ "$step_mode" == "attention-suite" ]]; then
        echo "attention_order=w,csa,hca"
        echo "backward_order=hca,csa,w"
        echo "independent_attention_graphs=true"
        echo "mode_serialization=cuda_event_happens_before"
        echo "mode_backward_completion_join=W:none,CSA:sparse+main+indexer+route,HCA:main+route"
        echo "overlap_accounting=same_mode_same_direction_non_route_compute"
        echo "gradient_accumulation=false"
        echo "profile_gradient_boundary=post_projection_magi_dsa_input"
        echo "backward_seed=precomputed_global_mean_scaled_dout_and_unit_dkl"
        echo "dout_scale=1/global_output_elements=2.3283064365386963e-10"
        echo "loss_capture=none"
        echo "projection_capture=pre_capture_once_per_attention"
        echo "token_layout_capture=pre_capture_once_per_attention"
        echo "parameter_gradient_allreduce=one_unified_after_three_backwards"
    elif [[ "$step_mode" == "pro-pair" ]]; then
        echo "attention_order=csa,hca"
        echo "backward_order=hca,csa"
        echo "representative_layer_ids=csa:2,hca:3"
        echo "representative_pair_semantics=independent_post_projection_graphs_serialized_in_layer_order"
        echo "main_stack_schedule=61_layers,30_CSA,31_HCA"
        echo "independent_attention_graphs=true"
        echo "mode_serialization=cuda_event_happens_before"
        echo "mode_backward_completion_join=CSA:sparse+main+indexer+route,HCA:main+route"
        echo "overlap_accounting=same_mode_same_direction_non_route_compute"
        echo "gradient_accumulation=false"
        echo "profile_gradient_boundary=post_projection_magi_dsa_input"
        echo "backward_seed=precomputed_global_mean_scaled_dout_and_unit_dkl"
        echo "dout_scale=1/global_output_elements=1.1641532182693481e-10"
        echo "loss_capture=none"
        echo "projection_capture=pre_capture_once_per_attention"
        echo "token_layout_capture=pre_capture_once_per_attention"
        echo "parameter_gradient_allreduce=one_unified_after_two_backwards"
        echo "major_kernel_balance_gate=indexer_score_topk_0.05_others_report_only"
        echo "indexer_d2d_accounting=separate"
        echo "indexer_d2d_attribution_source=same_balanced_aggregate_capture_no_replay"
        echo "indexer_d2d_cudnn_call_scopes=magi_dsa::CUDNN_CALL::{indexer_score,indexer_topk,selected_indexer_recompute,selected_attention_recompute,indexer_backward}"
        echo "pro_pair_required_summary_artifacts=INDEXER_D2D_PRO_PAIR.json,SUPPORT_OVERHEAD_PRO_PAIR.json,PRO_PAIR_COMMUNICATION_OVERLAP.json"
    fi
} >"$artifact_dir/COMMAND.txt"

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

cat >"$artifact_dir/CUDNN_FRONTEND.txt" <<EOF
version=$cudnn_frontend_version
revision=$cudnn_frontend_revision
source=official-unmodified
local_patches=none
EOF

cat >"$artifact_dir/FLASHMLA_DUAL_LSE_PATCH.txt" <<EOF
base_revision=$flashmla_base_revision
patch_revision=$flashmla_patch_revision
patch_sha256=$flashmla_patch_sha256
image_label_revision=$image_flashmla_patch_revision
image_label_sha256=$image_flashmla_patch_sha256
behavior=CSA sparse forward returns full sparse_lse and compressed-prefix lse_indexer from the same FlashMLA kernel invocation
EOF

cat >"$artifact_dir/FLASHMLA_PRO_H128_PATCH.txt" <<EOF
patch_revision=$flashmla_pro_patch_revision
patch_sha256=$flashmla_pro_patch_sha256
image_label_revision=$image_flashmla_pro_patch_revision
image_label_sha256=$image_flashmla_pro_patch_sha256
behavior=FlashMLA accepts DeepSeek-V4-Pro H128 with indexer_topk=1024
EOF

{
    echo "balanced_report=$msa_reference_report"
    cat <<'EOF'
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
dsa_forward_backward_outer=$Magi_DSA/capture_five_forward_backward_steps
dsa_forward_backward_phases=magi_dsa::forward,magi_dsa::backward,magi_dsa::parameter_gradient_allreduce
dsa_forward_backward_boundary=post_projection_magi_dsa_input
dsa_forward_backward_pre_capture=TOKEN_LAYOUT,profile_projection
dsa_forward_backward_seed=global_BF16_randn_scaled_by_inverse_global_output_elements_then_indexed_by_plan_local_query_global_rows,FP32_dkl_one
dsa_forward_backward_loss=none
dsa_attention_suite_outer=$Magi_DSA/capture_five_attention_suite_steps
dsa_attention_suite_order=W,CSA,HCA
dsa_attention_suite_backward_order=HCA,CSA,W
dsa_attention_suite_phases=magi_dsa::attention_suite::{w,csa,hca}::{forward,backward}
dsa_attention_suite_graphs=independent
dsa_attention_suite_mode_serialization=cuda_event_happens_before
dsa_attention_suite_overlap_accounting=same_mode_same_direction_non_route_compute
dsa_attention_suite_sendrecv=W:1F+1B,CSA:4F+4B,HCA:3F+3B
dsa_pro_pair_outer=$Magi_DSA/capture_five_pro_pair_steps
dsa_pro_pair_order=CSA,HCA
dsa_pro_pair_backward_order=HCA,CSA
dsa_pro_pair_phases=magi_dsa::pro_pair::{csa,hca}::{forward,backward}
dsa_pro_pair_graphs=independent_post_projection
dsa_pro_pair_layout=structural_balanced
dsa_pro_pair_layers=CSA:2,HCA:3
dsa_pro_pair_sendrecv=CSA:4F+4B,HCA:3F+3B,total:7F+7B
EOF
} >"$artifact_dir/MSA_REFERENCE.txt"

git rev-parse HEAD >"$artifact_dir/SOURCE_REVISION.txt"
git remote get-url origin >"$artifact_dir/SOURCE_REMOTE.txt"
git submodule status --recursive >"$artifact_dir/SUBMODULES.txt"
git status --short >"$artifact_dir/DIRTY_STATUS.txt"
docker image inspect "$image" >"$artifact_dir/IMAGE.json"
nvidia-smi -q >"$artifact_dir/HARDWARE.txt"
timeout --signal=TERM --kill-after=5s 60s docker run --rm --entrypoint bash "$image" -lc \
    'python -VV; python -m pip freeze; nsys --version' \
    >"$artifact_dir/ENVIRONMENT.txt" 2>&1
smoke_mode="retired"
profile_plans_json='["balanced"]'
profile_capture_order_json='["balanced"]'
profile_gradient_boundary="source_owner_x"
projection_capture="per_step"
token_layout_capture="per_step"
backward_seed="none"
dout_scale_json="null"
loss_capture="none"
attention_order_json='["csa"]'
backward_order_json='["csa"]'
gradient_accumulation_json="false"
independent_attention_graphs_json="false"
mode_serialization="none"
mode_backward_completion_join_json="null"
overlap_accounting="not_applicable"
parameter_gradient_allreduce="per_attention"
ratio_json="4"
ratios_json="[4]"
local_improvement_passes_json="$local_improvement_passes"
representative_layer_ids_json="null"
representative_pair_semantics_json="null"
structural_layout_config_json="null"
pro_model_contract_json="null"
cudnn_memcpy_attribution_scopes_json="null"
required_summary_artifacts_json="null"
gpu_clock_lock_json="null"
if [[ -n "$lock_gpu_clock_mhz" ]]; then
    gpu_clock_lock_json="$lock_gpu_clock_mhz"
fi
if [[ "$step_mode" == "forward-backward" ]]; then
    profile_plans_json='["balanced"]'
    profile_capture_order_json='["balanced"]'
    profile_gradient_boundary="post_projection_magi_dsa_input"
    projection_capture="pre_capture_once"
    token_layout_capture="pre_capture_once"
    backward_seed="precomputed_global_mean_scaled_dout_and_unit_dkl"
    dout_scale_json="2.3283064365386963e-10"
elif [[ "$step_mode" == "attention-suite" ]]; then
    profile_plans_json='["balanced"]'
    profile_capture_order_json='["balanced"]'
    profile_gradient_boundary="post_projection_magi_dsa_input"
    projection_capture="pre_capture_once_per_attention"
    token_layout_capture="pre_capture_once_per_attention"
    backward_seed="precomputed_global_mean_scaled_dout_and_unit_dkl"
    dout_scale_json="2.3283064365386963e-10"
    attention_order_json='["w", "csa", "hca"]'
    backward_order_json='["hca", "csa", "w"]'
    independent_attention_graphs_json="true"
    mode_serialization="cuda_event_happens_before"
    mode_backward_completion_join_json='{"w": [], "csa": ["sparse_backward_stream", "csa_main_stream", "csa_indexer_stream", "csa_route_stream"], "hca": ["hca_main_stream", "hca_route_stream"]}'
    overlap_accounting="same_mode_same_direction_non_route_compute"
    parameter_gradient_allreduce="one_unified_after_three_backwards"
    ratio_json="null"
    ratios_json="[0, 4, 128]"
elif [[ "$step_mode" == "pro-pair" ]]; then
    profile_plans_json='["balanced"]'
    profile_capture_order_json='["balanced"]'
    profile_gradient_boundary="post_projection_magi_dsa_input"
    projection_capture="pre_capture_once_per_attention"
    token_layout_capture="pre_capture_once_per_attention"
    backward_seed="precomputed_global_mean_scaled_dout_and_unit_dkl"
    dout_scale_json="1.1641532182693481e-10"
    attention_order_json='["csa", "hca"]'
    backward_order_json='["hca", "csa"]'
    independent_attention_graphs_json="true"
    mode_serialization="cuda_event_happens_before"
    mode_backward_completion_join_json='{"csa": ["sparse_backward_stream", "csa_main_stream", "csa_indexer_stream", "csa_route_stream"], "hca": ["hca_main_stream", "hca_route_stream"]}'
    overlap_accounting="same_mode_same_direction_non_route_compute"
    parameter_gradient_allreduce="one_unified_after_two_backwards"
    ratio_json="null"
    ratios_json="[4, 128]"
    local_improvement_passes_json="null"
    representative_layer_ids_json='{"csa": 2, "hca": 3}'
    representative_pair_semantics_json='"independent_post_projection_graphs_serialized_in_layer_order"'
    structural_layout_config_json='{"chunk_size": 512, "min_chunks_per_rank": 16, "uneven_shard": true}'
    pro_model_contract_json='{"source_revision": "b5968e9190ef611bbf34a7229255be88a0e937c1", "main_layer_count": 61, "csa_layer_count": 30, "hca_layer_count": 31, "hidden_size": 7168, "q_lora_rank": 1536, "num_query_heads": 128, "head_dim": 512, "indexer_heads": 64, "indexer_head_dim": 128, "indexer_topk": 1024, "window_size": 128}'
    cudnn_memcpy_attribution_scopes_json='["magi_dsa::CUDNN_CALL::indexer_score", "magi_dsa::CUDNN_CALL::indexer_topk", "magi_dsa::CUDNN_CALL::selected_indexer_recompute", "magi_dsa::CUDNN_CALL::selected_attention_recompute", "magi_dsa::CUDNN_CALL::indexer_backward"]'
    required_summary_artifacts_json='["INDEXER_D2D_PRO_PAIR.json", "SUPPORT_OVERHEAD_PRO_PAIR.json", "PRO_PAIR_COMMUNICATION_OVERLAP.json"]'
fi
cat >"$artifact_dir/WORKLOAD.json" <<EOF
{
  "attention_order": $attention_order_json,
  "backward_order": $backward_order_json,
  "case": "$case_name",
  "cp_size": 8,
  "cu_seqlens": [0, 131072],
  "cudnn_memcpy_attribution_scopes": $cudnn_memcpy_attribution_scopes_json,
  "dtype": "BF16",
  "backward_seed": "$backward_seed",
  "dout_scale": $dout_scale_json,
  "capture_order": $profile_capture_order_json,
  "gradient_accumulation": $gradient_accumulation_json,
  "gpu_clock_lock_mhz": $gpu_clock_lock_json,
  "independent_attention_graphs": $independent_attention_graphs_json,
  "layout_policy": "$layout_policy",
  "local_improvement_passes": $local_improvement_passes_json,
  "mode_serialization": "$mode_serialization",
  "mode_backward_completion_join": $mode_backward_completion_join_json,
  "overlap_accounting": "$overlap_accounting",
  "plans": $profile_plans_json,
  "loss_capture": "$loss_capture",
  "parameter_gradient_allreduce": "$parameter_gradient_allreduce",
  "profile_gradient_boundary": "$profile_gradient_boundary",
  "projection_capture": "$projection_capture",
  "profiler_attach_warmup_steps": $profiler_attach_warmup_steps,
  "rank_size": 8,
  "ratio": $ratio_json,
  "ratios": $ratios_json,
  "representative_layer_ids": $representative_layer_ids_json,
  "representative_pair_semantics": $representative_pair_semantics_json,
  "required_summary_artifacts": $required_summary_artifacts_json,
  "structural_layout_config": $structural_layout_config_json,
  "pro_model_contract": $pro_model_contract_json,
  "seed": 0,
  "smoke": "$smoke_mode",
  "step_mode": "$step_mode",
  "steps": 5,
  "token_layout_capture": "$token_layout_capture",
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
    --env TMPDIR=/cudnn-dsa-cache/tmp
    --env TEMP=/cudnn-dsa-cache/tmp
    --env TMP=/cudnn-dsa-cache/tmp
    --env XDG_CACHE_HOME=/cudnn-dsa-cache/xdg-cache
    --env TORCH_HOME=/cudnn-dsa-cache/torch
    --env TRITON_CACHE_DIR=/cudnn-dsa-cache/triton
    --env MAGI_DSA_PHASE_LOG=0
    --env NCCL_DEBUG=WARN
    --env PYTHONSAFEPATH=1
    --env PYTHONUNBUFFERED=1
    --env TORCH_DISTRIBUTED_DEBUG=INFO
    --volume "$artifact_dir:/profile-artifact"
    --volume "$cudnn_cache_dir:/cudnn-dsa-cache"
    --volume "$repo_root/scripts/profile:/workspace/MagiAttention/scripts/profile:ro"
    --workdir /profile-artifact
)

if [[ -n "$lock_gpu_clock_mhz" ]]; then
    lock_gpu_clocks "$lock_gpu_clock_mhz"
else
    cat >"$artifact_dir/CLOCK_CONTROL.txt" <<EOF
mode=dynamic
requested_graphics_clock_mhz=none
gpu_count=8
verified=false
reset_required=false
EOF
fi


profile_plan_order=(balanced)
for plan in "${profile_plan_order[@]}"; do
    run_stage \
        "$plan" \
        1800 \
        "$artifact_dir/$plan" \
        "${common_docker_args[@]}" \
        --entrypoint bash \
        "$image" \
        /workspace/MagiAttention/scripts/profile/run_plan.sh \
        --plan "$plan" \
        --artifact-dir "/profile-artifact/$plan" \
        --world-size 8 \
        --steps 5 \
        --tokens 131072 \
        --warmup 3 \
        --step-mode "$step_mode" \
        --layout-policy "$layout_policy" \
        --local-improvement-passes "$local_improvement_passes" \
        --profiler-attach-warmup-steps "$profiler_attach_warmup_steps"
done

if [[ "$step_mode" == "pro-pair" ]]; then
    timeout --signal=TERM --kill-after=5s 600s python3 - \
        "$artifact_dir/balanced" \
        >"$artifact_dir/NSYS_MEMCPY_PREFLIGHT.stdout" \
        2>"$artifact_dir/NSYS_MEMCPY_PREFLIGHT.stderr" <<'PY'
import json
import pathlib
import sys

plan_dir = pathlib.Path(sys.argv[1]).resolve()
global_path = plan_dir / "nsys_memcpy_attribution.jsonl"
summary_path = plan_dir / "NSYS_MEMCPY_ATTRIBUTION.json"
required_paths = [global_path, summary_path]
required_paths.extend(
    plan_dir / f"rank{rank}_nsys_memcpy_attribution.jsonl" for rank in range(8)
)
missing = [str(path) for path in required_paths if not path.is_file()]
if missing:
    raise SystemExit(f"missing Pro-pair memcpy attribution artifacts: {missing}")


def read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise SystemExit(f"{path}:{line_number} is not a JSON object")
        records.append(record)
    return records


records = read_jsonl(global_path)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_summary = {
    "plan": "balanced",
    "result": "PASS",
    "steps": 5,
    "world_size": 8,
    "logical_step_ranges": 40,
    "unattributed_bytes": 0,
    "unattributed_copy_count": 0,
    "unattributed_gpu_time_ms": 0.0,
    "unattributed_memcpy_activity_count": 0,
}
for field, expected in expected_summary.items():
    if summary.get(field) != expected:
        raise SystemExit(
            f"invalid Pro-pair memcpy attribution summary {field}: "
            f"expected {expected!r}, found {summary.get(field)!r}"
        )
for field in (
    "attribution_coverage",
    "attribution_byte_coverage",
    "attribution_copy_count_coverage",
):
    if float(summary.get(field, -1.0)) != 1.0:
        raise SystemExit(
            f"invalid Pro-pair memcpy attribution summary {field}: "
            f"expected 1.0, found {summary.get(field)!r}"
        )
if len(summary.get("global_pids", [])) != 8:
    raise SystemExit("Pro-pair memcpy attribution does not cover eight worker PIDs")
if int(summary.get("memcpy_activity_records", -1)) != len(records):
    raise SystemExit("Pro-pair global memcpy record count differs from its audit")
if int(summary.get("bytes", -1)) != sum(int(record["bytes"]) for record in records):
    raise SystemExit("Pro-pair global memcpy bytes differ from its audit")
if int(summary.get("copy_count", -1)) != sum(
    int(record["copy_count"]) for record in records
):
    raise SystemExit("Pro-pair global memcpy copy count differs from its audit")

for record in records:
    if record.get("record_type") != "magi_dsa_memcpy_attribution":
        raise SystemExit("Pro-pair global memcpy attribution has an invalid schema")
    if record.get("plan") != "balanced":
        raise SystemExit("Pro-pair global memcpy attribution has a non-balanced row")
    rank = int(record.get("rank", -1))
    step = int(record.get("step", -1))
    if rank not in range(8) or step not in range(5):
        raise SystemExit("Pro-pair global memcpy attribution is outside the CP8x5 grid")
    if record.get("attribution_kind") == "unattributed":
        raise SystemExit("Pro-pair global memcpy attribution contains an unowned row")

for rank in range(8):
    rank_records = read_jsonl(
        plan_dir / f"rank{rank}_nsys_memcpy_attribution.jsonl"
    )
    expected_rank_records = [
        record for record in records if int(record["rank"]) == rank
    ]
    if rank_records != expected_rank_records:
        raise SystemExit(
            f"rank {rank} Pro-pair memcpy attribution differs from the global file"
        )

print(
    "PASS: validated same-capture Pro-pair memcpy attribution "
    f"records={len(records)} bytes={summary['bytes']} copies={summary['copy_count']}"
)
PY
fi

if [[ -n "$lock_gpu_clock_mhz" ]]; then
    reset_locked_gpu_clocks
    echo "reset_verified=true" >>"$artifact_dir/CLOCK_CONTROL.txt"
fi

set +e
summary_script="$repo_root/scripts/profile/summarize_pro_pair.py"
timeout --signal=TERM --kill-after=5s 600s python3 \
    "$summary_script" \
    --artifact-dir "$artifact_dir" \
    --world-size 8 \
    --steps 5 \
    >"$artifact_dir/SUMMARIZE.stdout" 2>"$artifact_dir/SUMMARIZE.stderr"
summary_status=$?
set -e

if ((summary_status == 0)) && [[ "$step_mode" == "pro-pair" ]]; then
    set +e
    timeout --signal=TERM --kill-after=5s 600s python3 - \
        "$artifact_dir" \
        >"$artifact_dir/PRO_PAIR_ARTIFACT_CONTRACT.stdout" \
        2>"$artifact_dir/PRO_PAIR_ARTIFACT_CONTRACT.stderr" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
required_root = (
    "INDEXER_D2D_PRO_PAIR.json",
    "SUPPORT_OVERHEAD_PRO_PAIR.json",
    "PRO_PAIR_COMMUNICATION_OVERLAP.json",
)
required_plan = (
    "balanced/nsys_memcpy_attribution.jsonl",
    "balanced/NSYS_MEMCPY_ATTRIBUTION.json",
    *(f"balanced/rank{rank}_nsys_memcpy_attribution.jsonl" for rank in range(8)),
)
missing_or_empty = [
    relative
    for relative in (*required_root, *required_plan)
    if not (root / relative).is_file()
    or ((root / relative).stat().st_size == 0 and not relative.endswith(".jsonl"))
]
if missing_or_empty:
    raise SystemExit(
        f"incomplete final Pro-pair artifact contract: {missing_or_empty}"
    )
for relative in required_root:
    json.loads((root / relative).read_text(encoding="utf-8"))

inventory_path = root / "PRO_PAIR_ARTIFACTS.txt"
if inventory_path.exists():
    raise SystemExit(f"refusing to overwrite Pro-pair artifact inventory: {inventory_path}")
inventory_path.write_text(
    "capture_source=balanced/balanced_5steps_pro_pair.nsys-rep\n"
    "capture_replay=none\n"
    "extractor=single_export_from_same_aggregate_capture\n"
    + "\n".join(f"artifact={relative}" for relative in required_root)
    + "\n"
    + "\n".join(f"attribution_artifact={relative}" for relative in required_plan)
    + "\n",
    encoding="utf-8",
)
print("PASS: validated and listed final Pro-pair attribution artifacts")
PY
    pro_pair_artifact_status=$?
    set -e
    if ((pro_pair_artifact_status != 0)); then
        summary_status=$pro_pair_artifact_status
    fi
fi

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

# The per-rank and merged jsonl exports exist so the summarizer can validate
# collective counts, route order and attribution. Once it has passed there is
# nothing left to read them for, and they are four times the size of everything
# worth keeping. The nsys report and its sqlite stay, so any of it can be
# regenerated.
# Housekeeping runs after the verdict, so it is never allowed to change it.
# The rank handshake directory is written by the container, and NFS maps that
# root to nobody, which leaves it undeletable from the host by anyone. It is
# eight kilobytes; a run that already passed must not fail over it.
prune_artifacts() {
    find "$artifact_dir" \
        \( -name '*.jsonl' -o -name '*.stdout' -o -name '*.stderr' \
           -o -name '*.log' \) -delete
    rm -f "$artifact_dir/SHA256SUMS" "$artifact_dir/HARDWARE.txt" \
        "$artifact_dir/ENVIRONMENT.txt" "$artifact_dir/SUBMODULES.txt" \
        "$artifact_dir/IMAGE.json" "$artifact_dir"/*/NSYS_STATS.txt
    rm -f "$artifact_dir"/*/capture_done_rank*.json \
        "$artifact_dir"/*/ready_rank*.json
    find "$artifact_dir" -type d -empty -delete
}
kept_bytes_before="$(du -sk "$artifact_dir" | cut -f1)"
if ! prune_artifacts 2>/dev/null; then
    echo "note: some intermediates could not be removed; the result is complete" >&2
fi
kept_bytes_after="$(du -sk "$artifact_dir" | cut -f1)"
echo "pruned intermediates: ${kept_bytes_before}K -> ${kept_bytes_after}K"
echo "profile complete: $artifact_dir"
