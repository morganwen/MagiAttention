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
    echo "Usage: $0 --image <image-tag-or-id>" >&2
}

image=""
while (($# > 0)); do
    case "$1" in
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
if [[ -z "$image" ]]; then
    usage
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
task_suffix="$$"
containers=(
    "magi-dsa-preflight-env-$task_suffix"
    "magi-dsa-preflight-dsa-$task_suffix"
    "magi-dsa-preflight-flashmla-$task_suffix"
    "magi-dsa-preflight-nccl-$task_suffix"
)

cleanup() {
    local container
    for container in "${containers[@]}"; do
        if docker container inspect "$container" >/dev/null 2>&1; then
            docker container stop --time 5 "$container" >/dev/null 2>&1 || true
            docker container rm --force "$container" >/dev/null 2>&1 || true
        fi
    done
}
trap cleanup EXIT INT TERM

phase() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" >&2
}

common=(
    --rm
    --ipc=host
    --stop-timeout=5
    --volume "$repo_root:/workspace/MagiAttention:ro"
    --workdir /workspace/MagiAttention
)

phase "environment and eight-GPU visibility"
timeout --signal=TERM --kill-after=5s 180s docker run \
    "${common[@]}" \
    --name "${containers[0]}" \
    --gpus all \
    "$image" \
    python3 scripts/test/preflight_env.py

phase "cuDNN DSA score, top-k, and backward smoke"
timeout --signal=TERM --kill-after=5s 1800s docker run \
    "${common[@]}" \
    --name "${containers[1]}" \
    --gpus device=0 \
    "$image" \
    python3 scripts/test/preflight_cudnn_dsa.py

phase "FlashMLA sparse prefill smoke"
timeout --signal=TERM --kill-after=5s 600s docker run \
    "${common[@]}" \
    --name "${containers[2]}" \
    --gpus device=0 \
    "$image" \
    python3 scripts/test/preflight_flashmla.py

phase "eight-rank NCCL tensor collectives"
timeout --signal=TERM --kill-after=5s 60s docker run \
    "${common[@]}" \
    --env NCCL_DEBUG=INFO \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --name "${containers[3]}" \
    --gpus all \
    "$image" \
    torchrun --standalone --nproc-per-node=8 scripts/test/preflight_nccl.py

phase "Preflight passed"
