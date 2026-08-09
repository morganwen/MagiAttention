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

repo_root="$(git rev-parse --show-toplevel)"
image="${MAGI_DSA_IMAGE:-magi-dsa-v4:preflight-68c2f15}"
pack_sha="$(sha256sum "$repo_root/extensions/magi_attn_extensions/DSA/kernels/cutedsl/pack.py" | cut -c1-16)"
aot_dir="${MAGI_DSA_CUTE_AOT:-$repo_root/.cache/magi-dsa-v4/aot-cutlass-4.5.0-sm103-$pack_sha}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-cute-jit"
artifact_dir="$repo_root/artifacts/release/$run_id"
container_name="magi-dsa-cute-jit-$$"
validate_container_name="$container_name-validate"
mkdir -p "$artifact_dir" "$aot_dir" "$aot_dir/tmp" "$aot_dir/cuda" \
    "$aot_dir/cute-dsl" "$aot_dir/xdg-cache" "$aot_dir/torch" \
    "$aot_dir/triton"
# Docker root is intentionally mapped to nobody on this host. The ignored,
# content-addressed compiler cache must therefore be writable by that mapping.
chmod 0777 "$aot_dir" "$aot_dir/tmp" "$aot_dir/cuda" \
    "$aot_dir/cute-dsl" "$aot_dir/xdg-cache" "$aot_dir/torch" \
    "$aot_dir/triton"

cleanup() {
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        docker container stop --time 5 "$container_name" >/dev/null 2>&1 || true
        docker container rm --force "$container_name" >/dev/null 2>&1 || true
    fi
    if docker container inspect "$validate_container_name" >/dev/null 2>&1; then
        docker container stop --time 5 "$validate_container_name" >/dev/null 2>&1 || true
        docker container rm --force "$validate_container_name" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

{
    echo "artifact_dir=$artifact_dir"
    echo "aot_dir=$aot_dir"
    echo "image=$image"
    echo "python3 scripts/test/prewarm_dsa_pack.py"
} | tee "$artifact_dir/COMMAND.txt"

mapfile -t required_objects < <(python3 "$repo_root/scripts/test/dsa_pack_aot_manifest.py")
missing_objects=()
for object_name in "${required_objects[@]}"; do
    if [[ ! -f "$aot_dir/$object_name" ]]; then
        missing_objects+=("$object_name")
    fi
done
if ((${#missing_objects[@]} > 0)); then
    timeout --signal=TERM --kill-after=5s 1500s docker run \
        --rm \
        --name "$container_name" \
        --entrypoint python3 \
        --gpus all \
        --ipc=host \
        --stop-timeout=5 \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --env MAGI_DSA_CUTE_AOT_OUTPUT_DIR=/dsa-pack-aot \
        --env TMPDIR=/dsa-pack-aot/tmp \
        --env TEMP=/dsa-pack-aot/tmp \
        --env TMP=/dsa-pack-aot/tmp \
        --env CUDA_CACHE_PATH=/dsa-pack-aot/cuda \
        --env CUTE_DSL_CACHE_DIR=/dsa-pack-aot/cute-dsl \
        --env XDG_CACHE_HOME=/dsa-pack-aot/xdg-cache \
        --env TORCH_HOME=/dsa-pack-aot/torch \
        --env TRITON_CACHE_DIR=/dsa-pack-aot/triton \
        --env PYTHONPATH=/workspace/MagiAttention:/workspace/MagiAttention/extensions \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --volume "$aot_dir:/dsa-pack-aot" \
        --volume "$repo_root:/workspace/MagiAttention:ro" \
        --workdir /dsa-pack-aot \
        "$image" \
        /workspace/MagiAttention/scripts/test/prewarm_dsa_pack.py \
        2>&1 | tee "$artifact_dir/compile.log"
fi

for object_name in "${required_objects[@]}"; do
    if [[ ! -f "$aot_dir/$object_name" ]]; then
        echo "CuTe AOT export is missing required object: $object_name" >&2
        exit 1
    fi
done
object_count="$(find "$aot_dir" -maxdepth 1 -type f -name '*.o' | wc -l)"
timeout --signal=TERM --kill-after=5s 300s docker run \
    --rm \
    --name "$validate_container_name" \
    --entrypoint python3 \
    --gpus all \
    --ipc=host \
    --stop-timeout=5 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env MAGI_DSA_CUTE_AOT_DIR=/dsa-pack-aot \
    --env MAGI_DSA_CUTE_AOT_REQUIRED=1 \
    --env TMPDIR=/dsa-pack-aot/tmp \
    --env TEMP=/dsa-pack-aot/tmp \
    --env TMP=/dsa-pack-aot/tmp \
    --env CUDA_CACHE_PATH=/dsa-pack-aot/cuda \
    --env CUTE_DSL_CACHE_DIR=/dsa-pack-aot/cute-dsl \
    --env XDG_CACHE_HOME=/dsa-pack-aot/xdg-cache \
    --env TORCH_HOME=/dsa-pack-aot/torch \
    --env TRITON_CACHE_DIR=/dsa-pack-aot/triton \
    --env PYTHONPATH=/workspace/MagiAttention:/workspace/MagiAttention/extensions \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --volume "$aot_dir:/dsa-pack-aot" \
    --volume "$repo_root:/workspace/MagiAttention:ro" \
    --workdir /dsa-pack-aot \
    "$image" \
    /workspace/MagiAttention/scripts/test/prewarm_dsa_pack.py \
    2>&1 | tee "$artifact_dir/validate.log"

{
    echo "required_aot_object_count=${#required_objects[@]}"
    echo "total_aot_object_count=$object_count"
} | tee "$artifact_dir/SUMMARY.txt"
find "$aot_dir" -maxdepth 1 -type f -print0 | sort -z | xargs -0 -r sha256sum \
    >"$artifact_dir/AOT_MANIFEST.sha256"
