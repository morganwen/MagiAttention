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
pack_sha="$(sha256sum "$repo_root/magi_attention/kernel/cutedsl/dsa_pack.py" | cut -c1-16)"
aot_dir="${MAGI_DSA_CUTE_AOT:-$repo_root/.cache/magi-dsa-v4/aot-cutlass-4.5.0-sm103-$pack_sha}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-cute-jit"
artifact_dir="$repo_root/artifacts/release/$run_id"
container_name="magi-dsa-cute-jit-$$"
validate_container_name="$container_name-validate"
mkdir -p "$artifact_dir" "$aot_dir"
# Docker root is intentionally mapped to nobody on this host. The ignored,
# content-addressed compiler cache must therefore be writable by that mapping.
chmod 0777 "$aot_dir"

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

expected_objects=11
existing_objects="$(find "$aot_dir" -maxdepth 1 -type f -name '*.o' | wc -l)"
if ((existing_objects != expected_objects)); then
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
        --env PYTHONPATH=/workspace/Magi-DSA \
        --volume "$aot_dir:/dsa-pack-aot" \
        --volume "$repo_root:/workspace/Magi-DSA:ro" \
        --workdir /workspace/Magi-DSA \
        "$image" \
        scripts/test/prewarm_dsa_pack.py \
        2>&1 | tee "$artifact_dir/compile.log"
fi

object_count="$(find "$aot_dir" -maxdepth 1 -type f -name '*.o' | wc -l)"
if ((object_count != expected_objects)); then
    echo "CuTe AOT export produced $object_count objects, expected $expected_objects" >&2
    exit 1
fi
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
    --env PYTHONPATH=/workspace/Magi-DSA \
    --volume "$aot_dir:/dsa-pack-aot:ro" \
    --volume "$repo_root:/workspace/Magi-DSA:ro" \
    --workdir /workspace/Magi-DSA \
    "$image" \
    scripts/test/prewarm_dsa_pack.py \
    2>&1 | tee "$artifact_dir/validate.log"

echo "aot_object_count=$object_count" | tee "$artifact_dir/SUMMARY.txt"
find "$aot_dir" -maxdepth 1 -type f -print0 | sort -z | xargs -0 -r sha256sum \
    >"$artifact_dir/AOT_MANIFEST.sha256"
