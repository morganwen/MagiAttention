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
    echo "Usage: $0 --revision COMMIT --profile-artifact DIR --cp1-artifact DIR --cp2-artifact DIR [--image TAG]" >&2
}

revision=""
image=""
profile_artifact=""
cp1_artifact=""
cp2_artifact=""
while (($# > 0)); do
    case "$1" in
        --revision)
            revision="${2:-}"
            shift 2
            ;;
        --image)
            image="${2:-}"
            shift 2
            ;;
        --profile-artifact)
            profile_artifact="${2:-}"
            shift 2
            ;;
        --cp1-artifact)
            cp1_artifact="${2:-}"
            shift 2
            ;;
        --cp2-artifact)
            cp2_artifact="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    usage
    exit 2
fi
if [[ -z "$profile_artifact" || -z "$cp1_artifact" || -z "$cp2_artifact" ]]; then
    usage
    exit 2
fi
if [[ -z "$image" ]]; then
    image="magi-dsa-v4:release-${revision:0:12}"
fi

repo_root="$(git rev-parse --show-toplevel)"
if [[ "$(git -C "$repo_root" rev-parse HEAD)" != "$revision" ]]; then
    echo "HEAD does not match release revision $revision" >&2
    exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
    echo "release requires a clean worktree" >&2
    exit 1
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-dsa-v4-release"
artifact_dir="$repo_root/artifacts/release/$run_id"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite release artifact: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"

release_failed() {
    status=$?
    if ((status != 0)); then
        printf 'result=FAIL\nexit_status=%s\n' "$status" >"$artifact_dir/FAILED.txt"
    fi
    return "$status"
}
trap release_failed EXIT

{
    echo "bash scripts/image/build.sh --revision $revision --tag $image"
    echo "bash scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward --image $image --installed-wheel"
    echo "python3 scripts/image/finalize_release.py --artifact-dir $artifact_dir --revision $revision --image $image --correctness-artifact <generated> --profile-artifact $profile_artifact --cp1-artifact $cp1_artifact --cp2-artifact $cp2_artifact"
} >"$artifact_dir/COMMAND.txt"

echo "release stage=image-build revision=$revision image=$image"
bash "$repo_root/scripts/image/build.sh" \
    --revision "$revision" \
    --tag "$image" \
    2>&1 | tee "$artifact_dir/BUILD.log"

echo "release stage=installed-wheel-cp8 image=$image"
bash "$repo_root/scripts/test/run_multigpu.sh" \
    --world-size 8 \
    --case cp8-natural-backward \
    --image "$image" \
    --installed-wheel \
    2>&1 | tee "$artifact_dir/CP8.log"

correctness_artifact="$(
    awk -F= '$1 == "artifact_dir" {print substr($0, length($1) + 2); exit}' \
        "$artifact_dir/CP8.log"
)"
if [[ -z "$correctness_artifact" || ! -d "$correctness_artifact" ]]; then
    echo "could not resolve installed-wheel CP8 artifact from CP8.log" >&2
    exit 1
fi

echo "release stage=finalize correctness_artifact=$correctness_artifact"
python3 "$repo_root/scripts/image/finalize_release.py" \
    --artifact-dir "$artifact_dir" \
    --revision "$revision" \
    --image "$image" \
    --correctness-artifact "$correctness_artifact" \
    --profile-artifact "$profile_artifact" \
    --cp1-artifact "$cp1_artifact" \
    --cp2-artifact "$cp2_artifact" \
    | tee "$artifact_dir/FINALIZE.stdout"

trap - EXIT
echo "release complete: $artifact_dir"
