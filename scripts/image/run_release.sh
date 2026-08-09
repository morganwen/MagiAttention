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
    echo "Usage: $0 --revision COMMIT [--image TAG] [--profile-artifact DIR --cp1-artifact DIR --cp2-artifact DIR]" >&2
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
provided_artifacts=0
for artifact in "$profile_artifact" "$cp1_artifact" "$cp2_artifact"; do
    if [[ -n "$artifact" ]]; then
        provided_artifacts=$((provided_artifacts + 1))
    fi
done
if ((provided_artifacts != 0 && provided_artifacts != 3)); then
    usage
    exit 2
fi
generate_artifacts=0
if ((provided_artifacts == 0)); then
    generate_artifacts=1
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

resolve_artifact_dir() {
    local log_path="$1"
    awk -F= '$1 == "artifact_dir" {print substr($0, length($1) + 2); exit}' \
        "$log_path"
}

require_artifact_image_id() {
    local artifact_kind="$1"
    local artifact_path="$2"
    local expected_image_id="$3"
    python3 - "$artifact_kind" "$artifact_path" "$expected_image_id" <<'PY'
import json
import pathlib
import sys

kind = sys.argv[1]
artifact = pathlib.Path(sys.argv[2]).resolve()
expected = sys.argv[3]
if not artifact.is_dir():
    raise SystemExit(f"missing {kind} artifact directory: {artifact}")
if kind == "cp1":
    actual = json.loads((artifact / "SUMMARY.json").read_text(encoding="utf-8"))[
        "image_id"
    ]
elif kind in {"cp2", "cp8"}:
    first_field = (artifact / "IMAGE.txt").read_text(encoding="utf-8").split(maxsplit=1)[
        0
    ]
    actual = json.loads(first_field)
elif kind == "profile":
    image_records = json.loads(
        (artifact / "IMAGE.json").read_text(encoding="utf-8")
    )
    if not isinstance(image_records, list) or len(image_records) != 1:
        raise SystemExit("profile IMAGE.json does not contain one image record")
    actual = image_records[0]["Id"]
else:
    raise SystemExit(f"unsupported artifact kind: {kind}")
if actual != expected:
    raise SystemExit(
        f"{kind} artifact image ID differs: expected {expected}, found {actual}"
    )
print(f"{kind}_image_id={actual}")
PY
}

{
    echo "bash scripts/image/build.sh --revision $revision --tag $image"
    if ((generate_artifacts == 1)); then
        echo "bash scripts/test/run_cp1.sh --image $image"
        echo "bash scripts/test/run_multigpu.sh --world-size 2 --case csa-natural-backward --image $image --installed-wheel"
    else
        echo "reuse_cp1_artifact=$cp1_artifact"
        echo "reuse_cp2_artifact=$cp2_artifact"
        echo "reuse_profile_artifact=$profile_artifact"
    fi
    echo "bash scripts/test/run_multigpu.sh --world-size 8 --case cp8-natural-backward --image $image --installed-wheel"
    if ((generate_artifacts == 1)); then
        echo "bash scripts/profile/run_5step.sh --world-size 8 --cp-size 8 --case dsv4-pro-128k --plans balanced --steps 5 --step-mode pro-pair --layout-policy structural-balanced --skip-smoke --image $image"
    fi
    echo "python3 scripts/image/finalize_release.py --artifact-dir $artifact_dir --revision $revision --image $image --correctness-artifact <generated> --profile-artifact <resolved> --cp1-artifact <resolved> --cp2-artifact <resolved>"
} >"$artifact_dir/COMMAND.txt"

echo "release stage=image-build revision=$revision image=$image"
bash "$repo_root/scripts/image/build.sh" \
    --revision "$revision" \
    --tag "$image" \
    2>&1 | tee "$artifact_dir/BUILD.log"

release_image_id="$(docker image inspect "$image" --format '{{.Id}}')"
if [[ ! "$release_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "release image has an invalid ID: $release_image_id" >&2
    exit 1
fi
printf '%s\n' "$release_image_id" >"$artifact_dir/IMAGE_ID.txt"

if ((generate_artifacts == 1)); then
    echo "release stage=cp1 image=$image image_id=$release_image_id"
    bash "$repo_root/scripts/test/run_cp1.sh" \
        --image "$image" \
        2>&1 | tee "$artifact_dir/CP1.log"
    cp1_artifact="$(resolve_artifact_dir "$artifact_dir/CP1.log")"
    if [[ -z "$cp1_artifact" || ! -d "$cp1_artifact" ]]; then
        echo "could not resolve CP1 artifact from CP1.log" >&2
        exit 1
    fi

    echo "release stage=installed-wheel-cp2 image=$image image_id=$release_image_id"
    bash "$repo_root/scripts/test/run_multigpu.sh" \
        --world-size 2 \
        --case csa-natural-backward \
        --image "$image" \
        --installed-wheel \
        2>&1 | tee "$artifact_dir/CP2.log"
    cp2_artifact="$(resolve_artifact_dir "$artifact_dir/CP2.log")"
    if [[ -z "$cp2_artifact" || ! -d "$cp2_artifact" ]]; then
        echo "could not resolve installed-wheel CP2 artifact from CP2.log" >&2
        exit 1
    fi
fi

echo "release stage=installed-wheel-cp8 image=$image"
bash "$repo_root/scripts/test/run_multigpu.sh" \
    --world-size 8 \
    --case cp8-natural-backward \
    --image "$image" \
    --installed-wheel \
    2>&1 | tee "$artifact_dir/CP8.log"

correctness_artifact="$(resolve_artifact_dir "$artifact_dir/CP8.log")"
if [[ -z "$correctness_artifact" || ! -d "$correctness_artifact" ]]; then
    echo "could not resolve installed-wheel CP8 artifact from CP8.log" >&2
    exit 1
fi

if ((generate_artifacts == 1)); then
    echo "release stage=pro-pair-profile image=$image image_id=$release_image_id"
    bash "$repo_root/scripts/profile/run_5step.sh" \
        --world-size 8 \
        --cp-size 8 \
        --case dsv4-pro-128k \
        --plans balanced \
        --steps 5 \
        --step-mode pro-pair \
        --layout-policy structural-balanced \
        --skip-smoke \
        --image "$image" \
        2>&1 | tee "$artifact_dir/PROFILE.log"
    profile_artifact="$(resolve_artifact_dir "$artifact_dir/PROFILE.log")"
    if [[ -z "$profile_artifact" || ! -d "$profile_artifact" ]]; then
        echo "could not resolve Pro-pair profile artifact from PROFILE.log" >&2
        exit 1
    fi
fi

current_image_id="$(docker image inspect "$image" --format '{{.Id}}')"
if [[ "$current_image_id" != "$release_image_id" ]]; then
    echo "release image tag changed during validation: $release_image_id -> $current_image_id" >&2
    exit 1
fi
require_artifact_image_id cp1 "$cp1_artifact" "$release_image_id" \
    | tee "$artifact_dir/CP1_IMAGE_ID.txt"
require_artifact_image_id cp2 "$cp2_artifact" "$release_image_id" \
    | tee "$artifact_dir/CP2_IMAGE_ID.txt"
require_artifact_image_id cp8 "$correctness_artifact" "$release_image_id" \
    | tee "$artifact_dir/CP8_IMAGE_ID.txt"
require_artifact_image_id profile "$profile_artifact" "$release_image_id" \
    | tee "$artifact_dir/PROFILE_IMAGE_ID.txt"
{
    echo "resolved_cp1_artifact=$cp1_artifact"
    echo "resolved_cp2_artifact=$cp2_artifact"
    echo "resolved_cp8_artifact=$correctness_artifact"
    echo "resolved_profile_artifact=$profile_artifact"
    echo "release_image_id=$release_image_id"
} >>"$artifact_dir/COMMAND.txt"

echo "release stage=finalize correctness_artifact=$correctness_artifact"
python3 "$repo_root/scripts/image/finalize_release.py" \
    --artifact-dir "$artifact_dir" \
    --revision "$revision" \
    --image "$image" \
    --correctness-artifact "$correctness_artifact" \
    --profile-artifact "$profile_artifact" \
    --cp1-artifact "$cp1_artifact" \
    --cp2-artifact "$cp2_artifact"

trap - EXIT
echo "release complete: $artifact_dir"
