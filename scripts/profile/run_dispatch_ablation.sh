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
    echo "Usage: $0 [--image IMAGE] [--clock-mhz MHZ] [--pass4-repeats N]" >&2
}

image="${MAGI_DSA_IMAGE:-magi-dsa-v4:shared-greedy-dev}"
clock_mhz="1800"
pass4_repeats="3"
while (($# > 0)); do
    case "$1" in
        --image)
            image="${2:-}"
            shift 2
            ;;
        --clock-mhz)
            clock_mhz="${2:-}"
            shift 2
            ;;
        --pass4-repeats)
            pass4_repeats="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ ! "$clock_mhz" =~ ^[1-9][0-9]*$ || ! "$pass4_repeats" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi
if ((pass4_repeats > 8)); then
    echo "pass4 repeats must not exceed eight" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-dsv4-shared-greedy-dispatch-ablation"
artifact_dir="$repo_root/artifacts/profile/$run_id"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite dispatch-ablation artifact: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"

cat >"$artifact_dir/COMMAND.txt" <<EOF
bash scripts/profile/run_dispatch_ablation.sh --image $image --clock-mhz $clock_mhz --pass4-repeats $pass4_repeats
passes=0,1,4,8
profiler_attach_warmup_steps=1
formal_steps_per_run=5
world_size=8
EOF

declare -a summary_args=()

run_one() {
    local label="$1"
    local passes="$2"
    local log_path="$artifact_dir/${label}.log"
    set +e
    bash "$repo_root/scripts/profile/run_5step.sh" \
        --world-size 8 \
        --cp-size 8 \
        --case dsv4-flash-128k \
        --plans balanced \
        --steps 5 \
        --step-mode attention-suite \
        --layout-policy shared-greedy \
        --local-improvement-passes "$passes" \
        --profiler-attach-warmup-steps 1 \
        --lock-gpu-clock-mhz "$clock_mhz" \
        --skip-smoke \
        --image "$image" \
        2>&1 | tee "$log_path"
    local status=${PIPESTATUS[0]}
    set -e
    if ((status != 0)); then
        echo "dispatch-ablation run failed: label=$label status=$status" >&2
        exit "$status"
    fi
    local profile_dir
    profile_dir="$(sed -n 's/^profile complete: //p' "$log_path" | tail -n 1)"
    if [[ -z "$profile_dir" || ! -f "$profile_dir/SUMMARY_ATTENTION_SUITE.json" ]]; then
        echo "unable to resolve completed profile artifact for $label" >&2
        exit 1
    fi
    printf '%s=%s\n' "$label" "$profile_dir" >>"$artifact_dir/RUNS.txt"
    summary_args+=(--run "$label=$profile_dir")
}

run_one passes0 0
run_one passes1 1
run_one passes4-r0 4
run_one passes8 8
for ((repeat = 1; repeat < pass4_repeats; repeat++)); do
    run_one "passes4-r${repeat}" 4
done

python3 "$repo_root/scripts/profile/summarize_dispatch_ablation.py" \
    --output-dir "$artifact_dir" \
    "${summary_args[@]}"

git -C "$repo_root" rev-parse HEAD >"$artifact_dir/SOURCE_REVISION.txt"
git -C "$repo_root" status --short >"$artifact_dir/DIRTY_STATUS.txt"
python3 - "$artifact_dir" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest = root / "SHA256SUMS"
entries = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest:
        continue
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            hasher.update(chunk)
    entries.append(f"{hasher.hexdigest()}  {path.relative_to(root)}")
manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
PY
(
    cd "$artifact_dir"
    sha256sum -c SHA256SUMS
)
echo "dispatch ablation complete: $artifact_dir"
