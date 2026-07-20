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
    echo "Usage: $0 --plan {sequential|balanced} --artifact-dir DIR --world-size 8 --steps 5 --tokens 131072 --warmup N" >&2
}

plan=""
artifact_dir=""
world_size=""
steps=""
tokens=""
warmup=""
while (($# > 0)); do
    case "$1" in
        --plan)
            plan="${2:-}"
            shift 2
            ;;
        --artifact-dir)
            artifact_dir="${2:-}"
            shift 2
            ;;
        --world-size)
            world_size="${2:-}"
            shift 2
            ;;
        --steps)
            steps="${2:-}"
            shift 2
            ;;
        --tokens)
            tokens="${2:-}"
            shift 2
            ;;
        --warmup)
            warmup="${2:-}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ "$plan" != "sequential" && "$plan" != "balanced" ]]; then
    usage
    exit 2
fi
if [[ "$world_size" != "8" || "$steps" != "5" || "$tokens" != "131072" ]]; then
    usage
    exit 2
fi
if [[ ! "$warmup" =~ ^[1-9][0-9]*$ || -z "$artifact_dir" ]]; then
    usage
    exit 2
fi
if [[ ! -d "$artifact_dir" ]]; then
    echo "profile artifact directory does not exist: $artifact_dir" >&2
    exit 1
fi

control_dir="$artifact_dir/control"
mkdir -p "$control_dir"
report_base="$artifact_dir/${plan}_5steps"
report_path="${report_base}.nsys-rep"
sqlite_path="${report_base}.sqlite"
if [[ -e "$report_path" || -e "$sqlite_path" || -e "$control_dir/start" ]]; then
    echo "refusing to overwrite an existing profile capture in $artifact_dir" >&2
    exit 1
fi

session="magi_dsa_${plan}_$$_$(date -u +%Y%m%dT%H%M%SZ)"
launch_log="$artifact_dir/NSYS_LAUNCH.log"
start_log="$artifact_dir/NSYS_START.log"
stop_log="$artifact_dir/NSYS_STOP.log"
launch_pid=""
launch_pgid=""
capture_started=0

terminate_launch_group() {
    if [[ -n "$launch_pid" ]] && kill -0 "$launch_pid" 2>/dev/null; then
        echo "terminating recorded nsys launch process group pid=$launch_pid pgid=$launch_pgid" >&2
        kill -TERM -- "-$launch_pgid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            if ! kill -0 "$launch_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$launch_pid" 2>/dev/null; then
            kill -KILL -- "-$launch_pgid" 2>/dev/null || true
        fi
    fi
}

cleanup() {
    status=$?
    if ((status != 0)); then
        if ((capture_started == 1)); then
            timeout --signal=TERM --kill-after=5s 30s nsys stop --session="$session" \
                >>"$stop_log" 2>&1 || true
        fi
        terminate_launch_group
    fi
    return "$status"
}
trap cleanup EXIT INT TERM

{
    echo "session=$session"
    echo "plan=$plan"
    echo "nsys launch --session-new=$session --trace=cuda,nvtx --resolve-symbols=false --wait=all torchrun --standalone --nnodes=1 --nproc-per-node=$world_size benchmarks/dsa_v4/profile_5step.py --mode profile --plan $plan --artifact-dir $artifact_dir --seed 0 --tokens $tokens --steps $steps --warmup $warmup"
} >"$artifact_dir/PLAN_COMMAND.txt"

setsid nsys launch \
    --session-new="$session" \
    --trace=cuda,nvtx \
    --resolve-symbols=false \
    --wait=all \
    /usr/local/bin/torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="$world_size" \
        /workspace/Magi-DSA/benchmarks/dsa_v4/profile_5step.py \
        --mode profile \
        --plan "$plan" \
        --artifact-dir "$artifact_dir" \
        --seed 0 \
        --tokens "$tokens" \
        --steps "$steps" \
        --warmup "$warmup" \
    >"$launch_log" 2>&1 &
launch_pid="$!"
launch_pgid="$(ps -o pgid= -p "$launch_pid" | tr -d ' ')"
echo "nsys launch started pid=$launch_pid pgid=$launch_pgid session=$session" | tee -a "$launch_log"

ready_deadline=$((SECONDS + 1500))
while true; do
    ready_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'ready_rank*.json' | wc -l)"
    if ((ready_count == world_size)); then
        break
    fi
    if find "$artifact_dir" -maxdepth 1 -type f -name 'failure_rank*.json' | grep -q .; then
        echo "profile worker failed during setup/prewarm" >&2
        exit 1
    fi
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        wait "$launch_pid" || true
        echo "nsys launch exited before all ranks became ready" >&2
        exit 1
    fi
    if ((SECONDS >= ready_deadline)); then
        echo "profile setup/prewarm exceeded its 1500 second deadline" >&2
        exit 1
    fi
    sleep 1
done

timeout --signal=TERM --kill-after=5s 60s nsys start \
    --session="$session" \
    --capture-range=none \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=false \
    --output="$report_base" \
    >"$start_log" 2>&1
capture_started=1
sleep 5
date -u +%Y-%m-%dT%H:%M:%SZ >"$control_dir/start"

capture_deadline=$((SECONDS + 1500))
while true; do
    done_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'capture_done_rank*.json' | wc -l)"
    if ((done_count == world_size)); then
        break
    fi
    if find "$artifact_dir" -maxdepth 1 -type f -name 'failure_rank*.json' | grep -q .; then
        echo "profile worker failed during five-step capture" >&2
        exit 1
    fi
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        wait "$launch_pid" || true
        echo "nsys launch exited during five-step capture" >&2
        exit 1
    fi
    if ((SECONDS >= capture_deadline)); then
        echo "five-step capture exceeded its 1500 second deadline" >&2
        exit 1
    fi
    sleep 1
done

sleep 2
timeout --signal=TERM --kill-after=5s 600s nsys stop --session="$session" >"$stop_log" 2>&1
capture_started=0
date -u +%Y-%m-%dT%H:%M:%SZ >"$control_dir/capture_stopped"

worker_deadline=$((SECONDS + 600))
while kill -0 "$launch_pid" 2>/dev/null; do
    if ((SECONDS >= worker_deadline)); then
        echo "post-capture shadow verification exceeded 600 seconds" >&2
        exit 1
    fi
    sleep 1
done
wait "$launch_pid"
launch_pid=""
launch_pgid=""

result_count="$(find "$artifact_dir" -maxdepth 1 -type f -name 'result_rank*.json' | wc -l)"
if ((result_count != world_size)); then
    echo "profile produced $result_count rank results, expected $world_size" >&2
    exit 1
fi
if [[ ! -s "$report_path" ]]; then
    echo "Nsight did not produce the expected report: $report_path" >&2
    exit 1
fi

timeout --signal=TERM --kill-after=5s 600s nsys export \
    --type sqlite \
    --force-overwrite=false \
    --output "$sqlite_path" \
    "$report_path" \
    >"$artifact_dir/NSYS_EXPORT.log" 2>&1
timeout --signal=TERM --kill-after=5s 600s nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    "$report_path" \
    >"$artifact_dir/NSYS_STATS.txt" 2>&1

python /workspace/Magi-DSA/scripts/profile/extract_nsys.py \
    --plan "$plan" \
    --report "$report_path" \
    --sqlite "$sqlite_path" \
    --output-dir "$artifact_dir" \
    --steps "$steps" \
    --world-size "$world_size"

echo "profile plan complete: plan=$plan report=$report_path"
