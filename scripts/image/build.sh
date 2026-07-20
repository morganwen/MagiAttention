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
    echo "Usage: $0 --revision <40-char-commit> [--tag <image-tag>]" >&2
}

revision=""
image_tag=""
while (($# > 0)); do
    case "$1" in
        --revision)
            revision="${2:-}"
            shift 2
            ;;
        --tag)
            image_tag="${2:-}"
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

repo_root="$(git rev-parse --show-toplevel)"
if [[ "$(git -C "$repo_root" rev-parse HEAD)" != "$revision" ]]; then
    echo "HEAD does not match --revision $revision" >&2
    exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
    echo "Refusing a release image build from a dirty worktree" >&2
    exit 1
fi

if [[ -z "$image_tag" ]]; then
    image_tag="magi-dsa-v4:${revision:0:12}"
fi

timeout --signal=TERM --kill-after=5s 1800s docker build \
    --progress=plain \
    --build-arg "MAGI_ATTENTION_REVISION=$revision" \
    --file "$repo_root/docker/Dockerfile.dsa-v4" \
    --tag "$image_tag" \
    "$repo_root"

docker image inspect "$image_tag" --format '{{.Id}} {{json .Config.Labels}}'
timeout --signal=TERM --kill-after=5s 60s docker run --rm \
    --entrypoint python3 \
    --workdir /tmp \
    "$image_tag" \
    -c 'import os; from importlib import metadata; import magi_attention; from magi_attention.dsa_layer import MagiDSALayer; from magi_attention.dsa_runtime_mgr import MagiDSARuntimeMgr; assert metadata.version("magi-attention") == "1.1.1+g" + os.environ["MAGI_DSA_SOURCE_REVISION"]; assert "/site-packages/" in magi_attention.__file__; print(metadata.version("magi-attention")); print(magi_attention.__file__); print(MagiDSALayer.__name__, MagiDSARuntimeMgr.__name__)'
