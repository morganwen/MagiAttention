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
    --env PYTHONSAFEPATH=1 \
    --workdir /opt \
    "$image_tag" \
    -c 'import os; from importlib import metadata; from pathlib import Path; import magi_attention; import magi_attn_extensions; import magi_attn_extensions.DSA as magi_dsa; from magi_attn_extensions.DSA.modeling import MagiDSALayer; from magi_attn_extensions.DSA.runtime import MagiDSARuntimeMgr; assert metadata.version("magi-attention") == "1.1.1+g" + os.environ["MAGI_DSA_SOURCE_REVISION"]; assert metadata.version("magi_attn_extensions") == os.environ["MAGI_DSA_EXTENSION_VERSION"]; assert metadata.version("packaging") == "25.0"; assert metadata.version("pytest") == "8.4.2"; assert set(Path(magi_attention.__file__).parts).intersection({"site-packages", "dist-packages"}); assert set(Path(magi_attn_extensions.__file__).parts).intersection({"site-packages", "dist-packages"}); assert set(Path(magi_dsa.__file__).parts).intersection({"site-packages", "dist-packages"}); print(metadata.version("magi-attention")); print(metadata.version("magi_attn_extensions")); print(metadata.version("packaging")); print(metadata.version("pytest")); print(magi_attention.__file__); print(magi_dsa.__file__); print(MagiDSALayer.__name__, MagiDSARuntimeMgr.__name__)'
