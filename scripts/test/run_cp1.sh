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
    echo "Usage: $0 [--image IMAGE]" >&2
}

image="${MAGI_DSA_IMAGE:-magi-dsa-v4:preflight-68c2f15}"
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

repo_root="$(git rev-parse --show-toplevel)"
# Magi-DSA ships as a second distribution, so a valid CP1 checkout has to carry
# both. This replaces the old hardcoded host path with a structural check.
if [[ ! -f "$repo_root/pyproject.toml" || ! -f "$repo_root/extensions/setup.py" ]]; then
    echo "CP1 artifact generation requires a checkout with both the Core and the Extension distribution: $repo_root" >&2
    exit 1
fi

source_revision="$(git -C "$repo_root" rev-parse HEAD)"
dirty_status="$(git -C "$repo_root" status --short)"
if [[ -n "$dirty_status" ]]; then
    echo "CP1 artifact generation requires a clean worktree" >&2
    exit 1
fi
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
packaging_version="25.0"
pytest_version="8.4.2"

image_label() {
    docker image inspect "$image" --format "{{ index .Config.Labels \"$1\" }}"
}

require_image_label() {
    local label="$1"
    local expected="$2"
    local actual
    actual="$(image_label "$label")"
    if [[ "$actual" != "$expected" ]]; then
        echo "CP1 image label mismatch for $label: expected $expected, found ${actual:-<missing>}" >&2
        exit 1
    fi
}

require_image_label "org.magi-dsa.flashmla-revision" "$flashmla_base_revision"
require_image_label \
    "org.magi-dsa.flashmla-dual-lse-patch-revision" "$flashmla_patch_revision"
require_image_label \
    "org.magi-dsa.flashmla-dual-lse-patch-sha256" "$flashmla_patch_sha256"
require_image_label \
    "org.magi-dsa.flashmla-pro-h128-patch-revision" "$flashmla_pro_patch_revision"
require_image_label \
    "org.magi-dsa.flashmla-pro-h128-patch-sha256" "$flashmla_pro_patch_sha256"
require_image_label "org.magi-dsa.cudnn-backend" "$cudnn_backend_version"
require_image_label "org.magi-dsa.cudnn-frontend" "$cudnn_frontend_version"
require_image_label \
    "org.magi-dsa.cudnn-frontend-revision" "$cudnn_frontend_revision"
require_image_label "org.magi-dsa.cudnn-frontend-source" "official-unmodified"
require_image_label "org.magi-dsa.cudnn-frontend-local-patches" "none"
require_image_label "org.magi-dsa.cutlass-dsl" "$cutlass_dsl_version"
require_image_label "org.magi-dsa.quack-kernels" "$quack_version"
require_image_label "org.magi-dsa.tvm-ffi" "$tvm_ffi_version"
require_image_label "org.magi-dsa.magi-attention-revision" "$source_revision"
require_image_label "org.magi-dsa.install-mode" "python-wheel"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-cp1-cp1-kernel"
artifact_dir="$repo_root/artifacts/correctness/$run_id"
cache_dir="$repo_root/.cache/magi-dsa-v4/cp1-$source_revision"
if [[ -e "$artifact_dir" ]]; then
    echo "refusing to overwrite CP1 artifact directory: $artifact_dir" >&2
    exit 1
fi
mkdir -p \
    "$artifact_dir" \
    "$cache_dir/cuda" \
    "$cache_dir/cute-dsl" \
    "$cache_dir/magi-workspace" \
    "$cache_dir/quack" \
    "$cache_dir/pytest-root" \
    "$cache_dir/pytest-tmp" \
    "$cache_dir/torch-extensions" \
    "$cache_dir/torchinductor" \
    "$cache_dir/triton" \
    "$cache_dir/xdg"
chmod 0777 \
    "$artifact_dir" \
    "$cache_dir" \
    "$cache_dir/cuda" \
    "$cache_dir/cute-dsl" \
    "$cache_dir/magi-workspace" \
    "$cache_dir/quack" \
    "$cache_dir/pytest-root" \
    "$cache_dir/pytest-tmp" \
    "$cache_dir/torch-extensions" \
    "$cache_dir/torchinductor" \
    "$cache_dir/triton" \
    "$cache_dir/xdg"

failed=1
finish() {
    status=$?
    if ((failed == 1 && status != 0)); then
        printf 'result=FAIL\nexit_status=%s\n' "$status" >"$artifact_dir/FAILED.txt"
    fi
    return "$status"
}
trap finish EXIT

image_id="$(docker image inspect "$image" --format '{{.Id}}')"
{
    echo "artifact_dir=$artifact_dir"
    echo "case=cp1-kernel"
    echo "image=$image"
    echo "source_revision=$source_revision"
    echo "timeout_seconds=1800"
    echo "pytest=extensions/tests/dsa_v4/test_cp1_kernel.py"
    echo "packaging_version=$packaging_version"
    echo "pytest_version=$pytest_version"
    echo "pytest_import_mode=importlib"
    echo "pytest_plugin_autoload=disabled"
    echo "pytest_rootdir=/magi-cache/pytest-root"
    echo "pytest_basetemp=/magi-cache/pytest-tmp/$run_id"
    echo "source_mount=read-only"
    echo "package_import=installed-wheel"
} >"$artifact_dir/COMMAND.txt"
echo "artifact_dir=$artifact_dir"

printf '%s\n' "$source_revision" >"$artifact_dir/SOURCE_REVISION.txt"
git -C "$repo_root" remote get-url origin >"$artifact_dir/SOURCE_REMOTE.txt"
git -C "$repo_root" submodule status --recursive >"$artifact_dir/SUBMODULES.txt"
printf '%s\n' "$dirty_status" >"$artifact_dir/DIRTY_STATUS.txt"
docker image inspect "$image" >"$artifact_dir/IMAGE.json"
nvidia-smi -q >"$artifact_dir/HARDWARE.txt"
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
magi_source_revision=$source_revision
install_mode=python-wheel
validation=all_required_image_labels_exact
EOF

container_cache_args=(
    --env CUDA_CACHE_PATH=/magi-cache/cuda
    --env CUTE_DSL_CACHE_DIR=/magi-cache/cute-dsl
    --env MAGI_ATTENTION_WORKSPACE_BASE=/magi-cache/magi-workspace
    --env "MAGI_DSA_PACKAGING_VERSION=$packaging_version"
    --env "MAGI_DSA_PYTEST_VERSION=$pytest_version"
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONSAFEPATH=1
    --env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    --env QUACK_CACHE_DIR=/magi-cache/quack
    --env TEMP=/magi-cache/pytest-tmp
    --env TMP=/magi-cache/pytest-tmp
    --env TMPDIR=/magi-cache/pytest-tmp
    --env TORCH_EXTENSIONS_DIR=/magi-cache/torch-extensions
    --env TORCHINDUCTOR_CACHE_DIR=/magi-cache/torchinductor
    --env TRITON_CACHE_DIR=/magi-cache/triton
    --env XDG_CACHE_HOME=/magi-cache/xdg
    --volume "$cache_dir:/magi-cache"
    --workdir /magi-cache
)

timeout --signal=TERM --kill-after=5s 60s docker run --rm \
    --entrypoint bash \
    "${container_cache_args[@]}" \
    "$image" -lc 'python -VV; python -m pip freeze; nsys --version 2>&1 || true' \
    >"$artifact_dir/ENVIRONMENT.txt" 2>&1

docker run --rm \
    --entrypoint python3 \
    "${container_cache_args[@]}" \
    "$image" -c \
    'import json; import os; from importlib import metadata; import magi_attention; assert metadata.version("packaging") == os.environ["MAGI_DSA_PACKAGING_VERSION"]; assert metadata.version("pytest") == os.environ["MAGI_DSA_PYTEST_VERSION"]; print(json.dumps({"package_path": magi_attention.__file__, "package_version": metadata.version("magi-attention"), "packaging_version": metadata.version("packaging"), "pytest_version": metadata.version("pytest")}, sort_keys=True))' \
    >"$artifact_dir/INSTALLED_PACKAGE.json" \
    2>"$artifact_dir/INSTALLED_PACKAGE.stderr"

set +e
timeout --signal=TERM --kill-after=5s 1800s docker run --rm \
    --entrypoint python3 \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${container_cache_args[@]}" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PYTEST_ADDOPTS=-p\ no:cacheprovider \
    --volume "$artifact_dir:/cp1-artifact" \
    --volume "$repo_root:/workspace/MagiAttention:ro" \
    "$image" -m pytest \
    --import-mode=importlib \
    --rootdir=/magi-cache/pytest-root \
    --basetemp="/magi-cache/pytest-tmp/$run_id" \
    --junitxml=/cp1-artifact/PYTEST.xml \
    -q /workspace/MagiAttention/extensions/tests/dsa_v4/test_cp1_kernel.py \
    >"$artifact_dir/STDOUT.txt" 2>"$artifact_dir/STDERR.txt"
pytest_status=$?
set -e

set +e
python3 - "$artifact_dir" "$source_revision" "$image" "$image_id" \
    "$pytest_status" "$dirty_status" <<'PY'
import hashlib
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

artifact = pathlib.Path(sys.argv[1]).resolve()
revision = sys.argv[2]
image = sys.argv[3]
image_id = sys.argv[4]
pytest_status = int(sys.argv[5])
dirty = bool(sys.argv[6])
installed = json.loads((artifact / "INSTALLED_PACKAGE.json").read_text(encoding="utf-8"))
xml_path = artifact / "PYTEST.xml"
counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
if xml_path.is_file():
    root = ET.parse(xml_path).getroot()
    for field in counts:
        counts[field] = int(root.attrib.get(field, 0))
    if root.tag == "testsuites" and counts["tests"] == 0:
        for suite in root.findall("testsuite"):
            for field in counts:
                counts[field] += int(suite.attrib.get(field, 0))
result = (
    "PASS"
    if pytest_status == 0
    and counts["tests"] == 6
    and counts["failures"] == 0
    and counts["errors"] == 0
    and counts["skipped"] == 0
    and not dirty
    else "FAIL"
)
raw = {
    "case": "cp1-kernel",
    "pytest_exit_status": pytest_status,
    "junit_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest()
    if xml_path.is_file()
    else None,
    **counts,
}
summary = {
    "case": "cp1-kernel",
    "image": image,
    "image_contract": "PASS",
    "image_id": image_id,
    "installed_package": installed,
    "pytest_exit_status": pytest_status,
    "result": result,
    "source_dirty": dirty,
    "source_revision": revision,
    **counts,
}
(artifact / "RAW.json").write_text(
    json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(artifact / "SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
if result != "PASS":
    raise SystemExit(1)
PY
summary_status=$?
set -e

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
    if path.is_symlink():
        raise SystemExit(f"refusing symlink in CP1 artifact: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append(f"{digest}  {path.relative_to(root)}")
manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
PY
(cd "$artifact_dir" && sha256sum -c SHA256SUMS)

if ((pytest_status != 0 || summary_status != 0)); then
    echo "CP1 validation failed; preserved artifact: $artifact_dir" >&2
    exit 1
fi
failed=0
trap - EXIT
echo "CP1 complete: $artifact_dir"
