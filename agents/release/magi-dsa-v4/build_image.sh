#!/usr/bin/env bash
# Build a content-addressed Magi_DSA production image from Git objects only.
# The user's current worktree is never reset, stashed, cleaned, or used as a
# Docker COPY source.  Exact submodule archives are expanded into a temporary
# context, preventing ignored extensions and build caches from leaking in.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)

REVISION=HEAD
TAG=""
OUTPUT_DIR=""
RUN_FINAL_MATRIX=0
KEEP_CONTEXT=0

usage() {
  echo "usage: $0 [--revision REV] [--tag TAG] [--output-dir DIR] [--run-final-matrix] [--keep-context]" >&2
}

while (($#)); do
  case "$1" in
    --revision)
      REVISION=${2:?missing value for --revision}
      shift 2
      ;;
    --tag)
      TAG=${2:?missing value for --tag}
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR=${2:?missing value for --output-dir}
      shift 2
      ;;
    --run-final-matrix)
      RUN_FINAL_MATRIX=1
      shift
      ;;
    --keep-context)
      KEEP_CONTEXT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

REVISION=$(git -C "${REPO_ROOT}" rev-parse --verify "${REVISION}^{commit}")
if [[ ! ${REVISION} =~ ^[0-9a-f]{40}$ ]]; then
  echo "revision did not resolve to a full commit: ${REVISION}" >&2
  exit 1
fi
SOURCE_DATE_EPOCH=$(git -C "${REPO_ROOT}" show -s --format=%ct "${REVISION}")
PACKAGE_VERSION="1.1.1+dsa.${REVISION:0:12}"
TAG=${TAG:-"magi-dsa-v4-b300:${REVISION:0:12}"}
if [[ -z ${OUTPUT_DIR} ]]; then
  RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  OUTPUT_DIR="${REPO_ROOT}/agents/perf/magi-dsa-v4-release/${REVISION}-${RUN_STAMP}"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/magi-dsa-image.XXXXXXXX")
CONTEXT_DIR="${TMP_ROOT}/context"
SUBMODULE_MANIFEST="${TMP_ROOT}/submodules.txt"
mkdir -p "${CONTEXT_DIR}"
: > "${SUBMODULE_MANIFEST}"

cleanup() {
  if ((KEEP_CONTEXT)); then
    echo "kept clean Docker context: ${CONTEXT_DIR}" >&2
  else
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT

export_git_tree() {
  local repo=$1
  local commit=$2
  local destination=$3
  local prefix=$4
  local gitmodules
  local key
  local path
  local child_commit
  local child_repo

  mkdir -p "${destination}"
  git -C "${repo}" archive --format=tar "${commit}" | tar -xf - -C "${destination}"
  gitmodules=$(mktemp "${TMP_ROOT}/gitmodules.XXXXXXXX")
  if git -C "${repo}" show "${commit}:.gitmodules" > "${gitmodules}" 2>/dev/null; then
    while read -r key path; do
      [[ -n ${path} ]] || continue
      child_commit=$(git -C "${repo}" ls-tree "${commit}" -- "${path}" | awk '$1 == "160000" {print $3}')
      if [[ ! ${child_commit} =~ ^[0-9a-f]{40}$ ]]; then
        echo "cannot resolve gitlink ${prefix}${path} at ${commit}" >&2
        exit 1
      fi
      child_repo="${repo}/${path}"
      if ! git -C "${child_repo}" cat-file -e "${child_commit}^{commit}" 2>/dev/null; then
        echo "submodule object is unavailable: ${prefix}${path}@${child_commit}" >&2
        echo "initialize the frozen submodules in a disposable worktree, then retry" >&2
        exit 1
      fi
      printf '%s  %s%s\n' "${child_commit}" "${prefix}" "${path}" >> "${SUBMODULE_MANIFEST}"
      export_git_tree \
        "${child_repo}" \
        "${child_commit}" \
        "${destination}/${path}" \
        "${prefix}${path}/"
    done < <(git config -f "${gitmodules}" --get-regexp '^submodule\..*\.path$' || true)
  fi
  rm -f "${gitmodules}"
}

echo "exporting clean source revision ${REVISION}"
export_git_tree "${REPO_ROOT}" "${REVISION}" "${CONTEXT_DIR}" ""
printf '%s\n' "${REVISION}" > "${CONTEXT_DIR}/.magi-source-revision"
LC_ALL=C sort -u "${SUBMODULE_MANIFEST}" > "${CONTEXT_DIR}/.magi-submodules"

DOCKERFILE="${CONTEXT_DIR}/agents/release/magi-dsa-v4/Dockerfile.production"
if [[ ! -f ${DOCKERFILE} ]]; then
  echo "${REVISION} does not contain ${DOCKERFILE#${CONTEXT_DIR}/}" >&2
  echo "commit the release infrastructure before building the pinned image" >&2
  exit 1
fi

BUILD_LOG="${OUTPUT_DIR}/docker-build.log"
echo "building ${TAG} from archive context (log: ${BUILD_LOG})"
docker build \
  --no-cache \
  --pull=false \
  --progress=plain \
  --build-arg "MAGI_ATTENTION_REVISION=${REVISION}" \
  --build-arg "MAGI_ATTENTION_PACKAGE_VERSION=${PACKAGE_VERSION}" \
  --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
  --tag "${TAG}" \
  --file "${DOCKERFILE}" \
  "${CONTEXT_DIR}" 2>&1 | tee "${BUILD_LOG}"

IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${TAG}")
if [[ ! ${IMAGE_ID} =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "docker returned a non-content-addressed image ID: ${IMAGE_ID}" >&2
  exit 1
fi
LABEL_REVISION=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "${IMAGE_ID}")
LABEL_VERSION=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' "${IMAGE_ID}")
LABEL_BASE=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.base.digest"}}' "${IMAGE_ID}")
if [[ ${LABEL_REVISION} != "${REVISION}" ]]; then
  echo "image revision label mismatch: ${LABEL_REVISION} != ${REVISION}" >&2
  exit 1
fi
if [[ ${LABEL_VERSION} != "${PACKAGE_VERSION}" ]]; then
  echo "image package-version label mismatch: ${LABEL_VERSION} != ${PACKAGE_VERSION}" >&2
  exit 1
fi
if [[ ${LABEL_BASE} != "sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1" ]]; then
  echo "image base digest label mismatch: ${LABEL_BASE}" >&2
  exit 1
fi

printf '%s\n' "${IMAGE_ID}" > "${OUTPUT_DIR}/image_id.txt"
printf '%s\n' "${PACKAGE_VERSION}" > "${OUTPUT_DIR}/package_version.txt"
docker image inspect "${IMAGE_ID}" > "${OUTPUT_DIR}/image_inspect.json"

# This smoke has no source bind mount.  It proves the wheel and native modules
# load on exactly eight SM103 devices before the expensive matrix begins.
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env "USER=$(id -un)" \
  --env "LOGNAME=$(id -un)" \
  --env "MAGI_DSA_IMAGE_ID=${IMAGE_ID}" \
  --workdir /tmp \
  "${IMAGE_ID}" \
  python -c 'import os,pathlib,torch,magi_attention,nvidia.nvshmem; import magi_attention.magi_attn_comm; from magi_attention.api import MagiDSAConfig,MagiDSARuntimeMgr,calc_dsa; assert torch.cuda.device_count()==8; assert all(torch.cuda.get_device_capability(i)==(10,3) for i in range(8)); assert not pathlib.Path(magi_attention.__file__).resolve().is_relative_to(pathlib.Path("/opt/MagiAttention")); assert magi_attention.__version__ == os.environ["MAGI_ATTENTION_PACKAGE_VERSION"]; print(f"installed native wheel smoke: PASS ({magi_attention.__version__})")' \
  > "${OUTPUT_DIR}/installed-wheel-smoke.log" 2>&1

if ((RUN_FINAL_MATRIX)); then
  MATRIX_OUTPUT="${OUTPUT_DIR}/final-matrix"
  mkdir -p "${MATRIX_OUTPUT}"
  docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    --network host \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --env "USER=$(id -un)" \
    --env "LOGNAME=$(id -un)" \
    --env "MAGI_DSA_IMAGE_ID=${IMAGE_ID}" \
    --env "MAGI_DSA_EXPECTED_REVISION=${REVISION}" \
    --volume "${MATRIX_OUTPUT}:/artifacts" \
    --workdir /opt/magi-runtime \
    "${IMAGE_ID}" \
    python /opt/MagiAttention/agents/release/magi-dsa-v4/run_final_matrix.py \
      --matrix /opt/MagiAttention/agents/release/magi-dsa-v4/final_matrix.json \
      --output-dir /artifacts \
      --image-id "${IMAGE_ID}" \
      --expected-revision "${REVISION}"
fi

(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name artifact_manifest.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > artifact_manifest.sha256
)

echo "production image: ${IMAGE_ID}"
echo "build evidence: ${OUTPUT_DIR}"
