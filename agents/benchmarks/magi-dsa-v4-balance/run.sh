#!/usr/bin/env bash
# Run the frozen benchmark from the installed wheel in an immutable image.
# Only artifact directories are mounted; the host source checkout is never
# mounted into the container.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
ACTION=${1:-}

usage() {
  echo "usage: MAGI_DSA_IMAGE=IMAGE RUN_ID=ID $0 {packs|calibrate|measure|profile|validate}" >&2
}

if [[ -z ${ACTION} ]]; then
  usage
  exit 2
fi
if [[ ! ${ACTION} =~ ^(packs|calibrate|measure|profile|validate)$ ]]; then
  usage
  echo "unknown action: ${ACTION}" >&2
  exit 2
fi

IMAGE=${MAGI_DSA_IMAGE:?set MAGI_DSA_IMAGE to the immutable production image}
RUN_ID=${RUN_ID:?set RUN_ID to a unique artifact run ID}
if [[ ! ${RUN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
fi

IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${IMAGE}")
REVISION=$(docker image inspect \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "${IMAGE_ID}")
if [[ ! ${IMAGE_ID} =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "image is not content addressed: ${IMAGE_ID}" >&2
  exit 1
fi
if [[ ! ${REVISION} =~ ^[0-9a-f]{40}$ ]]; then
  echo "image has no full revision label: ${REVISION}" >&2
  exit 1
fi
if [[ -n ${EXPECTED_REVISION:-} && ${EXPECTED_REVISION} != "${REVISION}" ]]; then
  echo "EXPECTED_REVISION ${EXPECTED_REVISION} != image revision ${REVISION}" >&2
  exit 1
fi

PACKS_DIR=${PACKS_DIR:-"${REPO_ROOT}/agents/perf/magi-dsa-v4-balance/packs"}
PERF_DIR=${PERF_DIR:-"${REPO_ROOT}/agents/perf/magi-dsa-v4-balance/${RUN_ID}"}
PROFILE_DIR=${PROFILE_DIR:-"${REPO_ROOT}/agents/profiles/magi-dsa-v4-balance/${RUN_ID}"}
mkdir -p "${PACKS_DIR}" "${PERF_DIR}" "${PROFILE_DIR}"
PACKS_FILE=${PACKS_FILE:-"${PACKS_DIR}/packs-seed42.json"}
if [[ ${PACKS_FILE} != "${PACKS_DIR}"/* ]]; then
  echo "PACKS_FILE must be inside PACKS_DIR" >&2
  exit 2
fi
PACKS_NAME=${PACKS_FILE#"${PACKS_DIR}/"}

DRIVER=/opt/MagiAttention/agents/benchmarks/magi-dsa-v4-balance/driver.py
VALIDATOR=/opt/MagiAttention/agents/benchmarks/magi-dsa-v4-balance/validate.py
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-86400}

docker_run() {
  timeout --signal=TERM --kill-after=60 "${TIMEOUT_SECONDS}" \
    docker run --rm \
      --gpus all \
      --ipc=host \
      --network=host \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" \
      --env HOME=/tmp \
      --env "USER=$(id -un)" \
      --env "LOGNAME=$(id -un)" \
      --env PYTHONNOUSERSITE=1 \
      --env MAGI_ATTENTION_NATIVE_GRPCOLL=1 \
      --env MAGI_ATTENTION_HIERARCHICAL_COMM=0 \
      --env CUDA_DEVICE_MAX_CONNECTIONS=8 \
      --volume "${PACKS_DIR}:/magi-packs" \
      --volume "${PERF_DIR}:/magi-perf" \
      --volume "${PROFILE_DIR}:/magi-profile" \
      --workdir /opt/magi-runtime \
      "${IMAGE_ID}" "$@"
}

generate_packs() {
  docker_run python "${DRIVER}" packs \
    --output "/magi-packs/${PACKS_NAME}"
}

require_packs() {
  if [[ ! -f ${PACKS_FILE} ]]; then
    generate_packs
  fi
}

run_distributed() {
  local command=$1
  local run_dir=$2
  docker_run torchrun --standalone --nproc_per_node=8 \
    "${DRIVER}" "${command}" \
    --run-id "${RUN_ID}" \
    --run-dir "${run_dir}" \
    --packs "/magi-packs/${PACKS_NAME}" \
    --expected-revision "${REVISION}" \
    --expected-image-id "${IMAGE_ID}"
}

case "${ACTION}" in
  packs)
    generate_packs
    ;;
  calibrate)
    require_packs
    run_distributed calibrate /magi-perf
    ;;
  measure)
    require_packs
    run_distributed measure /magi-perf
    ;;
  profile)
    require_packs
    docker_run nsys profile \
      --force-overwrite=true \
      --trace=cuda,nvtx \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --output=/magi-profile/ratio4_pack0 \
      torchrun --standalone --nproc_per_node=8 \
        "${DRIVER}" profile \
        --run-id "${RUN_ID}" \
        --run-dir /magi-profile \
        --packs "/magi-packs/${PACKS_NAME}" \
        --expected-revision "${REVISION}" \
        --expected-image-id "${IMAGE_ID}"
    ;;
  validate)
    docker_run python "${VALIDATOR}" \
      --mode measure \
      --run-dir /magi-perf \
      --profile-dir /magi-profile \
      --expected-revision "${REVISION}" \
      --expected-image-id "${IMAGE_ID}" \
      --write-summary \
      --write-manifest \
      --verify-manifest
    ;;
esac
