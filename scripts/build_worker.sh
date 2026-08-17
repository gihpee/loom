#!/usr/bin/env bash
# Build (and optionally push) the public worker image.
#
#   scripts/build_worker.sh                 # build for THIS machine, load locally
#   scripts/build_worker.sh --push          # build linux/amd64 and push
#   scripts/build_worker.sh --push --platform linux/amd64,linux/arm64
#
# Why the platform matters: GPU hosts are almost always linux/amd64. An image
# built on Apple Silicon is linux/arm64, and pulling it on an amd64 server fails
# with "no matching manifest for linux/amd64".
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${LOOM_WORKER_IMAGE:-gihpee/loomworker:latest}"
DOCKERFILE="docker/worker.cuda.Dockerfile"
PLATFORM=""
PUSH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --push) PUSH=1; shift ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --cpu) DOCKERFILE="worker/Dockerfile.shard"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

host_arch=$(docker version --format '{{.Server.Arch}}' 2>/dev/null || uname -m)
echo "image      : ${IMAGE}"
echo "dockerfile : ${DOCKERFILE}"
echo "host arch  : ${host_arch}"

if [ "$PUSH" -eq 1 ]; then
  PLATFORM="${PLATFORM:-linux/amd64}"
  echo "platform   : ${PLATFORM} (push)"
  docker buildx build --platform "${PLATFORM}" -f "${DOCKERFILE}" -t "${IMAGE}" --push worker/
  echo
  echo "Pushed. Verify what the registry now serves:"
  echo "  docker manifest inspect ${IMAGE} | grep -A2 platform"
else
  if [ -n "$PLATFORM" ]; then
    echo "platform   : ${PLATFORM} (load)"
    docker buildx build --platform "${PLATFORM}" -f "${DOCKERFILE}" -t "${IMAGE}" --load worker/
  else
    echo "platform   : native"
    docker build -f "${DOCKERFILE}" -t "${IMAGE}" worker/
  fi
  echo
  echo "Built locally — 'docker run ${IMAGE} --key ...' on THIS machine needs no pull."
fi
