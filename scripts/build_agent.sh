#!/usr/bin/env bash
# Build (and optionally push) the public agent image.
#
#   scripts/build_agent.sh              # build for THIS machine, load locally
#   scripts/build_agent.sh --push       # build for amd64+arm64 and push
#
# The image is deliberately thin: an agent, and nothing that computes. Whatever
# a task needs is provisioned into that task's directory at run time
# (docs/WORKER_RUNTIME.md), so this image changes rarely — which matters,
# because a change here is the one kind of update a node's owner has to apply
# by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${LOOM_AGENT_IMAGE:-gihpee/loomagent}"
TAG="${LOOM_AGENT_TAG:-latest}"

if [ "${1:-}" = "--push" ]; then
  docker buildx build --platform linux/amd64,linux/arm64 \
    -t "$IMAGE:$TAG" -f agent/Dockerfile --push agent
else
  docker build -t "$IMAGE:$TAG" -f agent/Dockerfile agent
  docker images "$IMAGE:$TAG"
fi
