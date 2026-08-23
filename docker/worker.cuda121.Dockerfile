# Loom worker for hosts with an older NVIDIA driver — gihpee/loomworker-cuda121.
#
#   docker run -d --gpus all --restart unless-stopped --network host \
#     -v loom-hf:/root/.cache/huggingface \
#     gihpee/loomworker-cuda121 --key loom_<...>
#
# Why this image exists. The standard worker is built on vllm/vllm-openai,
# which tracks a CUDA newer than many installed drivers support. On such a host
# nothing of Loom ever runs: the NVIDIA container runtime refuses the container
# before the entrypoint, with
#
#   nvidia-container-cli: requirement error: unsatisfied condition: cuda>=...
#
# which reads like a Loom failure and is not one. Seen on driver 550.163.01
# (CUDA 12.4) with two RTX 3090s.
#
# This image is built on CUDA 12.1 instead, which needs only driver >= 530 —
# so it covers everything the standard image covers and a few years more.
#
# What it gives up: vLLM, and with it the `vllm` and `vllm_shard` backends. The
# `shard` backend runs pipeline stages on transformers, which is what a CUDA
# stage uses today anyway; deploy to this node with backend `shard`.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml ./
COPY loom_worker ./loom_worker

# The base carries torch and CUDA; everything a stage needs on top of that is
# listed here rather than inherited, so an upstream change cannot quietly
# remove one. transformers 4.51 is the floor for Qwen3.
#
# lattica: direct worker-to-worker links (docs/P2P_TRANSPORT.md). Its wheels
# are cp38-abi3, so the base's Python 3.11 is fine.
RUN pip install --no-cache-dir . \
      "transformers>=4.51,<5" \
      "accelerate>=1.0" \
      "safetensors>=0.4" \
      "huggingface_hub>=0.26" \
      "nvidia-ml-py>=12.0" \
      "lattica==1.0.21"

ENV LOOM_LOG_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface

# No EXPOSE: the worker only dials out, and inference is tunnelled back over
# that same outbound connection. Port 47100 (TCP+UDP) is worth opening anyway
# if this host can accept connections — that is what lets its activations go
# straight to the next stage instead of through the orchestrator.
ENTRYPOINT ["loom-worker"]
