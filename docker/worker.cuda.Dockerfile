# Public Loom worker image for NVIDIA hosts — published as gihpee/loomworker.
#
# Build context = worker/ ONLY: the image contains no orchestrator, planner or
# broker code. A GPU owner runs exactly one command:
#
#   docker run -d --gpus all --restart unless-stopped \
#     -v loom-hf:/root/.cache/huggingface \
#     gihpee/loomworker --key loom_<...>
#
# Base image already ships CUDA + PyTorch + vLLM, so the worker can serve real
# models out of the box; SGLang is optional (see below).
FROM vllm/vllm-openai:latest

WORKDIR /app
COPY pyproject.toml ./
COPY loom_worker ./loom_worker

# nvidia-ml-py: NVML-based hardware detection + VRAM watchdog enforcement.
RUN pip install --no-cache-dir . nvidia-ml-py>=12.0

# Optional: add SGLang support (large; enable if you plan to serve with sglang)
# RUN pip install --no-cache-dir "sglang[all]>=0.4"

ENV LOOM_LOG_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface

# No EXPOSE: the worker only dials out. Inference is tunnelled back over that
# same outbound connection, so no inbound port is ever needed.
ENTRYPOINT ["loom-worker"]
