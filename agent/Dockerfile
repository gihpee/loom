# Loom node agent — thin base image.
#
# It holds the LAUNCHER and nothing else that matters: no torch, no vLLM, no
# CUDA toolkit. Everything a task needs is provisioned into a task directory at
# run time (docs/WORKER_RUNTIME.md), and the agent itself arrives as a signed
# payload the launcher can replace without rebuilding this image
# (docs/WORKER_UPDATE.md).
#
# That is why this file must stay boring: every byte added here is a byte the
# node owner has to re-pull by hand one day.
#
#   docker run -d --gpus all --restart unless-stopped --network host \
#     -v loom-data:/var/lib/loom \
#     gihpee/loomagent --key loom_<...>
FROM python:3.12-slim

WORKDIR /app

# The user every task runs as. The agent runs as root so it can drop to this
# user per task; a task therefore cannot read the agent's join key, its files,
# or another task's directory. Without this user the agent refuses to run
# anything (see loom_agent/tasks/limits.py).
RUN useradd --system --no-create-home --shell /usr/sbin/nologin loom-task

# git is not needed to run, only to fetch payloads later; kept out on purpose.
COPY pyproject.toml ./
# loom_launcher/release_key.pub decides whose code may run on this machine.
# It is NOT in the repository: generate your own with
#   uv run --with cryptography python scripts/sign_release.py keygen
# An image built without it refuses every update — which is the safe state,
# and also means such a node never takes one.
COPY loom_launcher ./loom_launcher
COPY loom_agent ./loom_agent

# Both extras, deliberately:
#   [p2p]  the direct path between nodes. Without it every message between two
#          machines goes through the orchestrator — correct, and twice the
#          wide-area crossings. An image without it has no fast path at all.
#   [cuda] NVML, which is how the agent learns what cards it has and enforces
#          a task's memory quota. Harmless on a machine with no NVIDIA card:
#          detection just falls through to the next source.
RUN pip install --no-cache-dir ".[p2p,cuda]"

# Task directories, environment cache and agent payloads all live here. Mount a
# volume: without one, every restart re-provisions everything from scratch.
VOLUME ["/var/lib/loom"]

ENV LOOM_LOG_LEVEL=INFO \
    LOOM_ROOT=/var/lib/loom

# No EXPOSE: the agent only dials out. Everything comes back through that same
# outbound connection.
ENTRYPOINT ["loom-launcher"]
