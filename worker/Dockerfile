# Loom worker image. Public artifact for GPU providers.
# Build context = worker/ ONLY: the image never contains orchestrator or
# planning sources (enforced later by the worker-release orphan branch too).
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY loom_worker ./loom_worker

RUN pip install --no-cache-dir .

# The worker only dials OUT (hub-and-spoke); no ports are exposed except the
# backend serving port, which is announced dynamically to the orchestrator.
ENV LOOM_LOG_LEVEL=INFO
ENTRYPOINT ["loom-worker"]
