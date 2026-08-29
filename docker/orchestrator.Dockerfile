# Loom orchestrator image (private): planning library + gateway + client API.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# .[p2p] adds the rendezvous node (docs/P2P_TRANSPORT.md).
RUN pip install --no-cache-dir ".[p2p]"

EXPOSE 8000 9000
CMD ["python", "-m", "loom.orchestrator.server"]
