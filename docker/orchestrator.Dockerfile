# Loom orchestrator image (private): planning library + gateway + client API.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .

EXPOSE 8000 9000
CMD ["python", "-m", "loom.orchestrator.server"]
