# Loom orchestrator image: gateway, client API, admin.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Код стадии модели. Он не часть оркестратора и не устанавливается: узел
# получает его файлами вместе с задачей, потому что реестр пакетов посередине —
# это ещё одна вещь, которую надо поднимать, авторизовывать и держать живой.
COPY payloads ./payloads

# .[p2p] adds the rendezvous node (docs/P2P_TRANSPORT.md).
RUN pip install --no-cache-dir ".[p2p]"

ENV LOOM_PAYLOADS_DIR=/app/payloads

EXPOSE 8000 9000
CMD ["python", "-m", "loom.orchestrator.server"]
