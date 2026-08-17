#!/usr/bin/env bash
# Create .env from .env.example and fill in random secrets.
# Safe to re-run: an existing .env is never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  echo ".env already exists — keeping it. Key settings:"
  grep -E '^(LOOM_HTTP_PORT|LOOM_GRPC_PORT|LOOM_PUBLIC_ADDR|LOOM_ADMIN_TOKEN|LOOM_CATALOG)=' .env
  exit 0
fi

command -v openssl >/dev/null || { echo "openssl required"; exit 1; }
cp .env.example .env
admin_token=$(openssl rand -hex 16)
pg_password=$(openssl rand -hex 12)

# BSD sed (macOS) needs an argument to -i; GNU sed does not.
sedi() { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }
sedi "s/^LOOM_ADMIN_TOKEN=.*/LOOM_ADMIN_TOKEN=${admin_token}/" .env
sedi "s/^LOOM_PG_PASSWORD=.*/LOOM_PG_PASSWORD=${pg_password}/" .env
chmod 600 .env

cat <<MSG
.env created.

  admin token : ${admin_token}
  API + UI    : http://localhost:$(grep '^LOOM_HTTP_PORT=' .env | cut -d= -f2)/admin/ui
  workers dial: port $(grep '^LOOM_GRPC_PORT=' .env | cut -d= -f2)

Ports busy? Edit LOOM_HTTP_PORT / LOOM_GRPC_PORT in .env, then:
  docker compose -f docker-compose.prod.yml up -d --build
MSG
