#!/usr/bin/env bash
# Create .env from .env.example and fill in random secrets.
# Safe to re-run: an existing .env is never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  echo ".env already exists — keeping it. Key settings:"
  grep -E '^(LOOMA_HTTP_PORT|LOOMA_GRPC_PORT|LOOMA_PUBLIC_ADDR|LOOMA_ADMIN_TOKEN|LOOMA_CATALOG)=' .env
  exit 0
fi

command -v openssl >/dev/null || { echo "openssl required"; exit 1; }
cp .env.example .env
admin_token=$(openssl rand -hex 16)
pg_password=$(openssl rand -hex 12)

# BSD sed (macOS) needs an argument to -i; GNU sed does not.
sedi() { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }
sedi "s/^LOOMA_ADMIN_TOKEN=.*/LOOMA_ADMIN_TOKEN=${admin_token}/" .env
sedi "s/^LOOMA_DB_PASSWORD=.*/LOOMA_DB_PASSWORD=${pg_password}/" .env
# Строка подключения собирается из того же пароля: две записи одного
# секрета разъезжаются при первой же его смене.
sedi "s#^LOOMA_DATABASE_URL=.*#LOOMA_DATABASE_URL=postgresql://looma:${pg_password}@127.0.0.1:5432/looma#" .env
chmod 600 .env

cat <<MSG
.env created.

  admin token : ${admin_token}
  пароль базы : ${pg_password}
  панель      : http://localhost:$(grep '^LOOMA_WEB_PORT=' .env | cut -d= -f2)
  API         : http://localhost:$(grep '^LOOMA_HTTP_PORT=' .env | cut -d= -f2)
  узлы звонят : порт $(grep '^LOOMA_GRPC_PORT=' .env | cut -d= -f2)

Порты заняты? Поправьте их в .env и поднимите:
  docker compose up -d --build
MSG
