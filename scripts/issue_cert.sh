#!/usr/bin/env bash
# Первый выпуск сертификата. Один раз на домен, до первого `compose up`.
#
# Почему отдельно, а не в compose: проверка владения доменом ходит по http на
# порт 80, а его занимает пограничный nginx, который без сертификата не
# поднимется. Замкнутый круг разрывается тем, что первый выпуск делает
# отдельно стоящий certbot, сам слушающий 80.
#
#   scripts/issue_cert.sh
#
# Продление после этого идёт само, службой certbot из compose, и порт 80 у неё
# уже обслуживает работающий nginx.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "нет .env — скопируйте .env.example и заполните"; exit 1; }
set -a; . ./.env; set +a

: "${LOOMA_DOMAIN:?LOOMA_DOMAIN не задан в .env}"
: "${LOOMA_ACME_EMAIL:?LOOMA_ACME_EMAIL не задан в .env}"

PROJECT="$(basename "$PWD")"
LETSENCRYPT="${PROJECT}_letsencrypt"
ACME="${PROJECT}_acme"

echo "домен:  $LOOMA_DOMAIN"
echo "почта:  $LOOMA_ACME_EMAIL"
echo

# Тома создаются заранее: certbot запускается вне compose и сам их не заведёт.
docker volume create "$LETSENCRYPT" >/dev/null
docker volume create "$ACME" >/dev/null

if docker run --rm -v "$LETSENCRYPT:/etc/letsencrypt" certbot/certbot \
     certificates 2>/dev/null | grep -q "$LOOMA_DOMAIN"; then
  echo "сертификат на $LOOMA_DOMAIN уже есть — выпускать нечего."
  echo "Продлевает его служба certbot из compose."
  exit 0
fi

echo "Порт 80 должен быть свободен и доступен снаружи. Если compose поднят —"
echo "остановите пограничный слой: docker compose stop edge"
echo

docker run --rm -p 80:80 \
  -v "$LETSENCRYPT:/etc/letsencrypt" \
  -v "$ACME:/var/www/acme" \
  certbot/certbot certonly --standalone \
    -d "$LOOMA_DOMAIN" \
    --email "$LOOMA_ACME_EMAIL" \
    --agree-tos --no-eff-email --non-interactive

echo
echo "Готово. Дальше: docker compose up -d --build"
