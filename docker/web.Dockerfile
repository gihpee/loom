# Панель оператора: сборка и раздача. Node нужен только здесь — ни в образе
# оркестратора, ни на узлах его нет.

FROM node:22-alpine AS build
WORKDIR /web
# Сначала манифесты: слой с зависимостями переживает правку исходников.
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src ./src
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /web/dist /usr/share/nginx/html
# nginx:alpine подставляет переменные окружения в шаблоны при старте, так что
# порты берутся из .env и не дублируются в двух местах.
COPY web/nginx/default.conf.template /etc/nginx/templates/default.conf.template
# 127.0.0.1 верно при host-сети, которой compose и пользуется. Вынесено
# переменной, чтобы образ годился и при обычной сети docker.
ENV LOOMA_WEB_PORT=8080 \
    LOOMA_HTTP_PORT=8000 \
    LOOMA_API_HOST=127.0.0.1
