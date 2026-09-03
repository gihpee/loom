# Пограничный nginx. Отдельным образом от панели: у панели своя сборка с node,
# а этому нужен только конфиг.
FROM nginx:1.27-alpine

COPY docker/edge/default.conf.template /etc/nginx/templates/default.conf.template
# В conf.d, а не в templates: подставлять тут нечего, а nginx подхватит сам.
COPY docker/edge/upgrade.conf /etc/nginx/conf.d/upgrade.conf

ENV LOOMA_WEB_PORT=8080

# Перечитывать конфигурацию раз в половину суток. Certbot меняет файл
# сертификата в своём контейнере, а nginx держит открытым старый: без
# перезагрузки узлы и браузеры увидят просроченный сертификат ровно на 90-й
# день, и до этого дня ничто об этом не скажет.
CMD ["sh", "-c", "while :; do sleep 12h; nginx -s reload; done & nginx -g 'daemon off;'"]
