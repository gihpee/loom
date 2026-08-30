# Loom

Распределённые вычисления и инференс на машинах, до которых **нельзя
дозвониться**: домашний компьютер за роутером провайдера подключается одной
командой, ничего не настраивая.

Владелец GPU выполняет одну строку. Его машина сама открывает соединение к
оркестратору и держит его; команды, входные файлы, результаты и активации
инференса едут по этому же соединению. Ни одного входящего порта, ни проброса
на роутере, ни адреса, который надо где-то прописать.

- Узел: собрать, подключить, обновлять — [docs/AGENT_MANUAL.md](docs/AGENT_MANUAL.md)
- Архитектура узла: [docs/WORKER_RUNTIME.md](docs/WORKER_RUNTIME.md)
- Обновление парка: [docs/WORKER_UPDATE.md](docs/WORKER_UPDATE.md)
- Прямой канал между узлами: [docs/P2P_TRANSPORT.md](docs/P2P_TRANSPORT.md)
- Развёртывание: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Заимствования из Parallax: [NOTICE.md](NOTICE.md)

## Из чего состоит

```
src/loom/            оркестратор: кому какую работу, кто чем занят, кому какой релиз
agent/               то, что ставится на чужую машину. Тонкое: агент и больше ничего
payloads/loom_stage/ стадия модели. Ставится в задачу, когда просят инференс
relay/               circuit-relay для узлов, до которых не дозвониться напрямую
```

**Агент ничего не считает.** Работа приезжает задачей: ей выдаётся каталог,
в него разворачивается нужное окружение, там она и выполняется. Инференс,
обучение и чужой код — три случая одного механизма, поэтому образ агента
маленький и почти не меняется.

## Запуск

Оркестратор:

```bash
bash scripts/init_env.sh && docker compose up -d --build
```

Узел — команду с ключом выдаёт админка на `/admin`, вкладка Keys:

```bash
docker run -d --gpus all --restart unless-stopped --network host \
  -v loom-data:/var/lib/loom gihpee/loomagent --key loom_...
```

`--network host` не косметика: без него p2p-порт остаётся внутри контейнера и
прямой канал между узлами невозможен в принципе.

## Что делает клиент

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"привет"}]}'
```

Отвечает конвейер стадий, разложенный по нескольким чужим машинам. Ни одна из
них не открывала порта.

## Разработка

```bash
uv sync
uv run pytest tests -q                      # оркестратор и интеграция
cd agent && uv run pytest tests -q          # узел
cd payloads/loom_stage && uv run pytest tests -q
```

Протокол между оркестратором и узлами — один файл,
[src/loom/proto/agent.proto](src/loom/proto/agent.proto). После правки:

```bash
uv run --with grpcio-tools python scripts/gen_proto.py
```
