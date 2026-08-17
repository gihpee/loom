# Loom

Мультимодельный inference-маркетплейс. Ядро планирования адаптировано из
[Parallax](https://github.com/GradientHQ/parallax) (arXiv:2509.26182) — см.
`NOTICE.md`. Статус по фазам — `PROGRESS.md`.

**Полное описание архитектуры (wiki-onepager для онбординга):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).**
**Развёртывание реального стенда (Linux + NVIDIA): [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).**

## Модель больше одной карты (многостадийный инференс)

Ключевая возможность: модель режется по слоям между узлами. Phase-1 сам решает
раскладку, оркестратор рассылает стадиям их диапазоны, активации идут по
цепочке через туннели (воркерам не нужно видеть друг друга).

```bash
docker compose -f docker-compose.multistage.yml -p loomms up -d --build --scale worker=3
```

Проверить раскладку и спросить пайплайн:

```bash
curl -s localhost:8010/admin/models_view | python3 -m json.tool
```

Для прод-моделей укажите в каталоге `"backend_type": "shard"` — тогда модель,
не влезающая на одну карту, будет разложена по нескольким узлам.

## Как подключается GPU-машина

```bash
docker run -d --gpus all --restart unless-stopped -v loom-hf:/root/.cache/huggingface gihpee/loomworker --key loom_<выданный-ключ>
```

Это всё, что делает владелец GPU. Ключ выдаёт оркестратор
(`POST /admin/keys` или вкладка Keys в `/admin/ui`) и он несёт в себе адрес
оркестратора. Железо определяется автоматически (NVML/torch/nvidia-smi),
входящие порты не нужны — инференс идёт обратно по тому же исходящему
соединению, что открыл воркер.

## Состав (Фаза 0)

- `src/loom/planning` — программная библиотека планирования: Phase-1
  (упаковка слоёв в пайплайны, DP + water-filling) и Phase-2 (выбор цепочки,
  DAG DP). Один `Scheduler` = одна модель на выданном брокером sub-pool'е.
- `src/loom/perfmap` — централизованный Perf-map Store (замена DHT):
  `(model_id, node_id) → {layer_range, latency, health}`, `(src, dst) → rtt`;
  in-memory и Redis реализации + Postgres DDL каталога/биллинга.
- `src/loom/proto/worker_control.proto` — control-plane контракт
  (hub-and-spoke, push-команды воркеру).

## Ключевое отличие от Parallax

Capacity узла (`c_i`, сколько слоёв влезает) — **явный входной параметр**
(`ShardCapacity`), который Resource Broker вычисляет из выданной VRAM-квоты:

```python
from loom.planning import ModelInfo, Node, NodeHardwareInfo, Scheduler, ShardCapacity

mi = ModelInfo(model_name=..., num_layers=..., ...)
cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=12 * 2**30)  # квота от брокера
node = Node(node_id="w0", hardware=NodeHardwareInfo(...), model_info=mi, capacity=cap)

scheduler = Scheduler(mi, [node, ...], min_nodes_bootstrapping=2, routing_strategy="dp")
scheduler.bootstrap()                      # Phase-1: раскладка слоёв
path, latency = scheduler.request_router.find_optimal_path()  # Phase-2
```

Живые данные Phase-2 приходят не из DHT, а из Perf-map Store:

```python
from loom.perfmap import InMemoryPerfMapStore, sync_perfmap_to_scheduler
sync_perfmap_to_scheduler(store, scheduler, model_id="qwen3-0.6b")
```

## Быстрый старт (docker, мультимодельный)

```bash
docker compose up --build -d
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo-model-a","messages":[{"role":"user","content":"привет"}]}'

# Демо ре-балансировки: высокоприоритетная модель вытесняет низкоприоритетную
curl -s -X POST http://localhost:8000/admin/models \
  -H 'Content-Type: application/json' -d @configs/demo-model-c.json
curl -s http://localhost:8000/admin/status
```

Admin-дашборд для ручного тестирования: **http://localhost:8000/admin/ui**
(вкладки Nodes / Models / Perf-map / SLO+Rebalance / Test console; токен —
тот же `LOOM_ADMIN_TOKEN`, если задан).

По умолчанию воркеры используют echo-заглушку (test-only, без GPU). На
CUDA-хосте: образ воркера с `pip install .[vllm]` + каталог с
`"backend_type": "vllm"` (пример: `configs/qwen3-0.6b-vllm.json`).

Состав стека: `orchestrator` (gRPC ControlGateway :9000 + OpenAI API :8000 +
Model Registry + Resource Broker + Scheduler Pool), `worker` (закрытый образ,
только dial-out; масштабируется `--scale worker=N`), `redis`, `postgres`.

## Тесты

```bash
uv sync && uv run pytest
```

84 теста, в том числе:
- `tests/test_parity_with_original.py` — регрессионный паритет планировщика с
  оригинальным Parallax (нужен соседний checkout `../dllmi/parallax`);
- `tests/test_shard_parity.py` — 1/2/3/4 стадии дают тот же вывод, что модель
  целиком (крошечная модель генерируется локально, сеть не нужна);
- `tests/test_multistage_e2e.py` — полный стек: модель, не влезающая на узел,
  разложена по 3 узлам и отвечает через пайплайн.
