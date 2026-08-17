# План Фазы 1 — single-model MVP end-to-end (на ревью)

_Статус: черновик на ревью, код не пишется до аппрува._

## 0. Вводные, влияющие на архитектуру (проверено по коду Parallax)

1. **Executor Parallax не умеет CPU**: `executor/factory.py` поддерживает только
   `cuda` и `mlx`, иначе `ValueError`. В Linux-контейнере на dev-машине
   (Apple Silicon, Docker = Linux VM без GPU/Metal) оригинальный data plane
   Parallax **запуститься не может** в принципе.
2. **Baseline-планировщик Parallax запускается без executor'ов**: RPC
   `node_join` (`backend/server/rpc_connection_handler.py:34`) принимает
   декларативный JSON `{hardware: {tflops_fp16, memory_gb, ...}}`; решения о
   размещении читаются через `get_layer_allocation`/`cluster_status`. Значит
   «ванильный parallax serve» как baseline для паритета размещения можно
   поднять в контейнере и кормить синтетическими join'ами.
3. **Static wiring data plane существует в Parallax**: p2p server принимает
   явные `recv_from_peer_addr`/`send_to_peer_addr` (ZMQ PUSH/PULL) — передача
   активаций между стадиями не требует DHT.

## 1. Двухтрековый бенчмарк (следствие п.0)

Цель Фазы 1 (по согласованию): **функциональный и алгоритмический паритет** —
Perf-map Store + переиспользованные DP дают то же качество размещения, что и
DHT-путь Parallax. Методологический референс инъекции задержки — DeServe
(arXiv:2501.14784).

- **Track A — алгоритмический паритет + e2e-демо (работает на этой машине и в CI):**
  - **A1, placement/routing parity**: идентичная топология из 5 «узлов»
    (декларируемое железо) и идентичная RTT-матрица (env-параметр) подаются
    (а) в полный стек Loom (gRPC control plane → Perf-map → Phase-1/2 DP) и
    (б) в настоящий процесс планировщика Parallax в соседнем контейнере
    (Lattica + `node_join`-драйвер). Сравниваем: layer allocations, выбранные
    цепочки, оценку e2e-латентности. Критерий: точное совпадение allocations
    (как в Фазе 0) и совпадение выбранных цепочек.
  - **A2, e2e serving demo**: 3–5 воркер-контейнеров, реальная генерация
    токенов на CPU-референс-бэкенде (см. §3), multi-stage pipeline, netem-RTT
    профили {0, 25, 100 ms}; метрики TTFT/TPOT/throughput.
- **Track B — full-inference parity (тот же compose и скрипт, CUDA-хост):**
  флаг `LOOM_BENCH_DEVICE=cuda` включает vLLM-адаптер у воркеров и полный
  `parallax serve` в baseline. Запускается без изменения скриптов, когда
  появится GPU-машина (Фаза 3). RTT-матрица — тот же env-конфиг, куда позже
  подставляются измеренные значения.

**Явное отступление от буквы ТЗ Фазы 1** (прошу подтвердить): «паритет по
латентности/throughput с ванильным parallax serve на том же пуле и модели»
на этой машине физически недостижим (п.0.1) — real-inference сравнение
переносится в Track B; на Фазе 1 сдаётся Track A полностью + Track B в виде
готового к запуску, но не прогнанного на GPU кода.

## 2. Сетевая эмуляция

- `tc netem` в entrypoint'е каждого воркер-контейнера (`docker/netem-entrypoint.sh`,
  `cap_add: NET_ADMIN`), без sidecar'ов — pumba не нужен, задержка статична
  на время прогона.
- Конфиг — env: `RTT_MATRIX="w0:w1=25,w0:w2=100,..."` (мс, симметрично;
  генерируется из `benchmarks/topology.env.example`). Скрипт бенчмарка пишет
  в отчёт фактически измеренный RTT (ping между контейнерами), а не заданный.
  На Фазе 3 вместо синтетики в тот же env подставляются измеренные значения.

## 3. Worker v0

Каталог `worker/` — **самодостаточный**: свой `pyproject.toml`, не импортирует
`loom.planning`/`loom.orchestrator`; proto-стабы генерируются при сборке образа
из `proto/` (копия схем — единственная связь с остальным репо).

- `worker/loom_worker/main.py` — entrypoint: параметры только
  `LOOM_ORCH_ADDR`, `LOOM_NODE_TOKEN`, опц. GPU-девайсы.
- `gateway_client.py` — исходящее соединение (hub-and-spoke, NAT-friendly):
  прод-топология реализуется сразу, а не в Фазе 3, чтобы не переписывать
  handlers. Новый `proto/gateway.proto`:
  `ControlGateway.Attach(stream WorkerMessage) returns (stream ControlMessage)`;
  envelope переиспользует message-типы `worker_control.proto` (oneof по
  командам). Unary-сервис `WorkerControl` остаётся каноническим контрактом.
- `handlers.py` — исполнение LoadShard/UnloadShard/SetQuota/StartServing/
  StopServing/ReportTelemetry/Heartbeat. Никакой логики планирования.
- `backends/base.py` — единый интерфейс адаптера (фиксируется в этой фазе под
  Фазу 3): `prepare(shard) / start() / stop() / stats()`.
- `backends/vllm_adapter.py` — subprocess vLLM, квота через
  `--gpu-memory-utilization` (пересчёт из `vram_quota_bytes`). Полноценно
  прогоняется только в Track B.
- `backends/hf_cpu_adapter.py` — **референс-бэкенд** (наш код): загрузка слоёв
  `[start, end)` HF-чекпойнта на CPU (transformers), forward шарда, передача
  hidden states следующей стадии. Нужен, потому что ни vLLM-CUDA, ни MLX в
  контейнере на dev-машине недоступны; именно он даёт реальные токены в A2.
- `dataplane/zmq_pipe.py` — транспорт активаций: static prev/next ZMQ
  PUSH/PULL (адаптация схемы wiring из parallax p2p server — с атрибуцией);
  адреса стадий выдаёт оркестратор в `LoadShardRequest` (новые поля
  `prev_stage_addr`/`next_stage_addr` в proto).
- `watchdog.py` — v0: контроль RSS/VRAM квоты процесса бэкенда, kill при
  превышении (полноценно — Фаза 3, здесь каркас + RSS-лимит для CPU-демо).

## 4. Orchestrator v0 + Client API v0

- `src/loom/orchestrator/gateway.py` — gRPC-сервер `ControlGateway`; реестр
  подключённых воркеров, выдача команд, приём телеметрии → Perf-map Store.
- `src/loom/orchestrator/single_model.py` — один `model_id`: on join →
  пересчёт Phase-1 (библиотека Фазы 0) → diff → команды LoadShard/StartServing;
  периодический `sync_perfmap_to_scheduler` → Phase-2.
- `src/loom/api/app.py` — FastAPI `/v1/chat/completions` (стриминг SSE),
  модель захардкожена; маршрут запроса — `scheduler.request_router`.

## 5. Docker / git-изоляция воркера

- `docker-compose.yml` (корень): сервисы `orchestrator`, `redis`, `postgres`,
  `worker` (build: `worker/`, x5 реплик через `deploy.replicas`/шаблон),
  профили `dev` и `bench`.
- `docker-compose.parity.yml`: `parallax-scheduler` (образ из
  `docker/parallax-scheduler.Dockerfile`, исходники Parallax копируются при
  сборке из `../dllmi/parallax` — read-only, ничего не редактируем) +
  `join-driver` (наш скрипт `benchmarks/join_driver.py`, декларативные joins).
- `worker/Dockerfile` — публичный образ: только `worker/` + сгенерированные
  proto-стабы; без исходников оркестратора/планировщика.
- **Git**: `git init` в `loom/`, ветка `main`. CI-workflow
  `.github/workflows/worker-release.yml`: на push в `main` собирает содержимое
  `worker/` и коммитит его **одним squash-коммитом** в orphan-ветку
  `worker-release` (parent — предыдущий head этой ветки; общей истории с
  `main` нет вообще, приватный код физически не попадает ни в историю, ни в
  контекст сборки). Публикация образа в registry — только из
  `worker-release`. Registry по умолчанию — GHCR (вопрос §8).
- `.github/workflows/ci.yml` — pytest + сборка образов + smoke A1 на
  раннере (linux/amd64; A2 со скачиванием модели — nightly/manual, веса в
  кеше).

## 6. Полный список новых/изменяемых файлов

```
loom/
├── worker/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── loom_worker/
│       ├── main.py  gateway_client.py  handlers.py  state.py
│       ├── telemetry.py  watchdog.py
│       ├── backends/{base.py, vllm_adapter.py, hf_cpu_adapter.py}
│       └── dataplane/zmq_pipe.py
├── src/loom/proto/worker_control.proto      # + поля prev/next_stage_addr в LoadShardRequest
├── src/loom/proto/gateway.proto             # новый: reverse-channel envelope
├── src/loom/orchestrator/{__init__.py, gateway.py, controller.py,
│                          single_model.py, telemetry_ingest.py, config.py}
├── src/loom/api/{__init__.py, app.py, openai_types.py}
├── docker/{orchestrator.Dockerfile, parallax-scheduler.Dockerfile, netem-entrypoint.sh}
├── docker-compose.yml
├── docker-compose.parity.yml
├── benchmarks/{phase1_bench.py, join_driver.py, topology.env.example, README.md}
├── .github/workflows/{ci.yml, worker-release.yml}
└── tests/ (+ tests на handlers, gateway, hf_cpu_adapter, оба новых proto)
```

Жёсткие ограничения ТЗ: не нарушаются. Математика DP не трогается (orchestrator
вызывает библиотеку Фазы 0); `parallax/` — read-only (в baseline-образ исходники
копируются при docker build); в `worker/` нет кода принятия решений.

## 7. Порядок работ

1. proto v0.1 (gateway + stage-addr поля) + генерация стабов, тесты.
2. Worker v0 c hf_cpu-адаптером + zmq data plane; unit-тесты без докера.
3. Orchestrator v0 + API v0; интеграционный тест в docker-compose (1 машина,
   2 стадии, реальные токены).
4. netem + топология из env; A2-бенчмарк.
5. Baseline-образ parallax-scheduler + join-driver; A1-паритет.
6. vLLM-адаптер (код + unit-тесты с моком subprocess; прогон — Track B).
7. git init, CI, orphan-ветка worker-release, публикация образа.
8. Отчёт бенчмарка + обновление PROGRESS.md/NOTICE.md.

## 8. Вопросы на ревью

1. Подтвердить §1: real-inference parity уезжает в Track B (готовый код, без
   GPU-прогона в этой фазе); сдача Фазы 1 = A1 (паритет размещения через
   полные стеки) + A2 (e2e-токены на CPU-референс-бэкенде).
2. Ок ли baseline для A1 — процесс планировщика Parallax с декларативными
   join'ами (без executor'ов)? Это его штатный код и штатный RPC-путь.
3. Registry для публичного образа воркера: GHCR (default) или другой?
4. hf_cpu-адаптер остаётся в проде как третий бэкенд (полезен для CI) или
   помечаем как test-only?
