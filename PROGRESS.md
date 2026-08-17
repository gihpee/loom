# Loom — статус по фазам

_Обновлено: 2026-08-16_

## Фаза 0 — фундамент: ✅ выполнена

Deliverable: работающая библиотека планирования, вызываемая программно, с явным
capacity-параметром + схема БД + protobuf, покрытые unit-тестами с регрессионным
паритетом против оригинального Parallax.

Сделано:

1. **Исследование актуального кода Parallax** (`../dllmi/parallax/`), подтверждено:
   - `Node` биндит `ModelInfo` в конструкторе; `get_decoder_layer_capacity()`
     (node.py:274) считает c_i из `(hardware, model)`;
   - Phase-1 DP: `DynamicProgrammingLayerAllocator`, состояние
     `dp(i, open_residuals, finished_pipes)` (layer_allocation.py:848),
     water-filling в `adjust_pipeline_layers`;
   - Phase-2: `request_routing.py`, DAG DP читает `Node.layer_latency_ms` /
     `rtt_to_nodes`, которые наполняются через `Scheduler.enqueue_node_update`
     (в оригинале — из DHT-бродкастов Lattica) → точка замены источника данных;
   - `EndpointRegistry` (router/main.py) ключуется `base_url`, поля `model` нет
     (правка запланирована на Фазу 2);
   - `SchedulerManage` — синглтон процесса, сцеплен с Lattica; нода при join
     получает layer range через P2P shared state.
2. **`loom.planning`** — переиспользуемая библиотека: `ModelInfo` (as-is),
   Phase-1 DP + water-filling и Phase-2 DP (математика не тронута),
   `NodeManager`, `Scheduler` (программный, по инстансу на модель).
3. **Явный capacity-параметр**: `ShardCapacity` (`loom/planning/capacity.py`) —
   вычисляется Resource Broker'ом из `vram_quota_bytes`; `Node` делегирует
   capacity/KV-бюджет в него, а не в забинженный `ModelInfo`.
4. **Perf-map Store** (`loom/perfmap/`): интерфейс + in-memory + Redis-реализация,
   ключи с обязательным `model_id`; `sync.py` кормит планировщик через штатный
   `enqueue_node_update` (замена DHT). Postgres DDL: `perfmap/schema.sql`
   (каталог моделей/узлов, allocations, usage_records для биллинга).
5. **Protobuf** `loom/proto/worker_control.proto`: фиксированные RPC
   (LoadShard/UnloadShard/SetQuota/StartServing/StopServing/ReportTelemetry/
   Heartbeat) + CommandMeta (command_id, issued_at, signature — заготовка под
   replay protection Фазы 3). Компилируется protoc'ом (проверено).
6. **Тесты**: `uv run pytest` → **18 passed**, в т.ч.
   `tests/test_parity_with_original.py` — регрессионный паритет allocations
   loom vs оригинальный Parallax (DP и greedy, 24/48 слоёв, идентичные входы),
   плюс паритет capacity по всем 4 комбинациям (embed × lm_head), KV-памяти и
   `max_requests`.

Запуск тестов: `cd loom && uv sync && uv run pytest`.

## Фаза 1 — single-model MVP end-to-end: ✅ выполнена (в согласованном сокращённом скоупе)

Deliverable пересогласован (2026-08-16): вместо бенчмарка против
`parallax serve` — работающий код Worker v0 + Orchestrator v0 + Client API v0,
поднимающийся через `docker-compose` и отвечающий на `curl /v1/chat/completions`
на одной модели/одном узле. Бенчмарк/паритет с baseline отложены до GPU-хоста
(методологический план остаётся в `docs/phase1-plan.md`).

Сделано:

1. **Proto v0.1**: `gateway.proto` — hub-and-spoke реверс-канал
   (`ControlGateway.Attach`, bidi stream; воркер только dial-out, NAT-friendly);
   команды `WorkerControl` ходят в envelope. Стабы генерируются
   `scripts/gen_proto.py` в оба пакета и закоммичены.
2. **Worker v0** (`worker/` — самодостаточный, свой pyproject, не импортирует
   код оркестратора/планирования): gateway-клиент с реконнектом, handlers
   LoadShard/UnloadShard/SetQuota/StartServing/StopServing/ReportTelemetry/
   Heartbeat (идемпотентные, без логики планирования), бэкенды за единым
   интерфейсом (`backends/base.py`): **vllm** (subprocess `vllm serve`,
   квота → `--gpu-memory-utilization`; v0 — только полная модель) и **echo**
   (test-only заглушка для хостов без GPU), RSS-watchdog с kill при превышении
   квоты (VRAM/NVML — Фаза 3).
3. **Orchestrator v0**: gRPC ControlGateway + `SingleModelController` —
   register → planning `Node` (capacity из декларированного VRAM) → Phase-1
   bootstrap → push LoadShard/StartServing → приём ServingEndpoint; телеметрия →
   Perf-map Store → периодический sync в Phase-2; token-auth (shared secret v0).
4. **Client API v0**: FastAPI `/v1/chat/completions` (non-stream + SSE-стриминг,
   passthrough-прокси на отроученный эндпоинт), `/v1/models`, `/healthz`.
5. **Docker**: `worker/Dockerfile` (контекст сборки — только `worker/`),
   `docker/orchestrator.Dockerfile`, `docker-compose.yml`
   (orchestrator + redis + postgres со схемой + worker).
6. **Проверено вживую**: `docker compose up --build` → `curl` на
   `/v1/chat/completions` возвращает ответ (и non-stream, и stream);
   воркер переживает старт раньше оркестратора (reconnect). Тесты:
   **26 passed** (`uv run pytest`), включая e2e без докера
   (`tests/test_phase1_e2e.py`) и юниты воркера (watchdog kill-path,
   vllm cmdline/квота, отклонение неверного токена).

Известные ограничения v0 (осознанные, в плане следующих фаз):
- один pipeline-stage на модель (нет межстадийного data plane) — multi-stage
  отклоняется роутером с предупреждением;
- vLLM-адаптер не прогнан на реальном GPU (нет стенда) — юнит-тесты only;
- Redis/Postgres в compose подняты, но оркестратор v0 использует in-memory
  Perf-map (Redis подключается в Фазе 2 через `RedisPerfMapStore`);
- CI и orphan-ветка `worker-release` отложены по решению от 2026-08-16.

## Фаза 2 — мультимодельность: ✅ выполнена

Deliverable: 2+ модели одновременно на общем пуле + демонстрация
ре-балансировки (высокоприоритетная третья модель вытесняет низкоприоритетную).
Проверено и e2e-тестом (`tests/test_phase2_e2e.py`), и вживую в докере.

Сделано:

1. **Model Registry** (`orchestrator/registry.py`): CRUD каталога
   (`ModelSpec` = ModelInfo + demand_qps/priority/price_willing/
   target_pipelines), JSON-каталог на старте, admin-API для CRUD на лету.
2. **Resource Broker** (`orchestrator/broker.py`) — greedy FFD строго по
   спецификации ТЗ: `score = priority * price_willing * demand_qps`, сортировка
   по score, группировка по регионам (регион — по убыванию свободного VRAM),
   FFD-набор VRAM на k пайплайнов, деградация k до 1 при нехватке. Пайплайн не
   пересекает регионы (cross-region мост со штрафом — следующий шаг). Узел
   потребляется по байтам: остаток остаётся в пуле для моделей с меньшим score —
   это и есть multi-process co-location с VRAM-квотами (п.4 Фазы 2).
   Stickiness: сначала попытка уложить пайплайн целиком на прежние узлы модели,
   иначе чистый FFD (частичное «прилипание» хуже, чем никакое — дробит пайплайн).
3. **Scheduler Pool** (`orchestrator/pool.py`): по инстансу немодифицированного
   планировщика на модель; Phase-1 гоняется на sub-pool'е из грантов брокера
   (capacity = ShardCapacity из квоты). Rebalance v0 = полная переукладка
   инстанса по новым грантам.
4. **MultiModelController**: диф план→факт по (model, node, quota, layers);
   сначала awaited-вытеснения (StopServing+UnloadShard), потом асинхронные
   деплои (LoadShard+StartServing) — VRAM узла не переподписывается. Триггеры:
   model add/remove (admin), node join/leave, периодический таймер
   (LOOM_REBALANCE_S). Perfmap sync — по каждой модели из пула.
5. **Model-aware EndpointRegistry** (`api/endpoints.py`, адаптация из Parallax
   router с полем `model_id` — см. NOTICE) + стратегии LB as-is
   (`api/lb_strategy.py`). Роутинг: Phase-2 head → эндпоинт узла, fallback —
   LB по эндпоинтам той же модели.
6. **API**: `/v1/models` из каталога (+ флаг serving), роутинг
   `/v1/chat/completions` по `model` (400 без model при 2+ моделях, 404
   незнакомая, 503 нет capacity), `/admin/models` POST/DELETE,
   `/admin/status` (узлы/гранты/эндпоинты/unscheduled), опц. admin-токен.
7. **Namespace `model_id`**: worker state, телеметрия, Perf-map, эндпоинты —
   всё ключуется model_id (воркер уже в Фазе 1 держал shards по model_id).
8. Proto: `RegisterRequest.region` (брокер группирует по регионам,
   у воркера — env `LOOM_REGION`).

Демо в докере (1 узел 6 GB, каталог из 2 моделей по ~2.2 GB):
обе serving → `POST /admin/models` c demo-model-c (score 400) →
c и a отвечают 200, b вытеснена (unscheduled, 503). Тесты: **42 passed**,
включая юниты брокера (score-ordering/eviction/regions/k-degradation/
co-location) и model-aware реестра эндпоинтов.

Ограничения v0 фазы 2: полная переукладка модели при изменении её грантов
(SetQuota-без-рестарта — Фаза 3); cross-region мост не реализован (модель
остаётся unscheduled, если ни один регион не вмещает пайплайн).

## Фаза 3 — гетерогенные бэкенды + устойчивость: ✅ выполнена

Deliverable: воркер под управлением оркестратора на гетерогенном пуле
(cuda/mlx/cpu) + watchdog в действии: превышение квоты → процесс убит, узел не
падает целиком. Проверено e2e-тестами и вживую в докере (лог воркера:
`watchdog: backend pid=40 exceeded rss quota: used=25214976 > 1`; узел остался
connected, модель самовосстановилась на новом порту, соседняя модель не
затронута).

Сделано:

1. **SGLang-адаптер** (`worker/.../backends/sglang.py`): subprocess
   `sglang.launch_server`, квота → `--mem-fraction-static`. **MLX-адаптер**
   (`backends/mlx.py` + `mlx_launcher.py`): у `mlx_lm.server` нет CLI-флага
   лимита памяти, поэтому лаунчер применяет `mx.set_memory_limit(bytes)`
   in-process (API проверен по установленному MLX) и передаёт управление
   серверу; health — `/v1/models`. Оба за интерфейсом Фазы 1 — control-plane
   код не менялся (проверка: добавление адаптеров не тронуло ни handlers, ни
   gateway, ни orchestrator). Extras в worker/pyproject: `[sglang]`, `[mlx]`,
   `[cuda]`. На этой машине прогнаны юнитами (cmdline/квота/reject partial);
   реальный прогон движков — при появлении GPU-хоста.
2. **Watchdog**: для cuda — VRAM per-process через NVML (nvidia-ml-py,
   суммирование по дереву процессов; юнит с фейковым NVML), иначе — RSS
   (для MLX unified memory RSS = жёсткая страховка поверх мягкого
   `set_memory_limit`). Kill только дерева бэкенда; агент живёт. SetQuota
   обновляет лимит на лету; admin-эндпоинт `/admin/quota` — ops-инструмент
   (им же сделано живое демо).
3. **Самовосстановление**: в телеметрию добавлен `ShardTelemetry.status`;
   `failed` (отличаем от `loading`) → оркестратор забывает деплоймент,
   снимает эндпоинт и перекладывает модель следующим проходом брокера.
4. **Подписанные команды + replay protection**: HMAC-SHA256 поверх
   детерминированной сериализации ControlMessage с очищенным
   `meta.signature`; ключ v0 — onboarding-токен узла (keypair — Фаза 4).
   Воркер отклоняет: неверную подпись, tamper, повтор `command_id`
   (LRU-кэш), устаревшие команды (окно `issued_at` ±60с). Подпись — в единой
   точке (`WorkerSession.send_command`); проверка — перед исполнением любой
   команды кроме register_ack. Кросс-пакетная совместимость (стабы
   оркестратора ↔ стабы воркера) покрыта тестом.
5. **SLO-триггеры ре-балансировки**: API пишет TTFT/ошибки по модели;
   `slo_evaluate()` считает p95 по окну (60с): p95 > `slo_p95_ttft_ms` →
   score-boost модели (×2) и +1 целевой пайплайн на следующем проходе
   брокера; гистерезис — буст снимается при p95 < 0.7×SLO. Состояние SLO — в
   `/admin/status`.

Тесты: **54 passed** (весь набор фаз 0–3), включая e2e watchdog-kill +
self-heal, гетерогенный пул cuda/mlx/cpu, подпись/tamper/replay/stale, NVML-мок.

Ограничения: реальный прогон vLLM/SGLang/MLX-движков не выполнялся (нет
GPU-стенда; echo-заглушка подменяет движок в e2e); mTLS на транспорте — при
развёртывании (за пределами docker-compose dev-стека).

## КЛЮЧЕВАЯ ФУНКЦИЯ: многостадийный инференс (слои модели на разных узлах) — ✅ (2026-08-17)

То, ради чего взят Parallax: модель, не влезающая на одну карту, режется по
слоям между узлами. Работает end-to-end и доказано паритетом вывода.

Сделано:

1. **Shard-бэкенд** (`worker/loom_worker/shard/`) — единственный бэкенд,
   обслуживающий ЧАСТЬ модели:
   - `loader.py`: материализует только свои слои (safetensors, ремап
     `model.layers.{global}` → локальные `0..n-1`), опускает embed/lm_head на
     стадиях, где они не нужны, поддерживает tied embeddings;
   - `executor.py`: forward по своим слоям с per-request KV-кэшем, сэмплинг на
     последней стадии, сериализация активаций для провода;
   - `server.py`: HTTP-поверхность стадии (`/stage/forward`, на голове —
     `/v1/chat/completions`) + цикл генерации по кольцу стадий.
2. **Транспорт активаций** без прямой связи между воркерами: стадия отдаёт
   активации локальному relay агента (`stage_relay.py`), агент пушит
   `StageEnvelope` в свой туннель, `TunnelHub` оркестратора по таблице
   `(pipeline_id, stage) → node` доставляет их узлу следующей стадии. Обратный
   путь для токена — сразу на голову. Обоснование отступления от P2P (ТЗ его
   допускало) — в ARCHITECTURE §6.4b: прямое соединение требует
   hole-punching/открытых портов, что ломает принцип «владелец GPU ничего не
   настраивает».
3. **Оркестратор строит пайплайны**: `build_pipelines()` группирует раскладку
   Phase-1 в упорядоченные цепочки, присваивает `pipeline_id`/`stage_index`,
   рассылает топологию в `LoadShard`, публикует stage-маршруты в TunnelHub;
   клиентский эндпоинт публикует ТОЛЬКО голова (стадия 0).
4. **Брокер считает слои, а не байты**: `layers_fitting()`/`bytes_for_layers()`
   повторяют floor-округление Phase-1 на каждом узле. Без этого грант «ровно
   `pipeline_vram_bytes`» на N узлов терял до N−1 слоёв, и Phase-1 не мог
   покрыть модель (нашлось при первом же прогоне многостадийного теста).
5. **Устойчивость**: топология едет в телеметрии (`pipeline_id`,
   `stage_index`, `num_stages`), поэтому рестарт оркестратора восстанавливает и
   эндпоинты, и stage-маршруты; шард-процесс умирает при смерти агента
   (не оставляет VRAM) и мгновенно реагирует на SIGTERM.

Найденные и исправленные баги (все — из разряда «молчаливо неверный ответ»):
- **rotary embeddings**: `inv_freq` — не persistent-буфер, его нет в чекпойнте;
  создание на meta-device + `to_empty()` оставляло мусор. Симптом был
  диагностичным: позиция 0 точна, ошибка растёт с позицией. Теперь rotary
  строится напрямую на целевом девайсе, плюс `_assert_materialised()` ловит
  любой неинициализированный тензор;
- **KV-кэш игнорировался**: в transformers 5.x аргумент называется
  `past_key_values`, а `past_key_value` молча уходил в `**kwargs` — prefill был
  верен, а decode считал по пустому кэшу. Теперь имя определяется по сигнатуре
  слоя (совместимо с 4.x/5.x), и это покрыто тестом;
- watchdog убивал шард на старте: RSS процесса с torch не имеет отношения к
  VRAM-квоте → введена надбавка на рантайм (`LOOM_RSS_OVERHEAD_MB`, на CUDA с
  NVML не применяется).

Доказательства (все автоматические):
- `tests/test_shard_parity.py` (11 тестов): 1/2/3/4 стадии дают **побитово тот
  же** prefill-логит и **тот же greedy-вывод**, что модель целиком; отдельные
  тесты на использование KV-кэша и изоляцию состояния между запросами;
- `tests/test_multistage_e2e.py` (3 теста): полный стек, 3 воркера с квотой на
  2 слоя каждый → Phase-1 обязан построить 3-стадийный пайплайн; проверяются
  раскладка без дыр, единственный эндпоинт (голова), совпадение ответа с
  однопроцессным `transformers`, SSE-стриминг через пайплайн и публикация/
  очистка stage-маршрутов.

Проверено вживую в докере (`docker-compose.multistage.yml` +
`worker/Dockerfile.shard`): 3 воркер-контейнера с квотой на 2-3 слоя каждый →
Phase-1 разложил модель как `[0,3)` на одном контейнере и `[3,6)` на другом,
клиентский эндпоинт опубликован только у головы, и запрос через
`/v1/chat/completions` вернул **побитово тот же ответ**, что однопроцессный
`transformers` (31 prompt-токен, те же 6 сгенерированных токенов).

Ещё один баг, найденный только в докере: watchdog «смерти родителя» считал
`ppid == 1` признаком сироты, а в контейнере агент сам PID 1 — стадия убивала
себя через 2 секунды после старта. Теперь сравнивается только ИЗМЕНЕНИЕ ppid.

Тесты: **84 passed**.

Ограничения v0 (следующие шаги): одна последовательность на запрос (нет
continuous batching), активации во float32, каждый хоп через оркестратор,
исполнитель на transformers (не vLLM paged attention). Быстрый путь —
портирование `ParallaxVLLMModelRunner` (наследник vLLM `GPUModelRunner` +
monkey-patch `get_pp_indices`), но он требует CUDA-стенда для валидации.

## Внеплановое: прод-стенд «одна команда у владельца GPU» — ✅ (2026-08-17)

Цель (по требованию): публичный образ воркера + ключ от оркестратора;
владелец GPU делает только `docker run gihpee/loomworker --key {key}` и больше
ничего. Ни `LOOM_ADVERTISE_HOST`, ни декларации железа.

Сделано:

1. **Автодетект железа** (`worker/loom_worker/hwinfo.py`, адаптация
   `detect_node_hardware` из Parallax — см. NOTICE): NVML → torch →
   `nvidia-smi` для NVIDIA, `sysctl` для Apple Silicon, CPU-fallback; имя карты
   → TFLOPs/bandwidth по расширенной таблице (H100/H200/A100/L40S/L4/A10/
   A6000/V100/T4/RTX 30-50xx). Отдаются `vram_free` (что можно планировать) и
   `vram_total` (для расчёта `--gpu-memory-utilization`). Env остались только
   как override для тестов. Живая проверка: `device=mlx gpu=Apple M4
   vram_free=16.8GB tflops=8.5 (sysctl)` — без единой переменной окружения.
2. **Data-plane туннель** (`orchestrator/tunnel.py` + `dataplane.proto` +
   `worker/dataplane_client.py`): воркер держит второй исходящий стрим;
   оркестратор отправляет в него `HttpRequest`, воркер релеит на свой
   `127.0.0.1:<порт бэкенда>` и стримит ответ чанками назад. Мультиплекс по
   `request_id`, поддержка SSE, отмена, реконнект. Итог: **воркеру не нужен ни
   публичный адрес, ни открытый порт** — идеология Parallax (трафик по уже
   открытым соединениям), выраженная в hub-and-spoke. `ServingEndpoint` теперь
   несёт только локальный порт; эндпоинт в реестре — `tunnel://node:port`.
3. **Join-ключи** (`orchestrator/keys.py`): `loom_<base64url({key_id, secret,
   address})>` — ключ несёт и адрес оркестратора, и секрет; секрет служит
   **per-node ключом подписи команд**, поэтому отзыв ключа отбирает и право
   исполнять команды. `max_nodes`, отзыв, персист в JSON
   (`LOOM_KEYSTORE_PATH`). Admin: `POST/GET/DELETE /admin/keys` (ответ содержит
   готовую `docker run`-команду) + вкладка **Keys** в дашборде.
4. **Прод-обвязка**: `docker/worker.cuda.Dockerfile` (базовый образ
   `vllm/vllm-openai`, публикуется как `gihpee/loomworker`),
   `docker-compose.prod.yml` (оркестратор + redis + postgres, `LOOM_PUBLIC_ADDR`
   вшивается в ключи), каталоги `configs/catalog-qwen3-0.6b.json` и
   `catalog-qwen3-8b.json`, рунбук [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
5. **Настраиваемый param/KV split** (`LOOM_PARAM_MEM_RATIO`): практические
   числа — Qwen3-8B требует 25.4 ГБ при 0.6 и 20.3 ГБ при 0.75, то есть на
   карте 24 ГБ нужен 0.75 (в рунбуке есть таблица и способ посчитать свою
   модель).

Найденные и исправленные по ходу баги:
- `AttributeError` на удалённом поле proto молча убивал recv-цикл gateway —
  добавлено логирование ошибок обработчиков (gateway больше не «глохнет»);
- half-closed стрим выглядел живым туннелем/сессией, и запросы в него повисали
  бы — теперь сессия отцепляется сразу, как закрывается половина стрима;
- тесты, ходившие в API из чужого event loop, теперь используют loop
  оркестратора (как в проде: один процесс — один loop).

Ещё две правки надёжности реконнекта, найденные при живой проверке:
- после рестарта оркестратора воркер с уже поднятым шардом не пересообщал
  эндпоинт (команда идемпотентна, эффект терялся) → теперь `StartServing`
  повторно анонсирует эндпоинт, а `ShardTelemetry.local_port` в каждом
  heartbeat позволяет восстановить таблицу маршрутизации из телеметрии;
- при обрыве стрима «зависший» генератор старого соединения перехватывал
  сообщения нового → очередь исходящих стала per-connection (и в control, и в
  data plane).

Проверено вживую:
- воркер **вне docker-сети**, подключённый только ключом, зарегистрировался с
  реальным автодетектом (`Apple M4, 16.8 GB, sysctl`), поднял бэкенд и обслужил
  `/v1/chat/completions` (и стриминг) через туннель;
- **прод-образ** `gihpee/loomworker` (собран из `docker/worker.cuda.Dockerfile`,
  база `vllm/vllm-openai`, 33 GB) запущен одной командой
  `docker run ... --key loom_...` → зарегистрировался, получил модели и
  обслужил инференс через туннель;
- рестарт оркестратора → эндпоинты восстановились из телеметрии, инференс
  продолжил работать.

Тесты: **70 passed** (+14 новых: ключи, автодетект с моками NVML/smi, туннель,
отказ при пропавшем туннеле, re-sync эндпоинтов, admin-API ключей).

⚠️ Важно при сборке образа: protobuf-стабы генерируются закреплённым
`grpcio-tools==1.71.0` (gencode 5.29), потому что runtime protobuf в
vllm-образе — 6.x, а runtime не может быть старее gencode.

Осталось до реального GPU-стенда: TLS/mTLS на 9000 (сейчас insecure — закрывать
фаерволом), прогон vLLM на настоящей карте, многостадийный data plane для
моделей, не влезающих на один GPU.

## Внеплановое: admin-дашборд для ручного тестирования — ✅ (2026-08-16)

Не продовая фича; UI-слой поверх существующих admin-эндпоинтов, бизнес-логика
не менялась (broker/pool/controller — только чтение состояния).

- Страница `/admin/ui` ([src/loom/api/admin_ui.html](src/loom/api/admin_ui.html),
  vanilla HTML/JS, вкладки, автообновление 3с, admin-токен в
  localStorage → заголовок `X-Loom-Admin-Token`).
- Вкладки: **Nodes** (node_id, device, регион, декларированный VRAM, цена —
  n/a до Фазы 4, heartbeat, шарды со статусами из `ShardTelemetry.status`);
  **Models** (каталог + score, k actual/target, Phase-1 placement,
  эндпоинты); **Perf-map** (τ effective/measured по узлам, ρ-матрица RTT,
  превью Phase-2 цепочки); **SLO/Rebalance** (p95 vs порог, boost-флаг,
  кнопка force rebalance → `POST /admin/rebalance`); **Console**
  (форма → `/v1/chat/completions`, поддержка stream).
- Новые read-only эндпоинты: `GET /admin/nodes`, `GET /admin/models_view`,
  `GET /admin/perfmap/{model_id}`, `POST /admin/rebalance` (вызывает
  существующий `controller.rebalance`, как таймер). В контроллере добавлено
  только observability-состояние: `node_last_seen`, `shard_status`
  (заполняются из телеметрии).
- Проверено вживую в докере (все 5 вкладок, отправка запроса из консоли,
  force rebalance с выводом плана). Тесты: **56 passed**
  (`tests/test_admin_ui.py`: views, 403 без токена, rebalance).

## Фаза 4 — маркетплейс-минимум: ⬜ отложена (решение от 2026-08-16)

## Отклонения от жёстких ограничений
Нет. Математика Phase-1/Phase-2 не менялась (см. NOTICE.md); vector-knapsack не
реализовывался; в `parallax/` ничего не записано.
