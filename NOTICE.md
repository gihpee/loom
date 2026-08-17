# NOTICE — код, заимствованный из Parallax

Loom заимствует и адаптирует код проекта **Parallax**
(https://github.com/GradientHQ/parallax, arXiv:2509.26182). Локальный
read-only источник: `../dllmi/parallax/`. Этот файл дублирует (но не заменяет)
inline-комментарии атрибуции в каждом заимствованном модуле.

| Модуль в `loom/` | Источник в `parallax/` | Изменения |
|---|---|---|
| `src/loom/planning/model_info.py` | `src/scheduling/model_info.py` | Нет изменений в логике расчётов; заменён импорт логгера. |
| `src/loom/planning/capacity.py` | `src/scheduling/node.py` (`Node.get_decoder_layer_capacity`, `per_decoder_layer_kv_cache_memory`) | Capacity вынесен из `Node` в явный входной параметр `ShardCapacity` от Resource Broker; бюджет считается от `vram_quota_bytes` вместо полного объёма устройства. Формулы (param/kv ratio, вычет embedding/lm_head, tie_embedding, mlx_bit_factor) сохранены. |
| `src/loom/planning/node.py` | `src/scheduling/node.py` | `get_decoder_layer_capacity()` и KV-бюджет делегируют в явный `capacity: ShardCapacity`; `max_requests` считает KV-бюджет от квоты; убрана зависимость от torch. `model_info` сохранён только для roofline-оценки латентности (инстанс планировщика = одна модель). Roofline-математика не менялась. |
| `src/loom/planning/batching.py` | `src/parallax_utils/utils.py` (`compute_max_tokens_in_cache`, `derive_max_batch_size`, `compute_max_batch_size`) | Убраны torch/mlx и живой опрос памяти устройства; байтовый бюджет KV передаётся явно (из квоты брокера). Арифметика не менялась. |
| `src/loom/planning/layer_allocation.py` | `src/scheduling/layer_allocation.py` | Только переименованы импорты. Математика Phase-1 DP (`dp(i, open_residuals, finished_pipes)`) и water-filling не менялась; capacity приходит через параметризованный `Node`. |
| `src/loom/planning/node_management.py` | `src/scheduling/node_management.py` | Нет, скопировано as-is (переименованы импорты). |
| `src/loom/planning/request_routing.py` | `src/scheduling/request_routing.py` | Только переименованы импорты. Математика Phase-2 (DAG DP) не менялась; источник живых данных — Perf-map Store через `loom/perfmap/sync.py` вместо DHT. |
| `src/loom/planning/scheduler.py` | `src/scheduling/scheduler.py` | Только переименованы импорты; используется как программная библиотека (по инстансу на модель), не как синглтон процесса. |
| `worker/loom_worker/shard/loader.py` | `src/parallax/server/shard_loader.py` (загрузка среза слоёв, ремап ключей на локальные индексы) + `src/parallax/vllm/model_runner.py` (роли is_first_peer/is_last_peer) | Реализация на torch/transformers вместо MLX и вместо наследования vLLM `GPUModelRunner`: нужен device-agnostic исполнитель (cpu/cuda), проверяемый без GPU-стенда и не привязанный к внутренним API конкретной версии vLLM. Идея (грузить только свой диапазон, локальная индексация слоёв, опускать embed/lm_head на ненужных стадиях) сохранена. |
| `worker/loom_worker/shard/executor.py` | `src/parallax/server/executor/base_executor.py` (цикл стадии: принять hidden_states → прогнать слои → отдать дальше; выборка токена на последней стадии; per-request состояние) | Исполнение на torch/transformers; транспорт не ZMQ/Lattica, а релей через оркестратор; в v0 одна последовательность на запрос без continuous batching. |
| `worker/loom_worker/shard/server.py` | `src/parallax/server/executor/base_executor.py` (роль первого peer'а как владельца запроса) | HTTP-поверхность стадии вместо ZMQ-сокетов; OpenAI-совместимый `/v1/chat/completions` реализован здесь, а не отдельным vLLM Rust frontend'ом. |
| `worker/loom_worker/hwinfo.py` | `src/parallax/server/server_info.py` (`HardwareInfo.detect`, `_GPU_DB`/`_match_gpu_specs`, `_APPLE_PEAK_FP16`, `detect_node_hardware`) | Убрана обязательная зависимость от torch/mlx (основной путь — NVML, затем torch, затем `nvidia-smi`); неизвестная карта не бросает исключение, а получает консервативную оценку; добавлены vram_total/vram_free и detection_source; таблица карт расширена. |
| `src/loom/api/lb_strategy.py` | `src/router/lb_strategy.py` | Нет, скопировано as-is (формулы скоринга/выбора не менялись). |
| `src/loom/api/endpoints.py` | `src/router/main.py` (`Endpoint`, `EndpointMetrics`, `EndpointRegistry`) | Точечная правка по ТЗ: добавлено обязательное поле `model_id` (ключ реестра — `(model_id, base_url)`), матчинг запроса только на эндпоинты той же модели; обрезаны HTTP-конфигурация на лету, троттлинг-бакеты и httpx-клиент. |

Собственный код Loom (без заимствованной логики): `src/loom/logging_config.py`,
`src/loom/perfmap/*`, `src/loom/proto/*.proto`, `src/loom/orchestrator/*`
(registry, broker, pool, controller, gateway, keys, tunnel, signing, config,
server), весь `worker/` кроме `hwinfo.py`, тесты.
