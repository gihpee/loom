# NOTICE — код, заимствованный из Parallax

Loom заимствует и адаптирует код проекта **Parallax**
(https://github.com/GradientHQ/parallax, arXiv:2509.26182). Локальный
read-only источник: `../dllmi/parallax/`. Этот файл дублирует (но не заменяет)
inline-атрибуцию в шапке каждого заимствованного модуля.

Заимствованного стало меньше: планировщик Phase-1/Phase-2, роутер и
vLLM-адаптер удалены вместе со старой архитектурой. Осталось четыре модуля.

| Модуль в `loom/` | Источник в `parallax/` | Изменения |
|---|---|---|
| `payloads/loom_stage/loom_stage/loader.py` | `src/parallax/server/shard_loader.py` (загрузка среза слоёв, ремап ключей на локальные индексы) + `src/parallax/vllm/model_runner.py` (роли is_first_peer/is_last_peer) | Реализация на torch/transformers вместо MLX и вместо наследования vLLM `GPUModelRunner`: нужен device-agnostic исполнитель (cpu/cuda), проверяемый без GPU-стенда и не привязанный к внутренним API конкретной версии vLLM. Идея (грузить только свой диапазон, локальная индексация слоёв, опускать embed/lm_head на ненужных стадиях) сохранена. |
| `payloads/loom_stage/loom_stage/executor.py` | `src/parallax/server/executor/base_executor.py` (цикл стадии: принять hidden_states → прогнать слои → отдать дальше; выборка токена на последней стадии; per-request состояние) | Исполнение на torch/transformers; в v0 одна последовательность на запрос без continuous batching. |
| `payloads/loom_stage/loom_stage/server.py` | `src/parallax/server/executor/base_executor.py` (роль первой стадии как владельца запроса) | HTTP-поверхность стадии вместо ZMQ-сокетов; OpenAI-совместимый `/v1/chat/completions` реализован здесь, а не отдельным vLLM-фронтендом. Транспорт — не ZMQ и не прямой Lattica, а канал агента: стадия адресует **рангу**, а куда его везти, решает агент. |
| `agent/loom_agent/hwinfo.py` | `src/parallax/server/server_info.py` (`HardwareInfo.detect`, `_GPU_DB`/`_match_gpu_specs`, `_APPLE_PEAK_FP16`, `detect_node_hardware`) | Убрана обязательная зависимость от torch/mlx (основной путь — NVML, затем torch, затем `nvidia-smi`); неизвестная карта не бросает исключение, а получает консервативную оценку; добавлены vram_total/vram_free и detection_source; таблица карт расширена; VRAM суммируется по всем видимым картам. |

Всё остальное — собственный код Loom: `src/loom/*` целиком, `agent/*` кроме
`hwinfo.py`, `payloads/loom_stage/loom_stage/wire.py`, `relay/*`, тесты.
