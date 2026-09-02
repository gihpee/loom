# NOTICE — код, заимствованный из Parallax

Loom заимствует и адаптирует код проекта **Parallax**
(https://github.com/GradientHQ/parallax, arXiv:2509.26182). Локальный
read-only источник: `../dllmi/parallax/`. Этот файл дублирует (но не заменяет)
inline-атрибуцию в шапке каждого заимствованного модуля.

Заимствованного стало меньше: планировщик Phase-1/Phase-2 и роутер удалены
вместе со старой архитектурой. vLLM-адаптер возвращается по частям — ради
continuous batching, которого у собственного исполнителя нет и не будет.

| Модуль в `loom/` | Источник в `parallax/` | Изменения |
|---|---|---|
| `payloads/loom_stage/loom_stage/loader.py` | `src/parallax/server/shard_loader.py` (загрузка среза слоёв, ремап ключей на локальные индексы) + `src/parallax/vllm/model_runner.py` (роли is_first_peer/is_last_peer) | Реализация на torch/transformers вместо MLX и вместо наследования vLLM `GPUModelRunner`: нужен device-agnostic исполнитель (cpu/cuda), проверяемый без GPU-стенда и не привязанный к внутренним API конкретной версии vLLM. Идея (грузить только свой диапазон, локальная индексация слоёв, опускать embed/lm_head на ненужных стадиях) сохранена. |
| `payloads/loom_stage/loom_stage/executor.py` | `src/parallax/server/executor/base_executor.py` (цикл стадии: принять hidden_states → прогнать слои → отдать дальше; выборка токена на последней стадии; per-request состояние) | Исполнение на torch/transformers; в v0 одна последовательность на запрос без continuous batching. |
| `payloads/loom_stage/loom_stage/server.py` | `src/parallax/server/executor/base_executor.py` (роль первой стадии как владельца запроса) | HTTP-поверхность стадии вместо ZMQ-сокетов; OpenAI-совместимый `/v1/chat/completions` реализован здесь, а не отдельным vLLM-фронтендом. Транспорт — не ZMQ и не прямой Lattica, а канал агента: стадия адресует **рангу**, а куда его везти, решает агент. |
| `payloads/loom_stage/loom_stage/vllm_patch.py` | `src/parallax/vllm/monkey_patch_utils/weight_loader.py` + `src/parallax/vllm/monkey_patch.py` (снятие проверок инициализации `embed_tokens` и `lm_head` на стадиях, где их законно нет) | Одна функция вместо модуля с глобальным состоянием: стадия внутри процесса одна и не меняется. Добавлен явный отказ с названной причиной, когда внутренностей vLLM нет или они переехали, — вместо ImportError из глубины. Всё, что не про эти два веса, проходит насквозь: побитый чекпоинт обязан остаться ошибкой. |
| `payloads/loom_stage/loom_stage/vllm_runner.py` | `src/parallax/vllm/model_runner.py` (`ParallaxVLLMGroupCoordinator` — ответ «первый/последний ранг» по диапазону слоёв; `load_model` — подмена `get_pp_indices` на время загрузки; порядок поднятия из `initialize_vllm_model_runner`) | Решения (роль стадии, границы среза) вынесены в чистые функции и проверяются без vLLM и без карты. Подмена `get_pp_indices` возвращается на место через `try/finally` в любом случае: она глобальная, и оставленная после неудачи портит следующую загрузку. Негодный срез отвергается, а не берётся молча — иначе стадия считает чужие слои и отвечает связной чушью, нигде не падая. |
| `payloads/loom_stage/loom_stage/vllm_engine.py` | `src/parallax/vllm/model_runner.py` (`initialize_vllm_model_runner` — порядок поднятия: заплаты, распределённая группа, подмена группы конвейера, конфиги, загрузка) | Без LoRA, MoE-роутинга и спекулятивного декодирования: нечем проверить и незачем нести. Проверка карты — до всего остального, чтобы отказ назывался железом, а не приходил из глубины vLLM. Добавлена сверка числа собранных слоёв с запрошенным: подмена `get_pp_indices` — единственное, что удерживает vLLM от сборки всей модели, и её молчаливый провал даёт стадию, которая ест всю карту и не говорит ни слова. |
| `agent/loom_agent/hwinfo.py` | `src/parallax/server/server_info.py` (`HardwareInfo.detect`, `_GPU_DB`/`_match_gpu_specs`, `_APPLE_PEAK_FP16`, `detect_node_hardware`) | Убрана обязательная зависимость от torch/mlx (основной путь — NVML, затем torch, затем `nvidia-smi`); неизвестная карта не бросает исключение, а получает консервативную оценку; добавлены vram_total/vram_free и detection_source; таблица карт расширена; VRAM суммируется по всем видимым картам. |

Всё остальное — собственный код Loom: `src/loom/*` целиком, `agent/*` кроме
`hwinfo.py`, `payloads/loom_stage/loom_stage/wire.py`, `relay/*`, тесты.
