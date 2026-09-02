# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/model_runner.py — initialize_vllm_model_runner
# (порядок поднятия: заплаты, распределённая группа, подмена группы конвейера,
# конфиги, загрузка) и ParallaxVLLMModelRunner.load_model.
# Изменения: без LoRA, MoE-роутинга и спекулятивного декодирования — их тут
# нечем проверить и незачем нести; поднятие разбито на именованные шаги, чтобы
# отказ называл, на каком именно; проверка карты до всего остального.
"""Движок vLLM, собирающий только слои этой стадии.

Веха 1: он **грузит** свой срез и рассказывает, что загрузил. Шага модели и
батча тут ещё нет — они следующие, и до них надо убедиться, что приём вообще
работает на настоящей карте.

Почему так, а не всё сразу: приём держится на трёх вмешательствах во
внутренности vLLM (см. vllm_runner.py и vllm_patch.py), и любое из них может
разойтись с версией движка. Узнать это на загрузке — минуты; узнать на батче,
проделав всю работу, — недели.

Проверить на узле, ничего не разворачивая:

    python -m loom_stage.vllm_engine --weights Qwen/Qwen3-4B \\
        --start-layer 0 --end-layer 18 --num-model-layers 36
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from loom_stage.vllm_runner import (RunnerRefused, layer_range, replace_pipeline_group,
                                    stage_role, stage_runner_class)

logger = logging.getLogger("loom_stage.vllm_engine")

# Конфиг vLLM, установленный на всю жизнь процесса.
#
# У него есть глобальный «текущий конфиг», и части движка спрашивают его сами,
# без всяких аргументов: бэкенды внимания, CustomOp'ы. Вне контекста это падает
#     AssertionError: Current vLLM config is not set
# из места, которое к конфигу отношения не имеет — например, из раскладки
# KV-кэша.
#
# Держим открытым, а не оборачиваем каждый вызов: стадия в процессе одна и
# живёт столько же, сколько процесс, а забыть обернуть один вызов из десяти —
# ровно тот способ получить это исключение через неделю.
_CONFIG = None

# Сколько памяти карты отдать под веса и KV-кэш, когда квоты не назвали.
# Не 0.9, как у vLLM по умолчанию: узел чужой, и на нём живёт не только эта
# задача — рядом может стоять вторая стадия того же кластера.
DEFAULT_UTILISATION = 0.7


@dataclass
class LoadedShard:
    """Что получилось загрузить. Возвращается, чтобы это можно было показать
    и сравнить с тем, что просили."""

    start_layer: int
    end_layer: int
    num_layers: int
    is_first: bool
    is_last: bool
    dtype: str
    runner: object

    def as_dict(self) -> dict:
        return {
            "layers": [self.start_layer, self.end_layer],
            "of": self.num_layers,
            "first": self.is_first,
            "last": self.is_last,
            "dtype": self.dtype,
        }


def require_cuda() -> None:
    """Отказать до того, как что-то поднято.

    vLLM без карты не работает, и выясняется это глубоко внутри — сообщением,
    по которому не видно, что дело в железе, а не в модели или срезе.
    """
    try:
        import torch
    except ImportError as exc:
        raise RunnerRefused(f"нет torch ({exc})") from None
    if not torch.cuda.is_available():
        raise RunnerRefused(
            "vLLM работает только на CUDA, а карты на этом узле не видно. "
            "Для CPU и Apple есть движок torch — он медленнее и не умеет "
            "батчить, но считает везде")


def _config_with(kind, **options):
    """Собрать конфиг vLLM, отбросив поля, которых в этой версии нет.

    Поля конфигов переезжают между версиями, и лишний аргумент роняет всё
    поднятие целиком — сообщением про имя, а не про то, что версия другая.
    Отброшенное называется вслух: молча потерянный `enforce_eager` вернул бы
    захват графов и падение внутри него.
    """
    import dataclasses

    try:
        known = {field.name for field in dataclasses.fields(kind)}
    except TypeError:
        return kind(**options)
    dropped = sorted(set(options) - known)
    if dropped:
        logger.warning("%s не знает про %s — эта версия vLLM устроена иначе",
                       kind.__name__, ", ".join(dropped))
    return kind(**{name: value for name, value in options.items() if name in known})


def prepare_weights(weights: str, *, start_layer: int, end_layer: int,
                    is_first: bool, is_last: bool, dtype: str) -> str:
    """Положить рядом ровно те веса, которые нужны этой стадии.

    Две разные экономии, и обе заметные:

    **Скачивание.** Из репозитория берутся метаданные, а по ним — только те
    файлы safetensors, где лежат наши слои. Половина модели вместо целой.

    **Чтение.** vLLM, в отличие от нашего исполнителя, не терпит неполного
    чекпоинта: он перечисляет каждый файл из `model.safetensors.index.json` и
    открывает его, так что недостающий — ошибка, а не экономия. Поэтому рядом
    собирается «вид»: симлинки на нужные файлы плюс переписанный индекс, где
    упомянуты только они.

    Если урезать нечего — единственный файл, незнакомые имена ключей — вернётся
    исходный путь. Это не отказ: стадия просто прочитает больше, чем ей нужно.
    """
    from loom_stage.loader import ShardSpec, build_stage_checkpoint_view, resolve_model_path

    spec = ShardSpec(model_path=weights, start_layer=start_layer,
                     end_layer=end_layer, is_first=is_first, is_last=is_last,
                     dtype=dtype)
    local = resolve_model_path(weights, shard=spec)
    view = build_stage_checkpoint_view(local, spec)
    if view != local:
        logger.info("читаем урезанный чекпоинт: %s", view)
    else:
        logger.info("чекпоинт урезать нечем, читаем целиком: %s", local)
    return view


def _build_config(model_path: str, *, dtype: str, max_model_len: int,
                  utilisation: float, block_size: int, max_sequences: int,
                  max_batched_tokens: int):
    """Конфиги vLLM. Всё, чего мы не используем, названо явно нулём или None —
    молчаливое умолчание тут означало бы «как получится»."""
    import torch
    from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                             ParallelConfig, SchedulerConfig, VllmConfig)

    # Без torch.compile и без захвата CUDA-графов.
    #
    # Штатный движок перед захватом делает прогревочные прогоны и компилирует
    # всё заранее. Мы правим исполнителем напрямую, прогрева не делаем — и
    # первый же настоящий шаг запускает компиляцию ВНУТРИ захвата графа, где
    # нельзя даже прочитать состояние генератора:
    #
    #   RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during
    #   CUDA graph capture
    #
    # Захват тут и не нужен: он рассчитан на формы батчей, которые выбирает
    # сам vLLM, а у нас их выбирает первая стадия. Плата — eager вместо
    # скомпилированного, то есть медленнее на шаг; вернуть это можно, добавив
    # честный прогрев, но сначала конвейер должен просто заработать.
    model = _config_with(ModelConfig,
        model=model_path, tokenizer=model_path, tokenizer_mode="auto",
        trust_remote_code=True, dtype=dtype, seed=0,
        max_model_len=max_model_len, max_logprobs=1, enforce_eager=True,
    )
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=block_size,
                                 gpu_memory_utilization=utilisation,
                                 swap_space=0, cache_dtype="auto"),
        parallel_config=ParallelConfig(pipeline_parallel_size=1,
                                       tensor_parallel_size=1,
                                       distributed_executor_backend=None),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max(max_batched_tokens, model.max_model_len),
            max_num_seqs=max_sequences, max_model_len=model.max_model_len,
            is_encoder_decoder=False, enable_chunked_prefill=False),
        device_config=DeviceConfig(device=torch.device("cuda:0")),
        load_config=LoadConfig(load_format="auto"),
        lora_config=None, speculative_config=None, quant_config=None,
        kv_transfer_config=None, kv_events_config=None, additional_config={},
    )


def _start_distributed() -> None:
    """Поднять распределённое окружение vLLM из одного процесса.

    Настоящей группы у нас нет и не нужно: обмен между стадиями идёт через
    агента, а не через torch.distributed. Но vLLM без неё не собирается, так
    что она заводится вырожденной — один ранг, сам себе мир.
    """
    import os

    from vllm.distributed import parallel_state

    if parallel_state.model_parallel_is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "0")
    parallel_state.init_distributed_environment()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                             pipeline_model_parallel_size=1)


def load_shard(model_path: str, *, start_layer: int, end_layer: int,
               num_model_layers: int, dtype: str = "bfloat16",
               vram_quota_bytes: int = 0, max_model_len: int = 4096,
               block_size: int = 16, max_sequences: int = 64,
               max_batched_tokens: int = 16384) -> LoadedShard:
    """Собрать модель из одних только наших слоёв.

    Порядок шагов не переставляется, и каждый стоит там, где стоит:

    1. заплаты — до всякой загрузки;
    2. конфиг ставится текущим ДО распределённой группы. Свежий vLLM спрашивает
       конфиг уже внутри `initialize_model_parallel`, и без него падает на
       assert'е, в котором про конвейер нет ни слова: «Current vLLM config is
       not set... or a CustomOp was instantiated at module import time». Более
       ранние версии конфиг там не трогают, так что поставить его раньше —
       строго безопаснее, чем позже;
    3. группа конвейера подменяется после того, как vLLM собрал свою;
    4. срез слоёв навязывается только на время самой загрузки.
    """
    from loom_stage import vllm_patch

    require_cuda()
    is_first, is_last = stage_role(start_layer, end_layer, num_model_layers)
    logger.info("собираю слои [%d, %d) из %d: первая=%s, последняя=%s",
                start_layer, end_layer, num_model_layers, is_first, is_last)

    model_path = prepare_weights(model_path, start_layer=start_layer,
                                 end_layer=end_layer, is_first=is_first,
                                 is_last=is_last, dtype=dtype)
    vllm_patch.allow_missing_ends(is_first=is_first, is_last=is_last)

    utilisation = DEFAULT_UTILISATION
    if vram_quota_bytes > 0:
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        # Доля карты, а не байты: vLLM меряет именно так. Потолок 0.95 —
        # выше него он не оставляет места под собственные буферы.
        utilisation = min(0.95, max(0.05, vram_quota_bytes / max(1, total)))
        logger.info("квота %.1f ГБ на карте %.1f ГБ — беру %.2f",
                    vram_quota_bytes / 1024**3, total / 1024**3, utilisation)

    config = _build_config(model_path, dtype=dtype, max_model_len=max_model_len,
                           utilisation=utilisation, block_size=block_size,
                           max_sequences=max_sequences,
                           max_batched_tokens=max_batched_tokens)
    _hold_config(config)

    _start_distributed()
    replace_pipeline_group(start_layer, end_layer, num_model_layers)

    runner = stage_runner_class(start_layer, end_layer, num_model_layers)(
        vllm_config=config, device=config.device_config.device)
    with layer_range(start_layer, end_layer):
        runner.load_model()
    runner.prepare_cache(block_size=block_size, max_model_len=max_model_len)

    built = _count_layers(runner)
    wanted = end_layer - start_layer
    if built and built != wanted:
        raise RunnerRefused(
            f"просили {wanted} слоёв, а собралось {built}: приём разошёлся с "
            "этой версией vLLM, и считать она будет не то")
    logger.info("загружено слоёв: %s", built or "не удалось сосчитать")
    return LoadedShard(start_layer=start_layer, end_layer=end_layer,
                       num_layers=num_model_layers, is_first=is_first,
                       is_last=is_last, dtype=dtype, runner=runner)


def step(shard: LoadedShard, sequences, *, incoming=None, first_step: bool):
    """Один шаг движка над батчем, который выбрали снаружи.

    Возвращает то же, что и собственный исполнитель стадии: скрытые состояния
    на всех стадиях кроме последней, логиты — на последней. Различать их
    вызывающему не нужно.
    """
    from loom_stage import vllm_batch

    # Сначала то, что можно проверить, ничего не трогая: пустой батч и
    # отсутствующие тензоры — это не сбой движка, а неправильный вызов, и
    # звучать они должны так же.
    batch = list(sequences)
    if not batch:
        raise RunnerRefused("шаг без единой последовательности")
    if not shard.is_first and incoming is None:
        raise RunnerRefused(
            "неголовной стадии нечего считать: тензоры от предыдущей не пришли")

    runner = shard.runner
    form = vllm_batch.prefill if first_step else vllm_batch.decode
    scheduled = form(batch, runner)
    answer = runner.execute_model(scheduled, incoming if not shard.is_first else None)

    if shard.is_last:
        return None, _logits_from(runner, answer, expected=len(batch))
    return _hidden_from(runner, answer), None


class VllmEngine:
    """Стадия на vLLM, какой её видит `server.py`.

    Всё, что тут есть, — это обёртка вокруг `load_shard` и `step`: сам движок
    выше по файлу и ничего про конвейер Loom не знает. Класс нужен затем, что
    голове нельзя различать движки — она обязана звать одно и то же и получать
    одинаково устроенный ответ.

    Веса грузит он сам, из пути. Отдать ему уже собранную моделью стадию
    нельзя: она заняла бы карту вторым экземпляром тех же слоёв, и на карту,
    которой хватало впритык, стадия просто не поднялась бы.
    """

    #: Батч из нескольких — то, ради чего этот движок вообще нужен.
    batches = True

    def __init__(self, model_path: str, *, start_layer: int, end_layer: int,
                 num_model_layers: int, dtype: str = "bfloat16",
                 vram_quota_bytes: int = 0, max_requests: int = 64,
                 max_model_len: int = 4096, **_options) -> None:
        import torch

        self.torch = torch
        self.shard = load_shard(model_path, start_layer=start_layer,
                                end_layer=end_layer,
                                num_model_layers=num_model_layers, dtype=dtype,
                                vram_quota_bytes=vram_quota_bytes,
                                max_model_len=max_model_len,
                                max_sequences=max_requests)
        self.is_first = self.shard.is_first
        self.is_last = self.shard.is_last
        self._live: set = set()

    # ------------------------------------------------------------ счёт
    def step_batch(self, sequences, *, incoming=None, first_step: bool):
        """Один шаг над батчем. Возвращает `(карта тензоров, логиты)`.

        Ровно одно из двух не None: на последней стадии логиты, на прочих —
        тензоры. Так же отвечает и собственный исполнитель.
        """
        hidden, logits = step(self.shard, sequences,
                              incoming=self._incoming(incoming),
                              first_step=first_step)
        for sequence in sequences:
            self._live.add(sequence.request_id)
        if hidden is None:
            return None, logits
        return dict(hidden.tensors), None

    def _incoming(self, tensors):
        """Карта тензоров с провода — в то, что понимает vLLM.

        Превращение прячется здесь, а не у зовущего: голова обязана звать оба
        движка одинаково, а `IntermediateTensors` — тип vLLM, и знать о нём
        стадии на собственном исполнителе незачем.
        """
        from vllm.sequence import IntermediateTensors

        if tensors is None or isinstance(tensors, IntermediateTensors):
            return tensors
        device = getattr(self.shard.runner, "device", None)
        if device is not None:
            tensors = {name: value.to(device) for name, value in tensors.items()}
        return IntermediateTensors(tensors)

    def sample_batch(self, logits, sequences) -> list:
        """По токену на последовательность, в порядке батча."""
        from loom_stage import batch_wire

        rows = logits if getattr(logits, "dim", lambda: 1)() > 1 else logits[None]
        batch_wire.check_rows(list(sequences), int(rows.shape[0]))
        return [self.sample(row, temperature=sequence.temperature,
                            top_p=sequence.top_p, seed=sequence.seed)
                for row, sequence in zip(rows, sequences)]

    def sample(self, logits, *, temperature: float = 0.0, top_p: float = 1.0,
               seed: Optional[int] = None) -> int:
        """Выбор токена — тот же, что у собственного исполнителя.

        Не из vLLM: его сэмплер живёт внутри его же планировщика, которого мы
        как раз обходим. Одинаковый выбор на обоих движках стоит дороже, чем
        экономия на этих десяти строках, — иначе один и тот же промпт даёт
        разные ответы в зависимости от того, чем считали.
        """
        from loom_stage.executor import ShardExecutor

        return ShardExecutor.sample(self, logits, temperature=temperature,
                                    top_p=top_p, seed=seed)

    # ------------------------------------------------------------ уборка
    def free(self, request_id: str) -> None:
        from loom_stage import vllm_batch

        vllm_batch.release(self.shard.runner, request_id)
        self._live.discard(request_id)

    def active_requests(self) -> int:
        return len(self._live)

    def shutdown(self) -> None:
        shutdown()


def _hidden_from(runner, answer):
    """Промежуточные тензоры, как их отдала эта версия vLLM.

    Версии отличаются: одна возвращает их прямо, другая складывает в состояние
    исполнителя. Гадать не будем — если не нашли, скажем, ЧТО получили, чтобы
    первый же прогон на узле это назвал.
    """
    from vllm.sequence import IntermediateTensors

    if isinstance(answer, IntermediateTensors):
        return answer
    for name in ("intermediate_tensors", "hidden_states"):
        found = getattr(answer, name, None) or getattr(
            getattr(runner, "execute_model_state", None), name, None)
        if isinstance(found, IntermediateTensors):
            return found
    raise RunnerRefused(
        f"шаг не отдал промежуточных тензоров, а вернул {type(answer).__name__}; "
        "в этой версии vLLM они лежат где-то ещё")


def _logits_from(runner, answer, *, expected: int):
    """Логиты последней стадии — по строке на последовательность.

    Строка логитов не подписана именем запроса: соответствие держится только
    на порядке батча. Поэтому число строк сверяется с числом последовательностей
    здесь и сразу. Разойдись оно молча — токен уехал бы чужому клиенту, и
    заметить это можно было бы разве что по жалобе на бессвязный ответ.
    """
    state = getattr(runner, "execute_model_state", None)
    for source in (answer, state):
        found = getattr(source, "logits", None)
        if found is None:
            continue
        shape = list(getattr(found, "shape", []) or [])
        rows = shape[0] if len(shape) > 1 else 1
        if rows != expected:
            raise RunnerRefused(
                f"движок вернул {rows} строк логитов на {expected} "
                f"последовательностей (форма {shape}); соответствие строки и "
                "запроса держится только на порядке батча, а он уже разошёлся")
        return found
    raise RunnerRefused(
        f"шаг не отдал логитов, а вернул {type(answer).__name__}; "
        "в этой версии vLLM они лежат где-то ещё")


def _hold_config(config) -> None:
    """Установить текущий конфиг vLLM и не отпускать."""
    global _CONFIG
    import contextlib

    from vllm.config import set_current_vllm_config

    if _CONFIG is not None:
        return
    _CONFIG = contextlib.ExitStack()
    _CONFIG.enter_context(set_current_vllm_config(config))


def shutdown() -> None:
    """Разобрать распределённое окружение.

    Без этого torch на выходе жалуется на утечку — и жалуется по делу: группа
    держит дескрипторы и разделяемую память. Для одиночной проверки это
    безобидно, а для стадии, которая перезапускается по кругу при неудачном
    старте, нет.

    Ни один шаг не обязателен: разбираем то, что поднялось, и молчим про
    остальное. Падать на уборке — худшее, что можно сделать с процессом,
    который и так уходит.
    """
    global _CONFIG
    if _CONFIG is not None:
        try:
            _CONFIG.close()
        except Exception:
            logger.debug("контекст конфига не закрылся", exc_info=True)
        _CONFIG = None
    try:
        from vllm.distributed import parallel_state

        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    except Exception:
        logger.debug("разбор группы vLLM не прошёл", exc_info=True)
    try:
        import torch

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        logger.debug("разбор группы torch не прошёл", exc_info=True)


def _count_layers(runner) -> int:
    """Сколько слоёв на самом деле собралось.

    Проверка не ради аккуратности: подмена `get_pp_indices` — единственное,
    что удерживает vLLM от сборки всей модели, и её молчаливый провал даёт
    стадию, которая считает всё и ест всю карту.
    """
    model = getattr(runner, "model", None)
    for path in (("model", "layers"), ("layers",)):
        found = model
        for name in path:
            found = getattr(found, name, None)
            if found is None:
                break
        if found is not None:
            try:
                return sum(1 for layer in found if type(layer).__name__ != "PPMissingLayer")
            except TypeError:
                continue
    return 0


def _prompts(shard: LoadedShard, prompts: List[List[int]], *, incoming=None):
    """Prefill над батчем из скольких угодно последовательностей.

    Имена запросов задаются здесь и по порядку — тот же порядок повторит
    следующая стадия, собрав батч из тех же промптов. В этом и смысл проверки:
    состав батча не пересчитывается на каждой стадии, а повторяется, и если бы
    он разъехался, тензоры пришли бы не той длины.
    """
    from loom_stage.scheduler import Sequence

    batch = [Sequence(request_id=f"проверка-{index}", prompt_ids=list(ids))
             for index, ids in enumerate(prompts)]
    return step(shard, batch, incoming=incoming, first_step=True)


def _parse_prompts(text: str) -> List[List[int]]:
    """«1,2,3;4,5» -> [[1,2,3],[4,5]]. Пустые промпты отбрасываются: батч из
    пустой последовательности vLLM примет и посчитает ни за чем."""
    prompts = []
    for chunk in text.split(";"):
        ids = [int(piece) for piece in chunk.split(",") if piece.strip()]
        if ids:
            prompts.append(ids)
    if not prompts:
        raise RunnerRefused(f"в --prompt-ids не нашлось ни одного токена: {text!r}")
    return prompts


def _save_hidden(tensors, path: str) -> dict:
    """Сложить промежуточные тензоры в файл нашим же форматом провода.

    Через `wire`, а не pickle: это тот самый формат, которым они поедут между
    машинами, и проверить его заодно — бесплатно.
    """
    import json

    import torch

    from loom_stage import wire

    saved = {}
    blob = bytearray()
    for name, tensor in tensors.items():
        data, shape, dtype = wire.to_wire(torch, tensor)
        saved[name] = {"at": len(blob), "size": len(data), "shape": shape,
                       "dtype": dtype}
        blob.extend(data)
    with open(path, "wb") as handle:
        handle.write(blob)
    with open(path + ".json", "w") as handle:
        json.dump(saved, handle)
    return saved


def _load_hidden(path: str, device):
    """Обратно, в тензоры на карте."""
    import json

    import torch

    from loom_stage import wire
    from vllm.sequence import IntermediateTensors

    with open(path + ".json") as handle:
        layout = json.load(handle)
    blob = pathlib_read(path)
    restored = {}
    for name, where in layout.items():
        piece = blob[where["at"]:where["at"] + where["size"]]
        restored[name] = wire.from_wire(torch, piece, where["shape"],
                                        where["dtype"]).to(device)
    return IntermediateTensors(restored)


def pathlib_read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def main(argv: Optional[List[str]] = None) -> int:
    """Проверка на узле.

    Без аргументов про тензоры — только загрузка (веха 1). С `--dump-hidden`
    прогоняет промпт и складывает промежуточные тензоры в файл; с
    `--load-hidden` поднимает следующую стадию, читает их и печатает логиты.

    Двумя прогонами, а не одним процессом: группа конвейера у vLLM глобальная,
    и две стадии рядом дрались бы за неё. Заодно проверяется формат провода —
    тот самый, которым тензоры поедут между машинами.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="loom_stage.vllm_engine",
        description="Загрузить срез слоёв через vLLM и, по желанию, прогнать шаг")
    parser.add_argument("--weights", required=True, help="путь или репозиторий")
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--num-model-layers", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vram-quota-bytes", type=int, default=0)
    parser.add_argument("--prompt-ids", default="",
                        help="токены через запятую; несколько промптов — через "
                             "точку с запятой, и тогда шаг считает их батчем")
    parser.add_argument("--dump-hidden", default="",
                        help="куда сложить промежуточные тензоры")
    parser.add_argument("--load-hidden", default="",
                        help="откуда их взять — для стадии, которая не первая")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    answer: dict = {}
    try:
        shard = load_shard(args.weights, start_layer=args.start_layer,
                           end_layer=args.end_layer,
                           num_model_layers=args.num_model_layers,
                           dtype=args.dtype,
                           vram_quota_bytes=args.vram_quota_bytes)
        answer.update(shard.as_dict())

        if args.prompt_ids:
            prompts = _parse_prompts(args.prompt_ids)
            incoming = (_load_hidden(args.load_hidden, shard.runner.device)
                        if args.load_hidden else None)
            answer["последовательностей в батче"] = len(prompts)
            answer["токенов в батче"] = sum(len(ids) for ids in prompts)
            hidden, logits = _prompts(shard, prompts, incoming=incoming)
            if hidden is not None:
                answer["отдала"] = "скрытые состояния"
                answer["тензоры"] = sorted(hidden.tensors)
                if args.dump_hidden:
                    answer["сложено"] = _save_hidden(hidden.tensors, args.dump_hidden)
            if logits is not None:
                answer["отдала"] = "логиты"
                answer["форма логитов"] = list(getattr(logits, "shape", []))
                # По строке на последовательность, в порядке батча. Печатаем
                # выбранные токены: одинаковые токены на разных промптах —
                # первый признак, что батч склеился в одну последовательность.
                answer["токены"] = [int(row.argmax().item()) for row in
                                    (logits if logits.dim() > 1 else logits[None])]
    except RunnerRefused as exc:
        print(f"не вышло: {exc}")
        return 2
    finally:
        shutdown()
    print(json.dumps(answer, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
