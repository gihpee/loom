# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/model_runner.py — ParallaxVLLMGroupCoordinator
# (ответ «первый/последний ранг» по диапазону слоёв вместо номера процесса),
# ParallaxVLLMModelRunner.load_model (подмена get_pp_indices на время загрузки)
# и initialize_vllm_model_runner (порядок поднятия).
# Изменения: решения вынесены в чистые функции и проверяются без vLLM и без
# карты; подмена get_pp_indices возвращается на место через try/finally в любом
# случае; отказы называют причину вместо предупреждения в лог и продолжения с
# наполовину настроенным движком.
"""Заставить vLLM собрать только слои этой стадии.

Ключ ко всему — то, что конвейер у vLLM **уже есть**. Он спрашивает у себя две
вещи, и обе можно ответить по-своему:

    get_pp_indices(...)      какие слои строит этот ранг
    pp_group.is_first_rank   строить ли embed_tokens
    pp_group.is_last_rank    строить ли lm_head

Обычно на них отвечает номер процесса в распределённой группе. Мы отвечаем
**диапазоном слоёв** — и модель собирается срезом, без единой строчки про
конкретную архитектуру. Именно поэтому здесь нет и не будет файлов вида
`qwen3.py`: слои строит сам vLLM, мы только говорим ему, какие.

Распределённая группа при этом настоящая, но из одного процесса: vLLM без неё
не поднимается, а обмен между стадиями всё равно идёт не через неё, а через
агента.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Tuple

logger = logging.getLogger("loom_stage.vllm_runner")


class RunnerRefused(RuntimeError):
    """Собрать движок тут нельзя, и вот почему."""


# ------------------------------------------------------------------ решения
def stage_role(start_layer: int, end_layer: int, num_layers: int) -> Tuple[bool, bool]:
    """Первая ли это стадия и последняя ли. От этого зависит, что вообще
    строится: эмбеддинги у первой, голова у последней."""
    if num_layers <= 0:
        raise RunnerRefused("в конфиге модели нет числа слоёв")
    if not 0 <= start_layer < end_layer <= num_layers:
        raise RunnerRefused(
            f"срез [{start_layer}, {end_layer}) не помещается в модель "
            f"из {num_layers} слоёв")
    return start_layer == 0, end_layer == num_layers


@contextlib.contextmanager
def layer_range(start_layer: int, end_layer: int) -> Iterator[None]:
    """На время загрузки: сколько бы слоёв vLLM ни насчитал, строит он наши.

    Через try/finally, и это не аккуратность: подменённая функция — глобальная,
    и оставить её после неудачной загрузки значит испортить всё, что попробует
    грузить модель после нас, включая сообщение об ошибке.
    """
    try:
        import vllm.distributed.utils as utils
    except ImportError as exc:
        raise RunnerRefused(
            f"внутренности vLLM недоступны ({exc}); стадия рассчитана на "
            "версию, закреплённую в требованиях") from None

    original = utils.get_pp_indices

    def ours(num_layers: int, rank: int, world_size: int):
        return start_layer, end_layer

    utils.get_pp_indices = ours
    try:
        yield
    finally:
        utils.get_pp_indices = original


def coordinator_for(start_layer: int, end_layer: int, num_layers: int):
    """Группа, которая отвечает про первый и последний ранг по слоям.

    Класс собирается внутри функции: его базовый тип живёт в vLLM, которого на
    машине без карты нет вовсе, а модуль обязан импортироваться и там.
    """
    from vllm.distributed.parallel_state import GroupCoordinator

    is_first, is_last = stage_role(start_layer, end_layer, num_layers)

    class StageGroupCoordinator(GroupCoordinator):
        @property
        def is_first_rank(self) -> bool:
            return is_first

        @property
        def is_last_rank(self) -> bool:
            return is_last

    return StageGroupCoordinator


def _kv_spec_of(runner):
    """Спросить, какой нужен KV-кэш. У кого именно — зависит от версии.

    Мест два, и они менялись между версиями vLLM: метод бывает на самом
    исполнителе и бывает на модели, спрятанной под cudagraph-обёрткой (та не
    пропускает его наружу и падает с «not exists in the runnable of cudagraph
    wrapper»).

    У первоисточника здесь запасной путь: посчитать форму кэша по конфигу —
    число голов, размер головы. Мы так не делаем. Неверная форма кэша не
    падает, она портит внимание: ответы остаются связными и становятся
    неправильными, и найти это можно только по качеству.

    Поэтому спрашиваем везде, где он бывает, а не угадываем.
    """
    tried = []
    for name, candidate in _candidates(runner):
        tried.append(name)
        ask = getattr(candidate, "get_kv_cache_spec", None)
        if ask is None:
            continue
        try:
            return ask()
        except AttributeError:
            continue      # ещё одна обёртка, идём глубже
    raise RunnerRefused(
        "никто не рассказал про KV-кэш (смотрели: " + ", ".join(tried) + "). "
        "Похоже на это: " + _hints(runner) + ". Угадывать форму нельзя: "
        "неверная не падает, а портит внимание")


def _candidates(runner):
    """Кого спрашивать, от самого вероятного к самому глубокому."""
    # Сам исполнитель — в свежих версиях метод переехал сюда.
    yield "runner", runner
    get_model = getattr(runner, "get_model", None)
    if callable(get_model):
        try:
            yield "runner.get_model()", get_model()
        except Exception:
            pass
    model = getattr(runner, "model", None)
    seen = 0
    while model is not None and seen < 5:
        yield f"model{'.runnable' * seen}", model
        model = getattr(model, "runnable", None) or getattr(model, "module", None)
        seen += 1


def _hints(runner) -> str:
    """Что похожее нашлось поблизости.

    Чтобы следующий отказ называл, куда метод переехал, а не отправлял читать
    исходники vLLM. Этот приём здесь окупился уже дважды.
    """
    found = []
    for name, candidate in _candidates(runner):
        for attribute in dir(candidate):
            if "kv_cache" in attribute and callable(
                    getattr(candidate, attribute, None)):
                found.append(f"{name}.{attribute}")
        if len(found) > 8:
            break
    return ", ".join(found[:8]) or "ничего похожего"


def stage_runner_class(start_layer: int, end_layer: int, num_layers: int):
    """Исполнитель vLLM, знающий свой срез.

    Собирается внутри функции по той же причине, что и координатор: базовый
    тип живёт в vLLM, а модуль обязан импортироваться и там, где его нет.

    Добавляет к штатному ровно две вещи — заведение KV-кэша под наши слои и
    шаг, умеющий принять и отдать промежуточные тензоры.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    is_first, is_last = stage_role(start_layer, end_layer, num_layers)

    class StageRunner(GPUModelRunner):
        start_layer_index = start_layer
        end_layer_index = end_layer
        is_first_stage = is_first
        is_last_stage = is_last

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        # ------------------------------------------------------- KV-кэш
        def prepare_cache(self, *, block_size: int, max_model_len: int):
            """Завести кэш под наши слои и только под них.

            Спецификацию спрашиваем у самой модели: сколько голов и какого
            размера — знает она, а угадывать это по конфигу значит однажды
            угадать неверно и получить кэш не той формы. Такая ошибка не
            падает, она портит внимание.
            """
            import torch
            from vllm.v1.core.kv_cache_utils import (generate_scheduler_kv_cache_config,
                                                     get_kv_cache_configs)
            from vllm.v1.core.kv_cache_manager import KVCacheManager

            spec = _kv_spec_of(self)

            free, _total = torch.cuda.mem_get_info(self.device.index or 0)
            available = int(free * self.cache_config.gpu_memory_utilization)
            logger.info("под KV-кэш: %.1f ГБ из %.1f ГБ свободных",
                        available / 1024**3, free / 1024**3)

            configs = get_kv_cache_configs(vllm_config=self.vllm_config,
                                           kv_cache_specs=[spec],
                                           available_memory=[available])

            # У кэша две половины, и путать их нельзя.
            #
            # РАБОЧАЯ живёт в исполнителе: она выделяет сами тензоры и
            # связывает их со слоями внимания, попутно собирая attn_groups.
            # Без неё модель грузится, кэш «есть», а первый же шаг падает на
            #     IndexError: list index out of range
            # в attn_groups[0] — и по этому сообщению не догадаться, что
            # пропущен целый шаг инициализации.
            #
            # ПЛАНИРОВЩИКОВАЯ — это менеджер блоков: он решает, кому какие
            # блоки выдать, и именно его зовёт наша сборка батча.
            settle = getattr(self, "initialize_kv_cache", None)
            if settle is None:
                raise RunnerRefused(
                    "исполнитель не умеет initialize_kv_cache — без неё слои "
                    "внимания останутся без кэша, и первый же шаг упадёт")
            settle(configs[0])

            self.kv_cache_config = generate_scheduler_kv_cache_config(configs)
            self.kv_cache_manager = KVCacheManager(
                kv_cache_config=self.kv_cache_config, max_model_len=max_model_len,
                enable_caching=False, use_eagle=False, log_stats=False,
                enable_kv_cache_events=False, dcp_world_size=1,
                hash_block_size=block_size)
            logger.info("кэш разложен: групп внимания %d",
                        len(getattr(self, "attn_groups", []) or []))
            return self.kv_cache_manager

        # --------------------------------------------------------- шаг
        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          *args, **kwargs):
            """Шаг модели. Перед ним — буфер под входящие тензоры.

            vLLM не принимает их напрямую: он копирует пришедшее в СВОЙ буфер
            и нарезает по размеру батча. Буфера у неголовной стадии нет, пока
            его не завели, и шаг падает на

                assert self.intermediate_tensors is not None

            — утверждении, из которого не следует, что кто-то должен был этот
            буфер выделить.
            """
            if not self.is_first_stage:
                self._ensure_incoming()
            return super().execute_model(scheduler_output, intermediate_tensors,
                                         *args, **kwargs)

        def _ensure_incoming(self):
            """Завести буфер один раз и переиспользовать.

            Он размером с самый большой батч, и создавать его на каждом шаге
            значило бы выделять гигабайты в горячем пути.
            """
            if getattr(self, "intermediate_tensors", None) is not None:
                return
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype, device=self.device)
            logger.info("буфер под входящие тензоры заведён")

    return StageRunner


def replace_pipeline_group(start_layer: int, end_layer: int, num_layers: int) -> None:
    """Подменить группу конвейера на ту, что считает по слоям.

    Отдельным шагом после `initialize_model_parallel`: vLLM собирает свою
    группу сам, и переопределить в ней два свойства проще, чем построить свою
    с нуля со всем, что к ней прилагается.
    """
    import torch
    from vllm.distributed import parallel_state

    existing = parallel_state._PP
    if existing is None:
        raise RunnerRefused(
            "vLLM не поднял группу конвейера; порядок инициализации нарушен")

    made = coordinator_for(start_layer, end_layer, num_layers)(
        group_ranks=[existing.ranks],
        local_rank=existing.local_rank,
        torch_distributed_backend=torch.distributed.get_backend(existing.device_group),
        use_device_communicator=existing.use_device_communicator,
        use_message_queue_broadcaster=existing.mq_broadcaster is not None,
        group_name="pp",
    )
    parallel_state._PP = made
    logger.info("группа конвейера считает по слоям: первая=%s, последняя=%s",
                made.is_first_rank, made.is_last_rank)
