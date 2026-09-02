# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/batch_info.py — form_vllm_batch_prefill,
# form_vllm_batch_decode, _build_vllm_request, release_vllm_request:
# конструирование SchedulerOutput в обход планировщика vLLM.
# Изменения: без LoRA и мультимодальности — их тут нечем проверить; состав
# батча приходит одной структурой Sequence вместо объектов Request парallax;
# отказ выделить блоки поднимает исключение с причиной вместо возврата None
# (молчаливый None выше по стеку превращается в «шаг не сделан» без объяснения);
# освобождение блоков собрано в один вызов и не падает на неизвестном запросе.
"""Батч, который выбрал кто-то другой.

У vLLM v1 нет входа «прогони вот эти токены». Единственный путь в модель —
`execute_model(scheduler_output, ...)`, где `SchedulerOutput` — внутренняя
структура его собственного планировщика. Поэтому, чтобы конвейер работал,
эту структуру приходится собирать снаружи.

Зачем снаружи. Состав батча выбирает ПЕРВАЯ стадия: она видит очередь
запросов. Остальные обязаны прогнать ровно тот же набор в том же порядке —
иначе позиции в KV-кэше разъедутся, и ошибки не будет, будет чушь. Так что
планировщик у неголовных стадий не участвует вовсе: им говорят, что считать.

Два разных случая, и путать их нельзя:

    prefill  — первый шаг запроса: считается весь промпт, блоки выделяются
               под всю его длину, запрос для vLLM новый.
    decode   — каждый следующий: считается один токен, блок выделяется один,
               запрос для vLLM уже известен.

Это внутренние типы vLLM, и они меняются между версиями. Модуль намеренно
маленький и весь про одно: собрать структуру и освободить блоки.
"""

from __future__ import annotations

import logging
from typing import List

from loom_stage.scheduler import Sequence

logger = logging.getLogger("loom_stage.vllm_batch")

# Состав батча выбирает планировщик, поэтому и тип его элемента живёт там:
# он едет между стадиями и не должен зависеть от того, каким движком его
# посчитают. Здесь он только читается.
__all__ = ["Sequence", "BatchRefused", "prefill", "decode", "release"]


class BatchRefused(RuntimeError):
    """Батч собрать нельзя, и вот почему."""


def sampling_for(sequence: Sequence):
    """Параметры выборки в том виде, в каком их понимает vLLM."""
    from vllm.sampling_params import SamplingParams

    return SamplingParams(
        temperature=float(sequence.temperature),
        top_p=float(sequence.top_p),
        max_tokens=int(sequence.max_tokens),
        seed=sequence.seed,
    )


def _request_for(sequence: Sequence, runner, *, with_outputs: bool):
    from vllm.v1.request import Request

    made = Request(
        request_id=sequence.request_id,
        prompt_token_ids=list(sequence.prompt_ids),
        sampling_params=sampling_for(sequence),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        block_hasher=getattr(runner, "request_block_hasher", None),
        lora_request=None,
    )
    if with_outputs and sequence.output_ids:
        made.append_output_token_ids(list(sequence.output_ids))
    return made


def _cache(runner):
    manager = getattr(runner, "kv_cache_manager", None)
    if manager is None:
        raise BatchRefused(
            "у движка нет менеджера KV-кэша: он заводится после загрузки "
            "модели, и без него выделять блоки нечем")
    return manager


def _groups(runner) -> int:
    config = getattr(runner, "kv_cache_config", None)
    return len(getattr(config, "kv_cache_groups", []) or []) or 1


def prefill(sequences: List[Sequence], runner):
    """Первый шаг: посчитать промпты целиком."""
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    if not sequences:
        raise BatchRefused("пустой батч")
    manager = _cache(runner)
    made, scheduled, total = [], {}, 0
    allocated = []
    try:
        for sequence in sequences:
            request = _request_for(sequence, runner, with_outputs=False)
            computed_blocks, computed = manager.get_computed_blocks(request)
            fresh = max(len(sequence.prompt_ids) - computed, 0)
            if fresh:
                blocks = manager.allocate_slots(
                    request=request, num_new_tokens=fresh,
                    num_new_computed_tokens=computed,
                    new_computed_blocks=computed_blocks if computed else None)
                if blocks is None:
                    raise BatchRefused(
                        f"не хватило места в KV-кэше под {sequence.request_id} "
                        f"({len(sequence.prompt_ids)} токенов промпта)")
                all_blocks = computed_blocks + blocks if computed else blocks
            else:
                all_blocks = computed_blocks
            allocated.append(request)
            made.append(NewRequestData(
                req_id=sequence.request_id,
                prompt_token_ids=list(sequence.prompt_ids),
                mm_features=[], sampling_params=sampling_for(sequence),
                pooling_params=None, block_ids=all_blocks.get_block_ids(),
                num_computed_tokens=computed, lora_request=None,
                prompt_embeds=None))
            scheduled[sequence.request_id] = len(sequence.prompt_ids)
            total += len(sequence.prompt_ids)
    except Exception:
        # Половина выделенных блоков — хуже, чем ни одного: они не вернутся
        # сами, и следующий батч упрётся в память, которую никто не держит.
        for request in allocated:
            _free_quietly(manager, request)
        raise

    return SchedulerOutput(
        scheduled_new_reqs=made,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled, total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * _groups(runner),
        finished_req_ids=set(), free_encoder_mm_hashes=[],
        kv_connector_metadata=None)


def decode(sequences: List[Sequence], runner):
    """Каждый следующий шаг: один токен на последовательность."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    if not sequences:
        raise BatchRefused("пустой батч")
    manager = _cache(runner)
    ids, fresh_tokens, all_tokens = [], [], {}
    blocks, computed, produced, scheduled = [], [], [], {}

    for sequence in sequences:
        outputs = list(sequence.output_ids) or _outputs_known_to(runner, sequence)
        ids.append(sequence.request_id)
        all_tokens[sequence.request_id] = outputs
        # Вход этого шага — последний выданный токен, и только он.
        fresh_tokens.append([outputs[-1]] if outputs else [])

        request = _request_for(sequence, runner, with_outputs=True)
        request.num_computed_tokens = sequence.computed
        slot = manager.allocate_slots(request=request, num_new_tokens=1,
                                      num_new_computed_tokens=0)
        if slot is None:
            raise BatchRefused(
                f"не хватило места в KV-кэше на шаг {sequence.request_id}")
        blocks.append(slot.get_block_ids(allow_none=True))
        computed.append(sequence.computed)
        produced.append(len(outputs))
        scheduled[sequence.request_id] = 1

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=ids, resumed_req_ids=set(), new_token_ids=fresh_tokens,
            all_token_ids=all_tokens, new_block_ids=blocks,
            num_computed_tokens=computed, num_output_tokens=produced),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * _groups(runner),
        finished_req_ids=set(), free_encoder_mm_hashes=[],
        kv_connector_metadata=None)


def _outputs_known_to(runner, sequence: Sequence) -> List[int]:
    """Что движок сам помнит про этот запрос.

    Неголовной стадии токены не присылают: она их не видит и не сэмплирует.
    Но vLLM их помнит — и без этого её представление о длине разойдётся с
    головой, а это те самые разъехавшиеся позиции.
    """
    known = getattr(runner, "requests", {}) or {}
    state = known.get(sequence.request_id)
    return list(getattr(state, "output_token_ids", []) or []) if state else []


def release(runner, request_id: str) -> None:
    """Отпустить блоки запроса. Безопасно звать на незнакомом."""
    manager = getattr(runner, "kv_cache_manager", None)
    if manager is None:
        return
    known = getattr(runner, "requests", {}) or {}
    state = known.get(request_id)
    if state is not None:
        _free_quietly(manager, state)
        known.pop(request_id, None)


def _free_quietly(manager, request) -> None:
    try:
        manager.free(request)
    except Exception:
        logger.debug("блоки %s не отпустились", getattr(request, "request_id", "?"),
                     exc_info=True)
