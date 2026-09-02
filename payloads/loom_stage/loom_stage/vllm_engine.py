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
from typing import List, Optional, Tuple

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

    Порядок шагов не переставляется: заплаты ложатся до всякой загрузки,
    группа конвейера подменяется после того, как vLLM собрал свою, а срез
    навязывается только на время самой загрузки.
    """
    from loom_stage import vllm_patch

    require_cuda()
    is_first, is_last = stage_role(start_layer, end_layer, num_model_layers)
    logger.info("собираю слои [%d, %d) из %d: первая=%s, последняя=%s",
                start_layer, end_layer, num_model_layers, is_first, is_last)

    vllm_patch.allow_missing_ends(is_first=is_first, is_last=is_last)
    _start_distributed()
    replace_pipeline_group(start_layer, end_layer, num_model_layers)

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

    runner = shard.runner
    form = vllm_batch.prefill if first_step else vllm_batch.decode
    scheduled = form(list(sequences), runner)

    if not shard.is_first and incoming is None:
        raise RunnerRefused(
            "неголовной стадии нечего считать: тензоры от предыдущей не пришли")
    answer = runner.execute_model(scheduled, incoming if not shard.is_first else None)

    if shard.is_last:
        return None, _logits_from(runner, answer)
    return _hidden_from(runner, answer), None


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


def _logits_from(runner, answer):
    """Логиты последней стадии."""
    state = getattr(runner, "execute_model_state", None)
    for source in (answer, state):
        found = getattr(source, "logits", None)
        if found is not None:
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


def _one_prompt(shard: LoadedShard, prompt_ids: List[int], *, incoming=None):
    """Один prefill над одной последовательностью. Ровно то, чем проверяется
    веха 2: батч из одного — частный случай батча, а не отдельный путь."""
    from loom_stage.vllm_batch import Sequence

    sequence = Sequence(request_id="проверка", prompt_ids=list(prompt_ids))
    return step(shard, [sequence], incoming=incoming, first_step=True)


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
                        help="через запятую; без них шаг не делается")
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
            ids = [int(piece) for piece in args.prompt_ids.split(",") if piece.strip()]
            incoming = (_load_hidden(args.load_hidden, shard.runner.device)
                        if args.load_hidden else None)
            hidden, logits = _one_prompt(shard, ids, incoming=incoming)
            if hidden is not None:
                answer["отдала"] = "скрытые состояния"
                answer["тензоры"] = sorted(hidden.tensors)
                if args.dump_hidden:
                    answer["сложено"] = _save_hidden(hidden.tensors, args.dump_hidden)
            if logits is not None:
                answer["отдала"] = "логиты"
                answer["форма логитов"] = list(getattr(logits, "shape", []))
    except RunnerRefused as exc:
        print(f"не вышло: {exc}")
        return 2
    finally:
        shutdown()
    print(json.dumps(answer, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
