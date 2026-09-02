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

from loom_stage.vllm_runner import RunnerRefused, layer_range, replace_pipeline_group, stage_role

logger = logging.getLogger("loom_stage.vllm_engine")

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


def _build_config(model_path: str, *, dtype: str, max_model_len: int,
                  utilisation: float, block_size: int, max_sequences: int,
                  max_batched_tokens: int):
    """Конфиги vLLM. Всё, чего мы не используем, названо явно нулём или None —
    молчаливое умолчание тут означало бы «как получится»."""
    import torch
    from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                             ParallelConfig, SchedulerConfig, VllmConfig)

    model = ModelConfig(
        model=model_path, tokenizer=model_path, tokenizer_mode="auto",
        trust_remote_code=True, dtype=dtype, seed=0,
        max_model_len=max_model_len, max_logprobs=1,
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

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(vllm_config=config, device=config.device_config.device)
    with layer_range(start_layer, end_layer):
        runner.load_model()

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


def main(argv: Optional[List[str]] = None) -> int:
    """Проверка на узле: загрузить срез и сказать, что вышло."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="loom_stage.vllm_engine",
        description="Загрузить срез слоёв через vLLM и рассказать, что вышло")
    parser.add_argument("--weights", required=True, help="путь или репозиторий")
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--num-model-layers", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vram-quota-bytes", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        shard = load_shard(args.weights, start_layer=args.start_layer,
                           end_layer=args.end_layer,
                           num_model_layers=args.num_model_layers,
                           dtype=args.dtype,
                           vram_quota_bytes=args.vram_quota_bytes)
    except RunnerRefused as exc:
        print(f"не вышло: {exc}")
        return 2
    print(json.dumps(shard.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
