# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/monkey_patch_utils/weight_loader.py и
# src/parallax/vllm/monkey_patch.py — снятие проверок инициализации весов,
# которых на этой стадии конвейера быть не должно.
# Изменения: одна функция вместо модуля с глобальным состоянием (стадия внутри
# процесса одна и не меняется); сообщения на русском и объясняющие, почему
# пропуск законен; отдельная проверка, что версия vLLM вообще та, куда мы
# лезем.
"""Заплаты на vLLM: пусть он не требует того, чего у этой стадии нет.

vLLM грузит модель целиком и в конце проверяет, что все веса пришли из
чекпоинта. Стадия конвейера держит только свой кусок слоёв, и у неё законно
отсутствуют:

    embed_tokens   — есть только у первой стадии
    lm_head        — есть только у последней

Без заплаты загрузка падает с «not initialized from checkpoint» — сообщением,
которое выглядит как побитый чекпоинт, а не как «так и задумано».

Это временная мера, и так же она помечена у первоисточника: когда vLLM научится
конвейеру в нужном нам виде, заплата уйдёт. Поэтому она узкая — снимает ровно
две проверки и только на тех стадиях, где их отсутствие законно.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("looma_stage.vllm_patch")

_applied = False


class PatchRefused(RuntimeError):
    """Заплату наложить не на что, и вот почему."""


def allow_missing_ends(*, is_first: bool, is_last: bool) -> None:
    """Разрешить отсутствие эмбеддингов и головы там, где их не бывает.

    Идемпотентна: стадия в процессе одна, и накладывать заплату дважды нечего.
    """
    global _applied
    if _applied:
        return
    try:
        from vllm.model_executor.model_loader import default_loader
    except ImportError as exc:
        raise PatchRefused(
            f"внутренности загрузчика vLLM недоступны ({exc}); "
            "проверьте версию — стадия рассчитана на ту, что закреплена в "
            "требованиях") from None

    loader = default_loader.DefaultModelLoader
    original = loader.load_weights

    def load_weights(self, model, model_config):
        try:
            original(self, model, model_config)
        except ValueError as exc:
            text = str(exc)
            if "not initialized from checkpoint" not in text:
                raise
            missing_head = "model.embed_tokens.weight" in text
            missing_tail = "lm_head.weight" in text
            if missing_head and not is_first:
                logger.info("нет embed_tokens — так и должно быть, стадия не первая")
                return
            if missing_tail and not is_last:
                logger.info("нет lm_head — так и должно быть, стадия не последняя")
                return
            # Всё остальное — настоящая беда: либо чекпоинт неполон, либо
            # срез слоёв посчитан неверно. Молчать про это нельзя.
            raise

    loader.load_weights = load_weights
    _applied = True
    logger.info("загрузчик vLLM не потребует %s",
                "ничего лишнего" if (is_first and is_last)
                else ("lm_head" if is_first else
                      ("embed_tokens" if is_last else "embed_tokens и lm_head")))
