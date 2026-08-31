"""Что за модель просят и как её разрезать между узлами.

Здесь ровно два вопроса: сколько в модели слоёв, и кому какие достанутся.
Всё остальное про модель знает сама стадия — оркестратор её не загружает и
весов не видит.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loom.logging_config import get_logger

logger = get_logger(__name__)

HF_CONFIG = "https://huggingface.co/{repo}/resolve/main/config.json"


class ModelError(ValueError):
    """Про эту модель нельзя ответить на нужные вопросы, и вот почему."""


@dataclass(frozen=True)
class ModelInfo:
    repo: str
    num_layers: int
    hidden_size: int = 0
    architecture: str = ""

    def as_dict(self) -> dict:
        return {
            "repo": self.repo,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "architecture": self.architecture,
        }


def describe(repo: str, *, token: str = "") -> ModelInfo:
    """Прочитать config.json модели на HuggingFace.

    Одно короткое обращение к сети и никакой загрузки весов: чтобы разложить
    модель по узлам, достаточно знать число слоёв. Веса качает та стадия,
    которой они нужны, и только свой кусок.
    """
    repo = (repo or "").strip().strip("/")
    if not repo or "/" not in repo:
        raise ModelError(
            f"{repo!r} не похоже на имя модели: нужно 'владелец/название', "
            "например 'Qwen/Qwen3-8B'"
        )
    # Проверяем состав имени до того, как оно попадёт в URL: иначе кириллица
    # или пробел вылезают ошибкой кодировки из глубины urllib, где про модель
    # уже ничего не сказано.
    if not all(c.isascii() and (c.isalnum() or c in "-_./") for c in repo):
        raise ModelError(
            f"{repo!r} не может быть именем на HuggingFace: там латиница, цифры "
            "и -_./"
        )
    request = urllib.request.Request(HF_CONFIG.format(repo=repo))
    token = token or os.environ.get("HF_TOKEN", "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            config = json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ModelError(
                f"{repo} закрыта: нужен HF_TOKEN с доступом к ней"
            ) from None
        if exc.code == 404:
            raise ModelError(f"на HuggingFace нет {repo}") from None
        raise ModelError(f"HuggingFace ответил {exc.code} про {repo}") from None
    except (urllib.error.URLError, ValueError) as exc:
        raise ModelError(f"не удалось прочитать config.json у {repo}: {exc}") from None

    layers = config.get("num_hidden_layers") or config.get("n_layer")
    if not layers:
        raise ModelError(
            f"в config.json у {repo} не сказано число слоёв — такую модель "
            "нельзя разрезать между узлами"
        )
    architectures = config.get("architectures") or []
    return ModelInfo(
        repo=repo,
        num_layers=int(layers),
        hidden_size=int(config.get("hidden_size") or 0),
        architecture=architectures[0] if architectures else "",
    )


def split_layers(num_layers: int, stages: int,
                 weights: Optional[List[float]] = None) -> List[Tuple[int, int]]:
    """Кому какой диапазон слоёв.

    По умолчанию поровну. Если переданы веса (свободная VRAM узлов) — слои
    делятся пропорционально им: на стенде из 4090 и 3090 равный разрез
    упирается в меньшую карту, и половина большей простаивает.

    Каждой стадии достаётся хотя бы один слой: стадия без слоёв — это лишний
    сетевой переход, который ничего не считает.
    """
    if stages < 1:
        raise ModelError("нужна хотя бы одна стадия")
    if stages > num_layers:
        raise ModelError(
            f"в модели {num_layers} слоёв, а стадий просят {stages}: "
            "стадия без слоёв — это сетевой переход, который ничего не считает"
        )
    # Сначала каждому по одному слою — стадия без слоёв это сетевой переход,
    # который ничего не считает, — а остаток раздаём по долям. Так сумма сходится
    # по построению, без округления и починки округления после него.
    share = [1] * stages
    rest = num_layers - stages
    if weights and len(weights) == stages and sum(weights) > 0:
        portions = [w / sum(weights) for w in weights]
    else:
        portions = [1 / stages] * stages
    given = [int(rest * portion) for portion in portions]
    # Целые части розданы; дробные хвосты решают, кому достанутся оставшиеся
    # слои — начиная с того, у кого хвост длиннее.
    leftover = rest - sum(given)
    order = sorted(range(stages), key=lambda i: -(rest * portions[i] - given[i]))
    for i in order[:leftover]:
        given[i] += 1
    share = [one + extra for one, extra in zip(share, given)]

    ranges: List[Tuple[int, int]] = []
    start = 0
    for size in share:
        ranges.append((start, start + size))
        start += size
    return ranges


def stage_payload() -> Dict[str, bytes]:
    """Файлы стадии, которые уедут в задачу как её вход.

    Механизм общий для всех нагрузок — см. orchestrator/payloads.py; здесь
    только имя и то, во что превращается его отсутствие для вызывающего.
    """
    from loom.orchestrator.payloads import PayloadMissing, collect

    try:
        return collect("loom_stage", human="кода стадии", dirs=_payload_dirs())
    except PayloadMissing as exc:
        raise ModelError(str(exc)) from None


def _payload_dirs():
    """Где может лежать код стадии, в порядке доверия."""
    from loom.orchestrator.payloads import payload_dirs

    return payload_dirs("loom_stage")
