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

from looma.logging_config import get_logger

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
    from looma.orchestrator.payloads import PayloadMissing, collect

    try:
        return collect("looma_stage", human="кода стадии", dirs=_payload_dirs())
    except PayloadMissing as exc:
        raise ModelError(str(exc)) from None


def _payload_dirs():
    """Где может лежать код стадии, в порядке доверия."""
    from looma.orchestrator.payloads import payload_dirs

    return payload_dirs("looma_stage")


# --------------------------------------------------------------- движки
# Какой драйвер нужен vLLM, и почему именно этот.
#
# vLLM жёстко требует конкретную версию torch (0.28 — ровно 2.13.0). Сборки
# torch лежат на индексах по версиям CUDA, и версии там разные: на `cu124`
# последняя — 2.6.0, нужной нет вовсе. Узел с таким драйвером поставит vLLM
# «успешно»: pip увидит несовпадение и доставит torch с обычного PyPI, поверх
# правильной сборки. Каталог окружения останется с именем cu124, а внутри
# окажется torch под драйвер новее.
#
# Падает это не на установке, а через десять минут — при первом обращении к
# карте, сообщением «CUDA initialization: driver is too old», в котором ни
# слова про подмену. Поэтому решение принимается здесь, до запуска.
# Ниже 12.6 нет сборки torch, которую требует закреплённый vLLM (2.9.1 лежит
# на cu126 и новее; на cu124 последняя вообще 2.6.0). Узел с таким драйвером
# поставит vLLM «успешно», подменив torch колесом с PyPI, и упадёт только при
# первом обращении к карте.
MIN_CUDA_FOR_VLLM = (12, 6)

# Версия vLLM — часть нашего контракта, а не плавающая зависимость.
#
# Стадия лезет в его внутренности: подменяет `get_pp_indices`, снимает проверки
# инициализации весов, собирает `SchedulerOutput` в обход его планировщика,
# спрашивает спецификацию KV-кэша у исполнителя. Всё это приватные API, и они
# меняются между версиями молча.
#
# Без версии в строке требований это не просто «поставится свежий». Имя
# окружения на узле считается от СТРОК требований (agent/looma_agent/tasks/
# spec.py), поэтому `vllm` без пина даёт один и тот же отпечаток для любой
# версии: узел, собравший окружение месяц назад, держит vLLM той поры и не
# обновится никогда, а соседний соберёт сегодняшний. Две стадии одного
# конвейера — на разных версиях, под одним именем каталога.
#
# Меняя пин, мы меняем строку, отпечаток и, значит, окружение: узлы соберут
# новое вместо того, чтобы тихо остаться на старом. Это и есть механизм
# обновления, другого тут нет.
#
# 0.14.0, а не свежая: это единственная версия, на которой стадия проверена на
# живой карте от загрузки среза до логитов. Свежая уже показала одно
# расхождение во внутренностях (конфиг требуется раньше распределённой группы),
# и переход на неё — отдельная работа, а не побочный эффект установки.
VLLM_PIN = "vllm==0.14.0"

# Торч под vLLM пинится РОВНО той версией, которую он требует, и это не
# аккуратность.
#
# Агент ставит torch первым проходом из индекса по драйверу узла, а всё
# остальное — вторым, из обычного PyPI. Если версия там не совпадёт с той, что
# требует vLLM, второй проход молча доставит torch с PyPI поверх правильной
# сборки: каталог окружения останется с именем cu128, а внутри окажется чужое
# колесо. Падает это не на установке, а при первом обращении к карте.
#
# Совпадающий пин лишает второй проход повода что-либо менять. Версии здесь
# берутся из метаданных самого vLLM (`requires_dist`), а не подбираются.
VLLM_TORCH = ("torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1")

# Остальное намеренно без версий: мы зовём только их публичные API. `torch` без
# пина оставлен переносимому движку — там сборку выбирает агент по драйверу
# узла, и наборы версий на индексах cu124/cu126/cu128 разные, так что пин,
# годный для одного узла, сделал бы неустановимым другой.
STAGE_REQUIREMENTS = ("torch", "transformers", "safetensors", "huggingface-hub")


def stage_requirements(engine: str) -> List[str]:
    """Что поставить на узле под этот движок."""
    if engine != "vllm":
        return list(STAGE_REQUIREMENTS)
    # Свой torch вместо непинованного: иначе первый проход поставит с индекса
    # одну версию, а vLLM вторым проходом стянет другую с PyPI.
    rest = [name for name in STAGE_REQUIREMENTS if name != "torch"]
    return [*VLLM_TORCH, *rest, VLLM_PIN]


def cuda_tuple(version: str) -> Optional[Tuple[int, int]]:
    """«12.4» -> (12, 4). Ничего не разобрали — None, а не догадка."""
    parts = (version or "").strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def vllm_refusal(node: dict) -> str:
    """Почему на этом узле нельзя vLLM. Пусто — значит можно.

    Отказ, а не предупреждение: «можно, но сломается» — это то же самое, что
    нельзя, только узнаётся дороже.
    """
    need = ".".join(str(part) for part in MIN_CUDA_FOR_VLLM)
    reported = (node.get("cuda_version") or "").strip()
    found = cuda_tuple(reported)
    if found is None:
        return (f"{node.get('node_id', 'узел')} не сообщил версию CUDA — "
                f"похоже, карты на нём нет; vLLM нужна CUDA {need} или новее")
    if found < MIN_CUDA_FOR_VLLM:
        return (f"на {node.get('node_id', 'узле')} драйвер под CUDA {reported}, "
                f"а vLLM нужна {need} или новее: под {reported} нет сборки "
                "torch, которую он требует. Обновите драйвер узла или берите "
                "движок transformers")
    return ""
