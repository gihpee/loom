"""Веса моделей, которые скачиваются один раз на узел.

Без этого каждое развёртывание начинается с закачки: `HOME` у задачи указывает
в её собственный каталог, HuggingFace кладёт кэш туда же, а каталог живёт
ровно столько же, сколько задача. Одна и та же модель приезжала заново при
каждом перезапуске, а прошлые копии оставались лежать до тех пор, пока кто-то
не удалит группу вручную.

Ключ здесь не наш, и это сознательно. Кэш HuggingFace уже устроен правильно:
файлы лежат по хешу содержимого, снимки — по ревизии, и загрузка ЧАСТИ файлов
репозитория просто добавляет их в тот же снимок. Значит две стадии одной
модели с разными диапазонами слоёв делят метаданные и каждая приносит только
свои веса — без единой строчки с нашей стороны. Наша работа тут ровно одна:
дать этому кэшу пережить задачу и не дать ему съесть диск.

Что отсюда следует для вытеснения. Мы не знаем, какой задаче какой репозиторий
принадлежит: качает его сама задача, агент об этом не спрашивают. Поэтому
аренды по репозиториям нет, а есть отсрочка: недавно тронутое не трогаем.
Этого достаточно, потому что цена ошибки мала — веса нужны стадии только на
загрузке, дальше они в памяти карты, и удалённый файл стоит лишней закачки при
следующем старте, а не упавшего запроса.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("looma_agent.tasks.models")

# Сколько диска владельца узла отдать под веса. Больше, чем под окружения:
# одна модель бывает крупнее всех окружений вместе взятых, а альтернатива —
# качать её заново при каждом запуске.
DEFAULT_QUOTA_BYTES = int(os.environ.get("LOOMA_MODEL_QUOTA_BYTES", str(128 * 1024**3)))

# Сколько репозиторий считается «в работе» после последнего обращения. Покрывает
# скачивание и загрузку весов — то есть всё время, когда файлы кому-то нужны.
# Занизить это опаснее, чем завысить: снесённый на полпути снимок роняет
# загрузку стадии.
GRACE_S = float(os.environ.get("LOOMA_MODEL_GRACE_S", str(3600)))

# Как HuggingFace называет каталог репозитория внутри кэша.
REPO_PREFIX = "models--"


class ModelCache:
    """Каталог с весами, общий для всех задач узла."""

    def __init__(self, root: Path, *, quota_bytes: int = DEFAULT_QUOTA_BYTES,
                 grace_s: float = GRACE_S) -> None:
        self.root = root
        self.quota_bytes = max(0, int(quota_bytes))
        self.grace_s = max(0.0, float(grace_s))
        self.unusable = ""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Не отказ: без кэша задача просто скачает веса себе, как раньше.
            # Ронять из-за этого работу узла было бы хуже, чем медленно её
            # делать.
            self.unusable = f"кэш моделей в {self.root} недоступен ({exc.strerror})"
            logger.error("%s", self.unusable)

    # ------------------------------------------------------------ для задачи
    def env(self, inner_path: str) -> Dict[str, str]:
        """Переменные, которыми задача найдёт общий кэш.

        `inner_path` — путь, по которому каталог виден ИЗ задачи: в контейнере
        он другой, и путь снаружи ей ничего не говорит.

        Задаются все три имени: библиотеки читают то одно, то другое в
        зависимости от версии, а разойдись они — часть файлов уедет в кэш, а
        часть мимо, и заметно это станет по времени запуска, а не по ошибке.
        """
        if self.unusable:
            return {}
        return {
            "HF_HOME": inner_path,
            "HF_HUB_CACHE": f"{inner_path}/hub",
            "HUGGINGFACE_HUB_CACHE": f"{inner_path}/hub",
        }

    # ------------------------------------------------------------ вытеснение
    def sweep(self) -> int:
        """Убрать самое давнее, пока не уложимся в квоту. Вернуть сколько убрали."""
        if self.unusable or not self.quota_bytes:
            return 0
        entries = self._repositories()
        total = sum(size for _used, _path, size in entries)
        if total <= self.quota_bytes:
            return 0
        freed = 0
        deadline = time.time() - self.grace_s
        for used_at, path, size in entries:
            if total - freed <= self.quota_bytes:
                break
            if used_at > deadline:
                # Тронут недавно — возможно, прямо сейчас качается или
                # читается стадией. Снести такое значит уронить загрузку.
                continue
            logger.info("убираю веса %s (%.1f ГБ, не трогали %.0f мин) — кэш "
                        "моделей вышел за %.0f ГБ",
                        path.name, size / 1024**3, (time.time() - used_at) / 60,
                        self.quota_bytes / 1024**3)
            shutil.rmtree(path, ignore_errors=True)
            freed += size
        if total - freed > self.quota_bytes:
            logger.warning(
                "кэш моделей на %.1f ГБ больше квоты, но всё оставшееся трогали "
                "недавно — подожду следующего раза",
                (total - freed - self.quota_bytes) / 1024**3)
        return freed

    def _repositories(self) -> List[Tuple[float, Path, int]]:
        """Что лежит в кэше: (когда трогали, каталог, размер), давнее первым."""
        hub = self.root / "hub"
        found: List[Tuple[float, Path, int]] = []
        try:
            entries = list(hub.iterdir())
        except OSError:
            return found
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith(REPO_PREFIX):
                continue
            found.append((_used_at(entry), entry, _size(entry)))
        return sorted(found)

    def total_bytes(self) -> int:
        return sum(size for _used, _path, size in self._repositories())

    def snapshot(self) -> dict:
        return {
            "unusable": self.unusable,
            "bytes": self.total_bytes(),
            "quota_bytes": self.quota_bytes,
        }


def _used_at(repository: Path) -> float:
    """Когда репозиторием пользовались в последний раз.

    Берётся самое свежее время среди снимков, а не время самого каталога: он
    меняется только при добавлении ревизии, и по нему давно скачанная, но
    ежедневно используемая модель выглядела бы забытой.
    """
    latest = 0.0
    for candidate in (repository, repository / "snapshots", repository / "refs"):
        try:
            latest = max(latest, candidate.stat().st_mtime)
        except OSError:
            continue
    try:
        for snapshot in (repository / "snapshots").iterdir():
            try:
                latest = max(latest, snapshot.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return latest


def _size(repository: Path) -> int:
    """Сколько занимает репозиторий. Считаются blob'ы: снимки — это ссылки на
    них, и складывать те и другие значило бы посчитать всё дважды."""
    total = 0
    for base, _dirs, files in os.walk(repository):
        for name in files:
            path = os.path.join(base, name)
            try:
                stat = os.lstat(path)
            except OSError:
                continue
            if os.path.islink(path):
                continue
            total += stat.st_size
    return total
