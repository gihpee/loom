"""То, что оркестратор обязан помнить после перезапуска.

Узлы переживают перезапуск оркестратора: их задачи никто не останавливал, они
продолжают считать и держать карты. Оркестратор без этого файла просыпался с
пустой памятью — задачи шли, а сказать о них было нечего, и остановить их было
нельзя, потому что остановка адресуется по task_id, которого он больше не знал.

Здесь только то, что нельзя восстановить из доклада узла. Узел рассказывает
про task_id, состояние и время; про то, что вот эти две задачи — стадии одной
модели по имени `Qwen/Qwen3-4B`, знает только оркестратор, и знает лишь пока
жив его процесс. Отсюда файл.

Не база: состояние — это десятки записей, читается целиком один раз при старте
и переписывается целиком при изменении. SQLite добавила бы схему и её миграции
ради данных, которые помещаются в один `json.dumps`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from looma.logging_config import get_logger

logger = get_logger(__name__)

# Версия формата. Читатель, встретивший чужую, начинает с чистого листа, а не
# гадает по ключам: половина восстановленного состояния хуже, чем ничего.
FORMAT = 1


class StateStore:
    """Снимок состояния на диске. Пишется целиком, читается целиком."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """Прочитать снимок. Любая беда с файлом — это пустое состояние.

        Оркестратор, отказавшийся стартовать из-за побитого файла состояния,
        не даёт сделать ровно то единственное, что тут помогает — запустить
        оркестратор и разобраться.
        """
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("состояние в %s не читается (%s); начинаем с пустого",
                           self.path, exc)
            return {}
        if not isinstance(raw, dict) or raw.get("format") != FORMAT:
            logger.warning("состояние в %s другого формата; начинаем с пустого",
                           self.path)
            return {}
        return raw

    def save(self, snapshot: dict) -> None:
        """Записать снимок так, чтобы файла-полуфабриката не существовало.

        Через временный файл рядом и переименование: убитый посреди записи
        процесс оставляет либо прежний снимок, либо новый, но не половину
        нового — а именно её никто не сможет прочитать при следующем старте.
        """
        body = json.dumps({"format": FORMAT, **snapshot}, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", dir=self.path.parent, prefix=f".{self.path.name}.",
                suffix=".part", delete=False, encoding="utf-8")
            try:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, self.path)
        except OSError as exc:
            logger.warning("состояние не записалось в %s: %s", self.path, exc)
            self._discard(getattr(handle, "name", ""))

    @staticmethod
    def _discard(path: str) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass


def store_for(data_dir: Optional[str]) -> Optional[StateStore]:
    """Хранилище в каталоге данных, или ничего, если каталога нет.

    Тесты поднимают хаб без диска и не должны получить файл в рабочем каталоге
    от одного факта запуска.
    """
    if not data_dir:
        return None
    return StateStore(Path(data_dir) / "state.json")
