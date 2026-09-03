"""Подключение к Postgres и накат миграций.

Одно соединение на процесс не годится: HTTP обслуживается несколькими
корутинами сразу, и общий курсор превратил бы их в очередь. Пул решает это и
заодно переживает разрыв — Postgres перезапускают, а оркестратор нет.

Миграции — пронумерованные файлы и таблица с тем, что уже накатано. Без
инструмента вроде alembic намеренно: пока схема размером в три таблицы, его
конфигурация будет длиннее самой схемы, а понять по ней, что произошло с базой,
станет сложнее, а не проще.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("looma.store")

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class DatabaseUnavailable(RuntimeError):
    """К базе не подключиться. Отдельным типом, потому что ответ на это —
    не «упасть», а «сказать, что именно недоступно»."""


def database_url() -> str:
    """Строка подключения. Пусто — базы нет, и это законное состояние: тесты и
    локальный запуск обходятся без неё."""
    return os.environ.get("LOOMA_DATABASE_URL", "").strip()


class Database:
    """Пул соединений и накат схемы."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._pool = None

    @property
    def ready(self) -> bool:
        return self._pool is not None

    async def connect(self, *, minimum: int = 1, maximum: int = 10) -> None:
        try:
            import asyncpg
        except ImportError:
            raise DatabaseUnavailable(
                "нет пакета asyncpg — без него оркестратор не умеет в Postgres. "
                "Установите зависимость 'db' или уберите LOOMA_DATABASE_URL"
            ) from None
        try:
            self._pool = await asyncpg.create_pool(self.url, min_size=minimum,
                                                   max_size=maximum)
        except Exception as exc:
            raise DatabaseUnavailable(
                f"не подключиться к базе: {exc}. Проверьте LOOMA_DATABASE_URL "
                "и что служба postgres поднялась") from None
        logger.info("база подключена")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def acquire(self):
        """Соединение из пула. Использовать только как `async with`."""
        if self._pool is None:
            raise DatabaseUnavailable("база не подключена")
        return self._pool.acquire()

    # ------------------------------------------------------------ миграции
    async def migrate(self, folder: Optional[Path] = None) -> List[str]:
        """Накатить то, чего ещё нет. Возвращает применённое, по порядку.

        Каждый файл — в своей транзакции. Не все вместе: половина накатанной
        схемы хуже, чем ясная остановка на конкретном файле, имя которого видно
        в таблице.
        """
        folder = folder or MIGRATIONS
        applied: List[str] = []
        async with self.acquire() as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name TEXT PRIMARY KEY,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
            done = {row["name"] for row in
                    await connection.fetch("SELECT name FROM schema_migrations")}
            for path in sorted(folder.glob("*.sql")):
                if path.name in done:
                    continue
                logger.info("накатываю %s", path.name)
                async with connection.transaction():
                    await connection.execute(path.read_text())
                    await connection.execute(
                        "INSERT INTO schema_migrations (name) VALUES ($1)",
                        path.name)
                applied.append(path.name)
        return applied


def pending(done: set, folder: Optional[Path] = None) -> List[str]:
    """Какие миграции ещё не накатаны. Отдельной функцией — чтобы порядок и
    выбор можно было проверить, не поднимая базу."""
    folder = folder or MIGRATIONS
    return [path.name for path in sorted(folder.glob("*.sql"))
            if path.name not in done]
