#!/usr/bin/env python3
"""Завести учётную запись из рабочего дерева.

Настоящая команда живёт в пакете (`looma.accounts.bootstrap`), потому что в
образ оркестратора копируется только `src`. Здесь — обёртка для запуска без
установки пакета.

    scripts/create_admin.py ivan@looma.ru                          # из репозитория
    docker compose exec orchestrator \
        python -m looma.accounts.bootstrap ivan@looma.ru           # на сервере
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from looma.accounts.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
