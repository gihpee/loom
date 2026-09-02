"""Отдельность программы — условие, а не намерение.

Через полгода кто-нибудь «переиспользует» отсюда одну функцию из проекта, и
утилита перестанет ставиться клиенту, у которого нет репозитория Looma. Проверка
дешевле такого разговора.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "looma_connect"
# Всё, что живёт в репозитории Looma и чего у клиента не будет.
FORBIDDEN = ("looma.", "looma_agent", "looma_launcher", "looma_stage", "looma_ray")


def imported_names(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_ничего_не_импортируется_из_проекта():
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in imported_names(path.read_text()):
            assert not name.startswith(FORBIDDEN), \
                f"{path.name} импортирует {name} — утилита перестанет быть отдельной"


def test_зависимость_ровно_одна():
    """Ставится клиенту, который про Looma больше ничего знать не должен."""
    import tomllib

    config = tomllib.loads((PACKAGE.parent / "pyproject.toml").read_text())
    assert config["project"]["dependencies"] == ["websockets>=13"]
