"""Шов между стадией и тем, кто считает слои.

Проверяется не арифметика движка, а контракт: стадия обязана работать с любым,
не зная, какой перед ней.
"""

from __future__ import annotations

import pytest

from loom_stage.engine import Engine, EngineRefused, build


def test_нынешний_исполнитель_удовлетворяет_шву():
    """Иначе шов описывает не то, что есть, а то, что хотелось бы."""
    from loom_stage.executor import ShardExecutor

    for name in ("forward", "sample", "free", "serialize", "deserialize",
                 "active_requests"):
        assert hasattr(ShardExecutor, name), f"нет {name}"


def test_неизвестный_движок_называет_известные():
    with pytest.raises(EngineRefused, match="torch и vllm"):
        build("что-то своё", shard=None)


def test_имя_движка_не_чувствительно_к_регистру_и_пробелам():
    """Оно приходит из командной строки и из окружения."""
    from loom_stage.executor import ShardExecutor

    assert isinstance(build("  TORCH ", shard=_FakeShard()), ShardExecutor)


def test_пустое_имя_значит_переносимый_движок():
    """Умолчание — тот, что работает везде, а не тот, что быстрее."""
    from loom_stage.executor import ShardExecutor

    assert isinstance(build("", shard=_FakeShard()), ShardExecutor)


class _FakeShard:
    """Минимум, который трогает конструктор: он определяет, как в этой версии
    transformers называется аргумент KV-кэша, и для этого смотрит на слой."""

    class spec:
        is_first = True
        is_last = False

    class _Layer:
        def forward(self, hidden_states, past_key_values=None, **kwargs):
            raise AssertionError("модель тут не считает")

    layers = [_Layer()]
    model = None
