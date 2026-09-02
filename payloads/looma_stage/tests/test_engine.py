"""Шов между стадией и тем, кто считает слои.

Проверяется не арифметика движка, а контракт: стадия обязана работать с любым,
не зная, какой перед ней.
"""

from __future__ import annotations

import pytest

from looma_stage.engine import EngineRefused, build


CONTRACT = ("step_batch", "sample_batch", "sample", "free", "active_requests")


def test_переносимый_исполнитель_удовлетворяет_шву():
    """Иначе шов описывает не то, что есть, а то, что хотелось бы."""
    from looma_stage.executor import ShardExecutor

    for name in CONTRACT:
        assert hasattr(ShardExecutor, name), f"нет {name}"
    assert ShardExecutor.batches is False


def test_движок_vllm_удовлетворяет_тому_же_шву():
    """Проверяется класс, а не работающий движок: сам vLLM без карты не
    поднимется, но контракт должен сходиться и на машине без неё — иначе
    расхождение найдётся только на узле, посреди первого запроса."""
    from looma_stage.vllm_engine import VllmEngine

    for name in CONTRACT:
        assert hasattr(VllmEngine, name), f"нет {name}"
    assert VllmEngine.batches is True


def test_vllm_без_пути_к_весам_отказывает_внятно():
    with pytest.raises(EngineRefused, match="грузит их сам"):
        build("vllm")


def test_vllm_без_числа_слоёв_отказывает_внятно():
    """Без него он не знает, эта ли стадия последняя, и не соберёт lm_head."""
    with pytest.raises(EngineRefused, match="не соберёт lm_head"):
        build("vllm", model_path="/где-то/модель")


def test_переносимому_движку_нужна_собранная_стадия():
    with pytest.raises(EngineRefused, match="её не дали"):
        build("torch")


def test_неизвестный_движок_называет_известные():
    with pytest.raises(EngineRefused, match="torch и vllm"):
        build("что-то своё", shard=None)


def test_имя_движка_не_чувствительно_к_регистру_и_пробелам():
    """Оно приходит из командной строки и из окружения."""
    from looma_stage.executor import ShardExecutor

    assert isinstance(build("  TORCH ", shard=_FakeShard()), ShardExecutor)


def test_пустое_имя_значит_переносимый_движок():
    """Умолчание — тот, что работает везде, а не тот, что быстрее."""
    from looma_stage.executor import ShardExecutor

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
