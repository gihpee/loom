"""Решения, от которых зависит, что вообще соберётся.

vLLM тут нет и не нужно: проверяется арифметика среза и то, что подменённая
глобальная функция возвращается на место. Ошибка в первом даёт стадию, которая
считает чужие слои и молча отвечает чушью; во втором — испорченный процесс,
где следующая попытка загрузки грузит не то.
"""

from __future__ import annotations

import sys
import types

import pytest

from loom_stage.vllm_runner import RunnerRefused, layer_range, stage_role


# ------------------------------------------------------------------ роль
def test_срез_с_начала_делает_стадию_первой():
    assert stage_role(0, 12, 36) == (True, False)


def test_срез_до_конца_делает_стадию_последней():
    assert stage_role(24, 36, 36) == (False, True)


def test_одна_стадия_и_первая_и_последняя():
    """Модель целиком на одном узле — обычный случай, а не вырожденный."""
    assert stage_role(0, 36, 36) == (True, True)


def test_середина_не_строит_ни_эмбеддингов_ни_головы():
    assert stage_role(12, 24, 36) == (False, False)


@pytest.mark.parametrize("start, end, total", [
    (0, 37, 36),      # за край модели
    (-1, 12, 36),     # отрицательное начало
    (12, 12, 36),     # пустой срез
    (24, 12, 36),     # вывернутый
    (0, 12, 0),       # модель без слоёв
])
def test_негодный_срез_отвергается(start, end, total):
    """Молча взять не тот срез — значит получить стадию, которая считает чужие
    слои и отвечает связной чушью, не падая нигде."""
    with pytest.raises(RunnerRefused):
        stage_role(start, end, total)


# --------------------------------------------------------------- подмена
@pytest.fixture
def vllm_utils(monkeypatch):
    """Внутренности vLLM, которых на этой машине нет."""
    utils = types.ModuleType("vllm.distributed.utils")
    utils.get_pp_indices = lambda num_layers, rank, world_size: (0, num_layers)
    for name in ("vllm", "vllm.distributed"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.distributed.utils", utils)
    return utils


def test_на_время_загрузки_слои_наши(vllm_utils):
    with layer_range(12, 24):
        assert vllm_utils.get_pp_indices(36, 0, 1) == (12, 24)
        # Аргументы не важны: сколько бы vLLM ни насчитал, строит он наш срез.
        assert vllm_utils.get_pp_indices(999, 7, 8) == (12, 24)


def test_после_загрузки_всё_как_было(vllm_utils):
    было = vllm_utils.get_pp_indices
    with layer_range(12, 24):
        pass
    assert vllm_utils.get_pp_indices is было


def test_после_ПАДЕНИЯ_загрузки_тоже_как_было(vllm_utils):
    """Подменённая функция глобальная. Оставить её после неудачи — испортить
    всё, что попробует грузить модель следом, включая сообщение об ошибке."""
    было = vllm_utils.get_pp_indices
    with pytest.raises(RuntimeError, match="веса не те"):
        with layer_range(12, 24):
            raise RuntimeError("веса не те")
    assert vllm_utils.get_pp_indices is было


def test_без_vllm_отказ_называет_причину(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    with pytest.raises(RunnerRefused, match="версию"):
        with layer_range(0, 12):
            pass
