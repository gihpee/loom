"""Портовая арифметика: она же механизм обнаружения.

Ошибка здесь не выглядит как ошибка — два ранга просто не находят друг друга,
и это неотличимо от «нет связи между узлами».
"""

from __future__ import annotations

import pytest

from loom_ray.ports import (BASE, PortsRefused, crossing_for_group,
                            head_address, ports_for)


def test_диапазоны_рангов_не_пересекаются():
    """Пересекись они — второй ранг не смог бы забиндить свой порт, и выглядело
    бы это как «ray start не отработал»."""
    занято = set()
    for rank in range(8):
        ports = ports_for(rank)
        мои = set(ports.crossing()) | set(ports.local_only())
        assert not (мои & занято), f"ранг {rank} налезает на чужие порты"
        занято |= мои


def test_ранг_вычисляет_чужие_порты_не_спрашивая():
    """Смысл всей схемы: адреса не ищут, их считают — одинаково на всех узлах."""
    assert ports_for(3).gcs == ports_for(3, base=BASE).gcs
    assert head_address() == f"127.0.0.1:{ports_for(0).gcs}"
    # То, что посчитал ранг 5 про ранг 2, совпадает с тем, что ранг 2 знает о себе.
    assert ports_for(2).node_manager == ports_for(2).node_manager


def test_наружу_смотрят_только_те_порты_которым_надо():
    """Пробрасывать локальные — работа впустую, и она же лишние слушатели."""
    ports = ports_for(1)
    assert ports.metrics not in ports.crossing()
    assert ports.runtime_env_agent not in ports.crossing()
    assert ports.node_manager in ports.crossing()
    assert ports.object_manager in ports.crossing()
    assert ports.worker_first in ports.crossing()


def test_рабочих_портов_остаётся_достаточно():
    ports = ports_for(0)
    assert ports.worker_last - ports.worker_first + 1 >= 80


def test_слишком_мелкий_шаг_отвергается_сразу():
    """А не молча оставляет ранги без рабочих портов."""
    with pytest.raises(PortsRefused, match="рабочим портам"):
        ports_for(0, stride=5)


def test_выход_за_65535_называет_причину():
    with pytest.raises(PortsRefused, match="65535"):
        ports_for(500, base=60000, stride=100)


def test_отрицательный_ранг_отвергается():
    with pytest.raises(PortsRefused):
        ports_for(-1)


def test_карта_проброса_покрывает_всю_группу():
    карта = crossing_for_group(4)
    assert sorted(карта) == [0, 1, 2, 3]
    assert all(карта[r] for r in карта)


def test_доля_процессора_берётся_из_окружения(monkeypatch):
    """Её называет агент. Ноль означает «решай сам» — так ведёт себя Ray без
    флага, и это правильный ответ, когда доля неизвестна."""
    from loom_ray.cluster import _own_cpus

    monkeypatch.setenv("LOOM_TASK_CPUS", "8.0")
    assert _own_cpus() == 8
    monkeypatch.setenv("LOOM_TASK_CPUS", "0.5")
    assert _own_cpus() == 1, "меньше одного ядра Ray не поймёт"
    monkeypatch.delenv("LOOM_TASK_CPUS", raising=False)
    assert _own_cpus() == 0
    monkeypatch.setenv("LOOM_TASK_CPUS", "не число")
    assert _own_cpus() == 0
