"""Разрезать модель между узлами: сколько слоёв кому и что уедет на узел.

Сеть здесь трогает только один тест — остальное про арифметику разреза и про
отказы, которые оператор увидит раньше, чем что-то запустится.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from loom.orchestrator.models import ModelError, describe, split_layers, stage_payload


# ------------------------------------------------------------------- разрез
def test_слои_делятся_поровну():
    assert split_layers(36, 3) == [(0, 12), (12, 24), (24, 36)]


def test_остаток_достаётся_первым_стадиям():
    """Хвост округления не должен потеряться: сумма всегда равна числу слоёв."""
    ranges = split_layers(37, 3)
    assert ranges[0] == (0, 13)
    assert ranges[-1][1] == 37
    assert sum(e - s for s, e in ranges) == 37


def test_диапазоны_идут_подряд_без_дыр():
    """Пропущенный слой — не ошибка загрузки, а тихо неверный ответ."""
    ranges = split_layers(48, 5)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 48
    for before, after in zip(ranges, ranges[1:]):
        assert before[1] == after[0]


def test_разрез_по_свободной_vram():
    """На стенде из 4090 и 3090 равный разрез упирается в меньшую карту."""
    ranges = split_layers(36, 2, [24.0, 48.0])
    first, second = (e - s for s, e in ranges)
    assert second > first
    assert first + second == 36


@pytest.mark.parametrize("weights", [None, [1, 1, 1], [10, 1, 1], [1, 1, 10]])
def test_сумма_всегда_сходится(weights):
    ranges = split_layers(29, 3, weights)
    assert sum(e - s for s, e in ranges) == 29
    assert all(e > s for s, e in ranges), "стадия без слоёв — переход, который не считает"


def test_стадий_больше_чем_слоёв_отвергается():
    with pytest.raises(ModelError) as exc:
        split_layers(4, 8)
    assert "не считает" in str(exc.value)


# ------------------------------------------------------------------- имена
@pytest.mark.parametrize("bad", ["", "нетслеша", "нет/такой", "a b/c", "/"])
def test_негодное_имя_модели_отвергается_до_сети(bad):
    """Иначе кириллица вылезает ошибкой кодировки из глубины urllib."""
    with pytest.raises(ModelError):
        describe(bad)


# ----------------------------------------------------------------- нагрузка
def test_код_стадии_уезжает_вместе_с_задачей():
    """Узел, который эту модель не обслуживал, получает код с задачей —
    без пакетного реестра посередине."""
    payload = stage_payload()
    assert "loom_stage/server.py" in payload
    assert "loom_stage/loader.py" in payload
    assert all(name.startswith("loom_stage/") for name in payload)
    assert 0 < sum(len(v) for v in payload.values()) < 5 * 1024 * 1024


def test_пусковой_слой_в_нагрузку_не_попадает():
    """Он живёт в образе: сломанный запускатель нельзя починить, прислав ещё один."""
    assert not any("launcher" in name for name in stage_payload())


# --------------------------------------------------------------------- сеть
@pytest.mark.slow
def test_настоящая_модель_читается_с_huggingface():
    info = describe("Qwen/Qwen3-8B")
    assert info.num_layers == 36
    assert info.hidden_size == 4096
    assert "Qwen3" in info.architecture


def test_код_стадии_находится_и_в_установленном_пакете(tmp_path, monkeypatch):
    """В образе оркестратор живёт в site-packages, где никакого payloads/ рядом нет.

    Симптом со стенда: деплой отказал с путём
    /usr/local/lib/python3.12/payloads/... — путь считался от исходников и
    установку не пережил.
    """
    where = tmp_path / "payloads" / "loom_stage" / "loom_stage"
    where.mkdir(parents=True)
    (where / "server.py").write_text("# стадия\n")
    (where / "loader.py").write_text("# загрузчик\n")
    monkeypatch.setenv("LOOM_PAYLOADS_DIR", str(tmp_path / "payloads"))
    payload = stage_payload()
    assert set(payload) == {"loom_stage/server.py", "loom_stage/loader.py"}


def test_отсутствие_стадии_называет_где_искали(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_PAYLOADS_DIR", str(tmp_path / "нет-такого"))
    monkeypatch.setattr("loom.orchestrator.models._payload_dirs",
                        lambda: iter([tmp_path / "нет-такого"]))
    with pytest.raises(ModelError) as exc:
        stage_payload()
    assert "Искали в" in str(exc.value)
    assert "LOOM_PAYLOADS_DIR" in str(exc.value)
