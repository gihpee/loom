"""Батч на проводе.

Разъехавшийся состав не даёт ни исключения, ни расхождения форм — он даёт
связный бессмысленный ответ. Ловится он только по длинам, поэтому длины тут
проверяются нарочито тупо и в обе стороны.
"""

from __future__ import annotations

import pytest

from looma_stage import batch_wire
from looma_stage.batch_wire import BatchMismatch
from looma_stage.scheduler import Sequence


def seq(name="r1", prompt=3, output=(), **kwargs) -> Sequence:
    return Sequence(request_id=name, prompt_ids=list(range(prompt)),
                    output_ids=list(output), **kwargs)


# ------------------------------------------------------------- туда-обратно
def test_батч_переживает_дорогу():
    batch = [seq("a", prompt=3, temperature=0.7, top_p=0.9, seed=5, max_tokens=32),
             seq("b", prompt=2)]
    restored = batch_wire.unpack(batch_wire.pack(batch, first_step=True))
    assert [item.request_id for item in restored] == ["a", "b"]
    assert restored[0].prompt_ids == [0, 1, 2]
    assert (restored[0].temperature, restored[0].top_p, restored[0].seed) == (0.7, 0.9, 5)
    assert restored[0].max_tokens == 32


def test_порядок_не_нормализуется():
    """Порядок и есть соответствие между строкой логитов и запросом."""
    batch = [seq("я"), seq("а"), seq("б")]
    restored = batch_wire.unpack(batch_wire.pack(batch, first_step=True))
    assert [item.request_id for item in restored] == ["я", "а", "б"]


def test_на_декоде_промпт_не_едет_а_длина_едет():
    """Значения неголовной стадии не нужны — по ним нечего складывать; длина
    нужна: по ней считаются позиции."""
    packed = batch_wire.pack([seq("a", prompt=1000, output=[7])], first_step=False)
    assert packed[0]["prompt_ids"] == []
    restored = batch_wire.unpack(packed)
    assert len(restored[0].prompt_ids) == 1000
    assert restored[0].output_ids == [7]
    assert restored[0].computed == 1000


def test_выданные_токены_едут_всегда():
    packed = batch_wire.pack([seq("a", output=[7, 8])], first_step=False)
    assert batch_wire.unpack(packed)[0].output_ids == [7, 8]


# ------------------------------------------------------------------ отказы
def test_пустой_батч_отвергается():
    with pytest.raises(BatchMismatch, match="ни одного участника"):
        batch_wire.unpack([])


def test_участник_без_имени_отвергается():
    with pytest.raises(BatchMismatch, match="без имени запроса"):
        batch_wire.unpack([{"prompt_len": 3}])


# ------------------------------------------------------------------ длины
def test_ширина_на_префилле_это_весь_промпт():
    batch = [seq("a", prompt=3), seq("b", prompt=5)]
    assert batch_wire.widths(batch, first_step=True) == [3, 5]


def test_ширина_на_декоде_это_один_токен():
    batch = [seq("a", prompt=3), seq("b", prompt=5)]
    assert batch_wire.widths(batch, first_step=False) == [1, 1]


def test_несошедшиеся_токены_ловятся():
    """Границы внутри тензора ничем не помечены, но их сумма — помечена."""
    batch = [seq("a", prompt=3), seq("b", prompt=5)]
    batch_wire.check_tokens(batch, 8, first_step=True)          # сошлось
    with pytest.raises(BatchMismatch, match="состав разъехался"):
        batch_wire.check_tokens(batch, 7, first_step=True)


def test_отказ_называет_оба_числа():
    with pytest.raises(BatchMismatch, match="пришло 7 токенов на 2 "):
        batch_wire.check_tokens([seq("a", prompt=3), seq("b", prompt=5)], 7,
                                first_step=True)


def test_несошедшиеся_строки_логитов_ловятся():
    batch = [seq("a"), seq("b")]
    batch_wire.check_rows(batch, 2)
    with pytest.raises(BatchMismatch, match="какому запросу принадлежит строка"):
        batch_wire.check_rows(batch, 3)


def test_склеенный_в_одну_последовательность_батч_виден_по_строкам():
    """Худший случай: три промпта посчитались как один. Токенов столько же,
    а строк логитов — одна."""
    batch = [seq("a"), seq("b"), seq("c")]
    batch_wire.check_tokens(batch, 9, first_step=True)   # по токенам не видно
    with pytest.raises(BatchMismatch):                   # по строкам видно
        batch_wire.check_rows(batch, 1)
