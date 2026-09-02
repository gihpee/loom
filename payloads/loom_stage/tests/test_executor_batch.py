"""Батч у переносимого исполнителя.

Он его не ускоряет — прогоняет последовательности по одной. Но говорить на
проводе он обязан ровно то же, что и vLLM: иначе стадия должна знать, каким
движком считает соседняя.
"""

from __future__ import annotations

import types

import pytest
import torch

from loom_stage.batch_wire import BatchMismatch
from loom_stage.executor import ShardExecutor
from loom_stage.scheduler import Sequence


def executor(*, is_first=True, is_last=False, width=4):
    """Исполнитель со всем настоящим, кроме самого счёта."""
    made = ShardExecutor.__new__(ShardExecutor)
    made.torch = torch
    made.spec = types.SimpleNamespace(is_first=is_first, is_last=is_last)
    made.calls = []

    def forward(*, request_id, positions, input_ids=None, hidden=None):
        made.calls.append({"request_id": request_id, "positions": list(positions),
                           "input_ids": input_ids,
                           "hidden": None if hidden is None else tuple(hidden.shape)})
        if is_last:
            return None, torch.full((7,), float(len(positions)))
        return torch.ones(1, len(positions), width), None

    made.forward = forward
    return made


def seq(name="a", prompt=3, output=()):
    return Sequence(request_id=name, prompt_ids=list(range(prompt)),
                    output_ids=list(output))


# ------------------------------------------------------------ где считаем
def test_на_префилле_считается_весь_промпт():
    positions, ids = ShardExecutor._where(None, seq(prompt=3), first_step=True)
    assert positions == [0, 1, 2] and ids == [0, 1, 2]


def test_на_декоде_считается_один_последний_токен():
    """Позиции берутся из состояния самой последовательности: батч живёт
    дольше одного шага, и запросы в нём разной длины."""
    positions, ids = ShardExecutor._where(None, seq(prompt=3, output=[7, 8]),
                                          first_step=False)
    assert positions == [4] and ids == [8]


def test_позиция_декода_не_зависит_от_соседей_по_батчу():
    short = ShardExecutor._where(None, seq(prompt=2, output=[9]), first_step=False)
    long = ShardExecutor._where(None, seq(prompt=9, output=[9]), first_step=False)
    assert short[0] == [2] and long[0] == [9]


# ----------------------------------------------------------------- склейка
def test_первая_стадия_складывает_токены_подряд():
    """Так же, как настоящий батч: иначе следующая стадия нарежет не по тем
    границам, и ошибки не будет, будет чушь."""
    made = executor()
    hidden, logits = made.step_batch([seq("a", 3), seq("b", 2)], first_step=True)
    assert logits is None
    assert tuple(hidden["hidden_states"].shape) == (1, 5, 4)
    assert [c["request_id"] for c in made.calls] == ["a", "b"]


def test_каждая_последовательность_считается_отдельно():
    """Плотная каузальная маска дала бы каждой следующей видеть предыдущую."""
    made = executor()
    made.step_batch([seq("a", 3), seq("b", 2)], first_step=True)
    assert [c["positions"] for c in made.calls] == [[0, 1, 2], [0, 1]]


def test_последняя_стадия_отдаёт_строку_на_последовательность():
    made = executor(is_first=True, is_last=True)
    hidden, logits = made.step_batch([seq("a", 3), seq("b", 2)], first_step=True)
    assert hidden is None and tuple(logits.shape) == (2, 7)
    # Строка каждой — своя: одинаковые строки означали бы, что батч склеился.
    assert logits[0][0] == 3 and logits[1][0] == 2


# ----------------------------------------------------------------- нарезка
def test_средняя_стадия_режет_вход_по_составу_батча():
    made = executor(is_first=False)
    batch = [seq("a", 3), seq("b", 2)]
    made.step_batch(batch, incoming={"hidden_states": torch.zeros(5, 4)},
                    first_step=True)
    assert [c["hidden"] for c in made.calls] == [(1, 3, 4), (1, 2, 4)]


def test_вход_не_той_длины_ловится_а_не_режется():
    made = executor(is_first=False)
    with pytest.raises(BatchMismatch, match="разъехался"):
        made.step_batch([seq("a", 3), seq("b", 2)],
                        incoming={"hidden_states": torch.zeros(4, 4)},
                        first_step=True)


def test_вход_с_лишней_осью_понимается():
    """Один движок отдаёт [токены, ширина], другой [1, токены, ширина]."""
    made = executor(is_first=False)
    made.step_batch([seq("a", 3)],
                    incoming={"hidden_states": torch.zeros(1, 3, 4)},
                    first_step=True)
    assert made.calls[0]["hidden"] == (1, 3, 4)


def test_неголовная_стадия_без_тензоров_отказывается_считать():
    made = executor(is_first=False)
    with pytest.raises(ValueError, match="requires hidden states"):
        made.step_batch([seq("a")], incoming=None, first_step=True)


def test_тензоры_без_hidden_states_называют_что_пришло():
    made = executor(is_first=False)
    with pytest.raises(ValueError, match="residual"):
        made.step_batch([seq("a")], incoming={"residual": torch.zeros(3, 4)},
                        first_step=True)


# ---------------------------------------------------------------- выборка
def test_токен_на_последовательность_в_порядке_батча():
    made = executor(is_last=True)
    made.sample = lambda row, **kw: int(row.argmax().item())
    logits = torch.tensor([[0.0, 9.0, 0.0], [9.0, 0.0, 0.0]])
    assert made.sample_batch(logits, [seq("a"), seq("b")]) == [1, 0]


def test_строк_не_столько_сколько_запросов():
    made = executor(is_last=True)
    with pytest.raises(BatchMismatch):
        made.sample_batch(torch.zeros(3, 5), [seq("a"), seq("b")])
