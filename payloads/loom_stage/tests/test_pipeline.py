"""Голова конвейера и неголовная стадия.

Всё, что тут проверяется, ловит один и тот же класс поломок: расхождение
состава батча между стадиями. Оно не даёт ни исключения, ни несовпадения форм —
только другой ответ.
"""

from __future__ import annotations

import pytest

from loom_stage import batch_wire
from loom_stage.pipeline import Head, Stage
from loom_stage.scheduler import Full, Scheduler, Sequence


def FakeTensor(rows, width=4):
    """Настоящий тензор: в половине этих проверок он реально едет по проводу,
    и подделка ломалась бы на сериализации, а не на том, что проверяется."""
    import torch

    return torch.zeros(rows, width)


class FakeEngine:
    """Движок, который считает по числу строк и ничего не считает."""

    batches = True

    def __init__(self, *, is_last=False, token=7, fail=None):
        self.is_last = is_last
        self.token = token
        self.fail = fail
        self.steps = []
        self.freed = []

    def step_batch(self, sequences, *, incoming=None, first_step):
        if self.fail:
            raise RuntimeError(self.fail)
        self.steps.append((("prefill" if first_step else "decode"),
                           [s.request_id for s in sequences]))
        rows = sum(batch_wire.widths(list(sequences), first_step=first_step))
        if self.is_last:
            return None, FakeTensor(len(sequences))
        return {"hidden_states": FakeTensor(rows)}, None

    def sample_batch(self, logits, sequences):
        batch_wire.check_rows(list(sequences), int(logits.shape[0]))
        return [self.token] * len(sequences)

    def free(self, request_id):
        self.freed.append(request_id)


def seq(name="r1", prompt=3, max_tokens=1) -> Sequence:
    return Sequence(request_id=name, prompt_ids=list(range(prompt)),
                    max_tokens=max_tokens)


def head(**kwargs):
    sent = []
    engine = kwargs.pop("engine", None) or FakeEngine(is_last=True)
    return Head(engine, num_stages=kwargs.pop("num_stages", 1),
                send=sent.append, eos_ids=kwargs.pop("eos_ids", ()),
                **kwargs), sent, engine


# ------------------------------------------------------------ одна стадия
def test_запрос_доходит_до_токена():
    loop, _sent, _engine = head()
    ticket = loop.submit(seq("a"))
    assert loop.run_once() is True
    answer = ticket.next(1)
    assert answer["kind"] == "token" and answer["token_id"] == 7
    # Замер шага едет вместе с токеном, и число участников — рядом с ним:
    # стоимость посчитана на весь батч, делить её между запросами было бы
    # выдумкой.
    assert answer["batch"] == 1 and answer["head_ms"] >= 0


def test_считать_нечего_и_цикл_это_говорит():
    loop, _sent, _engine = head()
    assert loop.run_once() is False


def test_несколько_запросов_считаются_одним_шагом():
    """То, ради чего всё затевалось: один прогон весов на весь батч."""
    loop, _sent, engine = head()
    tickets = [loop.submit(seq(name)) for name in "абв"]
    loop.run_once()
    assert engine.steps == [("prefill", ["а", "б", "в"])]
    for ticket in tickets:
        assert ticket.next(1)["token_id"] == 7


def test_дошедший_до_предела_запрос_заканчивается():
    loop, _sent, _engine = head()
    ticket = loop.submit(seq("a", max_tokens=1))
    loop.run_once()
    ticket.next(1)
    assert ticket.next(1) == {"kind": "done", "finish_reason": "length"}


def test_конец_строки_обрывает_раньше_предела():
    loop, _sent, _engine = head(eos_ids={7})
    ticket = loop.submit(seq("a", max_tokens=100))
    loop.run_once()
    ticket.next(1)
    assert ticket.next(1)["finish_reason"] == "stop"


def test_законченный_запрос_освобождает_кэш():
    loop, _sent, engine = head()
    loop.submit(seq("a"))
    loop.run_once()          # посчитал и поставил на освобождение
    loop.run_once()          # отпустил перед следующим шагом
    assert engine.freed == ["a"]


def test_за_префиллом_идёт_декод():
    loop, _sent, engine = head()
    loop.submit(seq("a", max_tokens=5))
    loop.run_once()
    loop.run_once()
    assert [kind for kind, _ in engine.steps] == ["prefill", "decode"]


def test_мест_нет_и_клиент_узнаёт_об_этом_сразу():
    loop, _sent, _engine = head(scheduler=Scheduler(max_sequences=1))
    loop.submit(seq("a"))
    with pytest.raises(Full):
        loop.submit(seq("b"))
    assert "b" not in loop.tickets


# --------------------------------------------------------------- отказы
def test_упавший_шаг_валит_весь_батч():
    """Батч атомарен: разобрать, кому из них шаг не удался, нельзя — тензор
    был один на всех."""
    loop, _sent, _engine = head(engine=FakeEngine(is_last=True, fail="карта отвалилась"))
    tickets = [loop.submit(seq(name)) for name in "аб"]
    loop.run_once()
    for ticket in tickets:
        answer = ticket.next(1)
        assert answer["kind"] == "error" and "карта отвалилась" in answer["error"]


def test_упавший_батч_не_остаётся_в_планировщике():
    loop, _sent, _engine = head(engine=FakeEngine(is_last=True, fail="ой"))
    loop.submit(seq("a"))
    loop.run_once()
    assert loop.snapshot()["считаются"] == 0


def test_ушедший_клиент_отпускает_кэш():
    loop, sent, engine = head(num_stages=2)
    loop.submit(seq("a"))
    loop.cancel("a")
    loop.run_once()
    assert engine.freed == ["a"]
    assert sent[-1] == {"kind": "free", "request_id": "a", "target_stage": -1}


def test_движок_трогает_только_считающий_поток():
    """Он держит один набор буферов и общий на процесс контекст прохода: вызов
    из чужого потока посреди шага однажды убил живую стадию — «Forward context
    is not set», следом illegal instruction, и контекст CUDA испорчен до конца
    жизни процесса."""
    loop, _sent, engine = head()
    loop.submit(seq("a"))
    loop.cancel("a")                  # клиентский поток
    assert engine.freed == []         # ничего не тронул
    loop.run_once()
    assert engine.freed == ["a"]      # отпустил считающий


# ------------------------------------------------------- несколько стадий
def test_голова_отдаёт_состав_батча_дальше():
    loop, sent, _engine = head(num_stages=2, engine=FakeEngine(is_last=False))
    loop.submit(seq("a", prompt=3))
    loop.submit(seq("b", prompt=2))
    import threading
    threading.Timer(0.05, lambda: loop.on_returned(
        {"batch_id": loop._batch_id, "tokens": [7, 8]})).start()
    loop.run_once()

    message = sent[0]
    assert message["kind"] == "activations" and message["target_stage"] == 1
    assert [m["request_id"] for m in message["members"]] == ["a", "b"]
    assert message["first_step"] is True


def test_ответ_на_чужой_батч_выбрасывается():
    """Прими опоздавший ответ за свой — и каждый следующий токен уедет на шаг
    не туда: ошибки не будет, будет чушь с правильными формами."""
    loop, _sent, _engine = head(num_stages=2, engine=FakeEngine(is_last=False),
                                timeout_s=0.3)
    loop.submit(seq("a"))
    loop.on_returned({"batch_id": "прошлый", "tokens": [99]})
    ticket = loop.tickets["a"]
    loop.run_once()
    assert ticket.next(1)["kind"] == "error"


def test_молчащий_хвост_даёт_отказ_а_не_вечное_ожидание():
    loop, _sent, _engine = head(num_stages=2, engine=FakeEngine(is_last=False),
                                timeout_s=0.05)
    ticket = loop.submit(seq("a"))
    loop.run_once()
    answer = ticket.next(1)
    assert answer["kind"] == "error" and "не ответил" in answer["error"]


def test_хвост_вернул_не_столько_токенов_сколько_запросов():
    loop, _sent, _engine = head(num_stages=2, engine=FakeEngine(is_last=False),
                                timeout_s=0.5)
    tickets = [loop.submit(seq(name)) for name in "аб"]
    import threading
    threading.Timer(0.02, lambda: loop.on_returned(
        {"batch_id": loop._batch_id, "tokens": [7]})).start()
    loop.run_once()
    assert tickets[0].next(1)["kind"] == "error"


# --------------------------------------------------------- средняя стадия
def stage(**kwargs):
    sent = []
    engine = kwargs.pop("engine", None) or FakeEngine()
    return Stage(engine, stage_index=kwargs.pop("stage_index", 1),
                 is_last=kwargs.pop("is_last", False), send=sent.append), sent, engine


def activations(batch, *, first_step=True, rows=None):
    import torch

    tensors = {"hidden_states": torch.zeros(
        rows if rows is not None
        else sum(batch_wire.widths(batch, first_step=first_step)), 4)}
    return {"kind": "activations", "batch_id": "б1", "first_step": first_step,
            "members": batch_wire.pack(batch, first_step=first_step),
            "tensors": batch_wire.pack_tensors(torch, tensors)}


def test_средняя_стадия_передаёт_состав_не_трогая():
    """Пересобрать состав здесь значило бы дать ему шанс разойтись."""
    middle, sent, _engine = stage()
    batch = [seq("a", prompt=3), seq("b", prompt=2)]
    middle.on_activations(activations(batch))
    assert sent[0]["members"] == batch_wire.pack(batch, first_step=True)
    assert sent[0]["target_stage"] == 2


def test_последняя_стадия_возвращает_токены_голове():
    tail, sent, _engine = stage(is_last=True, engine=FakeEngine(is_last=True))
    tail.on_activations(activations([seq("a"), seq("b")]))
    assert sent[0]["kind"] == "tokens" and sent[0]["target_stage"] == 0
    assert sent[0]["tokens"] == [7, 7]


def test_время_счёта_копится_вдоль_конвейера():
    """Голова вычтет эту сумму из круга и увидит провод отдельно — не сравнивая
    при этом ничьи часы."""
    tail, sent, _engine = stage(is_last=True, engine=FakeEngine(is_last=True))
    message = activations([seq("a")])
    message["upstream_ms"] = 100.0
    tail.on_activations(message)
    assert sent[0]["upstream_ms"] >= 100.0


def test_разъехавшийся_состав_ловится_по_длине():
    """Тензор всё равно разложится, просто не по тем границам."""
    tail, sent, _engine = stage(is_last=True, engine=FakeEngine(is_last=True))
    tail.on_activations(activations([seq("a", prompt=3)], rows=5))
    assert sent[0]["kind"] == "error" and "разъехался" in sent[0]["error"]


def test_отказ_средней_стадии_едет_голове_а_не_дальше():
    middle, sent, _engine = stage(engine=FakeEngine(fail="нет памяти"))
    middle.on_activations(activations([seq("a")]))
    assert sent[0]["kind"] == "error" and sent[0]["target_stage"] == 0
    assert "стадия 1" in sent[0]["error"]


def test_батч_без_участников_отвергается():
    middle, sent, _engine = stage()
    middle.on_activations({"kind": "activations", "batch_id": "б", "members": []})
    assert sent[0]["kind"] == "error"
