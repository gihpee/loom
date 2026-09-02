"""Кого считать на этом шаге.

Отсюда берётся пропускная способность, и отсюда же берётся худший вид отказа:
выбитый из кэша запрос не падает, он продолжает отвечать связной чушью.
"""

from __future__ import annotations

import pytest

from loom_stage.scheduler import Full, Scheduler, Sequence


def request(request_id="r1", prompt=3, max_tokens=10) -> Sequence:
    return Sequence(request_id=request_id, prompt_ids=list(range(prompt)),
                    max_tokens=max_tokens)


# ------------------------------------------------------------- приём
def test_отказ_вместо_вытеснения_чужого():
    """Выбитый запрос не падает — он продолжает отвечать, читая пустой кэш.
    Это худший вид отказа: его не видно ни в логах, ни в кодах ответа."""
    scheduler = Scheduler(max_sequences=2)
    scheduler.add(request("a"))
    scheduler.add(request("b"))
    with pytest.raises(Full, match="предел"):
        scheduler.add(request("c"))


def test_отказ_называет_что_делать():
    scheduler = Scheduler(max_sequences=1)
    scheduler.add(request("a"))
    with pytest.raises(Full, match="Приходите позже"):
        scheduler.add(request("b"))


def test_промпт_длиннее_шага_отвергается_сразу():
    """Иначе он не влезет никогда, и запрос будет ждать вечно, ничего не
    сообщая."""
    scheduler = Scheduler(max_batch_tokens=10)
    with pytest.raises(Full, match="не помещается в шаг"):
        scheduler.add(request("a", prompt=11))


def test_место_освобождается_когда_запрос_кончился():
    scheduler = Scheduler(max_sequences=1)
    scheduler.add(request("a"))
    scheduler.next_batch()
    scheduler.finish("a")
    scheduler.add(request("b"))      # не бросает


# -------------------------------------------------------------- выбор
def test_новые_идут_вперёд_идущих():
    """Пока запрос не посчитал промпт, он не отвечает вовсе, а идущий уже
    выдаёт токены. Задержать выдачу на шаг дешевле, чем держать нового в
    тишине."""
    scheduler = Scheduler()
    scheduler.add(request("старый"))
    kind, batch = scheduler.next_batch()
    assert kind == "prefill"
    scheduler.accepted(7, "старый")

    scheduler.add(request("новый"))
    kind, batch = scheduler.next_batch()
    assert kind == "prefill"
    assert [item.request_id for item in batch] == ["новый"]


def test_несколько_промптов_в_одном_шаге():
    """То, ради чего всё затевалось: батч считается почти за то же время, что
    одна последовательность."""
    scheduler = Scheduler()
    for name in "абв":
        scheduler.add(request(name))
    kind, batch = scheduler.next_batch()
    assert kind == "prefill"
    assert len(batch) == 3


def test_батч_ограничен_токенами():
    """Один длинный промпт способен занять карту надолго, и остальным в этом
    шаге ждать нечего."""
    scheduler = Scheduler(max_batch_tokens=10)
    scheduler.add(request("a", prompt=6))
    scheduler.add(request("b", prompt=6))
    kind, batch = scheduler.next_batch()
    assert [item.request_id for item in batch] == ["a"]

    kind, batch = scheduler.next_batch()
    assert [item.request_id for item in batch] == ["b"]


def test_первый_берётся_даже_если_один_не_влезает():
    """Иначе запрос ровно по границе не посчитается никогда."""
    scheduler = Scheduler(max_batch_tokens=10)
    scheduler.add(request("ровный", prompt=10))
    kind, batch = scheduler.next_batch()
    assert [item.request_id for item in batch] == ["ровный"]


def test_батч_ограничен_местами():
    scheduler = Scheduler(max_sequences=2)
    scheduler.add(request("a"))
    scheduler.add(request("b"))
    _kind, batch = scheduler.next_batch()
    assert len(batch) == 2


def test_декод_идёт_когда_новых_нет():
    scheduler = Scheduler()
    scheduler.add(request("a"))
    scheduler.next_batch()
    kind, batch = scheduler.next_batch()
    assert kind == "decode"
    assert [item.request_id for item in batch] == ["a"]


def test_декод_батчем():
    scheduler = Scheduler()
    for name in "абв":
        scheduler.add(request(name))
    scheduler.next_batch()
    kind, batch = scheduler.next_batch()
    assert kind == "decode"
    assert len(batch) == 3


def test_досчитанный_запрос_из_декода_уходит():
    scheduler = Scheduler()
    scheduler.add(request("a", max_tokens=2))
    scheduler.next_batch()
    for token in (7, 8):
        scheduler.accepted(token, "a")
    kind, batch = scheduler.next_batch()
    assert (kind, batch) == ("", [])


def test_пустому_планировщику_нечего_считать():
    assert Scheduler().next_batch() == ("", [])


def test_порядок_прихода_сохраняется():
    """Ни приоритетов, ни коротких вперёд: пока их некому объяснять клиенту,
    любая такая политика — способ обидеть половину пользователей молча."""
    scheduler = Scheduler(max_sequences=10)
    for name in "абвгд":
        scheduler.add(request(name))
    _kind, batch = scheduler.next_batch()
    assert [item.request_id for item in batch] == list("абвгд")


def test_снимок_показывает_свободные_места():
    scheduler = Scheduler(max_sequences=3)
    scheduler.add(request("a"))
    assert scheduler.snapshot() == {"ждут": 1, "считаются": 0, "мест": 2}
    scheduler.next_batch()
    assert scheduler.snapshot() == {"ждут": 0, "считаются": 1, "мест": 2}
