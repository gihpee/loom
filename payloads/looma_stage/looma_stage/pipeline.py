# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/server/executor/base_executor.py — цикл стадии
# (принять активации -> прогнать свои слои -> отдать дальше), роль первой
# стадии как владельца запроса и как той, кто выбирает состав батча.
# Изменения: цикл вынесен из HTTP-слоя и проверяется без сети и без модели;
# транспорт активаций не ZMQ/Lattica, а релей через агента; состав батча едет
# отдельной структурой и сверяется по двум длинам на каждой стадии;
# переполнение — отказ клиенту, а не вытеснение чужого KV-кэша.
"""Голова конвейера: один цикл, который считает за всех.

До этого модуля каждый клиентский запрос сам гнал себя по конвейеру: поток
HTTP звал исполнителя, ждал токен с хвоста, звал снова. Параллельные клиенты
при этом делили пропускную способность, а не складывали её — они стояли в
очереди за одним и тем же замком вокруг модели.

Теперь считает один поток. Клиентский поток кладёт запрос в планировщик и
читает готовые токены из своей очереди; какими батчами они посчитались, его не
касается. Выигрыш весь отсюда: шаг движка над батчем из десяти стоит почти
столько же, сколько над одним, потому что тяжёлое в нём — прогон весов через
карту, и он один на весь батч.

Два свойства, на которых всё держится.

**Батч атомарен.** Последовательности в шаге считаются вместе и падают вместе.
Разделить их посреди конвейера нельзя: тензор один на всех, и вынуть из него
одну — значит сдвинуть границы остальных.

**В полёте один батч.** Голова не начинает следующий шаг, пока не вернулся
предыдущий. Настоящая конвейерная загрузка (пока хвост считает шаг N, голова
считает N+1) требует, чтобы кэш терпел два шага сразу, и это отдельная работа.
Здесь честнее медленно и правильно.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import uuid
from typing import Callable, Deque, Dict, List, Optional

from looma_stage import batch_wire
from looma_stage.scheduler import Full, Scheduler, Sequence

logger = logging.getLogger("looma_stage.pipeline")

__all__ = ["Head", "Ticket", "Full"]


class Ticket:
    """То, что клиентский поток держит, пока считается его запрос.

    Очередь, а не общий буфер: потоков-читателей столько же, сколько запросов,
    и каждый должен просыпаться от своего токена, а не от чужого.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.tokens: "queue.Queue[dict]" = queue.Queue()

    def next(self, timeout: float) -> dict:
        """Следующее событие: токен, конец или отказ."""
        return self.tokens.get(timeout=timeout)


class Head:
    """Планировщик, движок и один поток, который их сводит."""

    def __init__(self, engine, *, num_stages: int, send: Callable[[dict], None],
                 eos_ids, timeout_s: float = 120.0,
                 scheduler: Optional[Scheduler] = None,
                 on_step: Optional[Callable[[float, int], None]] = None) -> None:
        self.engine = engine
        self.num_stages = max(1, num_stages)
        self.send = send
        self.eos_ids = set(eos_ids or ())
        self.timeout_s = timeout_s
        # Собственный исполнитель батча не ускоряет: он прогоняет
        # последовательности по одной. Мест ему всё равно нужно много —
        # запросы обслуживаются вперемешку, по токену за круг.
        self.scheduler = scheduler or Scheduler()
        # Куда сообщать, сколько стоил шаг. Планировщик снаружи делит модель
        # между узлами по этому числу, и брать его больше неоткуда.
        self.on_step = on_step
        self.tickets: Dict[str, Ticket] = {}
        self._lock = threading.RLock()
        # Ответ хвоста ждём здесь. Очередь на единицу: в полёте один батч.
        self._returned: "queue.Queue[dict]" = queue.Queue()
        self._batch_id = ""
        # Кого отпустить. Движок трогает ТОЛЬКО считающий поток: он держит
        # один набор буферов и общий на процесс контекст прохода, и вызов из
        # чужого потока посреди шага уже убивал живую стадию — сначала
        # «Forward context is not set», следом illegal instruction, после
        # которой контекст CUDA испорчен до конца жизни процесса. Клиентский
        # поток поэтому не освобождает ничего сам, а кладёт имя сюда.
        self._to_release: Deque[str] = collections.deque()
        self._stop = threading.Event()
        self._woken = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- приём
    def submit(self, sequence: Sequence) -> Ticket:
        """Поставить запрос в очередь. Бросает `Full`, если мест нет."""
        ticket = Ticket(sequence.request_id)
        with self._lock:
            self.scheduler.add(sequence)      # бросает до регистрации билета
            self.tickets[sequence.request_id] = ticket
        self._woken.set()
        return ticket

    def cancel(self, request_id: str) -> None:
        """Клиент ушёл — или дочитал до конца и убирает за собой.

        Зовётся всегда, в том числе после нормального завершения, поэтому
        освобождение здесь идёт только если запрос ещё кому-то известен. Иначе
        каждый успешный ответ рассылал бы всем стадиям второе «забудьте про
        него» — безвредное, но заполняющее логи ровно там, где ищут поломку.
        """
        with self._lock:
            known = self.tickets.pop(request_id, None) is not None
        known = known or self.scheduler.running(request_id) is not None
        if not known:
            return
        self.scheduler.finish(request_id)
        self._release(request_id)

    def _release(self, request_id: str) -> None:
        """Поставить запрос в очередь на освобождение.

        Само освобождение делает считающий поток, между шагами: движок нельзя
        трогать, пока он считает.
        """
        with self._lock:
            self._to_release.append(request_id)
        self._woken.set()

    def _drain_releases(self) -> None:
        """Отпустить всё, что накопилось. Зовётся только считающим потоком."""
        while True:
            with self._lock:
                if not self._to_release:
                    return
                request_id = self._to_release.popleft()
            try:
                self.engine.free(request_id)
            except Exception:
                logger.exception("не удалось освободить состояние %s", request_id)
            if self.num_stages > 1:
                # -1 — «всем остальным»: кэш этого запроса держит каждая стадия.
                self.send({"kind": "free", "request_id": request_id,
                           "target_stage": -1})

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run_forever,
                                        name="stage-head", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._woken.set()

    # -------------------------------------------------------------- цикл
    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    # Считать нечего — ждём прихода, а не крутим цикл впустую.
                    self._woken.wait(timeout=0.05)
                    self._woken.clear()
            except Exception:
                logger.exception("шаг головы упал")
                time.sleep(0.05)

    def run_once(self) -> bool:
        """Один шаг. Возвращает False, если считать было нечего."""
        # До шага, а не после: освобождённые блоки нужны тому, кого сейчас
        # возьмут в батч.
        self._drain_releases()
        kind, batch = self.scheduler.next_batch()
        if not batch:
            return False
        first_step = kind == "prefill"
        try:
            tokens, spent = self._step(batch, first_step=first_step)
        except Exception as exc:
            # Батч атомарен: последовательности в нём считались вместе и
            # падают вместе. Разобрать, кому из них шаг не удался, нельзя —
            # тензор был один на всех.
            logger.exception("шаг над батчем из %d не удался", len(batch))
            for sequence in batch:
                self._fail(sequence.request_id, str(exc) or type(exc).__name__)
            return True
        if self.on_step is not None:
            self.on_step(spent["head_ms"], len(batch))
        for sequence, token in zip(batch, tokens):
            self._deliver(sequence, int(token), spent)
        return True

    def _step(self, batch: List[Sequence], *, first_step: bool):
        """Токены и то, во что шаг обошёлся.

        Замеры остались пошаговыми, а не позапросными, и это не небрежность:
        шаг посчитал весь батч сразу, и делить его стоимость между участниками
        было бы выдумкой. Число участников едет рядом, чтобы это было видно.
        """
        started = time.perf_counter()
        hidden, logits = self.engine.step_batch(batch, incoming=None,
                                                first_step=first_step)
        head_ms = (time.perf_counter() - started) * 1000
        if self.num_stages == 1:
            tokens = self.engine.sample_batch(logits, batch)
            return tokens, {"head_ms": (time.perf_counter() - started) * 1000,
                            "peer_ms": 0.0, "transport_ms": 0.0,
                            "batch": len(batch)}
        tokens, peer_ms = self._round_trip(batch, hidden, first_step=first_step)
        whole_ms = (time.perf_counter() - started) * 1000
        return tokens, {
            "head_ms": head_ms,
            "peer_ms": max(0.0, peer_ms),
            # Что осталось от круга за вычетом посчитанного здесь и там: провод,
            # два релея и переход через оркестратор. Ни одного сравнения часов:
            # длительность, измеренная тут, минус длительности, измеренные там.
            "transport_ms": max(0.0, whole_ms - head_ms - max(0.0, peer_ms)),
            "batch": len(batch),
        }

    def _round_trip(self, batch: List[Sequence], hidden, *, first_step: bool):
        """Отдать тензоры дальше и дождаться токенов с хвоста."""
        import torch

        batch_id = uuid.uuid4().hex
        self._batch_id = batch_id
        self.send({
            "kind": "activations",
            "batch_id": batch_id,
            "target_stage": 1,
            "first_step": first_step,
            "members": batch_wire.pack(batch, first_step=first_step),
            "tensors": batch_wire.pack_tensors(torch, hidden),
        })
        deadline = time.monotonic() + self.timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(
                    f"хвост не ответил за {self.timeout_s:g} с на батч из "
                    f"{len(batch)}")
            try:
                answer = self._returned.get(timeout=left)
            except queue.Empty:
                continue
            # Опоздавший ответ на прошлый батч — не ответ на этот. Прими его
            # за свой, и каждый следующий токен уедет на шаг не туда: ошибки
            # не будет, будет чушь с правильными формами.
            if answer.get("batch_id") != batch_id:
                logger.warning("выбрасываю ответ на чужой батч %s",
                               answer.get("batch_id"))
                continue
            if answer.get("kind") == "error":
                raise RuntimeError(answer.get("error", "конвейер не ответил"))
            tokens = list(answer.get("tokens") or [])
            batch_wire.check_rows(batch, len(tokens))
            return tokens, float(answer.get("upstream_ms") or 0.0)

    def on_returned(self, message: dict) -> None:
        """Хвост прислал токены. Зовётся из потока приёма сообщений."""
        self._returned.put(message)

    # ------------------------------------------------------------ выдача
    def _deliver(self, sequence: Sequence, token: int, spent: dict) -> None:
        self.scheduler.accepted(token, sequence.request_id)
        with self._lock:
            ticket = self.tickets.get(sequence.request_id)
        if ticket is None:
            # Клиент ушёл, пока считался шаг. Досчитывать некому.
            self._finish(sequence.request_id)
            return
        ticket.tokens.put({"kind": "token", "token_id": token, **spent})
        if token in self.eos_ids:
            self._end(sequence.request_id, ticket, "stop")
        elif sequence.done:
            self._end(sequence.request_id, ticket, "length")

    def _end(self, request_id: str, ticket: Ticket, reason: str) -> None:
        ticket.tokens.put({"kind": "done", "finish_reason": reason})
        self._finish(request_id)

    def _fail(self, request_id: str, error: str) -> None:
        with self._lock:
            ticket = self.tickets.get(request_id)
        if ticket is not None:
            ticket.tokens.put({"kind": "error", "error": error})
        self._finish(request_id)

    def _finish(self, request_id: str) -> None:
        with self._lock:
            self.tickets.pop(request_id, None)
        self.scheduler.finish(request_id)
        self._release(request_id)

    def snapshot(self) -> dict:
        return self.scheduler.snapshot()


class Stage:
    """Неголовная стадия: считает то, что прислали, и передаёт дальше.

    Ничего не выбирает. Состав батча приехал с предыдущей стадии, и повторить
    его надо точно — в том же порядке и той же длины. Всё, что тут делается
    сверх счёта, — это две сверки длин: единственное, чем разъехавшийся состав
    отличается от сошедшегося, потому что тензор всё равно разложится, просто
    не по тем границам.
    """

    def __init__(self, engine, *, stage_index: int, is_last: bool,
                 send: Callable[[dict], None],
                 on_step: Optional[Callable[[float, int], None]] = None) -> None:
        self.engine = engine
        self.stage_index = stage_index
        self.is_last = is_last
        self.send = send
        self.on_step = on_step

    def on_activations(self, message: dict) -> None:
        import torch

        batch_id = message.get("batch_id", "")
        try:
            batch = batch_wire.unpack(message.get("members") or [])
            first_step = bool(message.get("first_step"))
            tensors = batch_wire.unpack_tensors(torch, message.get("tensors") or {})
            batch_wire.check_tokens(batch, batch_wire.token_rows(tensors),
                                    first_step=first_step)
            started = time.perf_counter()
            hidden, logits = self.engine.step_batch(batch, incoming=tensors,
                                                    first_step=first_step)
            spent_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            logger.exception("стадия %d не смогла посчитать батч", self.stage_index)
            self.send({"kind": "error", "batch_id": batch_id, "target_stage": 0,
                       "error": f"стадия {self.stage_index}: {exc}"})
            return

        if self.on_step is not None:
            self.on_step(spent_ms, len(batch))
        # Сколько посчитали все стадии после головы: каждая прибавляет своё,
        # хвост привозит сумму. Голова вычтет её из круга и увидит провод
        # отдельно — не сравнивая при этом ничьи часы.
        upstream_ms = float(message.get("upstream_ms") or 0.0) + spent_ms
        if self.is_last:
            tokens = self.engine.sample_batch(logits, batch)
            self.send({"kind": "tokens", "batch_id": batch_id,
                       "target_stage": 0, "tokens": [int(t) for t in tokens],
                       "upstream_ms": upstream_ms})
            return
        self.send({
            "kind": "activations",
            "batch_id": batch_id,
            "target_stage": self.stage_index + 1,
            "first_step": message.get("first_step"),
            # Состав едет дальше как приехал: пересобирать его здесь значило бы
            # дать ему шанс разойтись.
            "members": message.get("members"),
            "upstream_ms": upstream_ms,
            "tensors": batch_wire.pack_tensors(torch, hidden),
        })

    def on_free(self, request_id: str) -> None:
        try:
            self.engine.free(request_id)
        except Exception:
            logger.exception("не удалось освободить состояние %s", request_id)
