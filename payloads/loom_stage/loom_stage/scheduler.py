"""Кого считать на этом шаге.

Пропускная способность берётся отсюда. Движок считает батч почти за то же
время, что одну последовательность: тяжёлое в шаге — это прогон весов через
карту, и он один на весь батч. Значит десять клиентов в одном шаге стоят почти
как один, а десять шагов подряд — вдесятеро.

Три решения, и каждое стоит объяснить.

**Prefill и decode не смешиваются.** В одном шаге либо считаются промпты
новых запросов, либо по токену у идущих. Смешанный батч быстрее, но требует
нарезки промптов на куски, а без неё длинный промпт задержал бы всех, кто в
том же шаге ждёт свой единственный токен. Начинаем с простого и честного.

**Отказ вместо вытеснения.** Когда мест нет, новый запрос получает отказ, а не
выбивает чужой KV-кэш. Выбитый не падает — он продолжает отвечать, читая
пустой кэш, то есть выдаёт связную чушь. Это худший вид отказа: его не видно
ни в логах, ни в кодах ответа.

**Порядок прихода.** Ни приоритетов, ни коротких вперёд: пока их некому
объяснять клиенту, любая такая политика — это способ обидеть половину
пользователей молча.
"""

from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("loom_stage.scheduler")

# Сколько последовательностей держать одновременно. Каждая занимает KV-кэш, и
# потолок здесь — это потолок памяти, а не вкуса.
MAX_SEQUENCES = 64
# Сколько токенов промпта считать за один шаг. Один длинный промпт способен
# занять карту надолго, и остальным в этом шаге ждать нечего.
MAX_BATCH_TOKENS = 8192


class Full(RuntimeError):
    """Мест нет. Не поломка — состояние, о котором клиенту говорят честно."""


@dataclass
class Sequence:
    """Одна последовательность в батче — то, что стадия про неё знает.

    Наш собственный тип, а не vLLM'ный: он едет между стадиями, и привязывать
    провод к внутренностям движка значило бы менять протокол вместе с его
    версией. Он же лежит в очереди планировщика — чтобы выбранный батч уходил
    в движок и на провод без перекладывания в третью структуру.
    """

    request_id: str
    prompt_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    seed: Optional[int] = None

    @property
    def computed(self) -> int:
        """Сколько токенов уже посчитано.

        На первом шаге декодирования это весь промпт; дальше — промпт плюс
        все выданные токены, кроме последнего: он и есть вход этого шага.
        """
        if not self.output_ids:
            return len(self.prompt_ids)
        return len(self.prompt_ids) + len(self.output_ids) - 1

    @property
    def done(self) -> bool:
        return len(self.output_ids) >= self.max_tokens


class Scheduler:
    """Очередь запросов и выбор тех, кого считать сейчас."""

    def __init__(self, *, max_sequences: int = MAX_SEQUENCES,
                 max_batch_tokens: int = MAX_BATCH_TOKENS) -> None:
        self.max_sequences = max(1, max_sequences)
        self.max_batch_tokens = max(1, max_batch_tokens)
        self._waiting: Deque[Sequence] = collections.deque()
        self._running: Dict[str, Sequence] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- приём
    def add(self, request: Sequence) -> None:
        """Принять запрос или отказать, назвав причину."""
        if len(request.prompt_ids) > self.max_batch_tokens:
            raise Full(
                f"промпт из {len(request.prompt_ids)} токенов не помещается в шаг "
                f"({self.max_batch_tokens}); разрежьте его или поднимите потолок")
        with self._lock:
            if len(self._waiting) + len(self._running) >= self.max_sequences:
                raise Full(
                    f"на этом узле уже {self.max_sequences} запросов — предел. "
                    "Приходите позже: выбивать чужой мы не станем")
            self._waiting.append(request)

    def finish(self, request_id: str) -> Optional[Sequence]:
        """Убрать запрос: он кончился или его бросили."""
        with self._lock:
            self._waiting = collections.deque(
                item for item in self._waiting if item.request_id != request_id)
            return self._running.pop(request_id, None)

    def accepted(self, token: int, request_id: str) -> None:
        """Записать выданный токен. Отсюда планировщик узнаёт, что запрос жив
        и сколько ему осталось."""
        with self._lock:
            request = self._running.get(request_id)
            if request is not None:
                request.output_ids.append(token)

    # ------------------------------------------------------------- выбор
    def next_batch(self) -> Tuple[str, List[Sequence]]:
        """Кого считать сейчас: («prefill»|«decode»|«», список).

        Новые вперёд идущих: пока запрос не посчитал промпт, он не отвечает
        вовсе, а идущий уже выдаёт токены. Задержать выдачу на шаг дешевле,
        чем держать нового в тишине.
        """
        with self._lock:
            fresh = self._take_fresh()
            if fresh:
                return "prefill", fresh
            alive = [item for item in self._running.values() if not item.done]
            if alive:
                return "decode", alive[:self.max_sequences]
            return "", []

    def _take_fresh(self) -> List[Sequence]:
        """Сколько новых промптов влезет в один шаг по токенам и по местам."""
        taken: List[Sequence] = []
        tokens = 0
        while self._waiting:
            candidate = self._waiting[0]
            length = len(candidate.prompt_ids)
            if taken and tokens + length > self.max_batch_tokens:
                break
            # Взятые уже лежат в _running, второй раз их считать нельзя —
            # иначе потолок в N мест пропускает N/2 запросов.
            if len(self._running) >= self.max_sequences:
                break
            self._waiting.popleft()
            self._running[candidate.request_id] = candidate
            taken.append(candidate)
            tokens += length
        return taken

    # ------------------------------------------------------------ отчёт
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ждут": len(self._waiting),
                "считаются": len(self._running),
                "мест": self.max_sequences - len(self._waiting) - len(self._running),
            }

    def running(self, request_id: str) -> Optional[Sequence]:
        with self._lock:
            return self._running.get(request_id)
