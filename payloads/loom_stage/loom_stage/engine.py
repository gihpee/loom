"""Что стадия требует от того, кто считает слои.

Движков будет два, и различаются они не тем, «как считать», а тем, **чем
задан батч**:

`torch`  — по одной последовательности на запрос. Просто, переносимо, работает
           на CPU и проверяется без карты. Пропускная способность равна
           скорости одного запроса: параллельные клиенты делят её, а не
           складывают.
`vllm`   — continuous batching: несколько последовательностей в одном шаге
           движка. Состав батча выбирает ПЕРВАЯ стадия, остальные обязаны
           прогнать ровно тот же набор — иначе позиции в KV-кэше разъедутся.

Интерфейс здесь намеренно узкий: ровно то, что сегодня вызывает `server.py`, и
ни строчкой больше. Широкий интерфейс притянул бы в стадию подробности того
движка, который его задал, и второй пришлось бы под них подгонять.

Одна деталь стоит отдельного слова. `forward` возвращает `(hidden, logits)`, и
ровно одно из них не None: на последней стадии логиты, на всех прочих —
скрытые состояния. Так устроен конвейер, и оба движка обязаны отвечать
одинаково, потому что `server.py` различать их не должен.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple


class Engine(Protocol):
    """То, что умеет прогнать слои этой стадии."""

    def forward(self, *, request_id: str, positions: List[int],
                input_ids: Optional[List[int]] = None,
                hidden: Optional[object] = None) -> Tuple[object, object]:
        """Прогнать шаг.

        `input_ids` приходят на первой стадии, `hidden` — на всех остальных.
        Возвращает `(hidden, None)` либо `(None, logits)` — см. шапку модуля.
        """

    def sample(self, logits, *, temperature: float, top_p: float,
               seed: Optional[int] = None) -> int:
        """Выбрать токен. Зовётся только на последней стадии."""

    def free(self, request_id: str) -> None:
        """Забыть состояние запроса. KV-кэш чужого запроса — чистая трата
        памяти, а на стадии её и так впритык."""

    def serialize(self, hidden) -> Tuple[bytes, List[int], str]:
        """Тензор в то, что уедет по проводу (см. wire.py)."""

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        """Обратно."""

    def active_requests(self) -> int:
        """Сколько запросов сейчас держит состояние."""


class EngineRefused(RuntimeError):
    """Этот движок тут работать не может, и вот почему."""


def build(kind: str, shard, *, max_requests: int = 64, **options) -> Engine:
    """Собрать движок по имени.

    Отказ здесь — обычное дело, а не поломка: vLLM не работает без карты, и
    сказать об этом на старте куда полезнее, чем упасть посреди первого
    запроса сообщением из его внутренностей.
    """
    kind = (kind or "torch").strip().lower()
    if kind == "torch":
        from loom_stage.executor import ShardExecutor

        return ShardExecutor(shard, max_requests=max_requests)
    if kind == "vllm":
        from loom_stage.vllm_engine import VllmEngine

        return VllmEngine(shard, max_requests=max_requests, **options)
    raise EngineRefused(
        f"неизвестный движок {kind!r}; эта стадия умеет torch и vllm")
