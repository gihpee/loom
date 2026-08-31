"""Кто на каком порту. Заодно — то, как ранги находят друг друга.

Ray устроен не звездой: воркеры ходят не только к голове, но и друг к другу за
объектами. Значит каждому рангу нужен адрес каждого — а спрашивать его негде,
потому что оркестратор в этот разговор не входит.

Поэтому адресов не ищут, их **вычисляют**. У ранга N свой непересекающийся
диапазон, и любой ранг получает порты любого другого арифметикой, ничего ни у
кого не спрашивая.

Из этого следует главное свойство: на одной машине всё работает сразу и без
посредников — ранги просто разговаривают по настоящему локалхосту. Между
машинами те же самые адреса начинает обслуживать агент, подставляя туда
туннель до нужного пира, и ни строчки здесь менять не надо.

Диапазоны портов, а НЕ разные loopback-адреса (127.0.7.N): часть компонентов
Ray биндится на 0.0.0.0 и заняла бы свой порт на всех адресах сразу, включая
чужие. С непересекающимися диапазонами такого столкновения не бывает.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

# Начало и шаг. Выше эфемерного диапазона Linux не лезем, ниже 1024 тоже:
# задача работает непривилегированной и низкий порт занять не сможет.
BASE = int(os.environ.get("LOOM_RAY_PORT_BASE", "20000"))
STRIDE = int(os.environ.get("LOOM_RAY_PORT_STRIDE", "100"))
# Сколько в конце диапазона отдать рабочим процессам Ray. Их много и они
# приходят-уходят, поэтому им отдаётся всё, что осталось после служебных.
FIRST_WORKER_OFFSET = 10


class PortsRefused(ValueError):
    """Так разложить порты нельзя, и вот почему."""


@dataclass(frozen=True)
class RankPorts:
    """Порты одного ранга. Считаются, а не выдаются."""

    rank: int
    gcs: int                 # голова; у остальных рангов не используется
    node_manager: int
    object_manager: int
    runtime_env_agent: int
    dashboard_listen: int
    dashboard_grpc: int
    metrics: int
    client_server: int      # зарезервирован; см. cluster.py
    worker_first: int
    worker_last: int

    def crossing(self) -> List[int]:
        """Порты, до которых обязаны дотянуться ДРУГИЕ узлы.

        Только они нуждаются в туннеле; остальное Ray дергает у себя же на
        локалхосте, и проброс для них был бы работой впустую.
        """
        return [self.gcs, self.node_manager, self.object_manager,
                *range(self.worker_first, self.worker_last + 1)]

    def local_only(self) -> List[int]:
        return [self.runtime_env_agent, self.dashboard_listen,
                self.dashboard_grpc, self.metrics, self.client_server]


def ports_for(rank: int, *, base: int = 0, stride: int = 0) -> RankPorts:
    """Разложить диапазон ранга. Одинаково на всех узлах — в этом смысл."""
    if rank < 0:
        raise PortsRefused(f"ранг не может быть отрицательным ({rank})")
    base = base or BASE
    stride = stride or STRIDE
    if stride <= FIRST_WORKER_OFFSET:
        raise PortsRefused(
            f"шаг {stride} не оставляет места рабочим портам: служебные "
            f"занимают первые {FIRST_WORKER_OFFSET}")
    start = base + rank * stride
    if start + stride > 65536:
        raise PortsRefused(
            f"ранг {rank} при основании {base} и шаге {stride} вышел за 65535; "
            "уменьшите LOOM_RAY_PORT_STRIDE или основание")
    return RankPorts(
        rank=rank,
        gcs=start,
        node_manager=start + 1,
        object_manager=start + 2,
        runtime_env_agent=start + 3,
        dashboard_listen=start + 4,
        dashboard_grpc=start + 5,
        metrics=start + 6,
        client_server=start + 7,
        worker_first=start + FIRST_WORKER_OFFSET,
        worker_last=start + stride - 1,
    )


def head_address(base: int = 0, stride: int = 0) -> str:
    """Куда подключаются все. Голова — всегда ранг 0, и это не соглашение
    между узлами, а следствие того же расчёта."""
    return f"127.0.0.1:{ports_for(0, base=base, stride=stride).gcs}"


def crossing_for_group(size: int, *, base: int = 0, stride: int = 0) -> dict:
    """Что агенту предстоит пробросить: ранг → его внешние порты.

    Считается здесь, а не в агенте: агент не должен знать, как Ray раскладывает
    порты, — иначе смена версии Ray станет обновлением парка.
    """
    return {rank: ports_for(rank, base=base, stride=stride).crossing()
            for rank in range(size)}
