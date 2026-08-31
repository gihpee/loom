"""Порты соседей, притворяющиеся местными.

Задача, которая ходит к соседям по РАНГУ, ничего этого не требует: агент возит
её сообщения сам. Требует чужой софт — прежде всего Ray, который собирает
кластер, обращаясь к адресам и портам, и переписать который нельзя, потому что
весь смысл в том, чтобы код клиента работал как есть.

Приём такой. Ранг N получает свой непересекающийся диапазон портов; агент на
каждом узле слушает у себя на 127.0.0.1 диапазоны ЧУЖИХ рангов и возит принятые
соединения в туннель до нужного пира. Все узлы видят одинаковую картину «ранг M
живёт на локалхосте», и Ray про NAT не узнаёт никогда.

Два следствия, оба существенные:

**Ранги на одной машине не проксируются вовсе.** Их Ray уже слушает эти порты
по-настоящему, и вклиниться туда значило бы не ускорить, а сломать: порт занят,
слушатель не встанет. Так что локальный сосед — это просто отсутствие работы.

**Агент не знает раскладку портов.** Её присылает сама задача (`/forward` на
канале), потому что раскладку определяет версия Ray, а не версия агента —
иначе обновление Ray стало бы обновлением парка.
"""

from __future__ import annotations

import logging
import selectors
import socket
import threading
import uuid
from typing import Callable, Dict, List, Optional

from loom_agent.p2p.tunnel import RemoteSide, TunnelRefused, pump

logger = logging.getLogger("loom_agent.tasks.forward")

# Сколько соединений держать в очереди на каждом слушателе. Ray открывает их
# пачками при сборке кластера.
BACKLOG = 32


class ForwardRefused(RuntimeError):
    """Пробросить не получилось, и вот почему."""


class Forwarder:
    """Слушатели чужих портов на этом узле, по задаче.

    Одна нить приёма на всех: диапазон ранга — это десятки портов, а группа из
    четырёх узлов даёт под три сотни слушателей. Нить на каждый — три сотни
    нитей, спящих в accept, ради работы, которой хватает одной.
    """

    def __init__(self, *, stub_for: Optional[Callable[[str], object]] = None,
                 allow_local: Optional[Callable[[List[int]], None]] = None) -> None:
        # Как достать стаб соседа. None означает «p2p нет» — тогда пробрасывать
        # некуда, и мы говорим об этом сразу, а не молча слушаем впустую.
        self.stub_for = stub_for
        # Чем открыть СВОИ порты входящим: соседи тянутся к нам так же.
        self.allow_local = allow_local or (lambda _ports: None)
        self._sel = selectors.DefaultSelector()
        self._by_task: Dict[str, List[socket.socket]] = {}
        self._targets: Dict[socket.socket, str] = {}   # слушатель → peer_id
        self._ports: Dict[socket.socket, int] = {}
        self._lock = threading.RLock()
        self._wake_r, self._wake_w = socket.socketpair()
        self._sel.register(self._wake_r, selectors.EVENT_READ)
        self._stop = threading.Event()
        self._loop: Optional[threading.Thread] = None

    # ------------------------------------------------------------- открытие
    def open(self, task_id: str, *, mine: List[int], remote: Dict[int, str],
             ports: Dict[int, List[int]]) -> dict:
        """Начать пробрасывать для этой задачи.

        `mine`   — порты нашего ранга: их надо открыть входящим.
        `remote` — ранг → peer_id тех, кто НЕ на этой машине.
        `ports`  — ранг → его порты, как их посчитала сама задача.
        """
        self.allow_local(list(mine))
        if not remote:
            # Все соседи на этой же машине: их Ray уже слушает эти порты
            # по-настоящему, и наше вмешательство только отняло бы их.
            return {"listening": 0, "ranks": []}
        if self.stub_for is None:
            raise ForwardRefused(
                "на этом узле нет прямого канала до соседей, а без него ранги "
                "друг друга не найдут")

        opened: List[socket.socket] = []
        try:
            for rank, peer_id in sorted(remote.items()):
                for port in ports.get(rank, []):
                    opened.append(self._listen(port, peer_id))
        except Exception:
            for sock in opened:
                self._drop(sock)
            raise
        with self._lock:
            self._by_task.setdefault(task_id, []).extend(opened)
        self._ensure_loop()
        self._wake()
        logger.info("задача %s: слушаю %d чужих портов для рангов %s",
                    task_id, len(opened), sorted(remote))
        return {"listening": len(opened), "ranks": sorted(remote)}

    def close(self, task_id: str) -> None:
        with self._lock:
            socks = self._by_task.pop(task_id, [])
        for sock in socks:
            self._drop(sock)
        self._wake()

    def close_all(self) -> None:
        with self._lock:
            tasks = list(self._by_task)
        for task_id in tasks:
            self.close(task_id)
        self._stop.set()
        self._wake()

    @property
    def listening(self) -> int:
        with self._lock:
            return sum(len(s) for s in self._by_task.values())

    # -------------------------------------------------------------- частное
    def _listen(self, port: int, peer_id: str) -> socket.socket:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            sock.close()
            raise ForwardRefused(
                f"порт {port} на этом узле занят ({exc}); если ранги делят "
                "машину, их диапазоны обязаны различаться") from None
        sock.listen(BACKLOG)
        sock.setblocking(False)
        with self._lock:
            self._targets[sock] = peer_id
            self._ports[sock] = port
        self._sel.register(sock, selectors.EVENT_READ)
        return sock

    def _drop(self, sock: socket.socket) -> None:
        try:
            self._sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        with self._lock:
            self._targets.pop(sock, None)
            self._ports.pop(sock, None)
        try:
            sock.close()
        except OSError:
            pass

    def _ensure_loop(self) -> None:
        if self._loop is not None and self._loop.is_alive():
            return
        self._stop.clear()
        self._loop = threading.Thread(target=self._accept_forever,
                                      name="loom-forward", daemon=True)
        self._loop.start()

    def _wake(self) -> None:
        try:
            self._wake_w.send(b"\x00")
        except OSError:
            pass

    def _accept_forever(self) -> None:
        while not self._stop.is_set():
            try:
                events = self._sel.select(timeout=1.0)
            except OSError:
                continue
            for key, _mask in events:
                if key.fileobj is self._wake_r:
                    try:
                        self._wake_r.recv(4096)
                    except OSError:
                        pass
                    continue
                self._accept(key.fileobj)

    def _accept(self, listener: socket.socket) -> None:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        with self._lock:
            peer_id = self._targets.get(listener)
            port = self._ports.get(listener)
        if peer_id is None or port is None:
            client.close()
            return
        client.setblocking(True)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=self._carry, args=(client, peer_id, port),
                         name=f"forward-{port}", daemon=True).start()

    def _carry(self, client: socket.socket, peer_id: str, port: int) -> None:
        remote = RemoteSide(self.stub_for(peer_id), uuid.uuid4().hex[:12], port)
        try:
            remote.open()
        except (TunnelRefused, Exception) as exc:
            # Отказ соседа — не наша поломка: Ray переоткроет соединение.
            # Но молчать нельзя, иначе «кластер не собрался» останется без
            # единого следа о том, почему.
            logger.warning("туннель к %s:%d не открылся: %s", peer_id[:12], port, exc)
            try:
                client.close()
            except OSError:
                pass
            return
        pump(client, remote, closed=threading.Event())
