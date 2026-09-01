"""Байтовый канал снаружи внутрь, поверх стрима, который узел открыл сам.

Здесь настоящий агент и настоящий gRPC: смысл канала в том, как он ведёт себя
с байтами и с отказами, а не в нашей арифметике.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from loom.orchestrator.agents import AgentError  # noqa: E402

from test_agent_gateway import Orchestrator, _Settings  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Echo:
    """Служба на узле, до которой тянется канал."""

    def __init__(self) -> None:
        self.port = free_port()
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.port))
        self._server.listen(8)
        self.clients: list = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            self.clients.append(client)
            threading.Thread(target=self._echo, args=(client,), daemon=True).start()

    def _echo(self, client: socket.socket) -> None:
        try:
            while True:
                piece = client.recv(65536)
                if not piece:
                    return
                client.sendall(piece)
        except OSError:
            return
        finally:
            client.close()

    def drop_clients(self) -> None:
        """Оборвать принятые соединения. Закрытие слушателя их не трогает —
        на этом и попался первый вариант теста."""
        for client in self.clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        self.clients.clear()

    def stop(self) -> None:
        self.drop_clients()
        try:
            self._server.close()
        except OSError:
            pass


@pytest.fixture
def stand(tmp_path, monkeypatch):
    """Оркестратор и один настоящий агент."""
    from conftest_agent import start_agent

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")
    orchestrator = Orchestrator().start()
    agent, thread = start_agent(orchestrator.port, tmp_path / "node")
    deadline = time.time() + 20
    while time.time() < deadline and not orchestrator.hub.sessions:
        time.sleep(0.05)
    assert orchestrator.hub.sessions, "агент не подключился"
    echo = Echo()
    yield orchestrator, agent, echo
    echo.stop()
    agent.stop()
    thread.join(timeout=10)
    orchestrator.stop()


def running_task(orchestrator, echo_port: int) -> str:
    """Задача, которой принадлежит порт: без неё канал открывать не для кого."""
    task = orchestrator.hub.submit(
        command=[sys.executable, "-c", "import time; time.sleep(120)"])
    orchestrator.wait_state(task.task_id, "running", timeout=60)
    return task.task_id


def through(orchestrator, task_id: str, port: int, payload: bytes,
            *, want: int = 0) -> bytes:
    """Прогнать байты по каналу, как это делал бы клиент."""
    async def talk():
        tunnel = orchestrator.hub.open_tunnel(task_id, port)
        try:
            tunnel.send(payload)
            got = bytearray()
            while len(got) < (want or len(payload)):
                piece = await tunnel.recv()
                if not piece:
                    break
                got.extend(piece)
            return bytes(got)
        finally:
            tunnel.close()

    return orchestrator.call(talk(), timeout=60)


def allow(agent, port: int) -> None:
    """Разрешение выдаёт задача — здесь делаем это за неё напрямую."""
    agent.commands.allowed_ports.add(port)


def test_байты_доходят_до_узла_и_обратно(stand):
    orchestrator, agent, echo = stand
    task_id = running_task(orchestrator, echo.port)
    allow(agent, echo.port)
    assert through(orchestrator, task_id, echo.port, b"loom") == b"loom"


def test_двоичное_переживает_дорогу(stand):
    """По каналу идёт чужой протокол, и он не текст."""
    orchestrator, agent, echo = stand
    task_id = running_task(orchestrator, echo.port)
    allow(agent, echo.port)
    payload = bytes(range(256)) * 16
    assert through(orchestrator, task_id, echo.port, payload) == payload


def test_неразрешённый_порт_закрыт(stand):
    """Иначе канал наружу означал бы доступ ко всему, что слушает на машине."""
    orchestrator, agent, echo = stand
    task_id = running_task(orchestrator, echo.port)
    # Разрешения нет намеренно.
    assert through(orchestrator, task_id, echo.port, b"x", want=1) == b""


def test_канал_к_чужой_задаче_не_открыть(stand):
    orchestrator, agent, echo = stand
    allow(agent, echo.port)
    with pytest.raises(AgentError, match="no task"):
        orchestrator.call(_open(orchestrator, "task-которой-нет", echo.port))


def test_закрытие_на_узле_доходит_до_нас(stand):
    """Клиент должен увидеть конец потока, а не зависнуть."""
    orchestrator, agent, echo = stand
    task_id = running_task(orchestrator, echo.port)
    allow(agent, echo.port)

    async def talk():
        tunnel = orchestrator.hub.open_tunnel(task_id, echo.port)
        tunnel.send("привет".encode())
        assert await tunnel.recv()          # эхо вернулось
        echo.drop_clients()                  # служба на узле оборвала соединение
        deadline = time.time() + 20
        while time.time() < deadline:
            if not await tunnel.recv():
                return True
        return False

    assert orchestrator.call(talk(), timeout=60)


async def _open(orchestrator, task_id: str, port: int):
    return orchestrator.hub.open_tunnel(task_id, port)
