"""How a task talks to the agent, and the agent to a task.

Plain HTTP on loopback, because a task may be anything: a Python process, a Go
binary, something inside an unpacked image. Every language can POST; nothing
else is true of every language.

Two directions, and the task only ever sees loopback in both:

  out   the task POSTs to the agent, saying which rank it is writing to. Where
        that rank actually lives — this machine, a peer, or the far side of the
        orchestrator — is the agent's problem and never the task's.
  in    the agent POSTs to the port the task was given.

That asymmetry is deliberate. A task that knew about peers would have to know
about NAT, relays and reconnections, and every payload would carry a copy of
it.
"""

from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

logger = logging.getLogger("loom_agent.tasks.channel")

TASK_HEADER = "X-Loom-Task"
RANK_HEADER = "X-Loom-To-Rank"
# A stage waiting on the next one is waiting on a whole forward pass, which on a
# large model is seconds. Short enough to notice a dead peer, long enough not to
# call a working one dead.
DELIVER_TIMEOUT_S = 120.0


class TaskChannel:
    """The loopback endpoint tasks send through."""

    def __init__(self, *, on_send: Callable[[str, int, bytes, str], None],
                 on_ready: Optional[Callable[[str, int], None]] = None,
                 on_forward: Optional[Callable[[str, dict], dict]] = None) -> None:
        self.on_send = on_send
        # Задача просит сделать порты соседей достижимыми у себя на локалхосте.
        # Просит она, а не мы: раскладку портов определяет её софт, и знать её
        # агенту значит обновлять агента при смене версии этого софта.
        self.on_forward = on_forward or (lambda _task, _body: {"listening": 0})
        # A task saying which port it actually bound. Authoritative: the agent
        # can only suggest one, and between suggesting and the task binding it
        # another process on the same machine may have taken it — which is not
        # hypothetical, it is how a multi-GPU host runs two agents.
        self.on_ready = on_ready or (lambda _task, _port: None)
        self.port = 0
        self._server: Optional[ThreadingHTTPServer] = None

    def start(self) -> int:
        channel = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.path == "/ready":
                    length = int(self.headers.get("Content-Length") or 0)
                    try:
                        import json as _json

                        port = int(_json.loads(self.rfile.read(length) or b"{}")["port"])
                    except (ValueError, KeyError, TypeError):
                        self._answer(400, b"which port?")
                        return
                    channel.on_ready(self.headers.get(TASK_HEADER, ""), port)
                    self._answer(200, b"")
                    return
                if self.path == "/forward":
                    length = int(self.headers.get("Content-Length") or 0)
                    try:
                        import json as _json

                        body = _json.loads(self.rfile.read(length) or b"{}")
                    except ValueError:
                        self._answer(400, b"not json")
                        return
                    try:
                        answer = channel.on_forward(
                            self.headers.get(TASK_HEADER, ""), body)
                    except Exception as exc:
                        # Задача обязана узнать причину: без проброса её соседи
                        # просто не найдутся, и выглядеть это будет как
                        # зависание, а не как отказ.
                        logger.warning("проброс для %s не вышел: %s",
                                       self.headers.get(TASK_HEADER, ""), exc)
                        self._answer(502, str(exc).encode())
                        return
                    import json as _json

                    self._answer(200, _json.dumps(answer).encode())
                    return
                if self.path != "/send":
                    self._answer(404, b"no such endpoint")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                payload = self.rfile.read(length)
                task_id = self.headers.get(TASK_HEADER, "")
                try:
                    to_rank = int(self.headers.get(RANK_HEADER, ""))
                except ValueError:
                    self._answer(400, b"which rank?")
                    return
                content_type = self.headers.get("Content-Type", "application/octet-stream")
                try:
                    channel.on_send(task_id, to_rank, payload, content_type)
                except Exception as exc:
                    logger.warning("could not route a message from %s: %s", task_id, exc)
                    self._answer(502, str(exc).encode())
                    return
                # Accepted, not delivered: the sender must not wait for a
                # network hop it cannot see, and a stage that blocked until the
                # next one answered would serialise the whole pipeline.
                self._answer(202, b"")

            def do_GET(self):
                self._answer(200 if self.path == "/health" else 404, b"")

            def _answer(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_port
        threading.Thread(target=self._server.serve_forever, name="task-channel",
                         daemon=True).start()
        logger.info("task channel on 127.0.0.1:%d", self.port)
        return self.port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()


def deliver(port: int, payload: bytes, *, content_type: str = "application/octet-stream",
            from_rank: int = 0, path: str = "/loom/message") -> None:
    """Hand a message to a task listening on loopback."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=payload, method="POST",
        headers={"Content-Type": content_type, "X-Loom-From-Rank": str(from_rank)},
    )
    urllib.request.urlopen(request, timeout=DELIVER_TIMEOUT_S).close()


def request(port: int, *, method: str, path: str, body: bytes,
            headers: dict, timeout_s: float = 600.0):
    """Forward an HTTP request to whatever the task is serving.

    Returns (status, headers, body). A task that is not listening yet is a
    normal state during startup, so the caller gets the error rather than an
    exception with nothing in it.
    """
    outgoing = {k: v for k, v in (headers or {}).items()
                if k.lower() not in ("host", "content-length", "connection")}
    call = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body or None,
        method=method or "GET", headers=outgoing,
    )
    try:
        with urllib.request.urlopen(call, timeout=timeout_s) as answer:
            return answer.status, dict(answer.headers), answer.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def request_stream(port: int, *, method: str, path: str, body: bytes,
                   headers: dict, timeout_s: float = 600.0):
    """То же, но частями: (status, headers), затем куски тела.

    Генерация длинного ответа занимает минуты. Дожидаться её целиком, чтобы
    показать первое слово, — значит выглядеть зависшим ровно столько же, и
    отличить «думает» от «умерло» станет нельзя.
    """
    outgoing = {k: v for k, v in (headers or {}).items()
                if k.lower() not in ("host", "content-length", "connection")}
    call = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body or None,
        method=method or "GET", headers=outgoing,
    )
    try:
        answer = urllib.request.urlopen(call, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        yield exc.code, dict(exc.headers or {})
        yield exc.read()
        return
    with answer:
        yield answer.status, dict(answer.headers)
        while True:
            # Небольшими кусками и без readline: SSE приходит событиями, и
            # ожидание полной строки задерживало бы каждое из них.
            chunk = answer.read1(16384) if hasattr(answer, "read1") else answer.read(16384)
            if not chunk:
                return
            yield chunk
