"""A pipeline stage with no model in it.

Stands in for the real inference stage while the transport is being proved. It
does what a stage does — receive from the rank before, do something, send to
the rank after, and serve HTTP at rank 0 — and nothing else, so a failure here
is a failure of the plumbing and never of a model.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RANK = int(os.environ.get("LOOM_RANK", "0"))
SIZE = int(os.environ.get("LOOM_GROUP_SIZE", "1"))
PORT = int(os.environ.get("LOOM_SERVE_PORT", "0"))
AGENT = os.environ.get("LOOM_AGENT_URL", "")
TASK = os.environ.get("LOOM_TASK_ID", "")

answers: "dict[str, str]" = {}
arrived = threading.Event()


def send_on(to_rank: int, body: dict) -> None:
    request = urllib.request.Request(
        f"{AGENT}/send", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "X-Loom-Task": TASK, "X-Loom-To-Rank": str(to_rank)},
    )
    urllib.request.urlopen(request, timeout=30).close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self._ok(b"")
        if self.path == "/loom/message":
            self._on_message(payload)
        elif self.path == "/ask":
            # Rank 0 only: start the pass and wait for it to come back round.
            arrived.clear()
            send_on(1, {"id": payload["id"], "text": payload["text"], "hops": [RANK]})

    def do_GET(self):
        if self.path.startswith("/answer/"):
            key = self.path.rsplit("/", 1)[1]
            if key in answers:
                self._ok(json.dumps(answers[key]).encode())
            else:
                self._ok(b'"pending"')
        else:
            self._ok(json.dumps({"rank": RANK, "size": SIZE}).encode())

    def _on_message(self, payload: dict) -> None:
        if payload.get("done"):
            answers[payload["id"]] = payload
            arrived.set()
            return
        payload["hops"] = payload.get("hops", []) + [RANK]
        if RANK + 1 < SIZE:
            send_on(RANK + 1, payload)
        elif RANK != 0:
            # Last stage: the answer goes back to the one that was asked, the
            # way a token goes back to stage 0.
            send_on(0, {"done": True, "id": payload["id"],
                        "text": payload["text"].upper(), "hops": payload["hops"]})

    def _ok(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


def announce(port: int) -> None:
    """Tell the agent which port was actually bound.

    The agent can only suggest one; between the suggestion and this bind
    another process on the same machine may have taken it, which is exactly
    what happens when a multi-GPU host runs two agents.
    """
    request = urllib.request.Request(
        f"{AGENT}/ready", data=json.dumps({"port": port}).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Loom-Task": TASK},
    )
    urllib.request.urlopen(request, timeout=30).close()


def main() -> None:
    for candidate in (PORT, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no port to serve on")
    announce(server.server_port)
    server.serve_forever()


if __name__ == "__main__":
    main()
