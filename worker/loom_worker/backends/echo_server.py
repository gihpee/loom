"""Standalone OpenAI-compatible echo server (see echo.py: TEST-ONLY).

Endpoints:
- GET  /health                  -> 200 once startup delay has elapsed
- POST /v1/chat/completions     -> echoes last user message (stream and non-stream)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STARTED_AT = time.time()
STARTUP_DELAY = 0.0
MODEL_ID = "echo"


def _completion_payload(content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(content.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": 2 * len(content.split()),
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _ready(self) -> bool:
        return (time.time() - STARTED_AT) >= STARTUP_DELAY

    def do_GET(self):
        if self.path == "/health":
            code = 200 if self._ready() else 503
            body = json.dumps({"status": "ok" if code == 200 else "starting"}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        messages = req.get("messages") or []
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        content = f"[echo:{MODEL_ID}] " + (user_texts[-1] if user_texts else "")

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            for i, word in enumerate(content.split(" ")):
                delta = {"content": word if i == 0 else " " + word}
                event = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self._write_chunk(f"data: {json.dumps(event)}\n\n")
            final = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self._write_chunk(f"data: {json.dumps(final)}\n\n")
            self._write_chunk("data: [DONE]\n\n")
            self._write_chunk("")
        else:
            body = json.dumps(_completion_payload(content)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _write_chunk(self, data: str) -> None:
        raw = data.encode()
        self.wfile.write(f"{len(raw):X}\r\n".encode() + raw + b"\r\n")
        self.wfile.flush()


def _exit_when_orphaned(poll_s: float = 2.0) -> None:
    """Exit if the process that spawned us is gone.

    Same guard as the pipeline stage server: an interrupted test run (or a
    killed agent) otherwise leaves these servers holding ports and CPU
    forever — dozens of them accumulated over a session and slowed everything
    down. Only a CHANGE of parent counts: the agent is PID 1 in a container.
    """
    original_ppid = os.getppid()

    def loop() -> None:
        while True:
            time.sleep(poll_s)
            if os.getppid() != original_ppid:
                os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def main() -> None:
    global STARTUP_DELAY, MODEL_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-id", default="echo")
    parser.add_argument("--startup-delay", type=float, default=0.0)
    args = parser.parse_args()
    STARTUP_DELAY = args.startup_delay
    MODEL_ID = args.model_id
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    _exit_when_orphaned()
    server.serve_forever()


if __name__ == "__main__":
    main()
