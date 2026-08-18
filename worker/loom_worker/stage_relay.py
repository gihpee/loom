"""Local relay: bridges the stage subprocess and the orchestrator tunnel.

The stage server cannot reach other workers (nobody can — they are all behind
NAT with no open ports). So it POSTs every inter-stage message to this loopback
relay, and the agent forwards it into its existing outbound tunnel. The
orchestrator then routes it to the node that owns the target stage.

    stage(head) -> agent relay -> tunnel -> orchestrator -> tunnel
                -> agent -> stage(next)  ->  ... -> back to head

Incoming messages travel the reverse path: the agent receives a StageEnvelope
on the tunnel and POSTs it to the local stage server's /stage/forward.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from loom_worker.proto import dataplane_pb2

logger = logging.getLogger("loom_worker.stage_relay")

_KIND_TO_PROTO = {
    "activations": dataplane_pb2.ACTIVATIONS,
    "token": dataplane_pb2.TOKEN,
    "free": dataplane_pb2.FREE,
    "error": dataplane_pb2.STAGE_ERROR,
}
_PROTO_TO_KIND = {v: k for k, v in _KIND_TO_PROTO.items()}


def envelope_from_json(payload: dict, *, pipeline_id: str, model_id: str, from_stage: int):
    """JSON from the stage process -> StageEnvelope for the wire."""
    kind = _KIND_TO_PROTO.get(payload.get("kind", ""), dataplane_pb2.STAGE_KIND_UNSPECIFIED)
    env = dataplane_pb2.StageEnvelope(
        pipeline_id=pipeline_id,
        model_id=model_id,
        request_id=payload.get("request_id", ""),
        target_stage=int(payload.get("target_stage", 0)),
        from_stage=from_stage,
        kind=kind,
        step=int(payload.get("step", 0)),
        token_id=int(payload.get("token_id", 0)),
        error=payload.get("error", ""),
        # Latency instrumentation: durations measured by each stage on its own
        # clock, so the head can split a token into compute and transport.
        compute_ms=float(payload.get("compute_ms") or 0.0),
        upstream_ms=float(payload.get("upstream_ms") or 0.0),
        hops=int(payload.get("hops") or 0),
    )
    if payload.get("positions"):
        env.positions.extend(int(p) for p in payload["positions"])
    if payload.get("tensor_b64"):
        env.tensor = base64.b64decode(payload["tensor_b64"])
        env.shape.extend(int(s) for s in payload.get("shape", []))
        env.dtype = payload.get("dtype", "float32")
    if payload.get("sampling"):
        # Sampling params ride along inside the request-scoped JSON the head
        # sends; carried through positions-agnostic fields to keep proto small.
        env.finish_reason = json.dumps(payload["sampling"])
    return env


def envelope_to_json(env) -> dict:
    """StageEnvelope from the wire -> JSON for the local stage process."""
    payload = {
        "kind": _PROTO_TO_KIND.get(env.kind, "unknown"),
        "request_id": env.request_id,
        "target_stage": env.target_stage,
        "from_stage": env.from_stage,
        "step": env.step,
        "positions": list(env.positions),
        "token_id": env.token_id,
        "error": env.error,
        "compute_ms": env.compute_ms,
        "upstream_ms": env.upstream_ms,
        "hops": env.hops,
    }
    if env.tensor:
        payload["tensor_b64"] = base64.b64encode(env.tensor).decode()
        payload["shape"] = list(env.shape)
        payload["dtype"] = env.dtype or "float32"
    if env.finish_reason:
        try:
            payload["sampling"] = json.loads(env.finish_reason)
        except json.JSONDecodeError:
            pass
    return payload


class StageRelayServer:
    """Loopback HTTP endpoint the stage process posts outgoing messages to."""

    def __init__(self, *, on_message: Callable[[dict], None]) -> None:
        self.on_message = on_message
        self._server: Optional[ThreadingHTTPServer] = None
        self.port = 0

    def start(self) -> int:
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self.send_error(400)
                    return
                try:
                    relay.on_message(payload)
                except Exception:
                    logger.exception("relay handler failed")
                body = b'{"ok":true}'
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(
            target=self._server.serve_forever, name="stage-relay", daemon=True
        ).start()
        logger.info("stage relay listening on 127.0.0.1:%d", self.port)
        return self.port

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/relay"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None


def post_to_stage(port: int, payload: dict, *, timeout: float = 60.0) -> None:
    """Deliver an inbound stage message to the local stage server."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/stage/forward",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.error("delivering stage message to local stage failed: %s", exc)
