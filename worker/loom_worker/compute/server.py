"""The compute backend: this node rents itself out for arbitrary work.

A sibling of the model backends, not a change to them. It answers a small HTTP
surface through the same tunnel everything else uses, so a client's task needs
no open port on the node and the node needs no new connection to the world.

    POST /task/run      start a task here
    POST /task/stop     stop it
    POST /task/status   how it is going
    POST /task/logs     what it has said
    GET  /health        what this node will and will not run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

from loom_worker.compute.runtime import (
    Task,
    TaskRefused,
    TaskSpec,
    available_runtime,
)

logger = logging.getLogger("loom_worker.compute.server")

TASKS: Dict[str, Task] = {}
LOCK = threading.RLock()
STATE: dict = {}


def run_task(payload: dict) -> dict:
    runtime = STATE["runtime"]
    if not runtime:
        raise TaskRefused(
            "this node runs no tasks: it has no Docker socket, and the weaker "
            "process runtime is off. Mount /var/run/docker.sock to run "
            "containers, or set LOOM_ALLOW_PROCESS_TASKS=1 to accept "
            "subprocesses — which share this machine's kernel and filesystem"
        )
    spec = TaskSpec(
        task_id=payload["task_id"],
        image=payload.get("image", ""),
        command=list(payload.get("command") or []),
        env=dict(payload.get("env") or {}),
        vram_bytes=int(payload.get("vram_bytes") or 0),
        ram_bytes=int(payload.get("ram_bytes") or 0),
        cpus=float(payload.get("cpus") or 1.0),
        gpus=int(payload.get("gpus") or 0),
        timeout_s=int(payload.get("timeout_s") or 3600),
        network=payload.get("network") or "none",
    )
    if STATE["allowed_images"] and not _permitted(spec.image):
        raise TaskRefused(
            f"this node only runs images matching {STATE['allowed_images']}; "
            f"{spec.image!r} is not one of them"
        )
    with LOCK:
        if spec.task_id in TASKS:
            raise TaskRefused(f"task {spec.task_id} is already here")
        task = Task(spec)
        TASKS[spec.task_id] = task
    task.start(runtime=runtime)
    return task.status()


def _permitted(image: str) -> bool:
    """Whether the node's owner allows this image.

    A marketplace where anyone may run anything on your machine is one nobody
    sane joins. The list is the owner's, set on their own worker, and empty
    means they accept the default of everything.
    """
    for pattern in STATE["allowed_images"]:
        if image == pattern or (pattern.endswith("*") and image.startswith(pattern[:-1])):
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            with LOCK:
                running = sum(1 for t in TASKS.values() if t.state == "running")
            self._json(200, {
                "ok": True,
                "runtime": STATE["runtime"] or "none",
                "running": running,
                "allowed_images": STATE["allowed_images"],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        task_id = payload.get("task_id", "")

        try:
            if self.path == "/task/run":
                self._json(200, run_task(payload))
                return
            with LOCK:
                task = TASKS.get(task_id)
            if task is None:
                self._json(404, {"error": f"no task {task_id!r} here"})
                return
            if self.path == "/task/status":
                self._json(200, task.status())
            elif self.path == "/task/logs":
                self._json(200, {
                    "task_id": task_id,
                    "logs": task.logs(tail=int(payload.get("tail") or 0)),
                })
            elif self.path == "/task/stop":
                task.stop()
                self._json(200, task.status())
            else:
                self._json(404, {"error": "not found"})
        except TaskRefused as exc:
            self._json(409, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            logger.exception("task request failed")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.environ.get("LOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(prog="loom-compute")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-id", default="")     # the job id, in Loom's terms
    parser.add_argument("--allowed-images", default=os.environ.get("LOOM_ALLOWED_IMAGES", ""))
    args = parser.parse_args(argv)

    STATE.update(
        runtime=available_runtime(),
        allowed_images=[p.strip() for p in args.allowed_images.split(",") if p.strip()],
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info(
        "compute node ready on port %d (runtime=%s, images=%s)",
        server.server_port, STATE["runtime"] or "none",
        STATE["allowed_images"] or "any",
    )
    if not STATE["runtime"]:
        logger.warning(
            "no runtime available: this node will refuse every task. Mount "
            "/var/run/docker.sock to run containers"
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
