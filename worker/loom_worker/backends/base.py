"""Backend adapter interface.

Fixed in Phase 1 so that adding SGLang/MLX in Phase 3 does not touch
control-plane code: a backend is a subprocess that serves an OpenAI-compatible
HTTP endpoint for one model shard, within a VRAM/RSS quota.
"""

from __future__ import annotations

import abc
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("loom_worker.backend")

# A cold start pulls the whole checkpoint from HuggingFace (tens of GB) before
# the server ever answers /health, so the readiness wait must be generous.
# Overridable because a slow link legitimately needs more.
READY_TIMEOUT_S = float(os.environ.get("LOOM_BACKEND_READY_TIMEOUT_S", "3600"))


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class BackendAdapter(abc.ABC):
    """Lifecycle: prepare() -> start() -> [serving] -> stop()."""

    def __init__(
        self,
        *,
        model_id: str,
        weights_uri: str,
        start_layer: int,
        end_layer: int,
        vram_quota_bytes: int,
        port: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.weights_uri = weights_uri
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.vram_quota_bytes = vram_quota_bytes
        self.port = port or pick_free_port()

    @abc.abstractmethod
    def prepare(self) -> None:
        """Fetch/validate weights, build launch config. May be slow."""

    def start(self) -> None:
        """Start the serving subprocess (non-blocking).

        Idempotent on purpose: a repeated StartServing (orchestrator retry,
        reconnect, rebalance race) must NOT spawn a second engine. Losing the
        handle to a live process would orphan a subprocess holding the whole
        VRAM quota, and the replacement would then fail to allocate.
        """
        if self.is_running():
            logger.warning(
                "%s: backend already running (pid=%s, port=%d); ignoring start",
                self.model_id,
                self.pid(),
                self.port,
            )
            return
        self._spawn()

    @abc.abstractmethod
    def _spawn(self) -> None:
        """Actually launch the subprocess. Called only when none is running."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Terminate the subprocess and release resources."""

    @abc.abstractmethod
    def pid(self) -> Optional[int]:
        """Subprocess pid (for the watchdog), or None if not running."""

    def is_running(self) -> bool:
        return self.pid() is not None

    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health_path(self) -> str:
        return "/health"

    def wait_healthy(
        self, timeout_s: Optional[float] = None, poll_s: float = 0.5, log_every_s: float = 30.0
    ) -> bool:
        """Poll the health endpoint until 200, process death, or timeout.

        Progress is logged periodically: a first start spends minutes fetching
        weights with nothing on the port yet, and silence there is indis-
        tinguishable from a hang.
        """
        timeout_s = READY_TIMEOUT_S if timeout_s is None else timeout_s
        started = time.time()
        deadline = started + timeout_s
        url = self.local_url() + self.health_path()
        next_log = started + log_every_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info(
                            "%s: backend healthy on port %d after %.0fs",
                            self.model_id,
                            self.port,
                            time.time() - started,
                        )
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if self.pid() is None:
                logger.error(
                    "%s: backend process exited before becoming healthy (%.0fs)",
                    self.model_id,
                    time.time() - started,
                )
                return False
            if time.time() >= next_log:
                logger.info(
                    "%s: backend still starting (%.0fs elapsed, pid=%s, port=%d)",
                    self.model_id,
                    time.time() - started,
                    self.pid(),
                    self.port,
                )
                next_log = time.time() + log_every_s
            time.sleep(poll_s)
        logger.error(
            "%s: backend did not become healthy within %.0fs", self.model_id, timeout_s
        )
        return False
