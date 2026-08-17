"""Backend adapter interface.

Fixed in Phase 1 so that adding SGLang/MLX in Phase 3 does not touch
control-plane code: a backend is a subprocess that serves an OpenAI-compatible
HTTP endpoint for one model shard, within a VRAM/RSS quota.
"""

from __future__ import annotations

import abc
import socket
import time
import urllib.error
import urllib.request
from typing import Optional


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

    @abc.abstractmethod
    def start(self) -> None:
        """Start the serving subprocess (non-blocking)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Terminate the subprocess and release resources."""

    @abc.abstractmethod
    def pid(self) -> Optional[int]:
        """Subprocess pid (for the watchdog), or None if not running."""

    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health_path(self) -> str:
        return "/health"

    def wait_healthy(self, timeout_s: float = 300.0, poll_s: float = 0.5) -> bool:
        """Poll the health endpoint until 200 or timeout."""
        deadline = time.time() + timeout_s
        url = self.local_url() + self.health_path()
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if self.pid() is None:
                return False  # process died
            time.sleep(poll_s)
        return False
