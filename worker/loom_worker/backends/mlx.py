"""MLX backend adapter (Apple Silicon): `mlx_lm.server` behind a memory limit.

mlx_lm.server has no CLI flag for a memory cap, but the MLX runtime exposes
`mx.set_memory_limit(bytes)` (verified against the installed MLX API). The
adapter therefore launches a thin wrapper (`loom_worker.backends.mlx_launcher`)
that applies the limit in-process before handing control to mlx_lm.server.

MLX uses unified memory, so the watchdog's RSS enforcement doubles as the
hard-kill safety net on top of the soft runtime limit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter


class MlxBackend(BackendAdapter):
    def __init__(self, *, extra_args: Optional[List[str]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        return [
            sys.executable,
            "-m",
            "loom_worker.backends.mlx_launcher",
            "--model",
            self.weights_uri,
            "--port",
            str(self.port),
            "--memory-limit-bytes",
            str(self.vram_quota_bytes),
            *self.extra_args,
        ]

    def health_path(self) -> str:
        # mlx_lm.server has no /health; /v1/models answers 200 once it is up.
        return "/v1/models"

    def prepare(self) -> None:
        if self.start_layer != 0:
            raise NotImplementedError(
                "MLX adapter v0 serves full models only; partial shard "
                f"[{self.start_layer}, {self.end_layer}) is not supported yet"
            )

    def _spawn(self) -> None:
        env = os.environ.copy()
        pkg_parent = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            self.command(), stdout=sys.stdout, stderr=sys.stderr, env=env
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def pid(self) -> Optional[int]:
        if self._proc is None or self._proc.poll() is not None:
            return None
        return self._proc.pid
