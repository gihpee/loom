"""TEST-ONLY echo backend.

Serves a minimal OpenAI-compatible /v1/chat/completions that echoes the last
user message. Exists so the full control plane (LoadShard -> StartServing ->
API proxy) can be exercised end-to-end on hosts without GPUs. Runs as a real
subprocess so lifecycle and the watchdog kill-path match production backends.
Not a production backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter


class EchoBackend(BackendAdapter):

    # A stub, so it "serves" any layer range asked of it. That is what lets
    # the control plane, the tunnel and multi-stage placement be tested on
    # machines with no GPU.
    serves_partial_shard = True
    def __init__(self, *, startup_delay_s: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.startup_delay_s = startup_delay_s
        self._proc: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        return [
            sys.executable,
            "-m",
            "loom_worker.backends.echo_server",
            "--port",
            str(self.port),
            "--model-id",
            self.model_id,
            "--startup-delay",
            str(self.startup_delay_s),
        ]

    def prepare(self) -> None:
        return None

    def _spawn(self) -> None:
        env = os.environ.copy()
        # Make loom_worker importable in the child even when the parent runs
        # from a source checkout (tests) rather than an installed package.
        pkg_parent = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            self.command(), stdout=sys.stdout, stderr=sys.stderr, env=env
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def pid(self) -> Optional[int]:
        if self._proc is None or self._proc.poll() is not None:
            return None
        return self._proc.pid
