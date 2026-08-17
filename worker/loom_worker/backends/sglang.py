"""SGLang backend adapter: `sglang.launch_server` subprocess with a VRAM quota.

Same contract as the vLLM adapter: the broker-granted absolute byte quota is
converted to SGLang's fraction of total device memory (--mem-fraction-static).

v0 limitation mirrors vLLM: full-model shards only; partial layer ranges need
the pipeline-stage executor machinery (multi-stage data plane).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter

GIB = 1024**3


class SglangBackend(BackendAdapter):
    def __init__(
        self,
        *,
        total_vram_bytes: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.total_vram_bytes = total_vram_bytes or int(
            float(os.environ.get("LOOM_TOTAL_VRAM_GB", "0")) * GIB
        )
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    def mem_fraction_static(self) -> float:
        if self.total_vram_bytes <= 0:
            # Unknown total: SGLang's own conservative default share.
            return 0.85
        frac = self.vram_quota_bytes / self.total_vram_bytes
        return max(0.05, min(0.95, frac))

    def command(self) -> List[str]:
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.weights_uri,
            "--port",
            str(self.port),
            "--host",
            "0.0.0.0",
            "--served-model-name",
            self.model_id,
            "--mem-fraction-static",
            f"{self.mem_fraction_static():.3f}",
            *self.extra_args,
        ]

    def prepare(self) -> None:
        if self.start_layer != 0:
            raise NotImplementedError(
                "SGLang adapter v0 serves full models only; partial shard "
                f"[{self.start_layer}, {self.end_layer}) is not supported yet"
            )

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            self.command(), stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy()
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
