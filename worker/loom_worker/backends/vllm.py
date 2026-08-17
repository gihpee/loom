"""vLLM backend adapter: `vllm serve` subprocess with a VRAM quota.

The quota arrives as absolute bytes (broker-granted); vLLM takes a fraction of
total device memory (--gpu-memory-utilization), so the adapter converts using
the device's total VRAM.

v0 limitation (single-model MVP): only full-model shards are supported
(start_layer=0, end_layer=num_layers). Serving a *partial* layer range on
vLLM requires the pipeline-stage executor machinery (hidden-state ingress/
egress), which is out of scope for Phase 1 and tracked for the multi-stage
data plane work.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter

GIB = 1024**3


class VllmBackend(BackendAdapter):
    def __init__(self, *, total_vram_bytes: Optional[int] = None, extra_args: Optional[List[str]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total_vram_bytes = total_vram_bytes or int(
            float(os.environ.get("LOOM_TOTAL_VRAM_GB", "0")) * GIB
        )
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    def gpu_memory_utilization(self) -> float:
        if self.total_vram_bytes <= 0:
            # Unknown total: fall back to vLLM's default share.
            return 0.9
        frac = self.vram_quota_bytes / self.total_vram_bytes
        return max(0.05, min(0.95, frac))

    def command(self) -> List[str]:
        return [
            "vllm",
            "serve",
            self.weights_uri,
            "--port",
            str(self.port),
            "--served-model-name",
            self.model_id,
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization():.3f}",
            *self.extra_args,
        ]

    def prepare(self) -> None:
        if self.start_layer != 0:
            raise NotImplementedError(
                "vLLM adapter v0 serves full models only; partial shard "
                f"[{self.start_layer}, {self.end_layer}) is not supported yet"
            )

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command(),
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy(),
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
