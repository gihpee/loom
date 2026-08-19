"""Pipeline-stage backend: serves `[start_layer, end_layer)` of a model.

Unlike the vLLM/SGLang adapters (which serve a whole model), this backend is
what enables Loom's core capability — a model split across several nodes. It
runs the stage server as a subprocess (so the watchdog and VRAM quota apply as
usual) and receives inter-stage traffic from the agent over loopback HTTP.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter


class ShardBackend(BackendAdapter):

    serves_partial_shard = True
    def __init__(
        self,
        *,
        topology: Optional[dict] = None,
        relay_url: str = "",
        device: str = "cpu",
        dtype: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.topology = topology or {
            "pipeline_id": "",
            "stage_index": 0,
            "num_stages": 1,
            "is_first": True,
            "is_last": True,
        }
        self.relay_url = relay_url
        self.device = device
        # bf16 halves weight memory on GPUs; CPU stays float32 for exactness.
        self.dtype = dtype or ("bfloat16" if device.startswith("cuda") else "float32")
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        return [
            sys.executable,
            "-m",
            "loom_worker.shard.server",
            "--model-id",
            self.model_id,
            "--weights-uri",
            self.weights_uri,
            "--start-layer",
            str(self.start_layer),
            "--end-layer",
            str(self.end_layer),
            "--stage-index",
            str(self.topology["stage_index"]),
            "--num-stages",
            str(self.topology["num_stages"]),
            "--pipeline-id",
            str(self.topology.get("pipeline_id", "")),
            "--port",
            str(self.port),
            "--relay-url",
            self.relay_url,
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            *self.extra_args,
        ]

    def prepare(self) -> None:
        return None  # any layer range is valid — that is the whole point

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
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def pid(self) -> Optional[int]:
        if self._proc is None or self._proc.poll() is not None:
            return None
        return self._proc.pid
