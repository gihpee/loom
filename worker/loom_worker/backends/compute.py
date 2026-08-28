"""Backend that rents this node out for arbitrary work.

Not a stage backend: a compute node holds no layers and takes no part in a
pipeline, so it deliberately does not inherit ShardBackend and is never handed
a topology. What it gets is the ordinary supervision every backend gets —
start, health, VRAM quota, watchdog — which is the whole reason arbitrary work
can be rented out at all without a second control plane.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from loom_worker.backends.base import BackendAdapter


class ComputeBackend(BackendAdapter):

    serves_partial_shard = False

    def __init__(self, *, allowed_images: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.allowed_images = allowed_images

    def command(self) -> List[str]:
        return [
            sys.executable, "-m", "loom_worker.compute.server",
            "--port", str(self.port),
            "--model-id", self.model_id,
            "--allowed-images", self.allowed_images,
        ]

    def prepare(self) -> None:
        return None
