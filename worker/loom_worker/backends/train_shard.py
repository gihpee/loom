"""Backend that trains a stage instead of serving one.

A separate adapter, and a separate server process behind it, so that nothing
about the inference path changes: the same node can run this backend for one
model and the ordinary shard backend for another, and the code that answers
user requests is untouched either way.

The watchdog and the VRAM quota apply exactly as they do to inference, which
matters more here — a training stage holds gradients and optimiser moments on
top of the weights, and is the likeliest thing on the fleet to run a card out
of memory.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Optional

from loom_worker.backends.shard import ShardBackend


class TrainShardBackend(ShardBackend):
    """Subclasses the serving stage backend on purpose.

    Not for the behaviour — every method that matters is overridden — but for
    the type. The agent decides who gets a pipeline topology and a relay url by
    asking `issubclass(cls, ShardBackend)`, and a stage backend that is not one
    comes up believing it is a lone stage 0 of 1 with nowhere to send its
    output. That exact bug cost a deployment once already, with the mlx stage
    engine; inheriting is how it cannot happen again.
    """

    def __init__(
        self,
        *,
        mode: str = "lora",
        rank: int = 16,
        alpha: float = 32.0,
        lr: float = 1e-4,
        micro_batches: int = 4,
        dataset: str = "",
        checkpoint_dir: str = "",
        checkpoint_every: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # float32, unlike inference. Gradients in bf16 lose the small updates
        # that fine-tuning is made of, and the memory saved is not worth a run
        # that quietly learns less than it should.
        if not kwargs.get("dtype"):
            self.dtype = "float32"
        self.mode = mode
        self.rank = rank
        self.alpha = alpha
        self.lr = lr
        self.micro_batches = micro_batches
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

    def command(self) -> List[str]:
        return [
            sys.executable, "-m", "loom_worker.train.server",
            "--model-id", self.model_id,
            "--weights-uri", self.weights_uri,
            "--start-layer", str(self.start_layer),
            "--end-layer", str(self.end_layer),
            "--stage-index", str(self.topology.get("stage_index", 0)),
            "--num-stages", str(self.topology.get("num_stages", 1)),
            "--pipeline-id", str(self.topology.get("pipeline_id", "")),
            "--port", str(self.port),
            "--relay-url", self.relay_url,
            "--device", self.device,
            "--dtype", self.dtype,
            "--mode", self.mode,
            "--rank", str(self.rank),
            "--alpha", str(self.alpha),
            "--lr", str(self.lr),
            "--micro-batches", str(self.micro_batches),
            "--dataset", self.dataset,
            "--checkpoint-dir", self.checkpoint_dir,
            "--checkpoint-every", str(self.checkpoint_every),
            *self.extra_args,
        ]

    def prepare(self) -> None:
        return None  # any layer range is trainable, same as any is servable
