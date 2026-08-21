"""Pipeline stage served by MLX on Apple Silicon.

Same process shape as the `shard` backend — a stage server on loopback that
the agent relays to — but the layers run on the Mac's GPU through Metal.

Kept as its own backend type for the same reason `vllm_shard` is: a pool holds
machines of different kinds, and the broker should place each model on what its
nodes can actually run. A CUDA node takes `vllm_shard` or `shard`, an Apple
node takes `mlx_shard`, and neither image has to know the other exists.

Native only, and not by choice. Metal is a macOS userspace framework; a Linux
container on a Mac is a guest in a VM with no GPU device of any kind — not even
/dev/dri. There is no passthrough to enable. See docs/MLX_STAGE.md for how the
worker is run as a launchd service instead, which is the closest thing to
`docker run --restart unless-stopped` that a Mac offers.
"""

from __future__ import annotations

import platform
from typing import List

from loom_worker.backends.shard import ShardBackend


class MlxShardBackend(ShardBackend):
    """A `shard` stage whose engine is MLX."""

    serves_partial_shard = True

    def __init__(self, *, num_model_layers: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # The tail stage is the one holding the last layer, and only the whole
        # model's depth says which that is.
        self.num_model_layers = num_model_layers or int(
            (self.topology or {}).get("num_model_layers") or 0
        )

    def command(self) -> List[str]:
        cmd = super().command()
        cmd += [
            "--engine",
            "mlx",
            "--num-model-layers",
            str(self.num_model_layers),
            # MLX unified memory: the grant is a share of the machine's RAM,
            # and the runtime enforces it itself so an over-allocation is a
            # refusal rather than the whole box starting to swap.
            "--vram-quota-bytes",
            str(self.vram_quota_bytes),
        ]
        # dtype is the checkpoint's business here. mlx_lm reads it from the
        # model (often a quantised one), and forcing a dtype would either be
        # ignored or would silently dequantise.
        return cmd

    def prepare(self) -> None:
        if platform.system() != "Darwin" or not platform.machine().startswith("arm"):
            raise NotImplementedError(
                "the mlx_shard backend runs on Apple Silicon only; use "
                "backend_type 'shard' or 'vllm_shard' on this node"
            )
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise NotImplementedError(
                "mlx and mlx_lm are not installed in this worker; install with "
                "'pip install loom-worker[mlx]' (see docs/MLX_STAGE.md)"
            ) from exc
        return None
