"""Pipeline stage served by vLLM instead of transformers.

Same process shape as the `shard` backend — a stage server on loopback that
the agent relays to — but the layers run on vLLM's GPU runner: paged KV cache,
CUDA graphs, fused kernels. See docs/VLLM_PIPELINE.md for how a stage of a
model is carved out of an engine that assumes it owns the whole thing.

Kept as a separate backend type rather than a flag on `shard` so a pool can
hold both: a CUDA node takes vllm_shard, a CPU or Apple node still takes shard,
and the broker places each model on what its nodes can actually run.
"""

from __future__ import annotations

from typing import List

from loom_worker.backends.shard import ShardBackend


class VllmShardBackend(ShardBackend):
    """A `shard` stage whose engine is vLLM."""

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
            "vllm",
            "--num-model-layers",
            str(self.num_model_layers),
            # The broker's grant, so vLLM sizes its KV cache to the share of the
            # card Loom actually handed to this model.
            "--vram-quota-bytes",
            str(self.vram_quota_bytes),
        ]
        # bf16 on GPU: the transformers default of float32 would double the
        # weights and halve the KV cache for nothing.
        if "--dtype" in cmd:
            cmd[cmd.index("--dtype") + 1] = "bfloat16"
        return cmd

    def prepare(self) -> None:
        if not str(self.device).startswith("cuda"):
            raise NotImplementedError(
                "the vllm_shard backend needs a CUDA device; use backend_type "
                "'shard' on CPU/Apple nodes"
            )
        return None
