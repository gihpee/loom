"""Backend adapters: subprocess wrappers exposing an OpenAI-compatible endpoint.

Adding a backend = adding one adapter class here. Control-plane code
(handlers, gateway, orchestrator) never changes for a new backend.
"""

from loom_worker.backends.base import BackendAdapter
from loom_worker.backends.echo import EchoBackend
from loom_worker.backends.mlx import MlxBackend
from loom_worker.backends.shard import ShardBackend
from loom_worker.backends.sglang import SglangBackend
from loom_worker.backends.vllm import VllmBackend
from loom_worker.backends.vllm_shard import VllmShardBackend

BACKENDS = {
    "vllm": VllmBackend,
    "sglang": SglangBackend,
    "mlx": MlxBackend,
    # The two backends that can serve a PART of a model, i.e. one stage of a
    # pipeline spread over several nodes: `shard` runs the layers on
    # transformers (portable, CPU included), `vllm_shard` on vLLM (CUDA only,
    # paged KV cache and CUDA graphs).
    "shard": ShardBackend,
    "vllm_shard": VllmShardBackend,
    # Test-only stub: lets the control plane and API be exercised end-to-end on
    # hosts without GPUs. Not a production backend.
    "echo": EchoBackend,
}


def make_backend(backend_type: str, **kwargs) -> BackendAdapter:
    try:
        cls = BACKENDS[backend_type]
    except KeyError:
        raise ValueError(f"Unsupported backend_type: {backend_type!r}") from None
    return cls(**kwargs)


__all__ = [
    "BackendAdapter",
    "VllmBackend",
    "SglangBackend",
    "MlxBackend",
    "ShardBackend",
    "VllmShardBackend",
    "EchoBackend",
    "BACKENDS",
    "make_backend",
]
