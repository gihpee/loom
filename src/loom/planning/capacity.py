# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/scheduling/node.py — Node.get_decoder_layer_capacity() and
# Node.per_decoder_layer_kv_cache_memory (capacity math).
# Изменения: capacity вынесен из Node в отдельный явный входной параметр
# (ShardCapacity), который Resource Broker вычисляет и передаёт в Phase-1 DP.
# Бюджет памяти считается от vram_quota_bytes (квота, выданная брокером этой
# модели на этом узле), а не от полного объёма памяти устройства
# (num_gpus * memory_gb), и не из забинженного в Node ModelInfo. Сама формула
# (param_mem_ratio / вычет embedding и lm_head с учётом tie_embedding /
# mlx_bit_factor / kvcache_mem_ratio) сохранена без изменений.
"""Explicit per-shard capacity input for Phase-1 layer allocation.

In the original Parallax design a ``Node`` is bound to exactly one
``ModelInfo`` and derives its decoder-layer capacity ``c_i`` internally. In
Loom the Resource Broker slices one physical node's VRAM between several
models; each model's scheduler instance must therefore receive ``c_i`` as an
explicit input computed from the *granted quota*, not from the whole device.

``ShardCapacity`` is that explicit input. It is deliberately model-agnostic
from the scheduler's point of view: it only carries byte sizes. The helper
constructor :meth:`ShardCapacity.from_model_info` reproduces exactly the byte
sizes the original Parallax code derived internally.
"""

from dataclasses import dataclass
from math import floor
from typing import Optional

from loom.planning.model_info import ModelInfo


@dataclass
class ShardCapacity:
    """Capacity budget granted to (node, model) by the Resource Broker.

    Attributes:
        vram_quota_bytes: VRAM budget granted to this model on this node.
        per_layer_param_bytes: parameter bytes of one decoder layer (already
            including the MLX bit factor when the target device is MLX).
        embedding_param_bytes: bytes of input embedding / LM head weights.
        tie_embedding: whether embedding and LM head share weights.
        param_mem_ratio: fraction of the quota reserved for parameters.
        kvcache_mem_ratio: fraction of the quota reserved for KV cache.
    """

    vram_quota_bytes: int
    per_layer_param_bytes: int
    embedding_param_bytes: int
    tie_embedding: bool = False
    param_mem_ratio: float = 0.5
    kvcache_mem_ratio: float = 0.3

    @classmethod
    def from_model_info(
        cls,
        model_info: ModelInfo,
        *,
        vram_quota_bytes: int,
        device: str = "cuda",
        param_mem_ratio: float = 0.5,
        kvcache_mem_ratio: float = 0.3,
    ) -> "ShardCapacity":
        """Build the explicit capacity input from a model footprint.

        Mirrors the byte-size derivation of the original
        ``Node.get_decoder_layer_capacity``: per-layer parameter bytes are
        ``decoder_layer_io_bytes(roofline=False)`` with the MLX bit factor
        applied for MLX devices; endpoint cost is ``embedding_io_bytes``.
        """
        per_layer = model_info.decoder_layer_io_bytes(roofline=False)
        if device == "mlx":
            per_layer = per_layer * model_info.mlx_bit_factor
        return cls(
            vram_quota_bytes=int(vram_quota_bytes),
            per_layer_param_bytes=per_layer,
            embedding_param_bytes=model_info.embedding_io_bytes,
            tie_embedding=bool(model_info.tie_embedding),
            param_mem_ratio=param_mem_ratio,
            kvcache_mem_ratio=kvcache_mem_ratio,
        )

    def decoder_layer_capacity(
        self, include_input_embed: bool = False, include_lm_head: bool = False
    ) -> int:
        """Return how many decoder layers fit in the parameter budget (c_i).

        Same formula as the original Parallax implementation, with the memory
        budget taken from the broker-granted quota.
        """
        available_memory_bytes = floor(self.vram_quota_bytes * self.param_mem_ratio)
        if include_input_embed:
            available_memory_bytes -= self.embedding_param_bytes
        if include_lm_head:
            if not (include_input_embed and self.tie_embedding):
                available_memory_bytes -= self.embedding_param_bytes
        return floor(available_memory_bytes / self.per_layer_param_bytes)

    def kv_cache_budget_bytes(self) -> int:
        """Total KV-cache byte budget within the quota."""
        return floor(self.vram_quota_bytes * self.kvcache_mem_ratio)

    def per_layer_kv_cache_memory(self, num_current_layers: int) -> Optional[int]:
        """KV-cache bytes available per hosted layer (None if no layers hosted)."""
        if num_current_layers <= 0:
            return None
        return floor(self.kv_cache_budget_bytes() / num_current_layers)
