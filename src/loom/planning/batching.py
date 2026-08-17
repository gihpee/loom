# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax_utils/utils.py — compute_max_tokens_in_cache,
# derive_max_batch_size, compute_max_batch_size.
# Изменения: убрана зависимость от torch/mlx и от локального опроса памяти
# устройства — available_cache_bytes всегда передаётся явно (в Loom он
# выводится из vram-квоты, выданной Resource Broker, а не из живого
# mem_get_info на узле); elem_bytes обязателен. Сама арифметика
# (per-token cache size, клампинг батча) сохранена без изменений.
"""Torch-free KV-cache batch-size derivation used by scheduling."""

from typing import Optional

from loom.logging_config import get_logger

logger = get_logger(__name__)


def compute_max_tokens_in_cache(
    *,
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    elem_bytes: int,
    available_cache_bytes: int,
) -> int:
    """Estimate max tokens storable in KV cache given the byte budget."""
    del kv_cache_memory_fraction  # already applied by the caller to the quota
    available_cache_size = int(available_cache_bytes)
    per_token_cache_size = (
        num_shard_layers * num_key_value_heads * (head_dim_k + head_dim_v) * elem_bytes
    )
    return max(0, available_cache_size // per_token_cache_size)


def derive_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    max_tokens_in_cache: Optional[int],
) -> int:
    """Derive final max_batch_size clamped by KV capacity if sequence length known."""
    max_batch_capacity: Optional[int] = None
    if max_sequence_len and max_tokens_in_cache:
        max_batch_capacity = max(1, max_tokens_in_cache // int(max_sequence_len))
    if requested_max_batch_size is None:
        if max_batch_capacity is None:
            logger.warning("Overriding max_batch_size to 16 due to no max_sequence_len provided")
            return 16
        return max_batch_capacity
    if max_batch_capacity is not None:
        return min(requested_max_batch_size, max_batch_capacity)
    return requested_max_batch_size


def compute_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    elem_bytes: int,
    kv_cache_budget_bytes: int,
    head_dim_k: Optional[int] = None,
    head_dim_v: Optional[int] = None,
) -> int:
    """Compute final max_batch_size by chaining KV capacity and clamping.

    ``kv_cache_budget_bytes`` is the explicit KV byte budget granted by the
    Resource Broker (quota * kvcache_mem_ratio).
    """
    max_tokens = compute_max_tokens_in_cache(
        kv_cache_memory_fraction=kv_cache_memory_fraction,
        num_shard_layers=num_shard_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim_k=head_dim_k if head_dim_k is not None else head_dim,
        head_dim_v=head_dim_v if head_dim_v is not None else head_dim,
        elem_bytes=elem_bytes,
        available_cache_bytes=kv_cache_budget_bytes,
    )
    return derive_max_batch_size(
        requested_max_batch_size=requested_max_batch_size,
        max_sequence_len=max_sequence_len,
        max_tokens_in_cache=max_tokens,
    )
