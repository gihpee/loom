"""Bring up an MLX model holding only this stage's layers.

Apple Silicon has no CUDA, so the transformers and vLLM engines are both out
on a Mac — but MLX runs on the GPU through Metal, and an mlx_lm model turns out
to be almost trivially sliceable:

    h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(self.layers, cache):
        h = layer(h, mask, c)
    return self.norm(h)

`self.layers` is a plain Python list, and `input_embeddings=` is already in the
signature — which is exactly what a middle stage needs. Compare with the CUDA
vLLM path, where the same capability took three patches into engine internals
(a scoped layer-range override, a tolerant weight loader and a retagged
pipeline group). Here nothing is patched at all.

Loading only this stage's weights needs two flags:

    load_model(path, lazy=True, strict=False)

`strict=False` tolerates the absent weights of other stages, `lazy=True` never
materialises them. Point it at the pruned checkpoint view Loom already builds
(shard/loader.py) and the node downloads only the safetensors its own layers
live in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("loom_worker.mlx_stage.runtime")

GIB = 1024**3


class MlxUnavailable(RuntimeError):
    """No usable MLX on this host. Only Apple Silicon has one."""


@dataclass
class MlxStageConfig:
    """Everything the stage needs to stand up its slice of the model."""

    model_path: str
    start_layer: int
    end_layer: int
    num_layers: int
    # The broker's grant. MLX uses unified memory, so this is a share of the
    # machine's RAM rather than of a separate card — and going over it does not
    # fail an allocation, it swaps, which is worse. The soft limit below turns
    # that into a refusal instead.
    memory_limit_bytes: int = 0

    @property
    def is_first(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last(self) -> bool:
        return self.end_layer >= self.num_layers


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except Exception:
        return False
    return True


def build_stage_model(config: MlxStageConfig):
    """Load the model and keep only `[start_layer, end_layer)` of it.

    Returns (model, model_config). The model object is the whole mlx_lm module
    tree — embeddings, norm and head included — because the first and last
    stages need those parts. What the middle of it holds is the sliced layer
    list, so the weights of everybody else's layers are never materialised.
    """
    if not mlx_available():
        raise MlxUnavailable(
            "mlx and mlx_lm are not importable; the mlx_shard backend runs "
            "natively on Apple Silicon (see docs/MLX_STAGE.md)"
        )
    import mlx.core as mx
    from mlx_lm.utils import load_model

    if config.memory_limit_bytes > 0:
        # A soft cap the runtime enforces itself. Without it MLX will happily
        # allocate past physical memory and let the OS swap, which on a
        # pipeline stage looks like the whole cluster stalling rather than one
        # node failing.
        mx.set_memory_limit(int(config.memory_limit_bytes))
        logger.info("MLX memory limit set to %.1f GB", config.memory_limit_bytes / GIB)

    model, model_config = load_model(
        _as_path(config.model_path),
        lazy=True,       # other stages' weights are never touched
        strict=False,    # ...and their absence is not an error
    )
    inner = _inner_model(model)
    total = len(inner.layers)
    if total < config.num_layers:
        logger.warning(
            "the checkpoint declares %d layers but the model built %d; "
            "using the built count",
            config.num_layers,
            total,
        )
    start, end = config.start_layer, min(config.end_layer, total)
    inner.layers = inner.layers[start:end]

    # Force this stage's weights into memory now. Lazy loading would otherwise
    # spread the cost across the first request, where it looks like the model
    # is inexplicably slow rather than still loading.
    mx.eval(inner.layers)
    logger.info(
        "MLX stage loaded: layers [%d, %d) of %d (first=%s last=%s)",
        start,
        end,
        total,
        config.is_first,
        config.is_last,
    )
    return model, model_config


def _inner_model(model):
    """The module that owns `layers`, `embed_tokens` and `norm`.

    mlx_lm wraps the transformer in an outer Model that adds the LM head, so
    the layer list lives one level down. A couple of architectures put it
    elsewhere; failing loudly here beats slicing the wrong list.
    """
    for attr in ("model", "transformer", "language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    if hasattr(model, "layers"):
        return model
    raise MlxUnavailable(
        f"cannot find the decoder layers of {type(model).__name__}; this "
        f"architecture is not supported by the mlx_shard backend yet"
    )


def _as_path(model_path: str):
    from pathlib import Path

    return Path(model_path)


def stage_config_from_env(
    *,
    model_path: str,
    start_layer: int,
    end_layer: int,
    num_layers: int,
    memory_limit_bytes: Optional[int] = None,
) -> MlxStageConfig:
    """Turn the LoadShard command plus env overrides into a runtime config."""
    override = os.environ.get("LOOM_MLX_MEMORY_LIMIT_GB", "").strip()
    limit = int(float(override) * GIB) if override else int(memory_limit_bytes or 0)
    return MlxStageConfig(
        model_path=model_path,
        start_layer=start_layer,
        end_layer=end_layer,
        num_layers=num_layers,
        memory_limit_bytes=limit,
    )
