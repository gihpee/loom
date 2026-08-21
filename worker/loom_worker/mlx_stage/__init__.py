"""Per-layer inference on Apple Silicon (MLX/Metal).

A Mac takes a range of a model's layers and serves it as one stage of a Loom
pipeline, alongside CUDA nodes. See docs/MLX_STAGE.md.
"""

from loom_worker.mlx_stage.executor import MlxStageExecutor
from loom_worker.mlx_stage.runtime import (
    MlxStageConfig,
    MlxUnavailable,
    build_stage_model,
    mlx_available,
    stage_config_from_env,
)

__all__ = [
    "MlxStageConfig",
    "MlxStageExecutor",
    "MlxUnavailable",
    "build_stage_model",
    "mlx_available",
    "stage_config_from_env",
]
