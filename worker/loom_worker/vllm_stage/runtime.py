# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/model_runner.py — сборка VllmConfig под одну
# стадию, ручной KVCacheConfig на слои стадии и KVCacheManager, вызов
# load_model под патчем диапазона слоёв.
# Изменения: конфиг собирается из брокерской квоты Loom (а не из аргументов
# CLI), убраны LoRA/MoE-ветки и TP (в Loom одна карта на стадию — параллелизм
# идёт по узлам), а несовместимость с версией vLLM превращается в явную
# ошибку с указанием, что проверять.
"""Bring up a vLLM model runner that owns one pipeline stage."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from loom_worker.vllm_stage.patches import (
    VllmIntegrationError,
    allow_missing_stage_weights,
    install_stage_pipeline_group,
    layer_range_for_model_build,
)

logger = logging.getLogger("loom_worker.vllm_stage.runtime")

GIB = 1024**3


@dataclass
class StageRuntimeConfig:
    """Everything the stage needs to stand up its slice of the model."""

    model_path: str
    start_layer: int
    end_layer: int
    num_layers: int
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    # Share of the card vLLM may use for weights + KV. The broker already
    # decided how much of this GPU belongs to the model; this is that decision
    # expressed the way vLLM wants it.
    gpu_memory_utilization: float = 0.85
    kv_block_size: int = 16
    max_num_seqs: int = 16
    max_batched_tokens: int = 8192
    enforce_eager: bool = False

    @property
    def is_first(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last(self) -> bool:
        return self.end_layer >= self.num_layers


def build_stage_runner(config: StageRuntimeConfig):
    """Create a GPUModelRunner holding only `[start_layer, end_layer)`.

    The order below is not arbitrary: the distributed environment must exist
    before the pipeline group can be replaced, the group must report this
    stage's role before the model is built (the model asks it what to
    construct), and only then can weights be loaded under the layer-range
    patch.
    """
    try:
        import torch
        import vllm.distributed.parallel_state as parallel_state
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            LoadConfig,
            ModelConfig,
            ParallelConfig,
            SchedulerConfig,
            VllmConfig,
        )
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        raise VllmIntegrationError(
            "vLLM is not importable; the vllm_shard backend requires the "
            "worker-vllm image (see docs/VLLM_PIPELINE.md)"
        ) from exc

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_config = ModelConfig(
        model=config.model_path,
        tokenizer=config.model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=config.dtype,
        seed=0,
        max_model_len=config.max_model_len,
        max_logprobs=1,
    )
    cache_config = CacheConfig(
        block_size=config.kv_block_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        swap_space=0,
        cache_dtype="auto",
    )
    # One card per stage: the parallelism Loom cares about lives between nodes,
    # and vLLM is told it is alone so it never tries to talk to peers itself.
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max(config.max_batched_tokens, config.max_model_len),
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device=device),
        load_config=LoadConfig(load_format="auto"),
    )

    if not parallel_state.model_parallel_is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("LOOM_VLLM_NCCL_PORT", "29591"))
        parallel_state.init_distributed_environment()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    install_stage_pipeline_group(
        start_layer=config.start_layer,
        end_layer=config.end_layer,
        num_layers=config.num_layers,
    )
    allow_missing_stage_weights(is_first_stage=config.is_first, is_last_stage=config.is_last)

    runner = GPUModelRunner(vllm_config=vllm_config, device=device)
    with layer_range_for_model_build(config.start_layer, config.end_layer):
        runner.load_model()
    logger.info(
        "vLLM stage loaded: layers [%d, %d) of %d (first=%s last=%s, dtype=%s)",
        config.start_layer,
        config.end_layer,
        config.num_layers,
        config.is_first,
        config.is_last,
        config.dtype,
    )
    return runner, vllm_config


def build_kv_cache(runner, vllm_config, config: StageRuntimeConfig):
    """Size and allocate the paged KV cache for this stage's layers only.

    vLLM normally derives this from the whole model; a stage holds a slice, so
    the layer names — and therefore the number of blocks that fit — are ours to
    compute. This is where paged attention comes from: instead of one
    contiguous cache per request, blocks are pooled and handed out on demand.
    """
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
    )

    kv_cache_specs = runner.get_kv_cache_spec()
    free_memory, _total = torch.cuda.mem_get_info(config_device_index())
    available = int(free_memory * config.gpu_memory_utilization)
    logger.info(
        "KV cache budget: %.2f GB of %.2f GB free",
        available / GIB,
        free_memory / GIB,
    )
    kv_cache_configs = get_kv_cache_configs(
        vllm_config=vllm_config,
        kv_cache_specs=[kv_cache_specs],
        available_memory=[available],
    )
    kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    runner.initialize_kv_cache(kv_cache_config)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=config.max_model_len,
        enable_caching=False,
        use_eagle=False,
        log_stats=False,
        enable_kv_cache_events=False,
    )
    logger.info(
        "KV cache ready: %d blocks x %d tokens = %d tokens of context in flight",
        kv_cache_config.num_blocks,
        config.kv_block_size,
        kv_cache_config.num_blocks * config.kv_block_size,
    )
    return manager, kv_cache_config


def config_device_index() -> int:
    import torch

    try:
        return torch.cuda.current_device()
    except Exception:  # pragma: no cover - no CUDA in tests
        return 0


def stage_config_from_env(
    *,
    model_path: str,
    start_layer: int,
    end_layer: int,
    num_layers: int,
    dtype: str,
    vram_quota_bytes: Optional[int] = None,
) -> StageRuntimeConfig:
    """Turn the LoadShard command plus env overrides into a runtime config."""
    utilisation = float(os.environ.get("LOOM_VLLM_GPU_UTILISATION", "0") or 0)
    if utilisation <= 0 and vram_quota_bytes:
        import torch

        try:
            _free, total = torch.cuda.mem_get_info(config_device_index())
            utilisation = min(0.92, vram_quota_bytes / total)
        except Exception:  # pragma: no cover - no CUDA in tests
            utilisation = 0.85
    return StageRuntimeConfig(
        model_path=model_path,
        start_layer=start_layer,
        end_layer=end_layer,
        num_layers=num_layers,
        dtype=dtype,
        max_model_len=int(os.environ.get("LOOM_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=utilisation or 0.85,
        kv_block_size=int(os.environ.get("LOOM_KV_BLOCK_SIZE", "16")),
        max_num_seqs=int(os.environ.get("LOOM_MAX_REQUESTS", "16")),
        enforce_eager=os.environ.get("LOOM_VLLM_EAGER", "").strip().lower()
        in ("1", "true", "yes"),
    )
