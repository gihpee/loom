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

import functools
import logging
import math
import os
import time
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
    max_batched_tokens: int = 4096
    # Eager by default. vLLM's torch.compile pass allocated 1.45 GB for
    # autotuning on the first request and OOMed a card that had already given
    # its memory to weights and KV; it also stalls that request for minutes.
    # Paged attention, FlashAttention and the fused kernels — the reasons to be
    # on vLLM at all — do not depend on it.
    enforce_eager: bool = True
    # Kept aside for activations, workspaces and allocator fragmentation.
    headroom_gb: float = 2.0
    # CUDA graphs WITHOUT torch.compile. `enforce_eager` turns off both, and
    # the two have very different costs: it was Inductor's autotuning that
    # allocated 1.45 GB and OOMed the card, while graph capture only replays
    # kernel launches. On a one-sequence decode those launches are most of the
    # per-layer overhead, so this is the main lever we have on tokens/s.
    # Opt-in until it has been measured on the stand.
    cuda_graphs: bool = False

    @property
    def is_first(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last(self) -> bool:
        return self.end_layer >= self.num_layers


def accepted_arguments(cls):
    """Keyword arguments this config class actually accepts, or None for "any".

    The constructor signature is the only honest source. `dataclasses.fields()`
    looks like the obvious one and is wrong: it omits InitVar pseudo-fields,
    which are init-only parameters. vLLM 0.27 declares SchedulerConfig's
    `max_model_len` and `is_encoder_decoder` exactly that way — required by
    __init__, invisible to fields() — so trusting fields() meant dropping two
    mandatory arguments and failing with "Field required".

    None means the constructor takes **kwargs and nothing should be dropped.
    """
    return _accepted_arguments_cached(cls)


@functools.lru_cache(maxsize=None)
def _accepted_arguments_cached(cls):
    """Cached because _construct runs on every decode step.

    Two config objects are rebuilt per token per stage, and inspect.signature
    is not cheap. A class's signature cannot change at runtime, so this is
    computed once.
    """
    import inspect

    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return {
        name
        for name, p in parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } - {"self"}


def _construct(cls, *, required=(), **kwargs):
    """Build a vLLM config, dropping keys this release no longer knows.

    vLLM's config dataclasses gain and lose fields between releases (0.27
    dropped `swap_space` from CacheConfig, for example), and a stray keyword
    aborts the whole start — after the checkpoint has already been downloaded.
    Optional keys are therefore dropped with a warning.

    `required` names the keys whose loss would change behaviour rather than
    just cosmetics — the VRAM quota, the context length. Those raise instead:
    silently serving with vLLM's own defaults would mean ignoring the broker's
    grant, and that shows up much later as an OOM nobody can explain.
    """
    accepted = accepted_arguments(cls)
    if accepted is None:
        return cls(**kwargs)  # takes **kwargs: nothing to filter
    missing_required = [k for k in required if k not in accepted]
    if missing_required:
        raise VllmIntegrationError(
            f"{cls.__name__} in this vLLM release does not accept "
            f"{missing_required}; loom_worker/vllm_stage/runtime.py must be "
            f"updated for it (see docs/VLLM_PIPELINE.md)"
        )
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        logger.warning(
            "%s: this vLLM release does not take %s — continuing without them",
            cls.__name__,
            ", ".join(dropped),
        )
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


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

    try:
        from vllm.config import set_current_vllm_config
    except ImportError as exc:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vllm.config.set_current_vllm_config is missing; the stage cannot "
            "publish its config to vLLM's globals"
        ) from exc

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    compilation_config = _stage_compilation_config(config)
    model_config = _construct(
        ModelConfig,
        required=("model", "max_model_len"),
        model=config.model_path,
        tokenizer=config.model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=config.dtype,
        seed=0,
        max_model_len=config.max_model_len,
        max_logprobs=1,
        # enforce_eager disables torch.compile AND cudagraphs together; with
        # graphs asked for, it must be off and the compilation config carries
        # the finer distinction.
        enforce_eager=config.enforce_eager and not config.cuda_graphs,
    )
    cache_config = _construct(
        CacheConfig,
        # The broker's grant lives in this one field; losing it would hand the
        # whole card to one model.
        required=("gpu_memory_utilization",),
        block_size=config.kv_block_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        swap_space=0,
        cache_dtype="auto",
        # Off, and it has to be: prefix caching keys blocks by their token ids,
        # and a middle stage never learns any — it receives activations, not
        # tokens, and fills the field with zeros. Two different requests of the
        # same length would hash alike and share each other's KV. The cache
        # manager is built with enable_caching=False for the same reason; these
        # two must agree.
        enable_prefix_caching=False,
    )
    # One card per stage: the parallelism Loom cares about lives between nodes,
    # and vLLM is told it is alone so it never tries to talk to peers itself.
    parallel_config = _construct(
        ParallelConfig,
        # No required fields: both default to 1, which is what a stage wants.
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        distributed_executor_backend=None,
    )
    scheduler_config = _construct(
        SchedulerConfig,
        # Scheduling limits are comfort, not correctness: if a field moved,
        # vLLM's own default is fine and the drop is logged.
        max_num_batched_tokens=max(config.max_batched_tokens, config.max_model_len),
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=False,
    )
    vllm_config = _construct(
        VllmConfig,
        required=("model_config", "cache_config"),
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=_construct(DeviceConfig, device=device),
        load_config=_construct(LoadConfig, load_format="auto"),
        **({"compilation_config": compilation_config} if compilation_config else {}),
    )

    # vLLM keeps "the config in force" in a global, and reads it in places that
    # look unrelated: initialize_model_parallel() asks for it, and so do custom
    # ops during forward. The stage process serves exactly one model, so the
    # context is entered once and never left — closing it would break inference
    # later with "Current vLLM config is not set".
    config_scope = set_current_vllm_config(vllm_config)
    config_scope.__enter__()

    if not parallel_state.model_parallel_is_initialized():
        # A stage is a world of one. The rendezvous is local and exists only
        # because vLLM's model code expects a torch.distributed group to be
        # there; no traffic ever crosses it.
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("LOOM_VLLM_NCCL_PORT", "29591"))
        # Explicit values instead of the env:// guessing vLLM logs as
        # "world_size=-1 rank=-1"; the backend stays vLLM's own default.
        parallel_state.init_distributed_environment(world_size=1, rank=0, local_rank=0)
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
    runner._loom_config_scope = config_scope  # keep it alive for the process
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


def _stage_compilation_config(config: StageRuntimeConfig):
    """CUDA graphs with the compiler left out, or None to keep vLLM's default.

    FULL capture records the whole stage as one graph, so it needs no piecewise
    splitting and therefore no Inductor pass — which is the expensive, memory
    hungry part we turned off after it OOMed a loaded card.
    """
    if not config.cuda_graphs:
        return None
    try:
        from vllm.config import CompilationConfig
        from vllm.config.compilation import CompilationMode, CUDAGraphMode
    except ImportError:
        logger.warning(
            "this vLLM release exposes no CompilationConfig; staying eager"
        )
        return None
    sizes = sorted({1, 2, 4, 8, max(1, config.max_num_seqs)})
    return _construct(
        CompilationConfig,
        mode=CompilationMode.NONE,          # no torch.compile, no Inductor
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=sizes,
        max_cudagraph_capture_size=max(sizes),
    )


def capture_cuda_graphs(runner, config: StageRuntimeConfig) -> None:
    """Record the decode graph, if this stage asked for one.

    Never fatal: a stage that cannot capture is slower, not broken, and losing
    a loaded model over a graph would be a bad trade.
    """
    if not config.cuda_graphs:
        return
    capture = getattr(runner, "capture_model", None)
    if not callable(capture):
        logger.warning("this vLLM release has no capture_model(); staying eager")
        return
    started = time.perf_counter()
    try:
        capture()
    except Exception:
        logger.exception(
            "CUDA graph capture failed; the stage keeps serving eagerly "
            "(unset LOOM_VLLM_CUDAGRAPH to silence this)"
        )
        return
    logger.info("CUDA graphs captured in %.1f s", time.perf_counter() - started)


def build_kv_cache(runner, vllm_config, config: StageRuntimeConfig):
    """Size and allocate the paged KV cache for this stage's layers only.

    The arithmetic mirrors what vLLM does for a whole model, adapted to a
    stage. It is worth spelling out, because getting it wrong does not fail at
    startup — it fails on the first request, when there is nothing left for
    activations:

        budget = min(quota, total x utilisation)   what this model may use
               - weights                           what it already took
               - headroom                          activations, workspaces
        budget = min(budget, what concurrency actually needs)

    The last line matters on a big card: sizing the pool to "all remaining
    memory" reserved 8.7 GB for 114k tokens of context nobody asked for, and
    left 700 MB for the forward pass itself.
    """
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
    )

    device_index = config_device_index()
    free_memory, total_memory = torch.cuda.mem_get_info(device_index)
    kv_cache_specs = runner.get_kv_cache_spec()

    weights = int(getattr(runner, "model_memory_usage", 0) or 0)
    requested = int(total_memory * config.gpu_memory_utilization)
    headroom = int(config.headroom_gb * GIB)
    budget = requested - weights - headroom

    needed = _kv_bytes_for_concurrency(kv_cache_specs, config)
    capped_by = "need"
    if needed and needed < budget:
        budget = needed
    else:
        capped_by = "budget"
    # Never take the last of what is actually free, whatever the arithmetic says.
    budget = min(budget, int(free_memory - headroom))

    logger.info(
        "KV cache sizing: quota %.1f GB, weights %.1f GB, headroom %.1f GB, "
        "free now %.1f GB -> %.1f GB for KV (limited by %s)",
        requested / GIB,
        weights / GIB,
        headroom / GIB,
        free_memory / GIB,
        budget / GIB,
        capped_by,
    )
    if budget <= 0:
        raise VllmIntegrationError(
            f"nothing left for the KV cache: the quota grants "
            f"{requested / GIB:.1f} GB, the weights of layers "
            f"[{config.start_layer}, {config.end_layer}) already take "
            f"{weights / GIB:.1f} GB and {config.headroom_gb:.1f} GB is kept "
            f"for activations. Give this model a bigger share "
            f"(LOOM_PARAM_MEM_RATIO) or split the model over more nodes"
        )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config=vllm_config,
        kv_cache_specs=[kv_cache_specs],
        available_memory=[budget],
    )
    kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    runner.initialize_kv_cache(kv_cache_config)

    manager = _construct(
        KVCacheManager,
        required=("kv_cache_config", "max_model_len"),
        kv_cache_config=kv_cache_config,
        max_model_len=config.max_model_len,
        # Both are mandatory in 0.27 and have no defaults: the scheduler's view
        # of a block and the block size used for prefix hashing.
        scheduler_block_size=config.kv_block_size,
        hash_block_size=config.kv_block_size,
        enable_caching=False,
        use_eagle=False,
        log_stats=False,
        enable_kv_cache_events=False,
    )
    tokens = kv_cache_config.num_blocks * config.kv_block_size
    logger.info(
        "KV cache ready: %d blocks x %d tokens = %d tokens of context "
        "(%d requests of %d tokens)",
        kv_cache_config.num_blocks,
        config.kv_block_size,
        tokens,
        tokens // max(1, config.max_model_len),
        config.max_model_len,
    )
    return manager, kv_cache_config


def _kv_bytes_for_concurrency(kv_cache_specs, config: StageRuntimeConfig) -> int:
    """How much cache this stage can actually put to use.

    Pool size beyond `max_num_seqs` full-length requests is memory taken away
    from the forward pass to hold context nobody can be serving.
    """
    per_block = 0
    for spec in kv_cache_specs.values():
        try:
            per_block += int(spec.page_size_bytes)
        except (AttributeError, TypeError):
            return 0  # unknown layout: fall back to the memory budget
    if per_block <= 0:
        return 0
    blocks_per_request = math.ceil(config.max_model_len / config.kv_block_size)
    return per_block * blocks_per_request * max(1, config.max_num_seqs)


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
        max_batched_tokens=int(
            os.environ.get("LOOM_MAX_BATCHED_TOKENS", "0")
            or os.environ.get("LOOM_MAX_MODEL_LEN", "4096")
        ),
        # Opt IN to compilation, not out: it is the thing that OOMed the card.
        enforce_eager=os.environ.get("LOOM_VLLM_COMPILE", "").strip().lower()
        not in ("1", "true", "yes"),
        headroom_gb=float(os.environ.get("LOOM_VLLM_HEADROOM_GB", "2.0")),
        cuda_graphs=os.environ.get("LOOM_VLLM_CUDAGRAPH", "").strip().lower()
        in ("1", "true", "yes"),
    )
