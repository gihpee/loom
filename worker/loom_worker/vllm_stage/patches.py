# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/monkey_patch.py, monkey_patch_utils/weight_loader.py
# и src/parallax/vllm/model_runner.py (ParallaxVLLMGroupCoordinator, подмена
# get_pp_indices на время load_model).
# Изменения: собрано в один модуль с явным контекстным менеджером вместо
# глобальных флагов, добавлено восстановление исходных функций (стадия может
# быть перезагружена в том же процессе), и патчи не падают, если vLLM другой
# версии — вместо этого поднимается понятная ошибка на старте, а не молча
# неверная модель.
"""Three patches that make vLLM serve a SLICE of a model.

vLLM knows how to split a model across GPUs (pipeline parallelism), but it
assumes one process owns the whole cluster: ranks are handed out by NCCL, and
stage boundaries follow from the rank. Loom's stages are independent workers
that never meet — they are matched by the orchestrator and talk over a tunnel.
So we keep vLLM's *model* machinery and replace its idea of "who am I in the
pipeline":

1. `get_pp_indices` -> this stage's [start, end) layer range, so the model
   builds only those layers instead of deriving them from a rank;
2. the weight loader stops treating a missing `embed_tokens` / `lm_head` as an
   error, because only the first / last stage owns them;
3. the pipeline-parallel group reports is_first_rank / is_last_rank from the
   layer range, which is what makes vLLM's forward accept `intermediate_tensors`
   as input and return them as output instead of doing NCCL send/recv.

After (3) the model is a pure function of (tokens | hidden states) -> (hidden
states | logits), and Loom moves those tensors itself.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator

logger = logging.getLogger("loom_worker.vllm_stage.patches")


class VllmIntegrationError(RuntimeError):
    """vLLM is missing or its internals moved. Always raised with what to check."""


@contextlib.contextmanager
def layer_range_for_model_build(start_layer: int, end_layer: int) -> Iterator[None]:
    """Make vLLM build exactly `[start_layer, end_layer)` while inside the block.

    Scoped on purpose: `get_pp_indices` is a module-level function used by every
    model, and leaving it patched would silently affect anything loaded later in
    the same process.
    """
    try:
        import vllm.distributed.utils as vllm_utils
    except ImportError as exc:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vLLM is not importable in this image; the vllm_shard backend needs "
            "the worker-vllm image (docker/worker.vllm.Dockerfile)"
        ) from exc

    original = getattr(vllm_utils, "get_pp_indices", None)
    if original is None:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vllm.distributed.utils.get_pp_indices is gone; the layer-slicing "
            "patch must be re-checked against this vLLM version"
        )

    def sliced_pp_indices(num_layers: int, rank: int, world_size: int):
        logger.debug(
            "layer slice requested (num_layers=%d) -> [%d, %d)",
            num_layers,
            start_layer,
            end_layer,
        )
        return start_layer, end_layer

    vllm_utils.get_pp_indices = sliced_pp_indices
    # The model code may have imported the symbol directly.
    patched_modules = []
    for module_name in ("vllm.model_executor.models.utils",):
        try:
            module = __import__(module_name, fromlist=["get_pp_indices"])
        except ImportError:
            continue
        if hasattr(module, "get_pp_indices"):
            patched_modules.append((module, module.get_pp_indices))
            module.get_pp_indices = sliced_pp_indices
    try:
        yield
    finally:
        vllm_utils.get_pp_indices = original
        for module, previous in patched_modules:
            module.get_pp_indices = previous


def allow_missing_stage_weights(*, is_first_stage: bool, is_last_stage: bool) -> None:
    """Stop the loader from failing over weights this stage does not own.

    A middle stage has no embedding matrix and no LM head; vLLM's default loader
    checks that every parameter was initialised from the checkpoint and raises
    otherwise. Only those two names are tolerated, and only where the stage's
    role says they belong to somebody else — anything else missing is still a
    real error.
    """
    try:
        from vllm.model_executor.model_loader import default_loader
    except ImportError as exc:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vllm.model_executor.model_loader.default_loader is not importable; "
            "the missing-weight patch must be re-checked for this vLLM version"
        ) from exc

    loader_cls = default_loader.DefaultModelLoader
    if getattr(loader_cls, "_loom_patched", False):
        loader_cls._loom_stage_roles = (is_first_stage, is_last_stage)
        return

    original_load_weights = loader_cls.load_weights

    def load_weights(self, model, model_config):
        first, last = loader_cls._loom_stage_roles
        try:
            return original_load_weights(self, model, model_config)
        except ValueError as exc:
            message = str(exc)
            if "not initialized from checkpoint" not in message:
                raise
            if "embed_tokens" in message and not first:
                logger.info("no embed_tokens on this stage — expected, it is not the head")
                return None
            if "lm_head" in message and not last:
                logger.info("no lm_head on this stage — expected, it is not the tail")
                return None
            raise

    loader_cls.load_weights = load_weights
    loader_cls._loom_patched = True
    loader_cls._loom_stage_roles = (is_first_stage, is_last_stage)
    logger.info(
        "weight loader patched (first=%s last=%s)", is_first_stage, is_last_stage
    )


def install_stage_pipeline_group(*, start_layer: int, end_layer: int, num_layers: int) -> None:
    """Teach vLLM's PP group that our position comes from the layer range.

    This is the patch that removes NCCL from the picture. vLLM's forward asks
    the pipeline group `is_first_rank` / `is_last_rank` to decide whether to
    embed tokens or accept `intermediate_tensors`, and whether to compute logits
    or hand hidden states onward. With a single-process group both would be
    True; deriving them from the layer range instead makes a middle stage behave
    like a middle stage — consuming and producing hidden states.
    """
    try:
        import vllm.distributed.parallel_state as parallel_state
        from vllm.distributed.parallel_state import GroupCoordinator
    except ImportError as exc:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vllm.distributed.parallel_state is not importable; the pipeline "
            "group patch must be re-checked for this vLLM version"
        ) from exc

    group = parallel_state._PP
    if group is None:  # pragma: no cover - requires vLLM
        raise VllmIntegrationError(
            "vLLM's pipeline group is not initialised yet; call this after "
            "initialize_model_parallel()"
        )

    class StagePipelineGroup(GroupCoordinator):
        """A one-process PP group whose position is the layer range it serves."""

        @property
        def is_first_rank(self) -> bool:
            return start_layer == 0

        @property
        def is_last_rank(self) -> bool:
            return end_layer >= num_layers

    import torch

    parallel_state._PP = StagePipelineGroup(
        group_ranks=[group.ranks],
        local_rank=group.local_rank,
        torch_distributed_backend=torch.distributed.get_backend(group.device_group),
        use_device_communicator=group.use_device_communicator,
        use_message_queue_broadcaster=getattr(group, "mq_broadcaster", None) is not None,
        group_name="pp",
    )
    logger.info(
        "pipeline group installed: layers [%d, %d) of %d -> first=%s last=%s",
        start_layer,
        end_layer,
        num_layers,
        start_layer == 0,
        end_layer >= num_layers,
    )
