# Loom worker with the vLLM pipeline engine — published as gihpee/loomworker-vllm.
#
# Same agent as the standard image, same single command for the GPU owner:
#
#   docker run -d --gpus all --restart unless-stopped \
#     -v loom-hf:/root/.cache/huggingface \
#     gihpee/loomworker-vllm --key loom_<...>
#
# What differs is the engine available to a pipeline stage. The standard image
# runs stages on transformers, which is portable but eager; this one adds the
# `vllm_shard` backend, where a stage's layers run on vLLM's GPU model runner
# (paged KV cache, CUDA graphs, fused kernels). See docs/VLLM_PIPELINE.md.
#
# Why a separate image rather than a flag: the vLLM engine reaches into engine
# internals that move between releases, so this image pins the vLLM it was
# tested against, while the standard image stays free to track the base.
FROM vllm/vllm-openai:latest

WORKDIR /app
COPY pyproject.toml ./
COPY loom_worker ./loom_worker

# Quoted on purpose: unquoted, the shell reads ">=12.0" as a redirect and
# silently installs whatever nvidia-ml-py version it likes.
RUN pip install --no-cache-dir . "nvidia-ml-py>=12.0"

# Record what the stage engine was built against. The patches in
# loom_worker/vllm_stage/ target these internals; a mismatch surfaces at
# startup with a message naming what to re-check, not as wrong numbers.
# python3, not python: the base image ships no `python` alias.
RUN python3 -c "\
import vllm, vllm.distributed.utils as u;\
from vllm.v1.worker.gpu_model_runner import GPUModelRunner;\
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData, CachedRequestData;\
from vllm.v1.core.kv_cache_manager import KVCacheManager;\
from vllm.model_executor.model_loader import default_loader;\
from vllm.distributed.parallel_state import GroupCoordinator;\
assert hasattr(u, 'get_pp_indices'), 'get_pp_indices moved';\
print('vLLM in image:', vllm.__version__)"

ENV LOOM_LOG_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface \
    LOOM_STAGE_ENGINE=vllm

# No EXPOSE: the worker only dials out, and inference is tunnelled back over
# that same outbound connection.
ENTRYPOINT ["loom-worker"]
