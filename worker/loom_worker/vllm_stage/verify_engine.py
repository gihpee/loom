"""Check that this vLLM build still exposes what the stage engine patches.

Run at image build time — and on a live worker when something looks wrong:

    python3 -m loom_worker.vllm_stage.verify_engine

Prints what it found before deciding. That matters because vLLM's internals
move between releases: when a check fails, the report says which names are
present, so the fix in `patches.py` / `runtime.py` is a five-minute edit rather
than an archaeology session.

The distinction it draws: a missing OPTIONAL field is dropped at runtime with a
warning (`_construct`), while a missing CRITICAL one is fatal here. Critical
means "losing it changes behaviour silently" — the broker's VRAM grant and the
context length size the KV cache, and quietly falling back to vLLM's own
defaults would surface later as an OOM nobody can trace.
"""

from __future__ import annotations

import importlib
import sys

# (module path, attribute) pairs the patches in patches.py reach for.
REQUIRED_SYMBOLS = [
    ("vllm.distributed.utils", "get_pp_indices"),
    ("vllm.distributed.parallel_state", "GroupCoordinator"),
    ("vllm.model_executor.model_loader.default_loader", "DefaultModelLoader"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ("vllm.v1.core.kv_cache_manager", "KVCacheManager"),
    ("vllm.v1.core.sched.output", "SchedulerOutput"),
    ("vllm.v1.core.sched.output", "NewRequestData"),
    ("vllm.v1.core.sched.output", "CachedRequestData"),
    ("vllm.v1.request", "Request"),
    ("vllm.sequence", "IntermediateTensors"),
]

# config class -> (fields we pass, fields whose loss is NOT survivable)
CONFIG_FIELDS = {
    "ModelConfig": (
        {"model", "tokenizer", "tokenizer_mode", "trust_remote_code", "dtype",
         "seed", "max_model_len", "max_logprobs"},
        {"model", "max_model_len"},
    ),
    "CacheConfig": (
        {"block_size", "gpu_memory_utilization", "swap_space", "cache_dtype"},
        # The broker's grant lives here; without it one model takes the card.
        {"gpu_memory_utilization"},
    ),
    "ParallelConfig": (
        {"pipeline_parallel_size", "tensor_parallel_size", "distributed_executor_backend"},
        set(),
    ),
    "SchedulerConfig": (
        {"max_num_batched_tokens", "max_num_seqs", "max_model_len",
         "is_encoder_decoder", "enable_chunked_prefill"},
        set(),
    ),
}


def _fields(cls) -> set:
    """What the constructor takes — see accepted_arguments on why not fields()."""
    from loom_worker.vllm_stage.runtime import accepted_arguments

    accepted = accepted_arguments(cls)
    return accepted if accepted is not None else set()


def main() -> int:
    import vllm

    print(f"vLLM in image: {vllm.__version__}")
    problems = []

    for module_path, name in REQUIRED_SYMBOLS:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            problems.append(f"{module_path} is not importable ({exc})")
            continue
        if not hasattr(module, name):
            problems.append(f"{module_path}.{name} is gone")

    from vllm import config as vllm_config

    for class_name, (wanted, critical) in CONFIG_FIELDS.items():
        cls = getattr(vllm_config, class_name, None)
        if cls is None:
            problems.append(f"vllm.config.{class_name} is gone")
            continue
        present = _fields(cls)
        missing = sorted(wanted - present)
        lost_critical = sorted(critical - present)
        status = "ok" if not missing else f"without {', '.join(missing)}"
        print(f"  {class_name}: {status}")
        if lost_critical:
            problems.append(
                f"vllm.config.{class_name} no longer takes {lost_critical} — "
                f"loom_worker/vllm_stage/runtime.py must be updated. "
                f"Fields it does take: {sorted(present)}"
            )

    if problems:
        print("\nvLLM engine check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nSee docs/VLLM_PIPELINE.md §3 for what each patched name is for.",
            file=sys.stderr,
        )
        return 1
    print("stage engine: all patched internals present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
