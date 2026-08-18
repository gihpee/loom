# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/server/executor/base_executor.py — роли стадий
# (is_first_peer/is_last_peer), цикл «принять hidden_states -> прогнать свои
# слои -> отдать дальше», выборка токена из hidden на последней стадии
# (_gen_token_id_from_hidden), а также per-request состояние KV.
# Изменения: исполнение на torch/transformers (у Parallax — MLX-модель или
# vLLM GPUModelRunner); транспорт активаций не ZMQ/Lattica, а релей через
# оркестратор (см. NOTICE и §data plane в docs); в v0 одна последовательность
# на пайплайн без continuous batching (у Parallax есть батчер).
"""Stage-local forward pass over `[start_layer, end_layer)` with a KV cache."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom_worker.shard.loader import ShardModel

logger = logging.getLogger("loom_worker.shard.executor")


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass
class RequestState:
    """Per-request KV cache and bookkeeping, owned by this stage."""

    request_id: str
    cache: object  # transformers Cache
    created_at: float = field(default_factory=time.time)
    seen_tokens: int = 0
    # Guards this request's own cache. Requests do not share state, so they
    # only need to be serialised against themselves — not against each other.
    lock: threading.RLock = field(default_factory=threading.RLock)


class ShardExecutor:
    """Runs this stage's layers. Thread-safe: one lock around the model."""

    def __init__(
        self,
        shard: ShardModel,
        *,
        max_requests: Optional[int] = None,
        static_cache: Optional[bool] = None,
        compile_model: Optional[bool] = None,
        max_cache_len: Optional[int] = None,
    ) -> None:
        import torch

        self.torch = torch
        self.shard = shard
        self.spec = shard.spec
        on_gpu = str(self.spec.device).startswith("cuda")
        # OFF by default, deliberately. Compiling a 20-layer stage of a 14B
        # model costs minutes on the first request (and again on every new
        # prompt length), while the CUDA graphs that would justify it are
        # skipped anyway because transformers' StaticCache mutates its buffers
        # in place. Enable with LOOM_COMPILE=1 to measure it on your own
        # hardware; the runtime falls back to eager if it misbehaves.
        self.compile_model = (
            _flag("LOOM_COMPILE", False) if compile_model is None else compile_model
        )
        # A preallocated KV cache is what lets graphs capture the decode step,
        # so it follows the compile switch. On its own it buys little and costs
        # real VRAM plus a hard ceiling on context length.
        self.static_cache = (
            _flag("LOOM_STATIC_CACHE", self.compile_model)
            if static_cache is None
            else static_cache
        )
        self.max_cache_len = int(
            max_cache_len
            if max_cache_len is not None
            else os.environ.get("LOOM_MAX_CACHE_LEN", "4096")
        )
        self.max_requests = int(
            max_requests
            if max_requests is not None
            else os.environ.get("LOOM_MAX_REQUESTS", "8" if self.static_cache else "64")
        )
        self._wire_dtype = self._pick_wire_dtype()
        self._log_capacity()
        if self.compile_model:
            self._compile()
        self._states: Dict[str, RequestState] = {}
        self._lock = threading.RLock()
        # transformers renamed the cache kwarg (`past_key_value` -> `past_key_values`)
        # and layer forwards swallow unknown kwargs via **kwargs, so passing the
        # wrong name silently disables the KV cache: prefill stays correct while
        # decode quietly reads an empty cache. Resolve the real name once.
        self._cache_kwarg = self._detect_cache_kwarg()

    def _detect_cache_kwarg(self) -> str:
        import inspect

        try:
            params = inspect.signature(type(self.shard.layers[0]).forward).parameters
        except (TypeError, ValueError, IndexError):
            return "past_key_values"
        for name in ("past_key_values", "past_key_value"):
            if name in params:
                return name
        raise RuntimeError(
            "decoder layer accepts neither past_key_values nor past_key_value; "
            "unsupported transformers version"
        )

    # --------------------------------------------------------------- runtime
    def _kv_bytes_per_token(self) -> int:
        cfg = self.shard.shard_config or self.shard.config
        heads = int(getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads"))
        head_dim = int(
            getattr(cfg, "head_dim", None)
            or int(getattr(cfg, "hidden_size")) // int(getattr(cfg, "num_attention_heads"))
        )
        layers = len(self.shard.layers)
        elem = 2 if self.shard.dtype in (self.torch.bfloat16, self.torch.float16) else 4
        return layers * heads * head_dim * 2 * elem  # K and V

    def _log_capacity(self) -> None:
        if not self.static_cache:
            logger.info(
                "executor: dynamic KV cache, up to %d concurrent requests", self.max_requests
            )
            return
        per_request = self._kv_bytes_per_token() * self.max_cache_len
        logger.info(
            "executor: static KV cache %d tokens = %.2f GB per request, "
            "%d concurrent max (%.2f GB total), compile=%s",
            self.max_cache_len,
            per_request / 1024**3,
            self.max_requests,
            per_request * self.max_requests / 1024**3,
            self.compile_model,
        )

    def _compile(self) -> None:
        """Install a torch.compile wrapper around the stage model.

        Inductor without CUDA graphs is the default on purpose. `reduce-overhead`
        captures graphs, and graph capture is incompatible with the way
        transformers' StaticCache mutates its buffers in place: inductor reports
        "skipping cudagraphs due to mutated inputs" and then generation dies with
        "accessing tensor output of CUDAGraphs that has been overwritten" after
        the first token. Anyone who wants to try graphs can still set
        LOOM_COMPILE_MODE=reduce-overhead, and the runtime fallback below keeps
        that experiment from taking the stage down.

        Shapes stay dynamic (torch decides): pinning them recompiles on every
        new prompt length, and a recompile in the middle of serving shows up as
        a two-minute TTFT.
        """
        torch = self.torch
        mode = os.environ.get("LOOM_COMPILE_MODE", "default")
        try:
            self.shard.compiled = torch.compile(self.shard.inner, mode=mode)
            logger.info(
                "executor: model compiled (mode=%s; first request pays warm-up)", mode
            )
        except Exception:
            logger.exception("torch.compile failed; continuing without it")
            self._disable_compile()

    def _disable_compile(self) -> None:
        self.compile_model = False
        self.shard.drop_compiled()

    def _run_stage(self, hidden, **kwargs):
        """Run the stage, degrading to eager execution if the compiled path fails.

        A compilation problem must cost speed, not availability: on a GPU box we
        cannot pre-test every model/driver/torch combination, so the first
        failure switches this stage back to eager and the request is retried
        instead of being returned as an error.
        """
        torch = self.torch
        if self.shard.compiled is not None:
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                # Tells CUDA-graph memory management that a new iteration starts,
                # so outputs of the previous one are not reused underneath us.
                torch.compiler.cudagraph_mark_step_begin()
            try:
                return self.shard.run_layers(hidden, **kwargs)
            except Exception:
                logger.exception(
                    "compiled forward failed; falling back to eager for this stage"
                )
                self._disable_compile()
        return self.shard.run_layers(hidden, **kwargs)

    def _new_cache(self):
        from transformers import DynamicCache

        if not self.static_cache:
            return DynamicCache()
        from transformers import StaticCache

        cfg = self.shard.shard_config
        if cfg is None:  # pragma: no cover - only for shards built by old code
            return DynamicCache()
        return StaticCache(
            config=cfg,
            max_cache_len=self.max_cache_len,
            device=self.shard.device,
            dtype=self.shard.dtype,
        )

    # ------------------------------------------------------------- lifecycle
    def _state(self, request_id: str) -> RequestState:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                if len(self._states) >= self.max_requests:
                    # Evict the oldest: a stage must not grow unbounded if a
                    # head disappears without sending FREE.
                    oldest = min(self._states.values(), key=lambda s: s.created_at)
                    self._states.pop(oldest.request_id, None)
                    logger.warning("evicted stale request state %s", oldest.request_id)
                state = RequestState(request_id=request_id, cache=self._new_cache())
                self._states[request_id] = state
            return state

    def free(self, request_id: str) -> None:
        with self._lock:
            self._states.pop(request_id, None)

    def active_requests(self) -> int:
        with self._lock:
            return len(self._states)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        *,
        request_id: str,
        positions: List[int],
        input_ids: Optional[List[int]] = None,
        hidden: Optional["object"] = None,
    ):
        """Run this stage.

        Args:
            positions: absolute positions of the tokens in this step (RoPE/KV).
            input_ids: token ids — only on the first stage.
            hidden: incoming hidden states tensor — on every other stage.

        Returns:
            On the last stage: (None, logits of the final position).
            Otherwise: (hidden_states, None).
        """
        torch = self.torch
        # A preallocated cache has a hard ceiling. Writing past it is not an
        # exception on CUDA, it is corruption, so refuse before the kernel runs.
        if self.static_cache and positions and positions[-1] >= self.max_cache_len:
            raise RuntimeError(
                f"context of {positions[-1] + 1} tokens exceeds the static KV cache "
                f"({self.max_cache_len}); raise LOOM_MAX_CACHE_LEN or set "
                f"LOOM_STATIC_CACHE=0 on the worker"
            )
        state = self._state(request_id)
        # Concurrency policy: requests hold only their own cache, so they can
        # run side by side — that is what fills the pipeline bubble while
        # another stage is busy. The exception is CUDA graphs (compile in
        # reduce-overhead mode), which replay through fixed buffers and must
        # not be entered twice at once.
        model_lock = self._lock if self.compile_model else state.lock
        with model_lock, torch.inference_mode():
            if self.spec.is_first:
                if input_ids is None:
                    raise ValueError("first stage requires input_ids")
                ids = torch.tensor([input_ids], dtype=torch.long, device=self.shard.device)
                h = self.shard.embed(ids)
            else:
                if hidden is None:
                    raise ValueError("non-first stage requires hidden states")
                h = hidden.to(device=self.shard.device, dtype=self.shard.dtype)
                if h.dim() == 2:
                    h = h.unsqueeze(0)

            pos = torch.tensor([positions], dtype=torch.long, device=self.shard.device)
            # The model's own forward builds the attention mask, applies rotary
            # and drives the cache — all of which depend on the cache type and
            # are silently wrong if reimplemented by hand.
            h = self._run_stage(
                h,
                position_ids=pos,
                cache_position=pos[0],
                use_cache=True,
                **{self._cache_kwarg: state.cache},
            )

            state.seen_tokens += len(positions)

            if not self.spec.is_last:
                return h, None

            h = self.shard.norm(h)
            logits = self.shard.lm_head(h[:, -1:, :])
            return None, logits[0, -1].float()

    # ---------------------------------------------------------------- sampling
    def sample(
        self,
        logits,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> int:
        """Greedy by default; nucleus sampling when temperature > 0."""
        torch = self.torch
        if temperature is None or temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / float(temperature), dim=-1)
        if top_p is not None and 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= top_p
            keep[0] = True
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum()
            generator = None
            if seed is not None:
                generator = torch.Generator(device=sorted_probs.device).manual_seed(seed)
            choice = torch.multinomial(sorted_probs, 1, generator=generator)
            return int(sorted_idx[choice].item())
        generator = None
        if seed is not None:
            generator = torch.Generator(device=probs.device).manual_seed(seed)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    # -------------------------------------------------------------- transport
    def _pick_wire_dtype(self):
        """Dtype for activations on the wire.

        The stage's own dtype by default: sending bf16 halves the payload of
        every hop, and the receiver casts to whatever it runs anyway, so a
        bf16 stage and an fp32 stage still interoperate. float32 stays the
        fallback because it is the one every device supports.
        """
        torch = self.torch
        override = os.environ.get("LOOM_WIRE_DTYPE", "").strip().lower()
        by_name = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if override in by_name:
            return by_name[override]
        if self.shard.dtype in (torch.bfloat16, torch.float16):
            return self.shard.dtype
        return torch.float32

    def serialize(self, tensor) -> Tuple[bytes, List[int], str]:
        """Tensor -> (bytes, shape, dtype) for the wire."""
        torch = self.torch
        wire = self._wire_dtype
        t = tensor.detach().to("cpu", dtype=wire).contiguous()
        if wire is torch.bfloat16:
            # numpy has no bfloat16; reinterpret the same 2 bytes as int16.
            raw = t.view(torch.int16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        return raw, list(t.shape), str(wire).replace("torch.", "")

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        torch = self.torch
        by_name = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        wire = by_name.get(dtype, torch.float32)
        # bytearray: frombuffer needs a writable buffer, and the copy is what
        # detaches the tensor from the transport's memory.
        flat = torch.frombuffer(bytearray(data), dtype=wire)
        return flat.reshape(tuple(shape))
