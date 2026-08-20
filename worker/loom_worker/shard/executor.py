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
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom_worker.shard.loader import ShardModel
from loom_worker.wire import from_wire, to_wire

logger = logging.getLogger("loom_worker.shard.executor")


@dataclass
class RequestState:
    """Per-request KV cache and bookkeeping, owned by this stage."""

    request_id: str
    cache: object  # transformers Cache
    created_at: float = field(default_factory=time.time)
    seen_tokens: int = 0


class ShardExecutor:
    """Runs this stage's layers. Thread-safe: one lock around the model."""

    def __init__(self, shard: ShardModel, *, max_requests: int = 64) -> None:
        import torch

        self.torch = torch
        self.shard = shard
        self.spec = shard.spec
        self.max_requests = max_requests
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

    # ------------------------------------------------------------- lifecycle
    def _state(self, request_id: str) -> RequestState:
        from transformers import DynamicCache

        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                if len(self._states) >= self.max_requests:
                    # Evict the oldest: a stage must not grow unbounded if a
                    # head disappears without sending FREE.
                    oldest = min(self._states.values(), key=lambda s: s.created_at)
                    self._states.pop(oldest.request_id, None)
                    logger.warning("evicted stale request state %s", oldest.request_id)
                state = RequestState(request_id=request_id, cache=DynamicCache())
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
        state = self._state(request_id)
        with self._lock, torch.no_grad():
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
            cache_position = pos[0]
            rotary = self.shard.rotary
            position_embeddings = rotary(h, pos) if rotary is not None else None

            layer_kwargs = {
                "attention_mask": None,  # v0: one sequence per request -> causal by default
                "position_ids": pos,
                "use_cache": True,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
                self._cache_kwarg: state.cache,
            }
            for layer in self.shard.layers:
                out = layer(h, **layer_kwargs)
                h = out[0] if isinstance(out, tuple) else out

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
    def serialize(self, tensor) -> Tuple[bytes, List[int], str]:
        """Tensor -> (bytes, shape, dtype) for the wire.

        The tensor's own dtype travels with it, so a bfloat16 stage sends half
        the bytes a float32 one does and neither has to know what the other
        runs (see loom_worker/wire.py).
        """
        return to_wire(self.torch, tensor)

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        # Whatever arrived, in the dtype it was sent as. The layers convert it
        # themselves on the first matmul, so there is nothing to cast here.
        return from_wire(self.torch, data, shape, dtype)
