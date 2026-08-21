"""Stage executor backed by MLX: same contract, Apple Silicon underneath.

`shard/server.py` drives a stage through five calls — forward, sample, free,
serialize, deserialize — so the pipeline transport, the generation loop, the
OpenAI surface and the latency instrumentation are shared with the other
engines, and only the arithmetic changes.

What makes this short is mlx_lm's own shape. A stage is a slice of a list:

    h = embed(tokens)                      # first stage only
    mask = create_attention_mask(h, cache[0])
    for layer, c in zip(my_layers, my_cache):
        h = layer(h, mask, c)
    h = norm(h); logits = head(h)          # last stage only

There is no engine to patch, no scheduler to reconstruct and no block manager
to drive — the CUDA vLLM path needed all three. MLX also keeps its own KV
cache per layer, advancing an offset internally, which is why nothing here
tracks positions: the cache knows where it is.

A Mac stage stands in the same pipeline as CUDA stages. That works because the
wire format carries the dtype with the tensor and both sides agree on the
names (see loom_worker/wire.py) — verified byte-identical in both directions.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom_worker.wire import mlx_from_wire, mlx_to_wire

logger = logging.getLogger("loom_worker.mlx_stage.executor")


@dataclass
class StageRequest:
    """What this stage remembers between the steps of one client request.

    The KV cache is per request AND per layer: MLX gives each layer its own
    cache object, and each one tracks how many tokens it has already seen.
    """

    request_id: str
    cache: list
    created_at: float = field(default_factory=time.time)
    steps: int = 0


class MlxStageExecutor:
    """Runs `[start_layer, end_layer)` of a model on Apple's GPU."""

    def __init__(self, model, config, spec) -> None:
        # Checked before anything is loaded or sliced: two sources say whether
        # this is the head — the topology the orchestrator sent and the layer
        # range itself — and a disagreement means the stage was wired wrong.
        # The symptom otherwise appears much later, as "the head stage was
        # given no token ids" from a stage that is plainly not the head.
        if bool(spec.is_first) != (config.start_layer == 0):
            raise ValueError(
                f"this stage holds layers [{config.start_layer}, "
                f"{config.end_layer}) but the topology calls it "
                f"{'the head' if spec.is_first else 'a later stage'}; the "
                f"pipeline wiring did not reach this worker"
            )

        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        self.mx = mx
        self._make_mask = create_attention_mask
        self.model = model
        self.config = config
        self.spec = spec
        self.inner = _inner_model(model)
        self.layers = list(self.inner.layers)
        self._requests: Dict[str, StageRequest] = {}
        self._lock = threading.RLock()
        # MLX evaluates lazily and its arrays are not thread safe. The stage
        # server hands each inter-stage message to its own worker, so the model
        # is entered by one caller at a time — the same rule the vLLM engine
        # needed, for the same reason.
        self._engine_lock = threading.RLock()

    # ------------------------------------------------------------- contract
    def active_requests(self) -> int:
        with self._lock:
            return len(self._requests)

    def free(self, request_id: str) -> None:
        """Drop this request's KV cache.

        Unified memory means a forgotten cache is RAM the machine cannot use
        for anything else, so this is not bookkeeping — it is the only thing
        that stops a long-running node from filling up.
        """
        with self._lock:
            self._requests.pop(request_id, None)

    def forward(
        self,
        *,
        request_id: str,
        positions: List[int],
        input_ids: Optional[List[int]] = None,
        hidden: Optional[object] = None,
    ):
        """One pipeline step for one request.

        Returns (hidden_states, None) on every stage but the last, and
        (None, logits) on the last — exactly what the other executors return,
        so the stage server cannot tell them apart.

        `positions` is accepted and unused: an MLX KV cache advances its own
        offset, so where the token sits is already known to the layers.
        """
        mx = self.mx
        with self._lock:
            state = self._requests.get(request_id)
            if state is None:
                state = StageRequest(request_id=request_id, cache=self._new_cache())
                self._requests[request_id] = state

        with self._engine_lock:
            h = self._incoming(input_ids, hidden)
            mask = self._make_mask(h, state.cache[0] if state.cache else None)
            for layer, cache in zip(self.layers, state.cache):
                h = layer(h, mask, cache)
            state.steps += 1

            if not self.spec.is_last:
                mx.eval(h)  # materialise before it leaves this process
                return h, None

            h = self.inner.norm(h)
            logits = self._head(h)
            mx.eval(logits)
            # Only the last position matters: the rest are prompt tokens whose
            # continuations were decided long ago.
            return None, logits[:, -1, :]

    def _incoming(self, input_ids: Optional[List[int]], hidden):
        """The hidden states this step starts from."""
        mx = self.mx
        if self.spec.is_first:
            if not input_ids:
                raise ValueError("the head stage was given no token ids")
            tokens = mx.array([list(input_ids)])
            return self.inner.embed_tokens(tokens)
        if hidden is None:
            raise ValueError("a non-first stage was given no hidden states")
        # A peer sends a flat (tokens, hidden) tensor; the layers want a batch
        # axis. Adding it here keeps the wire format free of a convention that
        # only this engine would care about.
        return hidden if hidden.ndim == 3 else hidden.reshape(1, *hidden.shape)

    def _head(self, h):
        """Logits from the final hidden states, tied embeddings or not."""
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            return lm_head(h)
        # tie_word_embeddings: the input embedding matrix doubles as the head.
        return self.inner.embed_tokens.as_linear(h)

    def _new_cache(self) -> list:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]

    def sample(
        self,
        logits,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> int:
        """Greedy by default; nucleus sampling when temperature > 0.

        Deliberately the same arithmetic as the other engines rather than
        mlx_lm's own samplers, so a model split across a Mac and a CUDA node
        picks tokens the same way wherever the last stage happens to land.
        """
        mx = self.mx
        flat = logits.reshape(-1)
        if temperature is None or temperature <= 0:
            return int(mx.argmax(flat).item())

        probs = mx.softmax(flat / float(temperature), axis=-1)
        if top_p is not None and 0 < top_p < 1.0:
            order = mx.argsort(-probs)
            ordered = probs[order]
            cumulative = mx.cumsum(ordered, axis=-1)
            keep = (cumulative - ordered) <= top_p
            ordered = ordered * keep
            ordered = ordered / mx.sum(ordered)
            if seed is not None:
                mx.random.seed(int(seed))
            choice = mx.random.categorical(mx.log(ordered + 1e-20))
            return int(order[choice].item())
        if seed is not None:
            mx.random.seed(int(seed))
        return int(mx.random.categorical(mx.log(probs + 1e-20)).item())

    # ------------------------------------------------------------- the wire
    def serialize(self, array) -> Tuple[bytes, List[int], str]:
        """MLX array -> (bytes, shape, dtype) for the next stage.

        The batch axis is dropped: peers exchange (tokens, hidden), and the
        receiving engine adds back whatever shape it wants.
        """
        if array.ndim == 3 and array.shape[0] == 1:
            array = array.reshape(array.shape[1], array.shape[2])
        return mlx_to_wire(self.mx, array)

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        return mlx_from_wire(self.mx, data, shape, dtype)


def _inner_model(model):
    for attr in ("model", "transformer", "language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    if hasattr(model, "layers"):
        return model
    raise RuntimeError(
        f"cannot find the decoder layers of {type(model).__name__}"
    )
