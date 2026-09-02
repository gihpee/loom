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

from loom_stage.loader import ShardModel
from loom_stage.wire import from_wire, to_wire

logger = logging.getLogger("loom_stage.executor")


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
            # Layers may live on different cards of this host. Where the card
            # changes, the state moves with it — a copy over PCIe measured in
            # microseconds for one token, against ~20 ms for the same boundary
            # placed on another machine. This is why a multi-GPU host should be
            # one node and not several.
            devices = self.shard.layer_devices or [self.shard.device] * len(
                self.shard.layers
            )
            here = devices[0]
            for layer, device in zip(self.shard.layers, devices):
                if device != here:
                    h, layer_kwargs = self._move(h, layer_kwargs, device)
                    here = device
                out = layer(h, **layer_kwargs)
                h = out[0] if isinstance(out, tuple) else out
            if self.spec.is_last and here != self.shard.devices[-1]:
                h, layer_kwargs = self._move(h, layer_kwargs, self.shard.devices[-1])

            state.seen_tokens += len(positions)

            if not self.spec.is_last:
                return h, None

            h = self.shard.norm(h)
            logits = self.shard.lm_head(h[:, -1:, :])
            return None, logits[0, -1].float()

    def _move(self, h, layer_kwargs, device):
        """Carry the running state to another card of this host.

        Everything a layer reads and that carries a device has to come along:
        the hidden state, the positions, and the rotary tables. Leaving one
        behind fails immediately with a device mismatch rather than silently,
        which is the one mercy of this class of bug.
        """
        moved = dict(layer_kwargs)
        for key in ("position_ids", "cache_position"):
            value = moved.get(key)
            if value is not None:
                moved[key] = value.to(device)
        embeddings = moved.get("position_embeddings")
        if embeddings is not None:
            moved["position_embeddings"] = tuple(e.to(device) for e in embeddings)
        return h.to(device), moved

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
        runs (see loom_stage/wire.py).
        """
        return to_wire(self.torch, tensor)

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        # Whatever arrived, in the dtype it was sent as. The layers convert it
        # themselves on the first matmul, so there is nothing to cast here.
        return from_wire(self.torch, data, shape, dtype)

    # ------------------------------------------------------------- батч
    #: Считать несколько последовательностей одним прогоном этот исполнитель
    #: не умеет: плотная каузальная маска дала бы каждой следующей видеть
    #: предыдущую. Он прогоняет их по одной и складывает результат так же,
    #: как сложил бы настоящий батч, — чтобы протокол между стадиями был один
    #: на оба движка. Пропускная способность от этого не растёт.
    batches = False

    def step_batch(self, sequences, *, incoming=None, first_step: bool):
        """Батч по одной последовательности за раз.

        Границы внутри общего тензора заданы только составом батча, поэтому
        нарезка входа и склейка выхода идут по тем же ширинам, что и у
        настоящего батча, и в том же порядке.
        """
        from loom_stage import batch_wire

        torch = self.torch
        batch = list(sequences)
        widths = batch_wire.widths(batch, first_step=first_step)
        pieces = None
        if not self.spec.is_first:
            pieces = self._slice_incoming(incoming, widths, batch,
                                          first_step=first_step)

        hiddens, rows = [], []
        for index, sequence in enumerate(batch):
            positions, input_ids = self._where(sequence, first_step=first_step)
            hidden, logits = self.forward(
                request_id=sequence.request_id, positions=positions,
                input_ids=input_ids if self.spec.is_first else None,
                hidden=None if self.spec.is_first else pieces[index])
            if self.spec.is_last:
                rows.append(logits)
            else:
                hiddens.append(hidden if hidden.dim() == 3 else hidden.unsqueeze(0))

        if self.spec.is_last:
            return None, torch.stack(rows)
        # По второй оси: токены разных последовательностей лежат подряд, как и
        # у настоящего батча, — иначе следующая стадия нарежет не по тем
        # границам и ошибки не будет, будет чушь.
        return {"hidden_states": torch.cat(hiddens, dim=1)}, None

    def _where(self, sequence, *, first_step: bool):
        """Какие позиции и какие токены считает эта последовательность.

        Позиции берутся из состояния самой последовательности, а не из счётчика
        снаружи: батч живёт дольше одного шага, и запросы в нём разной длины.
        """
        if first_step:
            return list(range(len(sequence.prompt_ids))), list(sequence.prompt_ids)
        last = sequence.output_ids[-1] if sequence.output_ids else (
            sequence.prompt_ids[-1] if sequence.prompt_ids else 0)
        return [sequence.computed], [last]

    def _slice_incoming(self, incoming, widths, batch, *, first_step: bool):
        """Разложить общий тензор обратно по последовательностям."""
        from loom_stage import batch_wire

        if not incoming:
            raise ValueError("non-first stage requires hidden states")
        tensor = incoming.get("hidden_states")
        if tensor is None:
            raise ValueError(
                f"в тензорах нет hidden_states, есть {sorted(incoming)}")
        flat = tensor if tensor.dim() == 2 else tensor[0]
        batch_wire.check_tokens(batch, int(flat.shape[0]), first_step=first_step)
        pieces, at = [], 0
        for width in widths:
            pieces.append(flat[at:at + width].unsqueeze(0))
            at += width
        return pieces

    def sample_batch(self, logits, sequences) -> List[int]:
        """По токену на последовательность, в порядке батча."""
        from loom_stage import batch_wire

        rows = logits if logits.dim() > 1 else logits[None]
        batch_wire.check_rows(list(sequences), int(rows.shape[0]))
        return [self.sample(row, temperature=sequence.temperature,
                            top_p=sequence.top_p, seed=sequence.seed)
                for row, sequence in zip(rows, sequences)]
