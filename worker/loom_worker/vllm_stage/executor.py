# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/vllm/batch_info.py (form_vllm_batch_prefill/decode —
# сборка SchedulerOutput вручную через KVCacheManager) и
# src/parallax/server/executor/vllm_executor.py (process_batch: подача
# intermediate_tensors, извлечение hidden_states/logits по роли стадии).
# Изменения: приведено к контракту исполнителя Loom (forward/sample/free/
# serialize), одна последовательность на вызов вместо их батчера, состояние
# запроса хранится здесь же (у Parallax его ведёт отдельный планировщик),
# и добавлен разбор ошибок нехватки KV-блоков в понятное сообщение.
"""Stage executor backed by vLLM: same contract, different engine.

`shard/server.py` drives a stage through five calls — forward, sample, free,
serialize, deserialize. Keeping that contract means the pipeline transport, the
generation loop, the OpenAI surface and the latency instrumentation are shared
with the transformers engine, and only the arithmetic underneath changes.

What vLLM brings that the transformers path cannot: paged KV cache (blocks are
pooled, so concurrent requests do not each reserve a worst-case buffer), CUDA
graphs, and fused kernels. What Loom keeps: which layers this stage owns, where
activations go next, and how tokens are sampled.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("loom_worker.vllm_stage.executor")


class KvCacheExhausted(RuntimeError):
    """No free blocks for this request; the caller should retry or shed load."""


@dataclass
class StageRequest:
    """What this stage remembers between the steps of one client request."""

    request_id: str
    prompt_len: int
    created_at: float = field(default_factory=time.time)
    generated: int = 0
    registered: bool = False
    vllm_request: object = None


class VllmStageExecutor:
    """Runs `[start_layer, end_layer)` of a model on vLLM's GPU runner."""

    def __init__(self, runner, kv_manager, kv_cache_config, config) -> None:
        import torch

        self.torch = torch
        self.runner = runner
        self.kv_manager = kv_manager
        self.kv_cache_config = kv_cache_config
        self.config = config
        self.spec = _SpecView(config)
        self._requests: Dict[str, StageRequest] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- contract
    def active_requests(self) -> int:
        with self._lock:
            return len(self._requests)

    def free(self, request_id: str) -> None:
        """Release the request's KV blocks back to the pool.

        Unlike a per-request contiguous cache, blocks freed here are
        immediately reusable by other requests — that is the whole point of
        paging, and forgetting to free would starve the stage.
        """
        with self._lock:
            state = self._requests.pop(request_id, None)
        if state is None:
            return
        try:
            if state.vllm_request is not None:
                self.kv_manager.free(state.vllm_request)
        except Exception:
            logger.exception("freeing KV blocks for %s failed", request_id)
        try:
            self.runner.requests.pop(request_id, None)
        except Exception:
            pass

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
        (None, logits) on the last — exactly what the transformers executor
        returns, so the stage server cannot tell them apart.
        """
        torch = self.torch
        with self._lock:
            state = self._requests.get(request_id)
            is_prefill = state is None
            if state is None:
                # Non-first stages never see token ids: the values are somebody
                # else's business, but the COUNT decides how many KV blocks to
                # reserve, and that we do know from the positions.
                prompt_len = len(input_ids) if input_ids else len(positions)
                state = StageRequest(request_id=request_id, prompt_len=prompt_len)
                self._requests[request_id] = state

        token_ids = list(input_ids) if input_ids else [0] * len(positions)
        intermediate = self._as_intermediate_tensors(hidden) if hidden is not None else None

        if is_prefill:
            scheduler_output = self._prefill_batch(state, token_ids)
        else:
            scheduler_output = self._decode_batch(state, token_ids)

        outputs = self.runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=intermediate,
        )
        state.generated += 1

        if not self.spec.is_last:
            hidden_states = self._extract_hidden(outputs)
            return hidden_states, None
        logits = self._extract_logits(outputs)
        return None, logits

    def sample(
        self,
        logits,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> int:
        """Greedy by default; nucleus sampling when temperature > 0.

        Sampling stays on Loom's side so both engines pick tokens the same way
        and their outputs stay comparable — the cost is microseconds against
        tens of milliseconds of layers.
        """
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

    def serialize(self, tensor) -> Tuple[bytes, List[int], str]:
        torch = self.torch
        t = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        return t.numpy().tobytes(), list(t.shape), "float32"

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        torch = self.torch
        by_name = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        wire = by_name.get(dtype, torch.float32)
        flat = torch.frombuffer(bytearray(data), dtype=wire)
        return flat.reshape(tuple(shape))

    # -------------------------------------------------------------- internals
    def _as_intermediate_tensors(self, hidden):
        """Wrap incoming activations the way vLLM's forward expects them."""
        from vllm.sequence import IntermediateTensors

        torch = self.torch
        h = hidden
        if h.dim() == 3:  # (batch, tokens, hidden) -> vLLM works on flat tokens
            h = h.reshape(-1, h.shape[-1])
        device = getattr(self.runner, "device", None)
        dtype = getattr(getattr(self.runner, "model_config", None), "dtype", None)
        h = h.to(device=device, dtype=dtype or torch.bfloat16)
        return IntermediateTensors({"hidden_states": h})

    def _new_vllm_request(self, state: StageRequest, token_ids: List[int]):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request as VllmRequest

        return VllmRequest(
            request_id=state.request_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            eos_token_id=None,
            arrival_time=state.created_at,
            block_hasher=getattr(self.runner, "request_block_hasher", None),
        )

    def _prefill_batch(self, state: StageRequest, token_ids: List[int]):
        """Register the request with vLLM and reserve blocks for the prompt."""
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

        vllm_request = self._new_vllm_request(state, token_ids)
        state.vllm_request = vllm_request
        state.registered = True

        blocks = self.kv_manager.allocate_slots(
            request=vllm_request,
            num_new_tokens=len(token_ids),
            num_new_computed_tokens=0,
        )
        if blocks is None:
            self.free(state.request_id)
            raise KvCacheExhausted(
                f"no KV blocks for a {len(token_ids)}-token prompt; lower "
                f"LOOM_MAX_REQUESTS or LOOM_MAX_MODEL_LEN on this worker"
            )

        new_request = NewRequestData(
            req_id=state.request_id,
            prompt_token_ids=token_ids,
            mm_features=[],
            sampling_params=vllm_request.sampling_params,
            pooling_params=None,
            block_ids=blocks.get_block_ids(),
            num_computed_tokens=0,
            lora_request=None,
            prompt_embeds=None,
        )
        return SchedulerOutput(
            scheduled_new_reqs=[new_request],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={state.request_id: len(token_ids)},
            total_num_scheduled_tokens=len(token_ids),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
        )

    def _decode_batch(self, state: StageRequest, token_ids: List[int]):
        """One more token for a request vLLM already knows about."""
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        vllm_request = state.vllm_request
        computed = state.prompt_len + state.generated - 1
        blocks = self.kv_manager.allocate_slots(
            request=vllm_request,
            num_new_tokens=1,
            num_new_computed_tokens=0,
        )
        if blocks is None:
            raise KvCacheExhausted(
                f"request {state.request_id} ran out of KV blocks at "
                f"{computed} tokens; raise LOOM_MAX_MODEL_LEN or shorten the context"
            )
        cached = CachedRequestData(
            req_ids=[state.request_id],
            resumed_from_preemption=[False],
            new_token_ids=[token_ids[-1:]],
            new_block_ids=[blocks.get_block_ids()],
            num_computed_tokens=[computed],
            num_output_tokens=[state.generated],
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={state.request_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
        )

    def _extract_hidden(self, outputs):
        """Hidden states for the next stage, whatever shape vLLM returned them in."""
        for attr in ("hidden_states", "last_hidden_state"):
            value = getattr(outputs, attr, None)
            if value is not None:
                return value
        if isinstance(outputs, dict) and "hidden_states" in outputs:
            return outputs["hidden_states"]
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise RuntimeError(
            "vLLM returned no hidden states for a middle stage; the pipeline "
            "group patch is the first thing to check (see docs/VLLM_PIPELINE.md)"
        )

    def _extract_logits(self, outputs):
        """Logits of the last position, as the last stage needs for sampling."""
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError(
                "vLLM returned no logits on the last stage; is_last_rank must be "
                "True there (see docs/VLLM_PIPELINE.md)"
            )
        if logits.dim() == 3:
            logits = logits[0, -1]
        elif logits.dim() == 2:
            logits = logits[-1]
        return logits.float()


class _SpecView:
    """The bits of ShardSpec the stage server reads off an executor."""

    def __init__(self, config) -> None:
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.is_first = config.is_first
        self.is_last = config.is_last
        self.device = "cuda"
        self.dtype = config.dtype
