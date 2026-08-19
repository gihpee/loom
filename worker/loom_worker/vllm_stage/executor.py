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
import os
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
    # Full token history: 0.27's CachedRequestData wants it for the connector.
    token_ids: List[int] = field(default_factory=list)
    # Blocks handed to this request so far — allocate_slots returns only the
    # NEW ones, so coverage has to be accumulated.
    block_count: int = 0


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
        # ONE engine, ONE caller at a time. vLLM's model runner keeps a single
        # set of persistent buffers — the input batch, the positions, the
        # landing buffer — and enters a process-wide forward context around the
        # model call. The stage server hands each inter-stage message to its
        # own thread, so two overlapping requests ran execute_model
        # concurrently: one thread's context exit cleared the global while the
        # other was still inside the model, which surfaced as "Forward context
        # is not set" and then, once attention had run on clobbered metadata,
        # as an illegal instruction that poisons the CUDA context for good.
        #
        # Concurrency for this engine has to come from batching several
        # requests into ONE execute_model call, never from calling it twice.
        self._engine_lock = threading.RLock()
        self._prepare_incoming_buffer()

    def _prepare_incoming_buffer(self) -> None:
        """Give a non-first stage the landing buffer vLLM copies arrivals into.

        `_preprocess` does not consume the tensors we hand it directly: it
        copies them into `runner.intermediate_tensors`, a persistent buffer
        sized for the largest batch, and asserts that buffer exists. vLLM
        normally allocates it during profile_run — which a stage never
        performs, so the assert fired on the first request instead.

        Allocated here rather than lazily: it is a fixed cost (max batch x
        hidden x one tensor per key), and a card that cannot spare it should
        say so at startup, not mid-request.
        """
        if self.spec.is_first:
            return  # the head embeds tokens; nothing arrives from upstream
        if getattr(self.runner, "intermediate_tensors", None) is not None:
            return
        model = getattr(self.runner, "model", None)
        factory = getattr(model, "make_empty_intermediate_tensors", None)
        if factory is None:
            logger.warning(
                "this model exposes no make_empty_intermediate_tensors; "
                "relying on vLLM to allocate the landing buffer itself"
            )
            return
        max_tokens = int(getattr(self.runner, "max_num_tokens", 0) or self.config.max_model_len)
        self.runner.intermediate_tensors = factory(
            batch_size=max_tokens,
            dtype=self.runner.model_config.dtype,
            device=self.runner.device,
        )
        keys = list(self.runner.intermediate_tensors.tensors)
        logger.info(
            "stage inbox ready: %s for up to %d tokens",
            ", ".join(keys),
            max_tokens,
        )

    # ------------------------------------------------------------- contract
    def active_requests(self) -> int:
        with self._lock:
            return len(self._requests)

    def free(self, request_id: str) -> None:  # noqa: D401 - see below
        """Release the request's KV blocks back to the pool.

        Unlike a per-request contiguous cache, blocks freed here are
        immediately reusable by other requests — that is the whole point of
        paging, and forgetting to free would starve the stage.
        """
        with self._lock:
            state = self._requests.pop(request_id, None)
        if state is None:
            return
        # Under the engine lock: releasing blocks and dropping the runner's
        # record of the request must not overlap a forward.
        with self._engine_lock:
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

        if is_prefill:
            state.token_ids = list(token_ids)

        # One step = one exclusive turn with the engine, sampling included:
        # execute_model parks its result and sample_tokens consumes it, so a
        # second caller slipping between them would take the wrong logits.
        with self._engine_lock:
            try:
                outputs = self.runner.execute_model(
                    scheduler_output=scheduler_output,
                    intermediate_tensors=intermediate,
                )
                state.generated += 1

                if not self.spec.is_last:
                    # A non-final stage gets IntermediateTensors back — every
                    # tensor in it belongs to the next stage, not just the
                    # hidden states.
                    return self._outgoing_tensors(outputs), None

                # The final stage returns None and parks its result: logits
                # live in `execute_model_state` until sample_tokens() consumes
                # it. Reading them and NOT calling sample_tokens makes the NEXT
                # step fail with "State error", so the pair is done together.
                parked = getattr(self.runner, "execute_model_state", None)
                logits = getattr(parked, "logits", None) if parked is not None else None
                if logits is None:
                    logits = self._extract_logits(outputs)
                else:
                    self.runner.sample_tokens(None)  # clears the parked state
                return None, self._last_position_logits(logits)
            except BaseException as exc:
                self._fail_fast_on_dead_device(exc)
                raise

    def _fail_fast_on_dead_device(self, exc: BaseException) -> None:
        """Leave the process when the CUDA context can no longer be trusted.

        An illegal memory access or illegal instruction kills the context, not
        just the request: every later call on this device fails the same way,
        so a stage that keeps running only serves errors. Exiting hands the
        problem to machinery that already handles it — the agent notices the
        backend died, reports the shard failed, and the orchestrator re-places
        it on a fresh process.
        """
        message = str(exc)
        fatal = (
            "an illegal instruction" in message
            or "an illegal memory access" in message
            or "CUDA error: unspecified launch failure" in message
        )
        if not fatal:
            return
        logger.critical(
            "CUDA context is unusable (%s); exiting so the stage is restarted "
            "instead of serving errors",
            message.splitlines()[0] if message else exc,
        )
        os._exit(70)

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

    # ------------------------------------------------------------- the wire
    # The envelope between stages carries one tensor, a shape and a dtype
    # string. A vLLM stage has to move a NAMED SET of tensors instead, so the
    # set is stacked along a new leading axis and the names ride in the dtype
    # field: "float32|hidden_states,residual". The transport, the relay and the
    # orchestrator stay untouched, and a torch-engine pipeline is unaffected
    # because it never produces a set.
    WIRE_SEPARATOR = "|"

    def serialize(self, payload) -> Tuple[bytes, List[int], str]:
        torch = self.torch
        tensors = getattr(payload, "tensors", None)
        if not isinstance(tensors, dict):
            flat = payload.detach().to("cpu", dtype=torch.float32).contiguous()
            return flat.numpy().tobytes(), list(flat.shape), "float32"

        names = list(tensors)
        stacked = torch.stack(
            [tensors[name].detach().to("cpu", dtype=torch.float32).contiguous()
             for name in names]
        ).contiguous()
        return (
            stacked.numpy().tobytes(),
            list(stacked.shape),
            "float32" + self.WIRE_SEPARATOR + ",".join(names),
        )

    def deserialize(self, data: bytes, shape: List[int], dtype: str):
        torch = self.torch
        name, _, joined = dtype.partition(self.WIRE_SEPARATOR)
        by_name = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        wire = by_name.get(name, torch.float32)
        flat = torch.frombuffer(bytearray(data), dtype=wire).reshape(tuple(shape))
        if not joined:
            return flat
        from vllm.sequence import IntermediateTensors

        names = joined.split(",")
        return IntermediateTensors({n: flat[i] for i, n in enumerate(names)})

    # -------------------------------------------------------------- internals
    def _as_intermediate_tensors(self, incoming):
        """Put what arrived on this device, in this model's dtype."""
        from vllm.sequence import IntermediateTensors

        torch = self.torch
        device = getattr(self.runner, "device", None)
        dtype = getattr(getattr(self.runner, "model_config", None), "dtype", None)
        dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16

        def prepare(tensor):
            # vLLM works on a flat token axis; a (batch, tokens, hidden) tensor
            # from a torch-engine peer is folded down to it.
            if tensor.dim() == 3:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            return tensor.to(device=device, dtype=dtype)

        tensors = getattr(incoming, "tensors", None)
        if isinstance(tensors, dict):
            return IntermediateTensors({k: prepare(v) for k, v in tensors.items()})
        return IntermediateTensors({"hidden_states": prepare(incoming)})

    def _new_vllm_request(self, state: StageRequest, token_ids: List[int]):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request as VllmRequest

        from loom_worker.vllm_stage.runtime import _construct

        # Sampling happens on Loom's side; this object exists so the KV cache
        # manager has something to account blocks against.
        return _construct(
            VllmRequest,
            required=("request_id", "prompt_token_ids"),
            request_id=state.request_id,
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
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
        # vLLM's scheduler advances this after scheduling a step; here we ARE
        # the scheduler, so we owe the request the same bookkeeping. See
        # _decode_batch for what leaving it at zero costs.
        vllm_request.num_computed_tokens = len(token_ids)
        self._check_block_coverage(state, blocks, computed=len(token_ids))

        from loom_worker.vllm_stage.runtime import _construct

        new_request = _construct(
            NewRequestData,
            required=("req_id", "block_ids", "num_computed_tokens"),
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
        return _construct(
            SchedulerOutput,
            required=("scheduled_new_reqs", "num_scheduled_tokens"),
            scheduled_new_reqs=[new_request],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={state.request_id: len(token_ids)},
            total_num_scheduled_tokens=len(token_ids),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    def _decode_batch(self, state: StageRequest, token_ids: List[int]):
        """One more token for a request vLLM already knows about."""
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        vllm_request = state.vllm_request
        # THE source of truth for where this request is. vLLM sizes the block
        # table from `request.num_computed_tokens + num_new_tokens`, so a
        # request stuck at zero asks for slots for exactly one token forever:
        # once the prompt's last block fills up, allocate_slots decides nothing
        # new is needed and returns no blocks. The block table then stops
        # growing, every position past it resolves to block 0, and the model
        # silently attends to a 16-slot circular buffer instead of its context
        # — fluent text that repeats itself and forgets the question.
        computed = int(vllm_request.num_computed_tokens)
        # The request must also own the token it is about to compute: num_tokens
        # is what caps caching and sliding-window trimming.
        append = getattr(vllm_request, "append_output_token_ids", None)
        if callable(append):
            append(list(token_ids[-1:]))
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
        vllm_request.num_computed_tokens = computed + 1
        self._check_block_coverage(state, blocks, computed=computed + 1)
        from loom_worker.vllm_stage.runtime import _construct

        state.token_ids.extend(token_ids[-1:])
        cached = _construct(
            CachedRequestData,
            required=("req_ids", "new_block_ids", "num_computed_tokens"),
            req_ids=[state.request_id],
            # A set of ids in 0.27, not a per-request flag list.
            resumed_req_ids=set(),
            new_token_ids=[token_ids[-1:]],
            all_token_ids={state.request_id: list(state.token_ids)},
            new_block_ids=[blocks.get_block_ids()],
            num_computed_tokens=[computed],
            num_output_tokens=[state.generated],
        )
        return _construct(
            SchedulerOutput,
            required=("scheduled_cached_reqs", "num_scheduled_tokens"),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={state.request_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    def _check_block_coverage(self, state: StageRequest, blocks, computed: int) -> None:
        """Refuse to compute a position the block table does not cover.

        This is the alarm the stage did not have. An under-allocated block
        table does not raise: the runner gathers block id 0 for any position
        past the last real block and attention quietly runs on the wrong
        memory. The output stays grammatical, so nothing looks broken — the
        model just loses the thread. Counting blocks costs nothing and turns
        that into a failure with a name.
        """
        try:
            state.block_count += sum(len(group) for group in blocks.get_block_ids())
        except Exception:  # pragma: no cover - unfamiliar block object
            return
        needed = -(-computed // self.config.kv_block_size)  # ceil
        if state.block_count >= needed:
            return
        raise RuntimeError(
            f"KV block table for {state.request_id} covers "
            f"{state.block_count * self.config.kv_block_size} tokens but the "
            f"request is at {computed}; the engine would read block 0 for the "
            f"rest and answer from a corrupted context "
            f"(see docs/VLLM_PIPELINE.md §5b)"
        )

    def _outgoing_tensors(self, outputs):
        """What the next stage must receive.

        Llama-family models (Qwen3 included) carry BOTH `hidden_states` and
        `residual` between stages: the residual stream is fused across layers,
        and a stage handed only the hidden states would silently compute
        something else. So whatever vLLM put in IntermediateTensors travels on
        — the wire format below carries a named set, not one tensor.
        """
        tensors = getattr(outputs, "tensors", None)
        if isinstance(tensors, dict) and tensors:
            return outputs
        if isinstance(outputs, dict) and outputs:
            from vllm.sequence import IntermediateTensors

            return IntermediateTensors(dict(outputs))
        raise RuntimeError(
            "vLLM returned no intermediate tensors for a middle stage; the "
            "pipeline group patch is what makes it do that "
            "(see docs/VLLM_PIPELINE.md §3.3)"
        )

    def _last_position_logits(self, logits):
        if logits is None:
            raise RuntimeError(
                "vLLM produced no logits on the last stage; is_last_rank must "
                "be True there (see docs/VLLM_PIPELINE.md §3.3)"
            )
        if logits.dim() == 3:
            logits = logits[0, -1]
        elif logits.dim() == 2:
            logits = logits[-1]
        return logits.float()

    def _extract_logits(self, outputs):
        """Fallback for releases where execute_model returns the logits itself."""
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        return logits


class _SpecView:
    """The bits of ShardSpec the stage server reads off an executor."""

    def __init__(self, config) -> None:
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.is_first = config.is_first
        self.is_last = config.is_last
        self.device = "cuda"
        self.dtype = config.dtype
