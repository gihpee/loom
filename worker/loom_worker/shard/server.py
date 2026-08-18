# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/server/executor/base_executor.py — цикл стадии
# (принять активации -> прогнать слои -> отдать следующей; на последней стадии
# сэмплировать токен) и роль первого peer'а как владельца клиентского запроса.
# Изменения: HTTP-поверхность вместо ZMQ-сокетов (следующая стадия достигается
# через relay агента, а не напрямую); OpenAI-совместимый /v1/chat/completions
# реализован здесь, а не отдельным vLLM Rust frontend'ом; в v0 одна
# последовательность на запрос без continuous batching.
"""Stage process: serves this shard and, on the head, drives generation.

HTTP surface (loopback only):
  GET  /health              — readiness
  POST /stage/forward       — incoming activations / token / free (from the agent)
  POST /v1/chat/completions — client requests (head stage only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

from loom_worker.shard.executor import ShardExecutor
from loom_worker.shard.loader import (
    ShardSpec,
    build_shard,
    build_stage_checkpoint_view,
    resolve_model_path,
)

logger = logging.getLogger("loom_worker.shard.server")

STATE: Dict[str, object] = {}

# Reasoning models (Qwen3 & co) spend their first few hundred tokens inside
# <think>, so a small cap truncates the answer before it starts — the reply
# looks broken even though the pipeline worked. Requests may still ask for
# fewer or more; this is only the default when the caller says nothing.
DEFAULT_MAX_TOKENS = int(os.environ.get("LOOM_MAX_TOKENS_DEFAULT", "2048"))


# --------------------------------------------------------------------- relay
def relay(payload: dict) -> None:
    """Hand a stage message to the local agent, which tunnels it onward."""
    relay_url = STATE["relay_url"]
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        relay_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.error("relay to agent failed: %s", exc)


# ------------------------------------------------------------- head bookkeeping
class PendingRequest:
    """Head-side state: waits for tokens coming back from the last stage."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.tokens: "queue.Queue[dict]" = queue.Queue()


PENDING: Dict[str, PendingRequest] = {}
PENDING_LOCK = threading.Lock()


def _tensor_payload(executor: ShardExecutor, hidden) -> dict:
    import base64

    data, shape, dtype = executor.serialize(hidden)
    return {
        "tensor_b64": base64.b64encode(data).decode(),
        "shape": shape,
        "dtype": dtype,
    }


def _decode_tensor(executor: ShardExecutor, payload: dict):
    import base64

    return executor.deserialize(
        base64.b64decode(payload["tensor_b64"]), payload["shape"], payload["dtype"]
    )


# ------------------------------------------------------------------- stage step
def handle_stage_message(msg: dict) -> None:
    """Process one inter-stage message on this stage."""
    executor: ShardExecutor = STATE["executor"]
    topology = STATE["topology"]
    kind = msg.get("kind")
    request_id = msg.get("request_id", "")

    if kind == "free":
        executor.free(request_id)
        with PENDING_LOCK:
            PENDING.pop(request_id, None)
        return

    if kind == "token":
        # Head stage: the last stage returned a sampled token.
        with PENDING_LOCK:
            pending = PENDING.get(request_id)
        if pending is not None:
            pending.tokens.put(msg)
        return

    if kind == "error":
        with PENDING_LOCK:
            pending = PENDING.get(request_id)
        if pending is not None:
            pending.tokens.put({"kind": "error", "error": msg.get("error", "stage failed")})
        return

    if kind != "activations":
        logger.warning("unknown stage message kind: %s", kind)
        return

    positions = list(msg.get("positions") or [])
    hidden = _decode_tensor(executor, msg)
    # Everything this stage adds to the token's latency: decode of the incoming
    # tensor is already done, so this is the forward (plus sampling on the tail).
    stage_started = time.perf_counter()
    try:
        out_hidden, logits = executor.forward(
            request_id=request_id, positions=positions, hidden=hidden
        )
    except Exception as exc:
        logger.exception("stage forward failed")
        relay(
            {
                "kind": "error",
                "request_id": request_id,
                "target_stage": 0,
                "error": str(exc),
            }
        )
        return

    if topology["is_last"]:
        sampling = msg.get("sampling") or {}
        token = executor.sample(
            logits,
            temperature=sampling.get("temperature", 0.0),
            top_p=sampling.get("top_p", 1.0),
            seed=sampling.get("seed"),
        )
        compute_ms = (time.perf_counter() - stage_started) * 1000
        relay(
            {
                "kind": "token",
                "request_id": request_id,
                "target_stage": 0,
                "token_id": token,
                "step": msg.get("step", 0),
                # Carried home so the head can subtract compute from the round
                # trip and see the transport on its own.
                "compute_ms": compute_ms,
                "upstream_ms": float(msg.get("upstream_ms") or 0.0),
                "hops": int(msg.get("hops") or 0) + 1,
            }
        )
    else:
        compute_ms = (time.perf_counter() - stage_started) * 1000
        relay(
            {
                "kind": "activations",
                "request_id": request_id,
                "target_stage": topology["stage_index"] + 1,
                "step": msg.get("step", 0),
                "positions": positions,
                "sampling": msg.get("sampling"),
                "compute_ms": compute_ms,
                # Accumulate so the tail carries the whole pipeline's compute.
                "upstream_ms": float(msg.get("upstream_ms") or 0.0) + compute_ms,
                "hops": int(msg.get("hops") or 0) + 1,
                **_tensor_payload(executor, out_hidden),
            }
        )


# ------------------------------------------------------------------ generation
def generate(
    messages: List[dict],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream_cb=None,
    template_kwargs: Optional[dict] = None,
) -> dict:
    """Head-stage generation loop; returns the completion text and token counts."""
    executor: ShardExecutor = STATE["executor"]
    topology = STATE["topology"]
    tokenizer = STATE["tokenizer"]
    eos_ids = set(STATE["eos_token_ids"])

    prompt_ids = _encode_chat(tokenizer, messages, template_kwargs)
    request_id = uuid.uuid4().hex
    pending = PendingRequest(request_id)
    with PENDING_LOCK:
        PENDING[request_id] = pending

    generated: List[int] = []
    finish_reason = "length"
    # Timings are measured here, at the only place that sees the whole request:
    # prefill, every hop between stages and sampling. A client can only guess
    # at these from the outside, and for a pipeline the guess is worst.
    started_at = time.perf_counter()
    first_token_at: Optional[float] = None
    token_times: List[float] = []
    # Per-token latency split. `head_ms` is this stage's own forward; `peer_ms`
    # is what the other stages reported doing; whatever is left of the round
    # trip is transport — the wire, the two relays and the orchestrator hop.
    head_times: List[float] = []
    peer_times: List[float] = []
    transport_times: List[float] = []
    try:
        positions = list(range(len(prompt_ids)))
        step_input = list(prompt_ids)
        sampling = {"temperature": temperature, "top_p": top_p}
        for step in range(max_tokens):
            step_started = time.perf_counter()
            hidden, logits = executor.forward(
                request_id=request_id, positions=positions, input_ids=step_input
            )
            head_ms = (time.perf_counter() - step_started) * 1000
            peer_ms = 0.0
            if topology["num_stages"] == 1:
                token = executor.sample(
                    logits, temperature=temperature, top_p=top_p
                )
                head_ms = (time.perf_counter() - step_started) * 1000
            else:
                relay(
                    {
                        "kind": "activations",
                        "request_id": request_id,
                        "target_stage": 1,
                        "step": step,
                        "positions": positions,
                        "sampling": sampling,
                        "compute_ms": head_ms,
                        "upstream_ms": head_ms,
                        "hops": 1,
                        **_tensor_payload(executor, hidden),
                    }
                )
                msg = pending.tokens.get(timeout=STATE["stage_timeout_s"])
                if msg.get("kind") == "error":
                    raise RuntimeError(msg.get("error", "pipeline error"))
                token = int(msg["token_id"])
                # What every stage after this one spent: the tail carries the
                # running total plus its own.
                peer_ms = float(msg.get("upstream_ms") or 0.0) - head_ms
                peer_ms += float(msg.get("compute_ms") or 0.0)

            now = time.perf_counter()
            head_times.append(head_ms)
            peer_times.append(max(0.0, peer_ms))
            # No clock comparison anywhere: a duration measured here minus
            # durations measured there. Both sides use their own monotonic
            # clock, so unsynchronised machines cannot skew this.
            transport_times.append(
                max(0.0, (now - step_started) * 1000 - head_ms - max(0.0, peer_ms))
            )
            if first_token_at is None:
                first_token_at = now
            token_times.append(now)
            generated.append(token)
            if stream_cb is not None:
                stream_cb(tokenizer.decode([token], skip_special_tokens=True))
            if token in eos_ids:
                finish_reason = "stop"
                break
            positions = [len(prompt_ids) + len(generated) - 1]
            step_input = [token]
    finally:
        # Release KV state on every stage.
        executor.free(request_id)
        with PENDING_LOCK:
            PENDING.pop(request_id, None)
        if topology["num_stages"] > 1:
            relay({"kind": "free", "request_id": request_id, "target_stage": -1})

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "text": text,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated),
        "finish_reason": finish_reason,
        "timings": _timings(
            started_at,
            first_token_at,
            token_times,
            len(prompt_ids),
            head_times=head_times,
            peer_times=peer_times,
            transport_times=transport_times,
        ),
    }


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def _timings(
    started_at: float,
    first_token_at: Optional[float],
    token_times: List[float],
    prompt_tokens: int,
    *,
    head_times: Optional[List[float]] = None,
    peer_times: Optional[List[float]] = None,
    transport_times: Optional[List[float]] = None,
) -> dict:
    """Per-request numbers, in the units people actually compare.

    - ttft: prefill of the whole prompt plus one trip through every stage;
    - decode: the steady-state part, which is what tokens/s should be measured
      over (mixing prefill in makes short answers look artificially slow);
    - inter-token percentiles: on a pipeline these expose the per-hop cost, so
      a tail that drifts up means the network between stages, not the GPU.
    """
    topology = STATE.get("topology") or {}
    ended_at = token_times[-1] if token_times else time.perf_counter()
    total_ms = (ended_at - started_at) * 1000
    ttft_ms = ((first_token_at - started_at) * 1000) if first_token_at else 0.0
    gaps = sorted(
        (token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))
    )
    decode_ms = total_ms - ttft_ms
    decoded = max(0, len(token_times) - 1)  # tokens produced after the first
    return {
        "ttft_ms": round(ttft_ms, 1),
        "total_ms": round(total_ms, 1),
        "decode_ms": round(decode_ms, 1),
        # Steady-state speed and the end-to-end speed a user actually feels.
        "decode_tokens_per_s": round(decoded / (decode_ms / 1000), 2) if decode_ms > 0 else 0.0,
        "tokens_per_s": (
            round(len(token_times) / (total_ms / 1000), 2) if total_ms > 0 else 0.0
        ),
        "prompt_tokens_per_s": (
            round(prompt_tokens / (ttft_ms / 1000), 1) if ttft_ms > 0 else 0.0
        ),
        "inter_token_ms_p50": round(_percentile(gaps, 0.50), 1),
        "inter_token_ms_p95": round(_percentile(gaps, 0.95), 1),
        "inter_token_ms_max": round(gaps[-1], 1) if gaps else 0.0,
        "stages": int(topology.get("num_stages", 1)),
        "pipeline_id": topology.get("pipeline_id", ""),
        # Where each token's time actually goes. Decode steps only: the first
        # entry is prefill, whose cost is already reported as ttft.
        **_latency_split(head_times, peer_times, transport_times),
    }


def _summary(values: Optional[List[float]], prefix: str) -> dict:
    """p50/p95/mean for one leg of the split, rounded for reading."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"{prefix}_ms_p50": round(_percentile(ordered, 0.50), 2),
        f"{prefix}_ms_p95": round(_percentile(ordered, 0.95), 2),
        f"{prefix}_ms_mean": round(sum(values) / len(values), 2),
    }


def _latency_split(
    head_times: Optional[List[float]],
    peer_times: Optional[List[float]],
    transport_times: Optional[List[float]],
) -> dict:
    """Per-token breakdown: this stage, the other stages, and the wire.

    Prefill is dropped from the samples: it is one long step whose cost is
    already visible as ttft, and leaving it in drags every percentile.
    """
    decode = slice(1, None)
    head = (head_times or [])[decode]
    peer = (peer_times or [])[decode]
    wire = (transport_times or [])[decode]
    if not head:
        return {}
    split = {}
    split.update(_summary(head, "head_compute"))
    split.update(_summary(peer, "peer_compute"))
    split.update(_summary(wire, "transport"))
    total = sum(head) + sum(peer) + sum(wire)
    if total > 0:
        # The one number that decides what to optimise next.
        split["transport_share"] = round(sum(wire) / total, 3)
    return split


def _as_token_ids(encoded) -> List[int]:
    """Normalise whatever a tokenizer returned into a flat list of token ids.

    Tokenizers are not consistent about this across versions: transformers 5
    made `return_dict=True` the default for `apply_chat_template`, so it hands
    back a BatchEncoding — and iterating THAT yields its keys ("input_ids",
    "attention_mask"), not the ids. Callers also see plain lists, batched
    lists-of-lists and tensors. Anything that is not ids raises, so the caller
    can fall back instead of feeding strings into torch.
    """
    if hasattr(encoded, "keys") and "input_ids" in encoded:  # BatchEncoding/dict
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):  # torch tensor / numpy array
        encoded = encoded.tolist()
    encoded = list(encoded)
    if encoded and isinstance(encoded[0], (list, tuple)):  # batch of one
        encoded = list(encoded[0])
    if not encoded:
        raise ValueError("tokenizer returned no tokens")
    bad = next((t for t in encoded if not isinstance(t, int) or isinstance(t, bool)), None)
    if bad is not None:
        raise TypeError(f"tokenizer returned {type(bad).__name__}, not token ids")
    return encoded


def _encode_chat(
    tokenizer, messages: List[dict], template_kwargs: Optional[dict] = None
) -> List[int]:
    """Apply the chat template when the model has one, else concatenate.

    `template_kwargs` is passed straight to the template, which is how
    reasoning models are configured per request — Qwen3 takes
    `enable_thinking: false` to skip the <think> block.
    """
    try:
        if getattr(tokenizer, "chat_template", None):
            return _as_token_ids(
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **(template_kwargs or {}),
                )
            )
    except Exception:
        logger.warning(
            "chat template failed; falling back to plain concatenation", exc_info=True
        )
    text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
    text += "\nassistant:"
    return _as_token_ids(tokenizer.encode(text))


# ------------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ready = STATE.get("ready", False)
            self._json(
                200 if ready else 503,
                {
                    "status": "ok" if ready else "loading",
                    "stage": STATE.get("topology", {}).get("stage_index"),
                    "layers": STATE.get("layer_range"),
                    "active_requests": (
                        STATE["executor"].active_requests() if ready else 0
                    ),
                },
            )
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return

        if self.path == "/stage/forward":
            # Run off the HTTP thread: a stage step can take a while and the
            # agent should not block on it.
            threading.Thread(
                target=handle_stage_message, args=(payload,), daemon=True
            ).start()
            self._json(202, {"accepted": True})
            return

        if self.path == "/v1/chat/completions":
            if not STATE["topology"]["is_first"]:
                self._json(404, {"error": "not the head stage"})
                return
            self._chat(payload)
            return

        self.send_error(404)

    def _chat(self, body: dict) -> None:
        messages = body.get("messages") or []
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
        )
        temperature = float(body.get("temperature") or 0.0)
        top_p = float(body.get("top_p") or 1.0)
        # Same field vLLM uses: {"chat_template_kwargs": {"enable_thinking": false}}
        # turns off the <think> block on Qwen3-style models.
        template_kwargs = body.get("chat_template_kwargs") or None
        if template_kwargs is not None and not isinstance(template_kwargs, dict):
            self._json(400, {"error": "chat_template_kwargs must be an object"})
            return
        model_id = STATE["model_id"]
        created = int(time.time())
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if not body.get("stream"):
            try:
                result = generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    template_kwargs=template_kwargs,
                )
            except Exception as exc:
                logger.exception("generation failed")
                self._json(500, {"error": {"message": str(exc), "type": "server_error"}})
                return
            self._json(
                200,
                {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result["text"]},
                            "finish_reason": result["finish_reason"],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                    },
                    # Non-standard, additive: clients that do not know the field
                    # ignore it, and ours renders it as generation stats.
                    "timings": result["timings"],
                },
            )
            return

        # Streaming (SSE, chunked).
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(piece: str) -> None:
            event = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            self._chunk(f"data: {json.dumps(event)}\n\n")

        try:
            result = generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream_cb=emit,
                template_kwargs=template_kwargs,
            )
            # The closing chunk carries the counts and timings (what OpenAI
            # calls stream_options.include_usage). Without it a streaming
            # client has to guess token counts from the number of deltas.
            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
                "timings": result["timings"],
            }
            self._chunk(f"data: {json.dumps(final)}\n\n")
            self._chunk("data: [DONE]\n\n")
        except Exception as exc:
            logger.exception("streaming generation failed")
            self._chunk(f"data: {json.dumps({'error': str(exc)})}\n\n")
        finally:
            self._chunk("")

    def _chunk(self, data: str) -> None:
        raw = data.encode()
        self.wfile.write(f"{len(raw):X}\r\n".encode() + raw + b"\r\n")
        self.wfile.flush()


def _build_vllm_executor(args, spec):
    """Stand up the vLLM engine for this stage and return (executor, config).

    Imported lazily: the transformers engine must keep working in images that
    have no vLLM at all (and on CPU, where it cannot even be imported).
    """
    from transformers import AutoConfig

    from loom_worker.vllm_stage.executor import VllmStageExecutor
    from loom_worker.vllm_stage.runtime import (
        build_kv_cache,
        build_stage_runner,
        stage_config_from_env,
    )

    config = AutoConfig.from_pretrained(spec.model_path)
    if hasattr(config, "text_config"):
        config = config.text_config
    num_layers = args.num_model_layers or int(getattr(config, "num_hidden_layers"))
    dtype = args.dtype if args.dtype in ("bfloat16", "float16") else "bfloat16"

    runtime_config = stage_config_from_env(
        model_path=spec.model_path,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        num_layers=num_layers,
        dtype=dtype,
        vram_quota_bytes=args.vram_quota_bytes or None,
    )
    runner, vllm_config = build_stage_runner(runtime_config)
    kv_manager, kv_cache_config = build_kv_cache(runner, vllm_config, runtime_config)
    executor = VllmStageExecutor(runner, kv_manager, kv_cache_config, runtime_config)
    return executor, config


def _watch_parent(poll_s: float = 2.0) -> None:
    """Exit if the agent that spawned us is gone.

    Without this an orphaned stage keeps holding GPU memory (and a KV cache)
    after its worker agent dies or is killed — VRAM would leak on the provider's
    machine until a reboot.
    """
    original_ppid = os.getppid()

    def loop() -> None:
        while True:
            time.sleep(poll_s)
            ppid = os.getppid()
            # Only a CHANGE of parent means the agent died (we got reparented).
            # Comparing against 1 is wrong: inside a container the agent itself
            # is usually PID 1, so that check killed every stage immediately.
            if ppid != original_ppid:
                logger.warning(
                    "agent process gone (ppid %s -> %s); shutting the stage down",
                    original_ppid,
                    ppid,
                )
                os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


# ----------------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Loom pipeline-stage server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--weights-uri", required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--relay-url", default="")
    parser.add_argument("--device", default=os.environ.get("LOOM_SHARD_DEVICE", "cpu"))
    parser.add_argument("--dtype", default=os.environ.get("LOOM_SHARD_DTYPE", "float32"))
    parser.add_argument(
        "--stage-timeout-s", type=float, default=float(os.environ.get("LOOM_STAGE_TIMEOUT_S", "120"))
    )
    parser.add_argument(
        "--engine",
        choices=("torch", "vllm"),
        default=os.environ.get("LOOM_STAGE_ENGINE", "torch"),
        help="what runs the layers: transformers (portable) or vLLM (fast, CUDA only)",
    )
    parser.add_argument(
        "--num-model-layers",
        type=int,
        default=int(os.environ.get("LOOM_NUM_MODEL_LAYERS", "0")),
        help="layers in the whole model; the vLLM engine needs it to know if "
        "this stage is the tail",
    )
    parser.add_argument(
        "--vram-quota-bytes",
        type=int,
        default=int(os.environ.get("LOOM_VRAM_QUOTA_BYTES", "0")),
        help="broker-granted share of the card, converted to vLLM's utilisation",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    is_first = args.stage_index == 0
    is_last = args.stage_index == args.num_stages - 1
    STATE["model_id"] = args.model_id
    STATE["relay_url"] = args.relay_url
    STATE["stage_timeout_s"] = args.stage_timeout_s
    STATE["layer_range"] = [args.start_layer, args.end_layer]
    STATE["topology"] = {
        "pipeline_id": args.pipeline_id,
        "stage_index": args.stage_index,
        "num_stages": args.num_stages,
        "is_first": is_first,
        "is_last": is_last,
    }
    STATE["ready"] = False

    # Serve /health immediately so the agent can watch progress while weights load.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, name="stage-http", daemon=True).start()

    spec = ShardSpec(
        model_path="",  # filled in below, once the weights are on disk
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_first=is_first,
        is_last=is_last,
        device=args.device,
        dtype=args.dtype,
    )
    if args.engine == "vllm":
        # Same selective download as the torch engine, then a view directory
        # whose index mentions only the files we fetched — vLLM opens every
        # file the index lists, so the index has to match what is on disk.
        spec.model_path = resolve_model_path(args.weights_uri, shard=spec)
        spec.model_path = build_stage_checkpoint_view(spec.model_path, spec)
        STATE["executor"], config = _build_vllm_executor(args, spec)
    else:
        # Fetch only the safetensors files this stage's layers live in.
        spec.model_path = resolve_model_path(args.weights_uri, shard=spec)
        shard, config = build_shard(spec)
        STATE["executor"] = ShardExecutor(shard)

    if is_first or is_last:
        from transformers import AutoTokenizer

        STATE["tokenizer"] = AutoTokenizer.from_pretrained(spec.model_path)
    eos = getattr(config, "eos_token_id", None)
    STATE["eos_token_ids"] = (
        [eos] if isinstance(eos, int) else list(eos or [])
    )
    STATE["ready"] = True
    # Exit immediately on SIGTERM: a redeploy must not wait out a 30s kill
    # timeout, and the freed VRAM should be reusable at once.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    _watch_parent()
    logger.info(
        "stage %d/%d ready on port %d (layers [%d, %d))",
        args.stage_index,
        args.num_stages,
        args.port,
        args.start_layer,
        args.end_layer,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
