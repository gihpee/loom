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
import collections
import json
import logging
import os
import queue
import signal
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, List, Optional

from loom_stage import pipeline
from loom_stage.loader import ShardSpec, build_shard, resolve_model_path
from loom_stage.scheduler import Full, Scheduler, Sequence

logger = logging.getLogger("loom_stage.server")

STATE: Dict[str, object] = {}

# Reasoning models (Qwen3 & co) spend their first few hundred tokens inside
# <think>, so a small cap truncates the answer before it starts — the reply
# looks broken even though the pipeline worked. Requests may still ask for
# fewer or more; this is only the default when the caller says nothing.
DEFAULT_MAX_TOKENS = int(os.environ.get("LOOM_MAX_TOKENS_DEFAULT", "2048"))


# --------------------------------------------------------------------- relay
AGENT_URL = os.environ.get("LOOM_AGENT_URL", "")
TASK_ID = os.environ.get("LOOM_TASK_ID", "")


def relay(payload: dict) -> None:
    """Hand a stage message to the local agent, which routes it onward.

    The stage says WHICH RANK it is writing to and nothing else. Whether that
    rank is on this machine, behind a direct link, or on the far side of the
    orchestrator is the agent's problem — and a stage that knew would have to
    know about NAT and relays too.

    The pipeline and model are stamped here because this process is the only
    one that knows them for certain: a node running stages of two models would
    otherwise have activations of one addressed with the other's id.
    """
    topology = STATE.get("topology") or {}
    payload.setdefault("pipeline_id", topology.get("pipeline_id", ""))
    payload.setdefault("model_id", STATE.get("model_id", ""))
    if not AGENT_URL:
        # A stage with nowhere to send was started outside an agent. Saying so
        # plainly beats urllib's "unknown url type: ''", which is what the
        # operator saw while this path reported a DIFFERENT error.
        logger.error(
            "this stage has no agent to hand %s to: it was started outside one",
            payload.get("kind", "a message"),
        )
        return
    to_rank = payload.get("target_stage", 0)
    request = urllib.request.Request(
        f"{AGENT_URL}/send", data=json.dumps(payload).encode(), method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Loom-Task": TASK_ID,
            # -1 means every other member: how a finished request tells the
            # whole pipeline to drop its cache for it.
            "X-Loom-To-Rank": str(to_rank),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            answer.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.error("handing a message to the agent failed: %s", exc)


def announce(port: int) -> None:
    """Tell the agent which port this stage actually bound.

    The agent can only suggest one; between the suggestion and this bind
    another process on the same machine may have taken it, which is exactly
    what happens when a multi-GPU host runs two agents.
    """
    if not AGENT_URL:
        return
    request = urllib.request.Request(
        f"{AGENT_URL}/ready", data=json.dumps({"port": port}).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Loom-Task": TASK_ID},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            answer.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.error("could not tell the agent this stage is on %d: %s", port, exc)


# ------------------------------------------------------------- head bookkeeping
# Кто ведёт конвейер, зависит от места стадии: голова (`pipeline.Head`) держит
# очередь и один считающий поток, все прочие (`pipeline.Stage`) считают то, что
# прислали. Оба живут в STATE, потому что до них добираются три разных потока:
# HTTP, приём сообщений и сам цикл.


def head() -> pipeline.Head:
    return STATE["head"]


# --------------------------------------------------------- measured speed
# Per-LAYER compute time, averaged over recent steps. The scheduler splits a
# model between nodes in proportion to how fast they are, and until this
# existed it had only a spec table to go on: an A30 that is missing from that
# table looks exactly like any other unknown card, so two nodes that differ
# fourfold in reality were handed twenty layers each. A number the node
# measured on the real model beats any table.
class StageSpeed:
    """Median ms per layer over recent steps.

    Median, not a moving average, and the difference matters here. This number
    decides how many layers this node is given, and that decision persists.
    With an EMA a single 300 ms hiccup — a GC pause, a contended card, one
    prefill among decodes — dragged the estimate an order of magnitude and
    would have moved layers off a perfectly good node. The median simply does
    not see one bad sample among many.
    """

    WINDOW = 64
    MIN_SAMPLES = 8

    def __init__(self) -> None:
        self._samples: Deque[float] = collections.deque(maxlen=self.WINDOW)
        self._lock = threading.Lock()

    def record(self, compute_ms: float, num_layers: int) -> None:
        if num_layers <= 0 or compute_ms <= 0:
            return
        with self._lock:
            self._samples.append(compute_ms / num_layers)

    def snapshot(self) -> Optional[float]:
        with self._lock:
            # A handful of steps is warm-up, and a warm-up number is worse than
            # none: it would tell the planner this node is slow and cost it its
            # layers. Silence keeps the planner on its own estimate.
            if len(self._samples) < self.MIN_SAMPLES:
                return None
            return statistics.median(self._samples)


SPEED = StageSpeed()


# ---------------------------------------------------------------- stage inbox
# Every inter-stage message is handled by ONE worker thread, in arrival order.
#
# This used to be a thread per message, which is what killed a live stage: the
# inference engine holds one set of persistent buffers and a process-wide
# forward context, so two steps running at once left one of them reading
# metadata the other had already torn down — "Forward context is not set",
# followed by an illegal instruction that poisons the CUDA context for good.
# Throughput has to come from batching several sequences into one engine step,
# never from calling the engine twice at the same time.
STAGE_INBOX: "queue.Queue[dict]" = queue.Queue()


def stage_inbox_loop() -> None:
    while True:
        msg = STAGE_INBOX.get()
        try:
            handle_stage_message(msg)
        except Exception:
            logger.exception("stage message failed: %s", msg.get("kind"))
        finally:
            STAGE_INBOX.task_done()


# ------------------------------------------------------------------- stage step
def _describe_failure(exc: BaseException, topology: dict) -> str:
    """Say which stage failed, and where, not just what it said.

    A pipeline error used to arrive as the bare text of the exception —
    "list index out of range" — with nothing to say which of four machines
    produced it or what it was doing. The head is the only place the operator
    looks, and it had no way to know either, so every failure started with
    grepping four sets of container logs.

    Carried instead: the stage and its layers (which machine to look at), the
    exception type (usually enough to name the bug), and the line that raised
    (where to look when it is not).
    """
    import traceback

    layers = STATE.get("layer_range") or [0, 0]
    where = ""
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        last = frames[-1]
        where = f" at {os.path.basename(last.filename)}:{last.lineno}"
    return (
        f"stage {topology.get('stage_index', '?')} of "
        f"{topology.get('num_stages', '?')} (layers [{layers[0]}, {layers[1]})): "
        f"{type(exc).__name__}: {exc}{where}"
    )


def handle_stage_message(msg: dict) -> None:
    """Одно межстадийное сообщение.

    Разбор кончается здесь: дальше сообщение уходит либо в голову, либо в
    стадию, и обе они ничего не знают ни про HTTP, ни про агента.
    """
    kind = msg.get("kind")
    topology = STATE["topology"]
    if not STATE.get("ready"):
        # Веса грузятся минутами, а сосед может успеть прислать активации
        # раньше. Сказать об этом внятно лучше, чем KeyError по имени, которого
        # ещё нет в STATE, — по нему не видно даже, что стадия просто не готова.
        logger.warning("сообщение %s пришло, пока стадия грузит веса — отбрасываю",
                       kind)
        return

    if kind == "free":
        request_id = msg.get("request_id", "")
        if topology["is_first"]:
            head().cancel(request_id)
        else:
            STATE["stage"].on_free(request_id)
        return

    if kind in ("tokens", "error") and topology["is_first"]:
        head().on_returned(msg)
        return

    if kind == "activations":
        if topology["is_first"]:
            # Голова активаций не принимает: она их только рассылает. Приход
            # такого сообщения означает, что кольцо замкнулось не туда.
            logger.error("голова получила активации — адресация конвейера сбита")
            return
        STATE["stage"].on_activations(msg)
        return

    logger.warning("неизвестный вид межстадийного сообщения: %s", kind)


# ------------------------------------------------------------------ generation
def generate(
    messages: List[dict],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream_cb=None,
    on_open=None,
    template_kwargs: Optional[dict] = None,
) -> dict:
    """Один клиентский запрос: поставить в очередь и читать токены.

    Конвейер этот поток больше не ведёт — его ведёт один общий цикл, который
    считает батчами. Отсюда видно только свой запрос, и это ровно то, что
    клиенту нужно знать.
    """
    loop = head()
    tokenizer = STATE["tokenizer"]

    prompt_ids = _encode_chat(tokenizer, messages, template_kwargs)
    sequence = Sequence(
        request_id=uuid.uuid4().hex,
        prompt_ids=prompt_ids,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # Бросает Full, если мест нет. Зовущий на потоке открывает поток только
    # после этой строки: начав его, отказать кодом ответа уже нельзя.
    ticket = loop.submit(sequence)
    if on_open is not None:
        on_open()

    generated: List[int] = []
    finish_reason = "length"
    started_at = time.perf_counter()
    first_token_at: Optional[float] = None
    token_times: List[float] = []
    # Разбивка осталась пошаговой: шаг считал весь батч сразу, и делить его
    # стоимость между участниками было бы выдумкой. Здесь она приписана тем
    # токенам, которые этот шаг и выдал.
    head_times: List[float] = []
    peer_times: List[float] = []
    transport_times: List[float] = []
    # Сколько последовательностей считалось вместе с этой. Единственное место,
    # откуда видно, работает ли батчинг вообще: ответы остаются правильными и
    # когда каждый запрос считается в одиночку — просто медленнее.
    batch_sizes: List[int] = []
    try:
        while True:
            try:
                event = ticket.next(STATE["stage_timeout_s"])
            except queue.Empty:
                raise RuntimeError(
                    f"конвейер молчит дольше {STATE['stage_timeout_s']:g} с")
            kind = event.get("kind")
            if kind == "error":
                raise RuntimeError(event.get("error", "конвейер не ответил"))
            if kind == "done":
                finish_reason = event.get("finish_reason", "length")
                break
            token = int(event["token_id"])
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_times.append(now)
            head_times.append(float(event.get("head_ms") or 0.0))
            peer_times.append(float(event.get("peer_ms") or 0.0))
            transport_times.append(float(event.get("transport_ms") or 0.0))
            batch_sizes.append(int(event.get("batch") or 1))
            generated.append(token)
            if stream_cb is not None:
                stream_cb(tokenizer.decode([token], skip_special_tokens=True))
    finally:
        # Клиент мог уйти посреди ответа, а мог дойти до конца — цикл в обоих
        # случаях должен перестать считать этот запрос и отпустить его кэш на
        # всех стадиях.
        loop.cancel(sequence.request_id)

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
            batch_sizes=batch_sizes,
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
    batch_sizes: Optional[List[int]] = None,
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
        # Средний и наибольший батч, в котором считался этот запрос. Единица
        # означает, что он всё время был на узле один: батчинг тут не при чём,
        # просто некого было к нему добавить.
        "batch_avg": (round(sum(batch_sizes) / len(batch_sizes), 2)
                      if batch_sizes else 1.0),
        "batch_max": max(batch_sizes) if batch_sizes else 1,
        # Where each token's time actually goes. Decode steps only: the first
        # entry is prefill, whose cost is already reported as ttft.
        **_latency_split(head_times, peer_times, transport_times),
        # The same measurements unsummarised, token by token. Percentiles say
        # what a request cost on average; they cannot show a rate drifting down
        # as the context grows, a stall in the middle, or the moment one stage
        # started lagging. That needs the series itself.
        "series": _timing_series(
            started_at, token_times, head_times, peer_times, transport_times
        ),
    }


# A long answer is a long series. Capped so one closing chunk cannot grow past
# what the tunnel will carry; the cap is far above any interactive request.
SERIES_MAX_POINTS = int(os.environ.get("LOOM_TIMING_SERIES_MAX", "4096"))


def _timing_series(
    started_at: float,
    token_times: List[float],
    head_times: Optional[List[float]],
    peer_times: Optional[List[float]],
    transport_times: Optional[List[float]],
) -> dict:
    """Per-token measurements as parallel arrays, aligned by index.

    Index 0 is the first token, so `gap_ms[0]` is the prefill (it equals ttft)
    and `head_ms[0]` is the prefill's compute. Consumers wanting steady-state
    decode drop index 0, exactly as the percentiles above do.

    Parallel arrays rather than a list of objects: the payload is a third of
    the size and it is what a plotting library wants anyway.
    """
    if not token_times:
        return {}
    count = len(token_times)
    step = max(1, -(-count // SERIES_MAX_POINTS))  # ceil division
    keep = range(0, count, step)

    def at(values: Optional[List[float]], index: int) -> float:
        if not values or index >= len(values):
            return 0.0
        return round(values[index], 2)

    elapsed, gaps = [], []
    for i in keep:
        elapsed.append(round((token_times[i] - started_at) * 1000, 1))
        previous = token_times[i - step] if i >= step else started_at
        gaps.append(round((token_times[i] - previous) * 1000, 2))
    return {
        "t_ms": elapsed,          # since the request started
        "gap_ms": gaps,           # since the previous point
        "head_ms": [at(head_times, i) for i in keep],
        "peer_ms": [at(peer_times, i) for i in keep],
        "wire_ms": [at(transport_times, i) for i in keep],
        "tokens": count,
        # >1 means points were sampled every Nth token, so gap_ms spans N tokens.
        "every_n_tokens": step,
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
                        STATE["engine"].active_requests() if ready else 0
                    ),
                    # Сколько ещё влезет. Клиент, получивший отказ, должен
                    # видеть отсюда, что узел действительно полон, а не молча
                    # гадать.
                    "queue": (
                        STATE["head"].snapshot()
                        if ready and STATE.get("head") is not None else None
                    ),
                    # None until enough steps have been seen; the planner then
                    # keeps using its roofline estimate instead of a warm-up
                    # number.
                    "layer_latency_ms": SPEED.snapshot(),
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

        if self.path in ("/loom/message", "/stage/forward"):
            # Off the HTTP thread (a step can take a while and the agent should
            # not block on it), but onto ONE queue, not a thread per message.
            STAGE_INBOX.put(payload)
            self._json(202, {"accepted": True})
            return

        if self.path == "/v1/chat/completions":
            if not STATE["topology"]["is_first"]:
                self._json(404, {"error": "not the head stage"})
                return
            if not STATE.get("ready"):
                # Веса грузятся минутами, и всё это время процесс уже слушает.
                # Без этой проверки запрос доходил до генерации и падал на
                # KeyError: 'executor' — ошибке, которая ничего не объясняет.
                self._json(503, {"error": {
                    "message": "стадия ещё загружает веса",
                    "type": "loading",
                }})
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
            except Full as exc:
                # 429, а не 500: узел исправен, он занят. Разница видна
                # клиенту — на 500 повторять запрос бессмысленно, на 429
                # осмысленно, и именно так поступают все библиотеки.
                self._json(429, {"error": {"message": str(exc),
                                           "type": "rate_limit_exceeded"}})
                return
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

        # Streaming (SSE, chunked). Заголовки уходят не отсюда, а из
        # `on_open` — то есть только после того, как запрос принят в очередь.
        # Начав поток, отказать кодом ответа уже нельзя: остаётся строка внутри
        # него, которую половина клиентов не покажет.
        def on_open() -> None:
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

        opened = {"да": False}

        def open_stream() -> None:
            on_open()
            opened["да"] = True

        try:
            result = generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream_cb=emit,
                on_open=open_stream,
                template_kwargs=template_kwargs,
            )
        except Full as exc:
            self._json(429, {"error": {"message": str(exc),
                                       "type": "rate_limit_exceeded"}})
            return
        except Exception as exc:
            logger.exception("streaming generation failed")
            # A failure raised HERE is the head's own. One raised on another
            # stage arrives already labelled (see _describe_failure) and is
            # re-raised verbatim, so the label survives the trip home.
            described = (
                str(exc)
                if str(exc).startswith("stage ")
                else _describe_failure(exc, STATE.get("topology") or {})
            )
            if not opened["да"]:
                # Поток не открылся — значит заголовков ещё не было, и отказать
                # можно кодом ответа, а не строкой внутри тела.
                self._json(500, {"error": {"message": described,
                                           "type": "server_error"}})
                return
            self._chunk(f"data: {json.dumps({'error': described})}\n\n")
            self._chunk("")
            return

        # The closing chunk carries the counts and timings (what OpenAI calls
        # stream_options.include_usage). Without it a streaming client has to
        # guess token counts from the number of deltas.
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
        self._chunk("")

    def _chunk(self, data: str) -> None:
        raw = data.encode()
        self.wfile.write(f"{len(raw):X}\r\n".encode() + raw + b"\r\n")
        self.wfile.flush()


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
def _build_engine(args, spec: ShardSpec):
    """Собрать то, что будет считать слои, и вернуть его вместе с конфигом.

    Конфиг нужен ровно за одним: за списком токенов конца строки. Читается он
    отсюда, а не из движка, потому что у vLLM своя модель конфигурации, а
    останавливать генерацию надо одинаково на обоих.
    """
    from transformers import AutoConfig

    from loom_stage import engine as engines

    if args.engine == "vllm":
        num_layers = args.num_model_layers or getattr(
            AutoConfig.from_pretrained(spec.model_path), "num_hidden_layers", 0)
        built = engines.build(
            "vllm", model_path=spec.model_path, start_layer=args.start_layer,
            end_layer=args.end_layer, num_model_layers=num_layers,
            dtype=args.dtype, vram_quota_bytes=args.vram_quota_bytes,
            max_requests=args.max_sequences)
        return built, AutoConfig.from_pretrained(spec.model_path)

    shard, config = build_shard(spec)
    return engines.build("torch", shard, max_requests=args.max_sequences), config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Loom pipeline-stage server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--weights-uri", required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    # Where this stage sits comes from the agent, which placed the group. The
    # flags stay for running a stage by hand, which is how a bad shard gets
    # reproduced without an orchestrator in the way.
    parser.add_argument("--stage-index", type=int,
                        default=int(os.environ.get("LOOM_RANK", "0")))
    parser.add_argument("--num-stages", type=int,
                        default=int(os.environ.get("LOOM_GROUP_SIZE", "1")))
    parser.add_argument("--pipeline-id", default=os.environ.get("LOOM_GROUP_ID", ""))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("LOOM_SERVE_PORT", "0")))
    parser.add_argument("--device", default=os.environ.get("LOOM_SHARD_DEVICE", "cpu"))
    parser.add_argument("--dtype", default=os.environ.get("LOOM_SHARD_DTYPE", "float32"))
    parser.add_argument(
        "--stage-timeout-s", type=float, default=float(os.environ.get("LOOM_STAGE_TIMEOUT_S", "120"))
    )
    parser.add_argument(
        "--engine",
        choices=("torch", "vllm"),
        default=os.environ.get("LOOM_STAGE_ENGINE", "torch"),
        help=(
            "чем считать слои: transformers (переносимо, работает на CPU) или "
            "vLLM (батч из нескольких последовательностей в одном шаге, только "
            "CUDA)"
        ),
    )
    parser.add_argument(
        "--num-model-layers",
        type=int,
        default=int(os.environ.get("LOOM_NUM_MODEL_LAYERS", "0")),
        help="layers in the whole model; the vLLM engine needs it to know if "
        "this stage is the tail",
    )
    parser.add_argument(
        "--max-sequences", type=int,
        default=int(os.environ.get("LOOM_MAX_SEQUENCES", "64")),
        help="сколько запросов держать одновременно; сверх этого — честный "
             "отказ, а не вытеснение чужого кэша")
    parser.add_argument(
        "--max-batch-tokens", type=int,
        default=int(os.environ.get("LOOM_MAX_BATCH_TOKENS", "8192")),
        help="сколько токенов промпта считать за один шаг")
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

    # Serve /health immediately so the agent can watch progress while weights
    # load — a stage takes minutes to come up and silence for minutes looks
    # exactly like a stage that died.
    for candidate in (args.port, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no port to serve on")
    args.port = server.server_port
    threading.Thread(target=server.serve_forever, name="stage-http", daemon=True).start()
    announce(args.port)
    threading.Thread(target=stage_inbox_loop, name="stage-inbox", daemon=True).start()

    spec = ShardSpec(
        model_path="",  # filled in below, once the weights are on disk
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        is_first=is_first,
        is_last=is_last,
        device=args.device,
        dtype=args.dtype,
    )
    # Fetch only the safetensors files this stage's layers live in.
    spec.model_path = resolve_model_path(args.weights_uri, shard=spec)
    engine, config = _build_engine(args, spec)
    STATE["engine"] = engine

    if is_first or is_last:
        from transformers import AutoTokenizer

        STATE["tokenizer"] = AutoTokenizer.from_pretrained(spec.model_path)
    eos = getattr(config, "eos_token_id", None)
    STATE["eos_token_ids"] = (
        [eos] if isinstance(eos, int) else list(eos or [])
    )

    layers = args.end_layer - args.start_layer
    if is_first:
        STATE["head"] = pipeline.Head(
            engine, num_stages=args.num_stages, send=relay,
            eos_ids=STATE["eos_token_ids"], timeout_s=args.stage_timeout_s,
            scheduler=Scheduler(max_sequences=args.max_sequences,
                                max_batch_tokens=args.max_batch_tokens),
            on_step=lambda ms, size: SPEED.record(ms, layers))
        STATE["head"].start()
    else:
        STATE["stage"] = pipeline.Stage(
            engine, stage_index=args.stage_index, is_last=is_last, send=relay,
            on_step=lambda ms, size: SPEED.record(ms, layers))
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
