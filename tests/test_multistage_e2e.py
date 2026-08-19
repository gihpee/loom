"""THE core deliverable: one model served by layers spread over several nodes.

Full stack, real gRPC, no shortcuts:
  client -> API -> tunnel -> head stage (layers [0,k)) -> orchestrator relay
         -> stage 1 -> ... -> last stage (LM head, sampling) -> back to head
         -> streamed to the client

The pool is sized so the model CANNOT fit on a single node: each worker's quota
holds only 2 of the model's 6 layers, so Phase-1 must build a 3-stage pipeline.
Correctness is asserted against a plain single-process transformers run.
"""

import json
import sys
from pathlib import Path

import pytest
from make_tiny_model import ensure_tiny_model
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from loom.orchestrator.registry import ModelRegistry, ModelSpec  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MODEL_ID = "tiny-shard"
PROMPT = "hello pipeline"
# Each worker's quota holds only 3 of the model's 6 layers, so the model can
# never be served by a single node — Phase-1 is forced to build a pipeline.
QUOTA_GB = 0.0008
LAYERS_PER_NODE = 3
NUM_LAYERS = 6


@pytest.fixture(scope="module")
def model_dir():
    return str(ensure_tiny_model())


@pytest.fixture(scope="module")
def reference_completion(model_dir):
    """What the whole model produces in one process, greedily."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    prompt_ids = tok.encode(f"user: {PROMPT}\nassistant:")
    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=0,
        )
    new_tokens = out[0, len(prompt_ids) :].tolist()
    return tok.decode(new_tokens, skip_special_tokens=True)


def shard_registry(model_dir) -> ModelRegistry:
    raw = json.loads((CONFIGS / "catalog-tiny-shard.json").read_text())["models"][0]
    raw["weights_uri"] = model_dir
    return ModelRegistry([ModelSpec.from_dict(raw)])


@pytest.fixture
def pipeline_stack(model_dir):
    orch = OrchestratorHarness(shard_registry(model_dir)).start()
    workers = [
        WorkerHarness(
            orch.grpc_port,
            join_key=orch.join_key,
            node_id=f"stage-node-{i}",
            memory_gb=QUOTA_GB,
        ).start()
        for i in range(3)
    ]
    try:
        yield orch, workers
    finally:
        for w in workers:
            w.stop()
        orch.stop()


def stage_layout(controller):
    """(node_id, stage_index, layer_range) for the model, ordered by stage."""
    rows = [
        (node_id, entry[4], (entry[1], entry[2]))
        for (mid, node_id), entry in controller.deployed.items()
        if mid == MODEL_ID
    ]
    return sorted(rows, key=lambda r: r[1])


def test_model_is_split_across_nodes_and_answers(pipeline_stack, reference_completion):
    orch, workers = pipeline_stack
    controller = orch.controller

    # 1. Phase-1 had to split the model: no single node can hold 6 layers.
    assert wait_until(lambda: len(stage_layout(controller)) >= 2, 60), (
        f"expected a multi-stage layout, got {stage_layout(controller)}"
    )
    layout = stage_layout(controller)
    assert len(layout) >= 2, f"expected several stages: {layout}"
    # No stage holds the whole model, and stages tile [0, 6) without gaps.
    ranges = [r for _, _, r in layout]
    assert all(end - start <= LAYERS_PER_NODE for start, end in ranges), ranges
    assert ranges[0][0] == 0 and ranges[-1][1] == NUM_LAYERS
    for (a_start, a_end), (b_start, b_end) in zip(ranges, ranges[1:]):
        assert a_end == b_start, f"gap between stages: {ranges}"
    # Every stage lives on a different node.
    assert len({node for node, _, _ in layout}) == len(layout)

    # 2. Only the head is routable; the pipeline is reachable end-to-end.
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 90), (
        "head endpoint never registered"
    )
    endpoints = controller.endpoints.candidates(MODEL_ID)
    assert len(endpoints) == 1, "only stage 0 may expose a client endpoint"
    head_node = layout[0][0]
    assert endpoints[0].node_id == head_node

    # 3. The pipeline produces the same text as the single-process model.
    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": PROMPT}],
            },
        )

    resp = orch.call_api(call, timeout=180)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert content == reference_completion, (
        f"pipeline output {content!r} != single-process {reference_completion!r}"
    )
    assert body["usage"]["completion_tokens"] == 8


def test_streaming_through_the_pipeline(pipeline_stack, reference_completion):
    orch, _ = pipeline_stack
    controller = orch.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 90)

    async def call(api):
        chunks = []
        async with api.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": PROMPT}],
            },
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                chunks.append(line)
        return chunks

    lines = orch.call_api(call, timeout=180)
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]
    assert data_lines, "no SSE chunks received"
    pieces = []
    for line in data_lines:
        payload = json.loads(line[len("data: ") :])
        pieces.append(payload["choices"][0]["delta"].get("content", ""))
    assert "".join(pieces) == reference_completion
    assert [l for l in lines if "[DONE]" in l], "stream not terminated"


def test_stage_routes_are_published_and_cleaned(pipeline_stack):
    orch, workers = pipeline_stack
    controller = orch.controller
    assert wait_until(lambda: len(stage_layout(controller)) >= 2, 60)
    layout = stage_layout(controller)

    # The hub knows which node serves each stage — that is what makes the
    # activations routable without workers talking to each other.
    # Derived from the live deployment, so it cannot disagree with the layout
    # above (it used to: a broker pass that ended early left this empty while
    # the stages kept serving).
    pipeline_ids = controller.model_pipelines.get(MODEL_ID, [])
    assert pipeline_ids, f"stages are deployed but no pipeline is named: {layout}"
    stages = controller.tunnel.pipeline_stages(pipeline_ids[0])
    assert set(stages) == set(range(len(layout)))
    assert len(set(stages.values())) == len(layout)

    # Losing a stage tears the pipeline down (no half-broken chain serving).
    before = len(stage_layout(controller))
    workers[-1].stop()
    assert wait_until(
        lambda: not controller.endpoints.candidates(MODEL_ID)
        or len(stage_layout(controller)) != before,
        60,
    )


def test_response_carries_generation_timings(pipeline_stack, reference_completion):
    """The head measures what only it can see: prefill, decode, per-hop cost.

    A client can time the round trip, but not tell prefill from decode, and on
    a pipeline the round trip also hides the tunnel. These numbers are what the
    admin console shows as generation stats.
    """
    orch, _ = pipeline_stack
    assert wait_until(lambda: bool(orch.controller.endpoints.candidates(MODEL_ID)), 90)

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 8,
            },
        )

    body = orch.call_api(call).json()
    t = body["timings"]
    assert t["stages"] == len(stage_layout(orch.controller))
    assert t["ttft_ms"] > 0 and t["total_ms"] >= t["ttft_ms"]
    assert t["decode_tokens_per_s"] > 0
    assert t["inter_token_ms_p50"] >= 0
    assert body["usage"]["completion_tokens"] == 8


def test_stream_closes_with_usage_and_timings(pipeline_stack, reference_completion):
    """Streaming clients must not have to count chunks to know the token cost."""
    orch, _ = pipeline_stack
    assert wait_until(lambda: bool(orch.controller.endpoints.candidates(MODEL_ID)), 90)

    async def call(api):
        async with api.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 8,
                "stream": True,
            },
        ) as resp:
            return [line async for line in resp.aiter_lines()]

    lines = orch.call_api(call)
    frames = [
        json.loads(l[len("data:"):].strip())
        for l in lines
        if l.startswith("data:") and "[DONE]" not in l
    ]
    closing = frames[-1]
    assert closing["usage"]["completion_tokens"] == 8
    assert closing["timings"]["stages"] >= 1
    assert closing["choices"][0]["finish_reason"]


def test_timings_split_a_token_into_compute_and_transport(pipeline_stack, reference_completion):
    """Every stage reports its own duration; the head derives the wire time.

    This is what turns "the pipeline got slower on separate machines" into an
    actionable number: if transport dominates, no runtime work will help.
    Durations are used rather than timestamps on purpose — the stages run on
    different hosts, and comparing their clocks would put the measurement
    error inside the measurement.
    """
    orch, _ = pipeline_stack
    assert wait_until(lambda: bool(orch.controller.endpoints.candidates(MODEL_ID)), 90)

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 12,
            },
        )

    t = orch.call_api(call).json()["timings"]

    for key in (
        "head_compute_ms_p50",
        "peer_compute_ms_p50",
        "transport_ms_p50",
        "transport_share",
    ):
        assert key in t, f"{key} missing from timings"

    assert t["head_compute_ms_p50"] > 0, "the head does run layers"
    assert t["peer_compute_ms_p50"] > 0, "a two-stage pipeline has a second stage"
    assert 0.0 <= t["transport_share"] <= 1.0

    # The parts must add up to the interval the head measured independently.
    parts = t["head_compute_ms_p50"] + t["peer_compute_ms_p50"] + t["transport_ms_p50"]
    assert parts == pytest.approx(t["inter_token_ms_p50"], rel=0.35), (
        f"split {parts:.2f} ms does not reconstruct the measured "
        f"{t['inter_token_ms_p50']:.2f} ms"
    )


def test_single_stage_reports_no_peer_time(pipeline_stack, reference_completion):
    """With one stage there is nobody to talk to, so transport must not appear."""
    from loom_worker.shard.server import _latency_split

    split = _latency_split([5.0, 4.0, 4.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert split["peer_compute_ms_p50"] == 0.0
    assert split["transport_ms_p50"] == 0.0
    assert split["transport_share"] == 0.0
    # Prefill (the first sample) is excluded: it is one long step already
    # reported as ttft, and it would drag every percentile.
    assert split["head_compute_ms_mean"] == pytest.approx(4.25)


def test_timings_carry_the_per_token_series(pipeline_stack, reference_completion):
    """Percentiles cannot show a rate drifting, stalling, or one stage lagging.

    The summary answers "what did a token cost on average"; a benchmark asks
    "when did it change". Both come from the same measurements, so the series
    is the same numbers unsummarised — and it is what the admin UI plots and
    exports for someone to overlay two runs elsewhere.
    """
    orch, _ = pipeline_stack
    assert wait_until(lambda: bool(orch.controller.endpoints.candidates(MODEL_ID)), 90)
    want = 12

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": want,
            },
        )

    t = orch.call_api(call).json()["timings"]
    series = t["series"]

    # Parallel arrays, index-aligned, one point per token.
    lengths = {key: len(series[key]) for key in
               ("t_ms", "gap_ms", "head_ms", "peer_ms", "wire_ms")}
    assert len(set(lengths.values())) == 1, f"ragged series: {lengths}"
    assert 1 < len(series["t_ms"]) <= want
    assert series["tokens"] == len(series["t_ms"]), "no thinning at this length"
    assert series["every_n_tokens"] == 1

    # Index 0 is the prefill, so its gap IS the time to first token.
    assert series["gap_ms"][0] == pytest.approx(t["ttft_ms"], rel=0.02)
    # Time only moves forward, and the last point is the end of the request.
    assert series["t_ms"] == sorted(series["t_ms"])
    assert series["t_ms"][-1] == pytest.approx(t["total_ms"], rel=0.02)

    # Each decode point reconstructs its own interval from its own parts —
    # the same identity the percentiles satisfy, held token by token.
    for i in range(1, len(series["t_ms"])):
        parts = series["head_ms"][i] + series["peer_ms"][i] + series["wire_ms"][i]
        assert parts == pytest.approx(series["gap_ms"][i], rel=0.35, abs=2.0), (
            f"token {i}: split {parts:.2f} ms does not reconstruct "
            f"{series['gap_ms'][i]:.2f} ms"
        )


def test_a_long_run_is_thinned_instead_of_growing_without_bound(monkeypatch):
    """One closing chunk must stay something the tunnel will carry."""
    from loom_worker.shard import server as stage_server

    monkeypatch.setattr(stage_server, "SERIES_MAX_POINTS", 10)
    token_times = [1.0 + 0.1 * i for i in range(250)]
    per_step = [5.0] * 250
    series = stage_server._timing_series(
        1.0, token_times, per_step, per_step, per_step
    )
    assert series["tokens"] == 250
    assert len(series["t_ms"]) <= 10
    assert series["every_n_tokens"] == 25, "the reader must know a point spans 25 tokens"
    # Thinned gaps still describe the span between the points that remain.
    assert series["gap_ms"][1] == pytest.approx(2500.0, rel=0.01)


def test_an_empty_run_has_no_series():
    from loom_worker.shard.server import _timing_series

    assert _timing_series(1.0, [], [], [], []) == {}
