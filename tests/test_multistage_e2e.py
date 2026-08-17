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
