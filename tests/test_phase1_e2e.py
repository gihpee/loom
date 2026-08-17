"""Phase 1 e2e (no docker): worker registers over real gRPC, orchestrator plans
via Phase-1, pushes LoadShard/StartServing, worker spawns the echo backend
subprocess, and /v1/chat/completions answers through the API proxy.

(Rewritten in Phase 2 on top of MultiModelController with a one-model catalog.)
"""

import json
from pathlib import Path

import pytest
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

from loom.orchestrator.registry import ModelRegistry, ModelSpec

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "demo-echo.json"
MODEL_ID = "qwen3-0.6b"


def one_model_registry() -> ModelRegistry:
    return ModelRegistry([ModelSpec.from_dict(json.loads(CONFIG_PATH.read_text()))])


@pytest.fixture
def stack():
    orch = OrchestratorHarness(one_model_registry()).start()
    worker = WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id="test-worker", memory_gb=8).start()
    try:
        yield orch
    finally:
        worker.stop()
        orch.stop()


def test_full_flow_answers_chat_completion(stack):
    controller = stack.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID))), (
        "endpoint never registered"
    )
    instance = controller.pool.get(MODEL_ID)
    node = instance.scheduler.get_node("test-worker")
    assert (node.start_layer, node.end_layer) == (0, 28)  # full model, one stage

    async def call_api(api):
        health = await api.get("/healthz")
        models = await api.get("/v1/models")
        chat = await api.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": [{"role": "user", "content": "hello loom"}]},
        )
        # Single-model catalog: "model" may be omitted.
        implicit = await api.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "implicit"}]},
        )
        missing = await api.post(
            "/v1/chat/completions", json={"model": "no-such-model", "messages": []}
        )
        return health, models, chat, implicit, missing

    health, models, chat, implicit, missing = stack.call_api(call_api)
    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert models.json()["data"][0]["id"] == MODEL_ID
    assert chat.status_code == 200
    assert "hello loom" in chat.json()["choices"][0]["message"]["content"]
    assert implicit.status_code == 200
    assert missing.status_code == 404


def test_streaming_passthrough(stack):
    controller = stack.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)))
    async def call_stream(api):
        chunks = []
        async with api.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "stream me"}],
            },
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                chunks.append(line)
        return chunks

    lines = stack.call_api(call_stream)
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert any("stream" in l for l in data_lines)
    assert data_lines[-1] == "data: [DONE]"


def test_invalid_join_key_rejected():
    """A worker without a valid key never joins the pool."""
    orch = OrchestratorHarness(one_model_registry()).start()
    bogus = orch.join_key[:-6] + "AAAAAA"  # same shape, wrong secret
    worker = WorkerHarness(
        orch.grpc_port, join_key=bogus, node_id="bad-worker", memory_gb=8
    )
    worker.start()
    try:
        assert not worker.client.wait_registered(3.0)
        assert "bad-worker" not in orch.servicer.sessions
        assert "bad-worker" not in orch.controller.nodes
    finally:
        worker.stop()
        orch.stop()


def test_revoked_key_rejected():
    orch = OrchestratorHarness(one_model_registry()).start()
    key = orch.keystore.issue(label="temp")
    assert orch.keystore.revoke(key.key_id)
    worker = WorkerHarness(
        orch.grpc_port, join_key=key.encode(), node_id="revoked-worker", memory_gb=8
    )
    worker.start()
    try:
        assert not worker.client.wait_registered(3.0)
        assert "revoked-worker" not in orch.servicer.sessions
    finally:
        worker.stop()
        orch.stop()
