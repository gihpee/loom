"""Phase 2 deliverable e2e: 2+ models on a shared pool, then a high-priority
third model arrives and evicts the low-priority one.

Topology: two workers x 2.5 GB, one region. A demo model needs ~1.9 GB of VRAM
per replica (28 layers of Qwen3-0.6B shape at the default param/KV split), so
each node holds exactly one replica and the pool fits exactly two models.
demo-model-c (score 400) must displace demo-model-b (score 3) but not
demo-model-a (score 10).
"""

import json
from pathlib import Path

import pytest
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

from loom.orchestrator.registry import ModelRegistry, ModelSpec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def catalog_registry() -> ModelRegistry:
    return ModelRegistry.from_catalog_file(CONFIGS / "catalog-demo.json")


def model_c_spec() -> ModelSpec:
    return ModelSpec.from_dict(json.loads((CONFIGS / "demo-model-c.json").read_text()))


@pytest.fixture
def stack():
    orch = OrchestratorHarness(catalog_registry()).start()
    workers = [
        WorkerHarness(
            orch.grpc_port, join_key=orch.join_key, node_id=f"w{i}", memory_gb=2.5
        ).start()
        for i in range(2)
    ]
    try:
        yield orch, workers
    finally:
        for w in workers:
            w.stop()
        orch.stop()


def serving_models(controller):
    return {m for m in controller.registry.ids() if controller.endpoints.candidates(m)}


def ask(orch, model_id: str):
    """Send a chat request through the orchestrator's own loop (tunnel-safe)."""

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "ping"}]},
        )

    return orch.call_api(call)


def test_two_models_share_pool_then_high_priority_evicts_low(stack):
    orch, workers = stack
    controller = orch.controller

    # Stage 1: both catalog models serve on the shared pool.
    assert wait_until(lambda: serving_models(controller) == {"demo-model-a", "demo-model-b"}, 40), (
        f"expected both models serving, got {serving_models(controller)}"
    )
    assert ask(orch, "demo-model-a").status_code == 200
    assert ask(orch, "demo-model-b").status_code == 200
    # Models landed on distinct nodes of the shared pool.
    nodes_a = {ep.node_id for ep in controller.endpoints.candidates("demo-model-a")}
    nodes_b = {ep.node_id for ep in controller.endpoints.candidates("demo-model-b")}
    assert nodes_a and nodes_b and nodes_a.isdisjoint(nodes_b)

    # Stage 2: deploy high-priority model C via the admin path.
    orch.submit(controller.add_model(model_c_spec()))

    assert wait_until(
        lambda: serving_models(controller) == {"demo-model-a", "demo-model-c"}, 40
    ), f"expected b evicted and c serving, got {serving_models(controller)}"
    assert controller.last_plan is not None
    assert controller.last_plan.unscheduled == ["demo-model-b"]

    # API reflects the new state.
    resp_c = ask(orch, "demo-model-c")
    assert resp_c.status_code == 200
    assert "demo-model-c" in resp_c.json()["choices"][0]["message"]["content"]
    assert ask(orch, "demo-model-a").status_code == 200
    assert ask(orch, "demo-model-b").status_code == 503  # evicted, no capacity

    # Worker-side truth: model B's backend is actually stopped, not just hidden.
    def worker_serving():
        return {
            model_id
            for w in workers
            for model_id, shard in w.state.snapshot().items()
            if shard.status.value == "serving"
        }

    assert wait_until(lambda: "demo-model-b" not in worker_serving(), 20)
    assert {"demo-model-a", "demo-model-c"} <= worker_serving()


def test_remove_model_frees_capacity(stack):
    orch, _ = stack
    controller = orch.controller
    assert wait_until(lambda: serving_models(controller) == {"demo-model-a", "demo-model-b"}, 40)

    orch.submit(controller.remove_model("demo-model-a"))
    assert wait_until(lambda: "demo-model-a" not in controller.registry.ids(), 10)
    assert wait_until(lambda: not controller.endpoints.candidates("demo-model-a"), 20)
    # Remaining model still answers; removed one 404s.
    assert ask(orch, "demo-model-b").status_code == 200
    assert ask(orch, "demo-model-a").status_code == 404
