"""Admin UI endpoints: read-only views + force rebalance + page serving."""

import json
from pathlib import Path

import pytest
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

from loom.orchestrator.registry import ModelRegistry, ModelSpec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def stack():
    registry = ModelRegistry.from_catalog_file(CONFIGS / "catalog-demo.json")
    orch = OrchestratorHarness(registry).start()
    workers = [
        WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id=f"ui-w{i}", memory_gb=3).start() for i in range(2)
    ]
    try:
        yield orch
    finally:
        for w in workers:
            w.stop()
        orch.stop()


def call(orch, method, path, **kwargs):
    """Issue an admin request on the orchestrator's loop."""

    async def go(api):
        return await getattr(api, method)(path, **kwargs)

    return orch.call_api(go)


def test_admin_views_and_rebalance(stack):
    controller = stack.controller
    assert wait_until(
        lambda: bool(controller.endpoints.candidates("demo-model-a"))
        and bool(controller.endpoints.candidates("demo-model-b")),
        40,
    )

    page = call(stack, "get", "/admin/ui")
    assert page.status_code == 200 and "loom admin" in page.text

    nodes = call(stack, "get", "/admin/nodes").json()["nodes"]
    assert set(nodes) == {"ui-w0", "ui-w1"}
    for n in nodes.values():
        assert n["connected"] is True
        assert n["vram_declared_gb"] == 3.0
    # Heartbeats flow every 1s in the harness; shard statuses arrive with them.
    assert wait_until(
        lambda: any(
            s["status"] == "serving"
            for n in call(stack, "get", "/admin/nodes").json()["nodes"].values()
            for s in n["shards"]
        ),
        15,
    )

    models = call(stack, "get", "/admin/models_view").json()["models"]
    assert set(models) == {"demo-model-a", "demo-model-b"}
    for m in models.values():
        assert m["k_actual"] >= 1
        assert m["placement"] and m["endpoints"]

    pm = call(stack, "get", "/admin/perfmap/demo-model-a").json()
    assert pm["route_preview"]["path"], "phase-2 preview should find a chain"
    assert pm["tau_effective"]
    assert call(stack, "get", "/admin/perfmap/nope").status_code == 404

    r = call(stack, "post", "/admin/rebalance")
    assert r.status_code == 200 and r.json()["ok"]
    assert set(r.json()["allocations"]) == {"demo-model-a", "demo-model-b"}


def test_admin_views_respect_token(stack):
    controller = stack.controller
    controller.config.admin_token = "s3cret"
    assert call(stack, "get", "/admin/nodes").status_code == 403
    assert call(stack, "post", "/admin/rebalance").status_code == 403
    ok = call(stack, "get", "/admin/nodes", headers={"X-Loom-Admin-Token": "s3cret"})
    assert ok.status_code == 200
    # The page itself is served without a token; it holds no data.
    assert call(stack, "get", "/admin/ui").status_code == 200
