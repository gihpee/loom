"""Orchestrator side of the launch-storm fix (see test_backend_lifecycle.py).

Two guards are pinned here:
  - a `failed` heartbeat that arrives while a deploy is still in flight is
    stale news about the PREVIOUS attempt and must not trigger re-placement;
  - a placement that failed is left alone for a cooldown instead of being
    retried on every broker pass — a retry costs a full checkpoint download.
"""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from loom.orchestrator.config import OrchestratorConfig
from loom.orchestrator.controller import MultiModelController
from loom.orchestrator.placement import Placement
from loom.orchestrator.registry import ModelRegistry

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GIB = 1024**3


def make_controller(**overrides):
    registry = ModelRegistry.from_catalog_file(CONFIGS / "catalog-demo.json")
    config = OrchestratorConfig(registry=registry, rebalance_interval_s=3600.0, **overrides)
    controller = MultiModelController(config)
    # These tests are about the retry cooldown, not about placement. A catalog
    # entry no longer deploys itself (see test_placement.py), so they opt into
    # brokered placement explicitly and carry on.
    for spec in registry.list():
        controller.placements[spec.model_id] = Placement.auto(spec.model_id)
    return controller


def telemetry(model_id: str, status: str):
    return SimpleNamespace(
        shards=[
            SimpleNamespace(
                model_id=model_id,
                start_layer=0,
                end_layer=24,
                current_requests=0,
                healthy=status == "serving",
                status=status,
                local_port=0,
                pipeline_id=f"{model_id}#0",
                stage_index=0,
                num_stages=1,
                avg_layer_latency_ms=0.0,
            )
        ]
    )


def session(node_id="n1"):
    return SimpleNamespace(node_id=node_id)


@pytest.mark.parametrize("elapsed,expect_replacement", [(0.0, False), (999.0, True)])
def test_failed_heartbeat_is_ignored_while_the_start_is_in_flight(elapsed, expect_replacement):
    ctrl = make_controller(deploy_grace_s=60.0)
    model_id = next(iter(ctrl.registry.ids()))
    key = (model_id, "n1")
    ctrl.deployed[key] = (4 * GIB, 0, 24, f"{model_id}#0", 0, 1)
    ctrl.deploy_started[key] = time.time() - elapsed

    rebalances = []
    ctrl.rebalance = lambda reason: rebalances.append(reason) or asyncio.sleep(0)
    asyncio.run(ctrl.on_telemetry(session(), telemetry(model_id, "failed")))

    assert (key not in ctrl.deployed) is expect_replacement
    assert bool(rebalances) is expect_replacement


def test_failed_placement_waits_out_its_cooldown():
    ctrl = make_controller(deploy_retry_s=300.0)
    model_id = next(iter(ctrl.registry.ids()))
    key = (model_id, "n1")

    deploys = []
    ctrl._deploy_on_worker = lambda m, n, e: deploys.append((m, n)) or asyncio.sleep(0)
    ctrl._mark_deploy_failed(model_id, "n1", "Free memory on device cuda:0 is not enough")

    layers = ctrl.registry.get(model_id).model_info.num_layers

    async def one_pass():
        # Pretend the broker wants this exact placement again.
        ctrl.broker.plan = lambda *a, **k: SimpleNamespace(
            allocations={model_id: {"n1": 4 * GIB}}, unscheduled=[]
        )
        ctrl.pool.rebuild = lambda *a, **k: None
        ctrl.pool.shard_plan = lambda mid: [("n1", 0, layers)]
        ctrl.pool.model_ids = lambda: []
        await ctrl.rebalance(reason="test")

    asyncio.run(one_pass())
    assert deploys == [], "re-deployed a placement that just failed"
    assert key not in ctrl.deployed

    # Cooldown elapsed -> the placement is attempted again.
    ctrl.deploy_failures[key] = (time.time() - 301.0, "old failure")
    asyncio.run(one_pass())
    assert deploys == [(model_id, "n1")]


def test_failure_reason_is_visible_in_the_models_view():
    ctrl = make_controller()
    model_id = next(iter(ctrl.registry.ids()))
    ctrl._mark_deploy_failed(model_id, "n1", "backend failed health check")
    view = ctrl.models_view()["models"][model_id]
    assert view["failures"][0]["node_id"] == "n1"
    assert view["failures"][0]["error"] == "backend failed health check"
    assert view["failures"][0]["retry_in_s"] > 0


def test_ui_renders_deploy_failures():
    html = (Path(__file__).resolve().parent.parent / "src/loom/api/admin_ui.html").read_text()
    assert "m.failures" in html
    assert "starting:" in html  # the new shard status has its own colour


def test_a_shard_that_was_serving_heals_without_waiting_out_the_grace():
    """The grace window covers unconfirmed starts only.

    Once a shard has reported `serving`, a later `failed` (watchdog kill, crash)
    is real: delaying re-placement by a minute would be a self-inflicted outage.
    """
    ctrl = make_controller(deploy_grace_s=60.0)
    model_id = next(iter(ctrl.registry.ids()))
    key = (model_id, "n1")
    ctrl.deployed[key] = (4 * GIB, 0, 24, f"{model_id}#0", 0, 1)
    ctrl.deploy_started[key] = time.time()  # just deployed...

    rebalances = []
    ctrl.rebalance = lambda reason: rebalances.append(reason) or asyncio.sleep(0)

    serving = telemetry(model_id, "serving")
    serving.shards[0].local_port = 41177
    asyncio.run(ctrl.on_telemetry(session(), serving))
    assert key not in ctrl.deploy_started, "a confirmed start is no longer in flight"

    asyncio.run(ctrl.on_telemetry(session(), telemetry(model_id, "failed")))
    assert key not in ctrl.deployed and rebalances


def test_the_endpoint_announcement_ends_the_grace_window():
    """A crash seconds after startup must heal at once.

    The worker announces its endpoint as soon as the backend answers /health —
    a whole heartbeat earlier than telemetry. Until this was honoured, a shard
    killed right after starting sat unusable for the length of the grace window.
    """
    ctrl = make_controller(deploy_grace_s=60.0)
    model_id = next(iter(ctrl.registry.ids()))
    key = (model_id, "n1")
    ctrl.deployed[key] = (4 * GIB, 0, 24, f"{model_id}#0", 0, 1)
    ctrl.deploy_started[key] = time.time()

    endpoint = SimpleNamespace(model_id=model_id, local_port=41177)
    asyncio.run(ctrl.on_endpoint(session(), endpoint))
    assert key not in ctrl.deploy_started

    rebalances = []
    ctrl.rebalance = lambda reason: rebalances.append(reason) or asyncio.sleep(0)
    asyncio.run(ctrl.on_telemetry(session(), telemetry(model_id, "failed")))
    assert key not in ctrl.deployed and rebalances


def test_a_failed_deploy_books_its_own_retry():
    """Recovery must not depend on an unrelated event happening later.

    A backend that dies on startup used to leave the model parked until some
    node joined or the periodic timer came round — in a pool that is otherwise
    idle, that is an outage nobody triggers the end of.
    """
    ctrl = make_controller(deploy_retry_s=0.1)
    model_id = next(iter(ctrl.registry.ids()))

    async def scenario():
        reasons = []

        async def fake_rebalance(reason):
            reasons.append(reason)

        ctrl.rebalance = fake_rebalance
        ctrl._mark_deploy_failed(model_id, "n1", "backend process exited")
        assert (model_id, "n1") in ctrl._retry_tasks
        await asyncio.sleep(1.0)
        return reasons

    reasons = asyncio.run(scenario())
    assert any("retry" in r for r in reasons), reasons
    assert not ctrl._retry_tasks, "retry task was not cleaned up"
