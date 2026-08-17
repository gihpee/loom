"""Phase 3 e2e:

1. Watchdog deliverable: quota breach -> backend process killed, the worker
   agent survives (keeps heartbeating), and the orchestrator self-heals the
   model back to serving.
2. Heterogeneous pool: workers declaring cuda / mlx / cpu devices serve a
   multi-model catalog together (echo backends stand in for the engines —
   adapter command construction is covered by unit tests).

All commands in these stacks are HMAC-signed and verified by the workers
(stack_utils enables the verifier), so the whole path runs under Phase-3
security for free.
"""

import json
from pathlib import Path

from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

from loom.orchestrator.registry import ModelRegistry, ModelSpec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def registry_from(*names):
    specs = []
    for name in names:
        raw = json.loads((CONFIGS / name).read_text())
        entries = raw["models"] if isinstance(raw, dict) and "models" in raw else [raw]
        specs.extend(ModelSpec.from_dict(e) for e in entries)
    return ModelRegistry(specs)


def test_watchdog_kill_keeps_node_alive_and_self_heals():
    orch = OrchestratorHarness(registry_from("demo-echo.json")).start()
    # rss_overhead_bytes=0: enforce the quota strictly, as NVML does on CUDA.
    worker = WorkerHarness(
        orch.grpc_port,
        join_key=orch.join_key,
        node_id="wd-node",
        memory_gb=8,
        rss_overhead_bytes=0,
    ).start()
    controller = orch.controller
    try:
        assert wait_until(lambda: bool(controller.endpoints.candidates("qwen3-0.6b")), 30)
        first_port = controller.endpoints.candidates("qwen3-0.6b")[0].base_url

        # Deliberate quota breach: 1 byte. The RSS watchdog must kill the
        # backend subprocess.
        ack = orch.submit(controller.set_quota("qwen3-0.6b", "wd-node", 1))
        assert ack.ok

        assert wait_until(
            lambda: (shard := worker.state.get("qwen3-0.6b")) is not None
            and shard.status.value == "failed",
            20,
        ), "watchdog did not kill the over-quota backend"

        # The node itself is alive: still connected, still heartbeating.
        assert "wd-node" in orch.servicer.sessions
        assert "wd-node" in controller.nodes

        # Self-healing: failed telemetry -> re-placement -> serving again
        # (with the broker-granted quota, not the poisoned override).
        assert wait_until(
            lambda: (shard := worker.state.get("qwen3-0.6b")) is not None
            and shard.status.value == "serving",
            30,
        ), "model did not self-heal after watchdog kill"
        assert wait_until(lambda: bool(controller.endpoints.candidates("qwen3-0.6b")), 10)
        second_port = controller.endpoints.candidates("qwen3-0.6b")[0].base_url
        assert second_port != first_port  # genuinely restarted backend
    finally:
        worker.stop()
        orch.stop()


def test_heterogeneous_pool_serves_catalog():
    orch = OrchestratorHarness(registry_from("catalog-demo.json")).start()
    workers = [
        WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id="w-cuda", memory_gb=3, device="cuda").start(),
        WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id="w-mlx", memory_gb=3, device="mlx").start(),
        WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id="w-cpu", memory_gb=3, device="cpu").start(),
    ]
    controller = orch.controller
    try:
        assert wait_until(
            lambda: bool(controller.endpoints.candidates("demo-model-a"))
            and bool(controller.endpoints.candidates("demo-model-b")),
            40,
        )
        # Both models landed somewhere on the mixed pool; devices differ.
        used_nodes = {
            ep.node_id
            for m in ("demo-model-a", "demo-model-b")
            for ep in controller.endpoints.candidates(m)
        }
        devices = {controller.nodes[n].hardware.device for n in used_nodes}
        assert len(used_nodes) >= 2
        assert len(devices) >= 2, f"expected mixed devices, got {devices}"
    finally:
        for w in workers:
            w.stop()
        orch.stop()


def test_slo_violation_boosts_and_rebalances():
    orch = OrchestratorHarness(registry_from("catalog-demo.json")).start()
    controller = orch.controller
    spec = controller.registry.get("demo-model-b")
    spec.slo_p95_ttft_ms = 100.0
    try:
        # Feed slow samples for model b: p95 far above 100ms.
        for _ in range(20):
            controller.record_request("demo-model-b", ttft_ms=500.0, error=False)
        assert controller.slo_evaluate() is True
        assert controller.slo_boosts.get("demo-model-b") == controller.config.slo_boost_factor
        # Re-evaluation without recovery keeps the boost, no flapping.
        assert controller.slo_evaluate() is False

        # Recovery below 70% of SLO lifts the boost. Enough fast samples must
        # arrive to push p95 down while the slow ones are still in the window.
        for _ in range(400):
            controller.record_request("demo-model-b", ttft_ms=10.0, error=False)
        assert wait_until(lambda: controller.slo_evaluate(), 5)
        assert "demo-model-b" not in controller.slo_boosts
    finally:
        orch.stop()
