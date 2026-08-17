"""Production-stand pieces: join keys, hardware auto-detection, data-plane tunnel.

The tunnel is exercised implicitly by every e2e test (endpoints are now
tunnel://), so here we assert the properties that make a real deployment work:
no advertised address, streaming over the tunnel, and failure when the tunnel
is absent.
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker import hwinfo  # noqa: E402
from loom_worker.joinkey import parse_join_key  # noqa: E402

from loom.api.app import create_app  # noqa: E402
from loom.orchestrator.keys import KeyStore, decode_key  # noqa: E402
from loom.orchestrator.registry import ModelRegistry, ModelSpec  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
MODEL_ID = "qwen3-0.6b"


def one_model_registry() -> ModelRegistry:
    raw = json.loads((CONFIGS / "demo-echo.json").read_text())
    return ModelRegistry([ModelSpec.from_dict(raw)])


# ------------------------------------------------------------------ join keys
def test_key_roundtrip_carries_address_and_secret():
    store = KeyStore(public_address="orch.example.com:9000")
    key = store.issue(label="gpu-box-1")
    encoded = key.encode()
    assert encoded.startswith("loom_")

    # Orchestrator side.
    parsed = decode_key(encoded)
    assert parsed["address"] == "orch.example.com:9000"
    assert store.validate(encoded, node_id="n1") == key.secret

    # Worker side parses the same string without importing orchestrator code.
    worker_view = parse_join_key(encoded)
    assert worker_view.address == "orch.example.com:9000"
    assert worker_view.secret == key.secret


def test_key_validation_rejects_tampered_revoked_and_garbage():
    store = KeyStore(public_address="host:9000")
    key = store.issue()
    encoded = key.encode()

    assert store.validate("loom_not-base64!") is None
    assert store.validate("plain-token") is None
    assert store.validate(encoded[:-4] + "AAAA") is None  # wrong secret
    assert store.validate(encoded) == key.secret
    assert store.revoke(key.key_id)
    assert store.validate(encoded) is None
    assert store.revoke("no-such-id") is False


def test_key_max_nodes_limit():
    store = KeyStore(public_address="host:9000")
    key = store.issue(max_nodes=2)
    encoded = key.encode()
    assert store.validate(encoded, node_id="a")
    assert store.validate(encoded, node_id="b")
    assert store.validate(encoded, node_id="c") is None  # third machine refused
    assert store.validate(encoded, node_id="a")  # already-known node still ok


def test_keystore_persists_across_restart(tmp_path):
    path = tmp_path / "keys.json"
    store = KeyStore(public_address="host:9000", path=path)
    encoded = store.issue(label="persisted").encode()
    assert path.exists()

    reopened = KeyStore(public_address="host:9000", path=path)
    assert reopened.validate(encoded) is not None
    assert [k.label for k in reopened.list()] == ["persisted"]
    assert not reopened.open_registration()  # keys exist -> registration closed


def test_open_registration_only_without_keys_or_token():
    assert KeyStore(public_address="h:1").open_registration() is True
    assert KeyStore(public_address="h:1", master_token="t").open_registration() is False
    store = KeyStore(public_address="h:1")
    store.issue()
    assert store.open_registration() is False


# ------------------------------------------------- hardware auto-detection
def test_gpu_db_matching():
    assert hwinfo.match_gpu_specs("NVIDIA A100-SXM4-80GB", 80)["tflops_fp16"] == 312.0
    assert hwinfo.match_gpu_specs("NVIDIA A100-PCIE-40GB", 40)["bandwidth_gbps"] == 1935.0
    assert hwinfo.match_gpu_specs("NVIDIA H100 80GB HBM3", 80)["tflops_fp16"] == 989.0
    assert hwinfo.match_gpu_specs("NVIDIA GeForce RTX 4090", 24)["tflops_fp16"] == 82.6
    # Unknown card must not crash the agent — conservative estimate instead.
    unknown = hwinfo.match_gpu_specs("Some Future GPU 9000", 48)
    assert unknown == hwinfo._FALLBACK_GPU


def test_detect_nvidia_via_nvml_path(monkeypatch):
    """NVML is the primary source; torch/nvidia-smi are fallbacks."""
    monkeypatch.setattr(
        hwinfo,
        "_nvidia_via_nvml",
        lambda: (2, "NVIDIA A100-SXM4-80GB", 80 * hwinfo.GIB, 60 * hwinfo.GIB),
    )
    monkeypatch.delenv("LOOM_DEVICE", raising=False)
    monkeypatch.delenv("LOOM_MEMORY_GB", raising=False)
    hw = hwinfo.detect_hardware()
    assert hw.device == "cuda"
    assert hw.num_gpus == 2
    assert hw.tflops_fp16 == 312.0
    assert hw.vram_total_bytes == 80 * hwinfo.GIB
    # Schedulable memory is what is FREE now, not the card's total.
    assert hw.memory_gb == pytest.approx(60.0, abs=0.1)
    assert hw.detection_source == "nvml"


def test_detect_falls_back_to_smi(monkeypatch):
    monkeypatch.setattr(hwinfo, "_nvidia_via_nvml", lambda: None)
    monkeypatch.setattr(hwinfo, "_nvidia_via_torch", lambda: None)
    monkeypatch.setattr(
        hwinfo,
        "_nvidia_via_smi",
        lambda: (1, "NVIDIA L40S", 48 * hwinfo.GIB, 48 * hwinfo.GIB),
    )
    monkeypatch.delenv("LOOM_DEVICE", raising=False)
    hw = hwinfo.detect_hardware()
    assert hw.detection_source == "nvidia-smi"
    assert hw.gpu_name == "NVIDIA L40S"


def test_env_overrides_detection(monkeypatch):
    monkeypatch.setattr(hwinfo, "_nvidia_via_nvml", lambda: None)
    monkeypatch.setattr(hwinfo, "_nvidia_via_torch", lambda: None)
    monkeypatch.setattr(hwinfo, "_nvidia_via_smi", lambda: None)
    monkeypatch.setenv("LOOM_DEVICE", "cuda")
    monkeypatch.setenv("LOOM_MEMORY_GB", "12")
    monkeypatch.setenv("LOOM_TFLOPS_FP16", "77")
    hw = hwinfo.detect_hardware()
    assert (hw.device, hw.memory_gb, hw.tflops_fp16) == ("cuda", 12.0, 77.0)
    assert hw.vram_free_bytes == 12 * hwinfo.GIB
    assert hw.detection_source == "env"


# ------------------------------------------------------------ data-plane tunnel
@pytest.fixture
def stack():
    orch = OrchestratorHarness(one_model_registry()).start()
    worker = WorkerHarness(
        orch.grpc_port, join_key=orch.join_key, node_id="tunnel-node", memory_gb=8
    ).start()
    try:
        yield orch, worker
    finally:
        worker.stop()
        orch.stop()


def test_inference_flows_through_tunnel_without_any_advertised_address(stack):
    orch, worker = stack
    controller = orch.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 30)

    # The endpoint is a routing handle, not a dialable address: nothing about
    # the worker's network location is known to the orchestrator.
    endpoint = controller.endpoints.candidates(MODEL_ID)[0]
    assert endpoint.base_url.startswith("tunnel://tunnel-node:")
    assert wait_until(lambda: controller.tunnel.is_connected("tunnel-node"), 15)
    assert controller.nodes_view()["nodes"]["tunnel-node"]["tunnel"] is True

    async def call(api):
        plain = await api.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": [{"role": "user", "content": "via tunnel"}]},
        )
        lines = []
        async with api.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "stream via tunnel"}],
            },
        ) as resp:
            async for line in resp.aiter_lines():
                lines.append(line)
        return plain, lines

    plain, lines = orch.call_api(call)
    assert plain.status_code == 200
    assert "via tunnel" in plain.json()["choices"][0]["message"]["content"]
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert data_lines and data_lines[-1] == "data: [DONE]"


def test_request_fails_cleanly_when_tunnel_is_gone(stack):
    orch, worker = stack
    controller = orch.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 30)

    # Drop only the data plane; the control plane stays up.
    worker.dataplane.stop()
    assert wait_until(lambda: not controller.tunnel.is_connected("tunnel-node"), 15)

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": [{"role": "user", "content": "x"}]},
        )

    resp = orch.call_api(call)
    assert resp.status_code == 502
    assert "unreachable" in resp.json()["error"]["message"]


def test_admin_key_endpoints(stack):
    orch, _ = stack
    async def call(api):
        issued = await api.post("/admin/keys", json={"label": "box-2", "max_nodes": 1})
        listed = await api.get("/admin/keys")
        key_id = issued.json()["key_id"]
        revoked = await api.delete(f"/admin/keys/{key_id}")
        missing = await api.delete("/admin/keys/nope")
        return issued, listed, revoked, missing

    issued, listed, revoked, missing = orch.call_api(call)
    assert issued.status_code == 200
    body = issued.json()
    assert body["key"].startswith("loom_")
    assert "docker run" in body["run_command"] and body["key"] in body["run_command"]
    assert body["address"] == orch.config.public_address
    # Secrets are never listed back.
    assert all("secret" not in k for k in listed.json()["keys"])
    assert revoked.status_code == 200
    assert missing.status_code == 404


def test_endpoint_resyncs_from_telemetry_after_orchestrator_restart(stack):
    """An orchestrator that lost its endpoint table recovers from heartbeats.

    Simulates a restart: wipe endpoints + deployed state while the worker keeps
    serving. The next telemetry must restore routing (and inference must work)
    without any new StartServing.
    """
    orch, worker = stack
    controller = orch.controller
    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 30)

    controller.endpoints.unregister(model_id=MODEL_ID)
    controller.deployed.clear()
    assert not controller.endpoints.candidates(MODEL_ID)

    assert wait_until(lambda: bool(controller.endpoints.candidates(MODEL_ID)), 20), (
        "endpoint was not re-synced from telemetry"
    )

    async def call(api):
        return await api.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": [{"role": "user", "content": "resynced"}]},
        )

    resp = orch.call_api(call)
    assert resp.status_code == 200
    assert "resynced" in resp.json()["choices"][0]["message"]["content"]
