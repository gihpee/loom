"""Operator-facing flow: deploy a model by HF name, see the dial address.

Covers what an operator does in the admin UI without touching JSON or IPs.
"""

import json
from pathlib import Path

import pytest
from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

from loom.orchestrator import public_addr
from loom.orchestrator.model_resolver import (
    ModelResolveError,
    model_info_from_hf_config,
)
from loom.orchestrator.registry import ModelRegistry, ModelSpec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

QWEN3_8B_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


# ------------------------------------------------- HF config -> ModelInfo
def test_maps_hf_config_to_model_info():
    info = model_info_from_hf_config(QWEN3_8B_CONFIG, model_name="Qwen/Qwen3-8B")
    assert (info.num_layers, info.hidden_dim, info.num_kv_heads) == (36, 4096, 8)
    assert info.head_size == 128
    assert info.param_bytes_per_element == 2  # bfloat16
    assert info.tie_embedding is False
    # Sanity: the derived weight footprint is in the right ballpark (~15 GB).
    weights = (
        info.num_layers * info.decoder_layer_io_bytes(roofline=False)
        + 2 * info.embedding_io_bytes
    )
    assert 13 * 1024**3 < weights < 18 * 1024**3


def test_infers_head_dim_and_kv_heads_when_absent():
    cfg = dict(QWEN3_8B_CONFIG)
    del cfg["head_dim"]
    del cfg["num_key_value_heads"]
    info = model_info_from_hf_config(cfg, model_name="m")
    assert info.head_size == 4096 // 32
    assert info.num_kv_heads == 32  # falls back to attention heads (MHA)


def test_handles_moe_and_multimodal_wrapper():
    moe = dict(QWEN3_8B_CONFIG)
    moe.update(num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
    info = model_info_from_hf_config(moe, model_name="moe")
    assert info.num_local_experts == 128 and info.num_experts_per_tok == 8

    wrapped = {"text_config": QWEN3_8B_CONFIG, "architectures": ["SomeVLM"]}
    assert model_info_from_hf_config(wrapped, model_name="vlm").num_layers == 36


def test_rejects_unsupported_config():
    with pytest.raises(ModelResolveError):
        model_info_from_hf_config({"hidden_size": 8}, model_name="broken")


# ------------------------------------------------- dial address detection
def test_explicit_address_wins(monkeypatch):
    monkeypatch.setenv("LOOM_PUBLIC_ADDR", "loom.example.com:9000")
    addr = public_addr.resolve_public_address(9000)
    assert addr.address == "loom.example.com:9000"
    assert addr.source == "env"
    assert addr.reachable_externally is True
    assert addr.warning is None


def test_private_explicit_address_warns(monkeypatch):
    monkeypatch.setenv("LOOM_PUBLIC_ADDR", "192.168.1.5:9000")
    addr = public_addr.resolve_public_address(9000)
    assert addr.reachable_externally is False
    assert "cannot reach" in addr.warning


def test_tunnel_is_used_when_present(monkeypatch):
    """With no env var, a co-located tunnel supplies the public endpoint."""
    monkeypatch.delenv("LOOM_PUBLIC_ADDR", raising=False)
    monkeypatch.setattr(public_addr, "_from_ngrok", lambda port: "5.tcp.ngrok.io:12345")
    addr = public_addr.resolve_public_address(9000)
    assert addr.address == "5.tcp.ngrok.io:12345"
    assert addr.source == "ngrok"
    assert addr.reachable_externally is True


def test_loopback_fallback_warns(monkeypatch):
    monkeypatch.delenv("LOOM_PUBLIC_ADDR", raising=False)
    monkeypatch.setattr(public_addr, "_from_ngrok", lambda port: None)
    monkeypatch.setattr(public_addr, "_host_ip", lambda: None)
    addr = public_addr.resolve_public_address(9000)
    assert addr.address == "127.0.0.1:9000"
    assert addr.reachable_externally is False
    assert addr.warning


# ------------------------------------------------- admin API surface
@pytest.fixture
def stack():
    registry = ModelRegistry.from_catalog_file(CONFIGS / "catalog-demo.json")
    orch = OrchestratorHarness(registry).start()
    worker = WorkerHarness(
        orch.grpc_port, join_key=orch.join_key, node_id="deploy-node", memory_gb=6
    ).start()
    try:
        yield orch
    finally:
        worker.stop()
        orch.stop()


def test_connect_endpoint_gives_address_and_image(stack):
    async def call(api):
        return await api.get("/admin/connect")

    body = stack.call_api(call).json()
    assert body["dial_address"] == stack.config.public_address
    assert body["worker_image"] == "gihpee/loomworker"


def test_deploy_from_hf_is_offline_safe(stack, monkeypatch):
    """The endpoint resolves architecture from config.json; stub the fetch."""
    import loom.api.app as app_module

    async def fake_spec_from_hf(repo, **kwargs):
        info = model_info_from_hf_config(QWEN3_8B_CONFIG, model_name=repo)
        spec = ModelSpec(
            model_id=kwargs.get("model_id") or "qwen3-8b",
            weights_uri=repo,
            backend_type=kwargs.get("backend_type", "shard"),
            model_info=info,
            priority=kwargs.get("priority", 1.0),
            target_pipelines=kwargs.get("target_pipelines", 1),
        )
        return spec, QWEN3_8B_CONFIG

    monkeypatch.setattr(app_module, "spec_from_hf", fake_spec_from_hf)

    async def call(api):
        return await api.post(
            "/admin/models/from_hf",
            json={"repo": "Qwen/Qwen3-8B", "backend_type": "shard", "priority": 3},
        )

    resp = stack.call_api(call)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["added"] == "qwen3-8b"
    assert body["detected"]["num_layers"] == 36
    assert body["sizing"]["weights_gb"] > 10
    # Layer-based sizing is what tells the operator how many cards are needed.
    assert body["sizing"]["layers_per_card"]["24GB"] > 0
    assert stack.controller.registry.get("qwen3-8b") is not None


def test_deploy_from_hf_requires_repo(stack):
    async def call(api):
        return await api.post("/admin/models/from_hf", json={})

    resp = stack.call_api(call)
    assert resp.status_code == 400
    assert "repo" in resp.json()["error"]["message"]


def test_ui_exposes_deploy_and_connect():
    html = (Path(__file__).resolve().parent.parent / "src/loom/api/admin_ui.html").read_text()
    for needle in (
        "/admin/models/from_hf",
        "/admin/connect",
        "deployModel",
        "removeModel",
        "docker run -d --gpus all",
    ):
        assert needle in html, needle
