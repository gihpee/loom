"""Model-aware EndpointRegistry: requests match endpoints of the SAME model only."""

import pytest

from loom.api.endpoints import EndpointRegistry


def test_model_namespacing():
    reg = EndpointRegistry()
    reg.register(model_id="m1", base_url="http://w0:8100", node_id="w0")
    reg.register(model_id="m2", base_url="http://w0:8200", node_id="w0")
    reg.register(model_id="m1", base_url="http://w1:8100", node_id="w1")

    assert {ep.base_url for ep in reg.candidates("m1")} == {
        "http://w0:8100",
        "http://w1:8100",
    }
    assert [ep.base_url for ep in reg.candidates("m2")] == ["http://w0:8200"]
    assert reg.select("m2").model_id == "m2"
    assert reg.select("unknown") is None
    assert reg.list_models() == ["m1", "m2"]


def test_unregister_by_node_and_model():
    reg = EndpointRegistry()
    reg.register(model_id="m1", base_url="http://w0:8100", node_id="w0")
    reg.register(model_id="m1", base_url="http://w1:8100", node_id="w1")
    reg.register(model_id="m2", base_url="http://w0:8200", node_id="w0")

    assert reg.unregister(model_id="m1", node_id="w0") == 1
    assert [ep.node_id for ep in reg.candidates("m1")] == ["w1"]
    # m2 on the same node untouched.
    assert len(reg.candidates("m2")) == 1
    assert reg.unregister(model_id="m2") == 1
    assert reg.candidates("m2") == []


def test_round_robin_prefers_less_loaded():
    reg = EndpointRegistry(strategy="round_robin")
    a = reg.register(model_id="m", base_url="http://a:1", node_id="a")
    b = reg.register(model_id="m", base_url="http://b:1", node_id="b")
    reg.mark_request_start(a)  # a becomes loaded
    assert reg.select("m") is b


def test_metrics_lifecycle():
    reg = EndpointRegistry()
    ep = reg.register(model_id="m", base_url="http://a:1")
    reg.mark_request_start(ep)
    assert ep.metrics.inflight == 1 and ep.metrics.total_requests == 1
    reg.mark_request_end(ep, error=True, ttft_ms=100.0)
    assert ep.metrics.inflight == 0
    assert ep.metrics.total_errors == 1
    assert ep.metrics.ema_ttft_ms == pytest.approx(100.0)
    reg.mark_request_end(ep)  # never below zero
    assert ep.metrics.inflight == 0


def test_register_idempotent():
    reg = EndpointRegistry()
    e1 = reg.register(model_id="m", base_url="http://a:1/")
    e2 = reg.register(model_id="m", base_url="http://a:1")
    assert e1 is e2
    assert len(reg.candidates("m")) == 1


def test_rejects_bad_url():
    reg = EndpointRegistry()
    with pytest.raises(ValueError):
        reg.register(model_id="m", base_url="w0:8100")
