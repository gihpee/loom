"""Phase 0: a node that connects, says what it is, and stays connected."""

from __future__ import annotations

import threading
import time

import pytest

from conftest import make_join_key
from fake_orchestrator import FakeOrchestrator

from looma_agent.config import parse_args
from looma_agent.identity import BadJoinKey, parse_join_key
from looma_agent.main import Agent
from looma_agent.proto import agent_pb2


@pytest.fixture
def orchestrator():
    fake = FakeOrchestrator()
    fake.start()
    yield fake
    fake.stop()


@pytest.fixture
def agent(orchestrator, tmp_path):
    config = parse_args([
        "--key", make_join_key(f"127.0.0.1:{orchestrator.port}"),
        "--node-id", "test-node",
        "--root", str(tmp_path),
        "--heartbeat-interval", "0.2",
        "--reconnect-delay", "0.2",
    ])
    running = Agent(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    yield running
    running.stop()
    thread.join(timeout=5)


def test_registers_with_detected_hardware(orchestrator, agent):
    assert orchestrator.wait_registered(), "the node never registered"
    registration = orchestrator.registrations[0]
    assert registration.node_id == "test-node"
    assert registration.join_key.startswith("looma_")
    assert registration.agent_version
    # Hardware is detected, not declared: whatever this machine is, the field
    # must carry the source that answered rather than being left empty.
    assert registration.hardware.detection_source


def test_heartbeats_keep_arriving(orchestrator, agent):
    assert orchestrator.wait_registered()
    deadline = time.time() + 5
    while time.time() < deadline:
        if len(orchestrator.telemetry) >= 2:
            assert orchestrator.telemetry[0].node_id == "test-node"
            assert orchestrator.telemetry[0].reported_at_unix_ms > 0
            return
        time.sleep(0.1)
    pytest.fail("no telemetry arrived; the node would look dead to the orchestrator")


def test_reconnects_after_the_stream_dies(orchestrator, agent):
    """A stream that dies without a clean close must not end the node."""
    assert orchestrator.wait_registered()
    first = len(orchestrator.registrations)
    orchestrator.drop_next_stream()
    orchestrator.reset_registration_flag()
    # Force the live stream down so the next Attach is the dropped one.
    agent.client.stop()
    agent.client._stop.clear()
    threading.Thread(target=agent.client.run_forever, daemon=True).start()
    deadline = time.time() + 15
    while time.time() < deadline:
        if len(orchestrator.registrations) > first:
            return
        time.sleep(0.1)
    pytest.fail("the node did not re-register after the stream broke")


def test_the_node_says_whether_it_can_take_work(orchestrator, agent):
    """A node that cannot isolate a task declares it instead of failing each one."""
    assert orchestrator.wait_registered()
    readiness = orchestrator.registrations[0].readiness
    assert "python" in readiness.environment_kinds
    if not readiness.accepts_tasks:
        assert readiness.refusal, "a node that refuses work must say why"


# ------------------------------------------------------------------ identity
def test_join_key_carries_the_address():
    key = parse_join_key(make_join_key("10.0.0.1:50051"))
    assert key.address == "10.0.0.1:50051"
    assert key.secret == "secret"


@pytest.mark.parametrize("bad", ["", "nope", "looma_!!!!", "looma_"])
def test_bad_join_keys_say_what_to_do(bad):
    with pytest.raises(BadJoinKey) as exc:
        parse_join_key(bad)
    assert "looma_" in str(exc.value) or "damaged" in str(exc.value)
