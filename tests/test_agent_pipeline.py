"""A job spread over several nodes, over one stream each.

This is the mechanism the inference pipeline will be built on: a task sends to
a RANK, and where that rank lives — this machine, a peer, the far side of the
orchestrator — is decided by the agent and never by the task.

The stage here holds no model on purpose. A failure in this file is a failure
of the transport, which is the thing being built.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from loom.orchestrator.agents import AgentError

from test_agent_gateway import Orchestrator  # noqa: E402

STAGE = str(Path(__file__).resolve().parent / "stage_fixture" / "pipeline_stage.py")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def two_nodes(tmp_path, monkeypatch):
    """Two agents on one orchestrator — the smallest real pipeline."""
    from conftest_agent import start_agent

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")   # force the path through the orchestrator
    orchestrator = Orchestrator().start()
    running = []
    for index in range(2):
        agent, thread = start_agent(orchestrator.port, tmp_path / f"node{index}",
                                    node_id=f"node-{index}")
        running.append((agent, thread))
    deadline = time.time() + 20
    while time.time() < deadline and len(orchestrator.hub.sessions) < 2:
        time.sleep(0.05)
    assert len(orchestrator.hub.sessions) == 2, "both agents should have attached"
    yield orchestrator
    for agent, thread in running:
        agent.stop()
        thread.join(timeout=10)
    orchestrator.stop()


def start_pipeline(orchestrator, size: int = 2, port: int = 0, spread: bool = True):
    """A pipeline across `size` nodes.

    Nodes are named rather than chosen when spreading, because a task with no
    resource requirements has no reason to go anywhere in particular — and
    stages on ONE node is the faster arrangement, not a mistake. Naming them is
    how this file tests the path between machines at all.
    """
    port = port or free_port()
    group = orchestrator.hub.submit_group(
        size=size,
        command=[sys.executable, STAGE],
        serve_port=port,
        timeout_s=120,
        node_ids=[f"node-{i}" for i in range(size)] if spread else None,
    )
    for rank in range(size):
        orchestrator.wait_state(group.tasks[rank], "running", timeout=60)
    return group, port


def test_a_group_is_placed_where_it_was_told(two_nodes):
    group, _port = start_pipeline(two_nodes)
    assert len(group.tasks) == 2
    assert set(group.nodes.values()) == {"node-0", "node-1"}


def test_a_group_with_nothing_to_spread_for_may_share_a_node(two_nodes):
    """Two stages on one machine cost no network at all. That is the good case,
    not a placement failure."""
    group = two_nodes.hub.submit_group(
        size=2, command=[sys.executable, "-c", "import time; time.sleep(3)"],
        timeout_s=60)
    assert len(set(group.nodes.values())) in (1, 2)
    assert len(group.tasks) == 2


def test_a_group_that_cannot_be_placed_whole_is_not_placed_at_all(two_nodes):
    """A pipeline missing a stage does not run slower — it does not run."""
    before = set(two_nodes.hub.tasks)
    with pytest.raises(AgentError) as exc:
        two_nodes.hub.submit_group(size=3, command=["true"], resources={"gpus": 4})
    assert "place member" in str(exc.value)
    assert set(two_nodes.hub.tasks) == before, "a failed group left tasks behind"


def test_a_message_travels_between_nodes_and_comes_back(two_nodes):
    """Rank 0 -> rank 1 -> rank 0, each hop crossing the orchestrator."""
    group, _port = start_pipeline(two_nodes)
    answer = ask(two_nodes, group, "hello pipeline")
    assert answer["text"] == "HELLO PIPELINE"
    assert answer["hops"] == [0, 1], f"the message took {answer['hops']}"


def test_a_longer_pipeline_visits_every_rank_in_order(tmp_path, monkeypatch):
    from conftest_agent import start_agent

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")
    orchestrator = Orchestrator().start()
    running = []
    try:
        for index in range(3):
            agent, thread = start_agent(orchestrator.port, tmp_path / f"n{index}",
                                        node_id=f"node-{index}")
            running.append((agent, thread))
        deadline = time.time() + 20
        while time.time() < deadline and len(orchestrator.hub.sessions) < 3:
            time.sleep(0.05)
        group, _port = start_pipeline(orchestrator, size=3)
        answer = ask(orchestrator, group, "three stages")
        assert answer["hops"] == [0, 1, 2]
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=10)
        orchestrator.stop()


def test_an_http_request_reaches_a_task_that_opened_no_port(two_nodes):
    """How a model on somebody's home machine answers the internet.

    The node has no reachable address and opened nothing; the request goes in
    over the connection it made.
    """
    group, _port = start_pipeline(two_nodes)
    status, _headers, body = two_nodes.call(
        two_nodes.hub.request(group.tasks[0], method="GET", path="/", timeout_s=30))
    assert status == 200
    assert json.loads(body) == {"rank": 0, "size": 2}


def test_asking_a_task_that_serves_nothing_says_so(two_nodes):
    record = two_nodes.hub.submit(command=[sys.executable, "-c", "import time; time.sleep(5)"],
                                  timeout_s=30)
    two_nodes.wait_state(record.task_id, "running", timeout=30)
    with pytest.raises(AgentError) as exc:
        two_nodes.call(two_nodes.hub.request(record.task_id, timeout_s=10))
    assert "not serving" in str(exc.value)


def test_a_message_for_a_rank_that_is_not_there_does_not_kill_the_node(two_nodes):
    """A stray message is a bug somewhere else; this node still works after it."""
    from loom_agent.proto import agent_pb2

    group, _port = start_pipeline(two_nodes)
    two_nodes.hub.route(agent_pb2.TaskMessage(
        group_id=group.group_id, from_rank=0, to_rank=7, payload=b"nowhere"))
    time.sleep(0.5)
    answer = ask(two_nodes, group, "still here")
    assert answer["text"] == "STILL HERE"


def test_stopping_a_group_stops_every_member(two_nodes):
    group, _port = start_pipeline(two_nodes)
    two_nodes.hub.stop_group(group.group_id, reason="the client changed their mind")
    for rank in range(2):
        record = two_nodes.wait_state(group.tasks[rank], "cancelled", "failed", timeout=60)
        assert record.state == "cancelled"


def ask(orchestrator, group, text: str, timeout: float = 60.0) -> dict:
    """Push a request in at rank 0 and wait for it to come back round."""
    key = f"q{int(time.time() * 1000)}"
    orchestrator.call(orchestrator.hub.request(
        group.tasks[0], method="POST", path="/ask",
        body=json.dumps({"id": key, "text": text}).encode(),
        headers={"Content-Type": "application/json"}, timeout_s=30))
    deadline = time.time() + timeout
    while time.time() < deadline:
        _status, _headers, body = orchestrator.call(orchestrator.hub.request(
            group.tasks[0], method="GET", path=f"/answer/{key}", timeout_s=30))
        answer = json.loads(body)
        if answer != "pending":
            return answer
        time.sleep(0.2)
    pytest.fail("the message never came back round the pipeline")
