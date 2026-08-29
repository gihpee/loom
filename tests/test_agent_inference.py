"""A real model, split across nodes, answering through the agent transport.

Everything below is real: a real (tiny) Llama, real weights sliced by layer, a
real forward pass per stage, real activations crossing the orchestrator between
machines. The stage payload is the one that ships; the only thing standing in
for production is the size of the model.

The point is the thing phase 7 exists for — that inference is now a task like
any other, and the pipeline it forms uses the same transport a client's own
code would.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from make_tiny_model import ensure_tiny_model
from test_agent_gateway import Orchestrator

PAYLOAD = str(Path(__file__).resolve().parent.parent / "payloads" / "loom_stage")


@pytest.fixture(scope="module")
def tiny_model():
    return str(ensure_tiny_model())


def stage_command(model: str, start: int, end: int) -> list:
    return [
        sys.executable, "-m", "loom_stage.server",
        "--model-id", "tiny", "--weights-uri", model,
        "--start-layer", str(start), "--end-layer", str(end),
        "--device", "cpu", "--dtype", "float32",
    ]


def start_nodes(orchestrator, tmp_path, count: int):
    from conftest_agent import start_agent

    running = []
    for index in range(count):
        running.append(start_agent(orchestrator.port, tmp_path / f"n{index}",
                                   node_id=f"node-{index}"))
    deadline = time.time() + 30
    while time.time() < deadline and len(orchestrator.hub.sessions) < count:
        time.sleep(0.05)
    assert len(orchestrator.hub.sessions) == count
    return running


def wait_ready(orchestrator, group, timeout: float = 300.0) -> None:
    """Weights take a while. Silence for a while is not the same as failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        states = [orchestrator.hub.tasks[t].state for t in group.tasks.values()]
        if any(s in ("failed", "cancelled") for s in states):
            logs = orchestrator.call(orchestrator.hub.logs(list(group.tasks.values())[0]))
            pytest.fail(f"a stage died: {states}\n{logs}")
        if all(s == "running" for s in states):
            healthy = 0
            for task_id in group.tasks.values():
                try:
                    status, _headers, body = orchestrator.call(
                        orchestrator.hub.request(task_id, path="/health", timeout_s=20))
                    if status == 200 and json.loads(body).get("status") == "ok":
                        healthy += 1
                except Exception:
                    break
            if healthy == len(group.tasks):
                return
        time.sleep(1.0)
    pytest.fail("the pipeline never came up")


def ask(orchestrator, group, prompt: str, *, max_tokens: int = 4) -> dict:
    status, _headers, body = orchestrator.call(orchestrator.hub.request(
        group.tasks[0], method="POST", path="/v1/chat/completions",
        body=json.dumps({
            "model": "tiny",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json"},
        timeout_s=300,
    ))
    assert status == 200, body[:400]
    return json.loads(body)


@pytest.mark.slow
def test_a_model_split_over_two_nodes_answers(tmp_path, monkeypatch, tiny_model):
    """Layers 0-2 on one machine, 3-6 on another, one answer out of the pair."""
    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 2)
    try:
        group = orchestrator.hub.submit_group(
            size=2,
            command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD},
            serve_port=1,          # non-zero: "serve, and tell me on what"
            timeout_s=600,
            node_ids=["node-0", "node-1"],
            # The split is the orchestrator's decision, because only it knows
            # what each node can hold. Here it is fixed because the model is.
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, group)

        answer = ask(orchestrator, group, "hello")
        assert answer["choices"][0]["message"]["content"] is not None
        assert answer["usage"]["completion_tokens"] >= 1
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_a_model_on_one_node_answers_without_touching_the_network(tmp_path, monkeypatch,
                                                                  tiny_model):
    """Both stages on one machine: the fast arrangement, and the same code."""
    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 1)
    try:
        group = orchestrator.hub.submit_group(
            size=2,
            command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD},
            serve_port=1,
            timeout_s=600,
            node_ids=["node-0", "node-0"],
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, group)
        answer = ask(orchestrator, group, "hello")
        assert answer["choices"][0]["message"]["content"] is not None
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()


@pytest.mark.slow
def test_splitting_the_model_does_not_change_the_answer(tmp_path, monkeypatch, tiny_model):
    """The test that matters more than "it runs".

    A pipeline that answers is not the same as a pipeline that answers
    CORRECTLY: an off-by-one in the layer split, a lost KV cache entry or a
    dropped position id all still produce fluent-looking output. Greedy
    decoding is deterministic, so one stage and two must emit the same tokens
    from the same prompt — and if they do not, the split is wrong.
    """
    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOM_P2P", "0")
    orchestrator = Orchestrator().start()
    running = start_nodes(orchestrator, tmp_path, 2)
    try:
        whole = orchestrator.hub.submit_group(
            size=1, command=stage_command(tiny_model, 0, 6),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0"],
        )
        wait_ready(orchestrator, whole)
        one_stage = ask(orchestrator, whole, "hello", max_tokens=8)
        orchestrator.hub.stop_group(whole.group_id)

        split = orchestrator.hub.submit_group(
            size=2, command=stage_command(tiny_model, 0, 3),
            env={"PYTHONPATH": PAYLOAD}, serve_port=1, timeout_s=600,
            node_ids=["node-0", "node-1"],
            per_rank=[
                {"command": stage_command(tiny_model, 0, 3)},
                {"command": stage_command(tiny_model, 3, 6)},
            ],
        )
        wait_ready(orchestrator, split)
        two_stages = ask(orchestrator, split, "hello", max_tokens=8)

        assert two_stages["choices"][0]["message"]["content"] == \
            one_stage["choices"][0]["message"]["content"], \
            "the same prompt gave different answers whole and split"
    finally:
        for agent, thread in running:
            agent.stop()
            thread.join(timeout=15)
        orchestrator.stop()
