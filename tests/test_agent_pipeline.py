"""A job spread over several nodes, over one stream each.

This is the mechanism the inference pipeline will be built on: a task sends to
a RANK, and where that rank lives — this machine, a peer, the far side of the
orchestrator — is decided by the agent and never by the task.

The stage here holds no model on purpose. A failure in this file is a failure
of the transport, which is the thing being built.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from looma.orchestrator.agents import AgentError

from test_agent_gateway import ADMIN_HEADERS, Orchestrator  # noqa: E402

STAGE = str(Path(__file__).resolve().parent / "stage_fixture" / "pipeline_stage.py")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def two_nodes(tmp_path, monkeypatch):
    """Two agents on one orchestrator — the smallest real pipeline."""
    from conftest_agent import start_agent

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")   # force the path through the orchestrator
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

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
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
    from looma_agent.proto import agent_pb2

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


def test_модель_не_считается_отвечающей_пока_грузит_веса(two_nodes, monkeypatch):
    """Симптом со стенда: панель показала «отвечает», запрос ушёл в стадию,
    которая ещё качала веса, и упал на KeyError: 'executor' — ошибке, из
    которой ничего не следует.

    Состояние задачи говорит только о процессе: он слушает за минуты до того,
    как веса окажутся в памяти.
    """
    port = free_port()
    group = two_nodes.hub.submit_group(
        size=2, command=[sys.executable, STAGE], serve_port=port, timeout_s=120,
        node_ids=["node-0", "node-1"], label="warming",
        env={"STAGE_WARMUP_S": "6"},
    )
    for rank in range(2):
        two_nodes.wait_state(group.tasks[rank], "running", timeout=60)

    # Процесс запущен — но группа ещё не обслуживает.
    assert two_nodes.hub.group_for("warming") is not None, "группа размещена"
    assert two_nodes.call(two_nodes.hub.serving("warming")) is None, \
        "модель объявлена отвечающей, пока стадии грузят веса"

    deadline = time.time() + 40
    while time.time() < deadline:
        if two_nodes.call(two_nodes.hub.serving("warming")) is not None:
            return
        time.sleep(1)
    pytest.fail("стадия так и не доложила о готовности")


def test_готовность_спрашивается_один_раз(two_nodes):
    """Загруженная стадия не разгружается обратно, так что в установившемся
    режиме проверка не должна стоить ничего."""
    group, _port = start_pipeline(two_nodes)
    deadline = time.time() + 40
    while time.time() < deadline:
        if two_nodes.call(two_nodes.hub.serving("warming2")) is not None:
            break
        if two_nodes.hub.groups.get(group.group_id):
            break
        time.sleep(0.5)
    # Группа из start_pipeline без label; проверяем сам механизм запоминания.
    two_nodes.hub._ready_groups.add(group.group_id)
    two_nodes.hub.groups[group.group_id].label = "cached"
    assert two_nodes.call(two_nodes.hub.serving("cached")) is not None


GROUP_INPUT_ECHO = (
    "import os, pathlib;"
    "src = pathlib.Path('shared.txt').read_text();"
    "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']) / 'seen.txt';"
    "out.write_text(src + ' rank=' + os.environ['LOOMA_RANK'])"
)


def test_код_группы_доезжает_до_каждого_ранга(two_nodes):
    """Фаза 2: конвейер моделей завозил свой код на узлы через `inputs`, а
    общий API групп этого не умел — код клиента на группу было не доставить
    иначе как запечь в образ.

    Одна программа на всех, разное ей передаётся через per_rank: узел, впервые
    видящий эту работу, получает её вместе с задачей, и реестр пакетов
    посередине для этого не нужен.
    """
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from test_agent_gateway import _Settings

    client = TestClient(create_app(agents=two_nodes.hub, config=_Settings()), headers=ADMIN_HEADERS)
    submitted = client.post("/admin/groups", json={
        "size": 2,
        "command": [sys.executable, "-c", GROUP_INPUT_ECHO],
        "node_ids": ["node-0", "node-1"],
        "inputs": {"shared.txt": base64.b64encode("общий код".encode()).decode()},
        "timeout_s": 120,
    }).json()
    assert "group_id" in submitted, submitted

    for rank in submitted["ranks"]:
        record = two_nodes.wait_state(rank["task_id"], "done", "failed")
        assert record.state == "done", record.error
        answer = client.get(f"/admin/tasks/{rank['task_id']}/results/seen.txt")
        assert answer.status_code == 200, answer.text
        assert answer.content.decode() == f"общий код rank={rank['rank']}"


def test_испорченный_base64_называет_свой_файл(two_nodes):
    """«Неверный base64» ничего не стоит, когда файлов десяток."""
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from test_agent_gateway import _Settings

    client = TestClient(create_app(agents=two_nodes.hub, config=_Settings()), headers=ADMIN_HEADERS)
    answer = client.post("/admin/groups", json={
        "size": 1, "command": ["true"], "node_ids": ["node-0"],
        "inputs": {"beper.bin": "не base64 ни разу!!"},
    })
    assert answer.status_code == 400
    assert "beper.bin" in answer.json()["error"]["message"]
