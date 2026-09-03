"""The two halves meeting: a real agent, a real orchestrator, one stream.

Everything below runs an actual gRPC server, an actual agent process-manager
and actual subprocesses. The point of the phase is that the orchestrator can
place a task on a node it cannot dial and get a file back, so nothing here is
stubbed on either side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from looma.orchestrator.agents import AgentError, AgentHub, add_agent_gateway_to_server


class Orchestrator:
    """The agent gateway on its own event loop, driven from the test thread."""

    def __init__(self) -> None:
        self.hub = AgentHub()
        self.loop = asyncio.new_event_loop()
        self.port = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Orchestrator":
        self._thread.start()
        assert self._ready.wait(20), "the gateway never came up"
        return self

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())
        self.loop.run_forever()

    async def _serve(self) -> None:
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=8))
        add_agent_gateway_to_server(self.server, self.hub)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        await self.server.start()
        self._ready.set()

    def call(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)

    def wait_node(self, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.hub.sessions:
                return True
            time.sleep(0.05)
        return False

    def wait_state(self, task_id: str, *states, timeout: float = 180.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            record = self.hub.tasks.get(task_id)
            if record is not None and record.state in states:
                return record
            time.sleep(0.1)
        record = self.hub.tasks.get(task_id)
        pytest.fail(f"task stayed in {record.state if record else 'nowhere'}, "
                    f"never reached {states}")


@pytest.fixture
def stand(tmp_path, monkeypatch):
    """An orchestrator and one node attached to it."""
    from conftest_agent import start_agent

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    agent, thread = start_agent(orchestrator.port, tmp_path)
    assert orchestrator.wait_node(), "the agent never attached"
    yield orchestrator, agent
    agent.stop()
    thread.join(timeout=10)
    orchestrator.stop()


WRITE_RESULT = (
    "import os, pathlib;"
    "data = pathlib.Path('input.txt').read_text();"
    "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']);"
    "(out / 'answer.txt').write_text(data.upper());"
    "print('done with', len(data), 'bytes')"
)


def test_a_node_appears_with_what_it_can_do(stand):
    orchestrator, _agent = stand
    nodes = orchestrator.hub.node_list()
    assert len(nodes) == 1
    node = nodes[0]
    assert node["agent_version"]
    assert "python" in node["environment_kinds"]
    if not node["accepts_tasks"]:
        assert node["refusal"], "a node that takes no work must say why"


def test_a_task_goes_out_and_a_file_comes_back(stand):
    """From the orchestrator's side: place, wait, collect. The whole phase."""
    orchestrator, _agent = stand
    payload = b"the client's data"
    record = orchestrator.hub.submit(
        command=[sys.executable, "-c", WRITE_RESULT],
        inputs={"input.txt": payload},
        timeout_s=120,
    )
    done = orchestrator.wait_state(record.task_id, "done", "failed")
    assert done.state == "done", done.error
    assert [f.name for f in done.results] == ["answer.txt"]

    collected = orchestrator.call(orchestrator.hub.collect(record.task_id, "answer.txt"))
    assert collected == payload.upper()


def test_submitting_does_not_wait_for_the_task(stand):
    """An HTTP request must not be open while a node provisions an environment."""
    orchestrator, _agent = stand
    started = time.time()
    record = orchestrator.hub.submit(
        command=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_s=60)
    assert time.time() - started < 2.0
    assert record.state == "pending"
    orchestrator.wait_state(record.task_id, "done", "failed", "cancelled")


def test_logs_come_back_through_the_stream(stand):
    orchestrator, _agent = stand
    record = orchestrator.hub.submit(
        command=[sys.executable, "-c", "print('what the task said')"], timeout_s=60)
    orchestrator.wait_state(record.task_id, "done", "failed")
    text = orchestrator.call(orchestrator.hub.logs(record.task_id))
    assert "what the task said" in text


def test_a_task_can_be_stopped_from_the_orchestrator(stand):
    orchestrator, _agent = stand
    record = orchestrator.hub.submit(
        command=[sys.executable, "-c", "import time; time.sleep(120)"], timeout_s=300)
    orchestrator.wait_state(record.task_id, "running")
    orchestrator.hub.stop(record.task_id, reason="the client changed their mind")
    stopped = orchestrator.wait_state(record.task_id, "cancelled", "failed", timeout=60)
    assert stopped.state == "cancelled"
    assert "changed their mind" in stopped.error


def test_releasing_forgets_the_task(stand):
    orchestrator, agent = stand
    record = orchestrator.hub.submit(command=[sys.executable, "-c", "pass"], timeout_s=60)
    orchestrator.wait_state(record.task_id, "done", "failed")
    orchestrator.hub.release(record.task_id)
    assert record.task_id not in orchestrator.hub.tasks
    deadline = time.time() + 20
    while time.time() < deadline:
        if agent.tasks.get(record.task_id) is None:
            return
        time.sleep(0.1)
    pytest.fail("the node still holds a released task")


# ----------------------------------------------------------------- placement
def test_a_task_no_node_can_take_says_why(stand):
    orchestrator, _agent = stand
    with pytest.raises(AgentError) as exc:
        orchestrator.hub.submit(command=["true"], resources={"gpus": 64})
    assert "gpu" in str(exc.value).lower()


def test_a_second_task_does_not_get_a_card_the_first_holds(stand):
    """Telemetry arrives seconds later; two submissions in one breath must not
    both see the same card free."""
    orchestrator, _agent = stand
    node = orchestrator.hub.node_list()[0]
    if node["gpus_total"] < 1:
        pytest.skip("this machine has no GPU to double-book")
    orchestrator.hub.submit(command=[sys.executable, "-c", "import time; time.sleep(10)"],
                            resources={"gpus": node["gpus_total"]}, timeout_s=60)
    with pytest.raises(AgentError):
        orchestrator.hub.submit(command=["true"], resources={"gpus": 1})


def test_asking_for_a_node_that_is_not_here_is_refused(stand):
    orchestrator, _agent = stand
    with pytest.raises(AgentError) as exc:
        orchestrator.hub.submit(command=["true"], node_id="not-a-node")
    assert "not connected" in str(exc.value)


def test_a_task_on_a_node_that_left_cannot_be_collected(stand):
    orchestrator, agent = stand
    record = orchestrator.hub.submit(command=[sys.executable, "-c", "pass"], timeout_s=60)
    orchestrator.wait_state(record.task_id, "done", "failed")
    orchestrator.hub.sessions.clear()
    with pytest.raises(AgentError) as exc:
        orchestrator.call(orchestrator.hub.collect(record.task_id, "anything"))
    assert "not connected" in str(exc.value)


# ------------------------------------------------------------------- the API
def test_the_http_api_runs_a_task_and_hands_back_the_file(stand):
    """From the button to the result: what the phase is finished by.

    Goes through the real FastAPI app, so the encoding the admin page uses is
    the encoding under test.
    """
    from fastapi.testclient import TestClient

    from looma.api.app import create_app

    orchestrator, _agent = stand
    app = create_app(agents=orchestrator.hub, config=_Settings())
    client = TestClient(app, headers=ADMIN_HEADERS)

    listed = client.get("/admin/agents").json()
    assert [n["node_id"] for n in listed["nodes"]] == ["test-agent"]

    submitted = client.post("/admin/tasks", json={
        "command": [sys.executable, "-c", WRITE_RESULT],
        "inputs": {"input.txt": base64.b64encode(b"from the api").decode()},
        "timeout_s": 120,
    }).json()
    assert "task_id" in submitted, submitted
    task_id = submitted["task_id"]

    record = orchestrator.wait_state(task_id, "done", "failed")
    assert record.state == "done", record.error

    shown = client.get(f"/admin/tasks/{task_id}").json()
    assert [f["name"] for f in shown["results"]] == ["answer.txt"]

    downloaded = client.get(f"/admin/tasks/{task_id}/results/answer.txt")
    assert downloaded.status_code == 200
    assert downloaded.content == b"FROM THE API"

    logs = client.get(f"/admin/tasks/{task_id}/logs").json()
    assert "done with" in logs["text"]

    assert client.delete(f"/admin/tasks/{task_id}").json() == {"released": task_id}
    assert client.get(f"/admin/tasks/{task_id}").status_code == 404


def test_the_api_explains_a_task_it_cannot_place(stand):
    from fastapi.testclient import TestClient

    from looma.api.app import create_app

    orchestrator, _agent = stand
    client = TestClient(create_app(agents=orchestrator.hub, config=_Settings()), headers=ADMIN_HEADERS)
    answer = client.post("/admin/tasks", json={"command": ["true"],
                                               "resources": {"gpus": 64}})
    assert answer.status_code == 409
    assert "gpu" in answer.json()["error"]["message"].lower()


#: Чем тесты представляются. Общим, чтобы добавление маршрута не требовало
#: трогать двенадцать мест.
ADMIN_TOKEN = "test-admin-token"
ADMIN_HEADERS = {"X-Looma-Admin-Token": ADMIN_TOKEN}


class _Settings:
    """Оркестратор с настоящим админским токеном.

    Раньше здесь была пустая строка, и это работало ровно потому, что пустой
    токен означал «пускать всех» — то есть тесты опирались на дыру. Теперь
    отсутствие настройки означает «никого», и представляться нужно даже им.
    """

    admin_token = ADMIN_TOKEN


# ------------------------------------------------------------------ releases
def test_a_node_in_the_wave_is_told_what_to_run(tmp_path, monkeypatch):
    """The release reaches the agent inside the ordinary registration ack."""
    from looma.orchestrator.releases import ReleaseStore

    from conftest_agent import start_agent

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    monkeypatch.delenv("LOOMA_AGENT_INCOMING", raising=False)

    orchestrator = Orchestrator().start()
    store = ReleaseStore(tmp_path / "releases")
    store.publish(version="9.9.9", signature=b"\x02" * 64, archive=b"payload bytes")
    store.set_wave(100)
    orchestrator.hub.releases = store
    orchestrator.hub.release_base_url = "http://127.0.0.1:9"

    agent, thread = start_agent(orchestrator.port, tmp_path / "node")
    try:
        assert orchestrator.wait_node()
        # Started by hand, so there is no launcher that could install anything.
        # Saying so beats downloading something nobody will read.
        deadline = time.time() + 15
        while time.time() < deadline and not agent.updater.last_refusal:
            time.sleep(0.1)
        assert agent.updater.status().state == "refused"
        # And the node stayed up rather than restarting for an update it cannot
        # apply.
        assert orchestrator.hub.sessions
    finally:
        agent.stop()
        thread.join(timeout=10)
        orchestrator.stop()


def test_a_node_outside_the_wave_is_left_alone(tmp_path, monkeypatch):
    from looma.orchestrator.releases import ReleaseStore

    from conftest_agent import start_agent

    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    orchestrator = Orchestrator().start()
    store = ReleaseStore(tmp_path / "releases")
    store.publish(version="9.9.9", signature=b"\x02" * 64, archive=b"payload")
    store.set_wave(0)          # published, not rolling out
    orchestrator.hub.releases = store
    orchestrator.hub.release_base_url = "http://127.0.0.1:9"

    agent, thread = start_agent(orchestrator.port, tmp_path / "node")
    try:
        assert orchestrator.wait_node()
        time.sleep(1.5)
        assert agent.updater.last_refusal == "", "a node outside the wave was offered a release"
    finally:
        agent.stop()
        thread.join(timeout=10)
        orchestrator.stop()


def test_the_archive_is_served_to_anyone_who_asks(tmp_path):
    """Unauthenticated on purpose: a node has a join key, not an admin token,
    and the payload is signed, so the bytes are worthless without the key."""
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from looma.orchestrator.releases import ReleaseStore

    store = ReleaseStore(tmp_path / "releases")
    store.publish(version="0.4.0", signature=b"\x03" * 64, archive=b"the payload")
    client = TestClient(create_app(releases=store, config=_Settings()), headers=ADMIN_HEADERS)

    got = client.get("/agent/release/0.4.0.tar.gz")
    assert got.status_code == 200
    assert got.content == b"the payload"
    assert client.get("/agent/release/0.9.9.tar.gz").status_code == 404


def test_publishing_and_advancing_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from looma.orchestrator.releases import ReleaseStore

    store = ReleaseStore(tmp_path / "releases")
    client = TestClient(create_app(releases=store, config=_Settings()), headers=ADMIN_HEADERS)

    published = client.post("/admin/release", json={
        "version": "0.5.0",
        "signature": ("aa" * 64),
        "archive": base64.b64encode(b"bytes").decode(),
    }).json()
    assert published["version"] == "0.5.0"
    assert published["wave_percent"] == 0, "publishing must not roll out"

    assert client.post("/admin/release/wave", json={"percent": 25}).json()["wave_percent"] == 25
    assert client.get("/admin/release").json()["release"]["wave_percent"] == 25
    client.post("/admin/release/withdraw")
    assert client.get("/admin/release").json()["release"]["wave_percent"] == 0


def test_the_api_refuses_an_unsigned_release(tmp_path):
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from looma.orchestrator.releases import ReleaseStore

    client = TestClient(create_app(releases=ReleaseStore(tmp_path / "r"), config=_Settings()), headers=ADMIN_HEADERS)
    answer = client.post("/admin/release", json={
        "version": "0.5.0", "signature": "",
        "archive": base64.b64encode(b"bytes").decode()})
    assert answer.status_code == 400
    assert "signature" in answer.json()["error"]["message"]
