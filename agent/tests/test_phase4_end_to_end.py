"""Phase 4: a task travels from the orchestrator to a result, over one stream.

Everything here goes through a real gRPC connection to a real agent running
real processes. The point of the phase is that the pieces built separately in
phases 1-3 actually meet.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from conftest import make_join_key
from fake_orchestrator import FakeOrchestrator

from looma_agent.config import parse_args
from looma_agent.control import tasks as tasks_mod
from looma_agent.main import Agent


@pytest.fixture
def orchestrator():
    fake = FakeOrchestrator()
    fake.start()
    yield fake
    fake.stop()


@pytest.fixture
def node(orchestrator, tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setenv("LOOMA_P2P", "0")
    config = parse_args([
        "--key", make_join_key(f"127.0.0.1:{orchestrator.port}"),
        "--node-id", "node-1",
        "--root", str(tmp_path),
        "--heartbeat-interval", "0.3",
        "--reconnect-delay", "0.2",
    ])
    agent = Agent(config)
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    assert orchestrator.wait_registered(), "the node never registered"
    yield agent
    agent.stop()
    thread.join(timeout=10)


WRITE_RESULT = (
    "import os, pathlib;"
    "data = pathlib.Path('input.txt').read_text();"
    "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']);"
    "(out / 'answer.txt').write_text(data.upper());"
    "print('worked on', len(data), 'bytes')"
)


# ------------------------------------------------------------- the round trip
def test_a_task_with_input_produces_a_result_that_comes_back(orchestrator, node):
    """The whole phase in one test, and the thing the old path could not do."""
    payload = b"the client's data"
    orchestrator.run_task("t1", [sys.executable, "-c", WRITE_RESULT],
                          inputs={"input.txt": payload})

    final = orchestrator.wait_finished("t1")
    assert final is not None, "no terminal state ever arrived"
    assert final.state == "done", final.error
    assert [f.name for f in final.results] == ["answer.txt"]

    orchestrator.collect("t1", "answer.txt")
    assert orchestrator.wait_collected("t1", "answer.txt")
    assert bytes(orchestrator.results["t1/answer.txt"]) == payload.upper()


def test_a_task_is_acknowledged_before_it_is_carried_out(orchestrator, node):
    """The stream must answer at once; the work happens elsewhere."""
    orchestrator.run_task("t2", [sys.executable, "-c", "pass"], command_id="cmd-2")
    deadline = time.time() + 10
    while time.time() < deadline:
        if any(a.command_id == "cmd-2" and a.ok for a in orchestrator.acks):
            return
        time.sleep(0.05)
    pytest.fail("the run command was never acknowledged")


def test_the_states_a_task_passes_through_are_reported(orchestrator, node):
    orchestrator.run_task("t3", [sys.executable, "-c", "import time; time.sleep(0.5)"])
    assert orchestrator.wait_finished("t3") is not None
    names = orchestrator.state_names("t3")
    assert names[0] == "provisioning"
    assert "running" in names
    assert names[-1] == "done"


def test_a_large_input_arrives_whole(orchestrator, node):
    """Chunking must not lose or reorder anything."""
    payload = bytes(range(256)) * 8192  # 2 MB, and order-sensitive
    program = (
        "import os, pathlib, hashlib;"
        "raw = pathlib.Path('blob.bin').read_bytes();"
        "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']);"
        "(out / 'digest.txt').write_text(hashlib.sha256(raw).hexdigest())"
    )
    orchestrator.run_task("t4", [sys.executable, "-c", program],
                          inputs={"blob.bin": payload}, timeout_s=120)
    final = orchestrator.wait_finished("t4")
    assert final.state == "done", final.error

    orchestrator.collect("t4", "digest.txt")
    assert orchestrator.wait_collected("t4", "digest.txt")
    import hashlib

    assert bytes(orchestrator.results["t4/digest.txt"]).decode() == \
        hashlib.sha256(payload).hexdigest()


def test_several_input_files_arrive(orchestrator, node):
    program = (
        "import os, pathlib;"
        "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']);"
        "(out / 'joined.txt').write_text("
        "pathlib.Path('a.txt').read_text() + pathlib.Path('b/c.txt').read_text())"
    )
    orchestrator.run_task("t5", [sys.executable, "-c", program],
                          inputs={"a.txt": b"first", "b/c.txt": b"second"})
    final = orchestrator.wait_finished("t5")
    assert final.state == "done", final.error
    orchestrator.collect("t5", "joined.txt")
    assert orchestrator.wait_collected("t5", "joined.txt")
    assert bytes(orchestrator.results["t5/joined.txt"]) == b"firstsecond"


# ------------------------------------------------------------- environments
def test_a_task_runs_in_a_provisioned_environment(orchestrator, node):
    orchestrator.run_task(
        "e1", ["python", "-c", "import sys; print(sys.prefix)"],
        environment={"kind": "python"}, timeout_s=180,
    )
    final = orchestrator.wait_finished("e1", timeout=240)
    assert final.state == "done", final.error


def test_an_environment_that_cannot_be_built_fails_the_task_with_a_reason(orchestrator, node):
    """The reason reaches the orchestrator, not just the node's own log."""
    orchestrator.run_task("e2", ["true"], environment={"kind": "wasm"})
    final = orchestrator.wait_finished("e2")
    assert final.state == "failed"
    assert "wasm" in final.error


# ------------------------------------------------------------------ control
def test_a_running_task_can_be_stopped(orchestrator, node):
    from looma_agent.proto import agent_pb2

    orchestrator.run_task("s1", [sys.executable, "-c", "import time; time.sleep(120)"],
                          timeout_s=300)
    deadline = time.time() + 20
    while time.time() < deadline and "running" not in orchestrator.state_names("s1"):
        time.sleep(0.1)
    orchestrator.send(agent_pb2.ServerMessage(stop_task=agent_pb2.StopTask(
        command_id="stop-1", task_id="s1", reason="the client changed their mind")))
    final = orchestrator.wait_finished("s1", timeout=60)
    assert final.state == "cancelled"
    assert "changed their mind" in final.error


def test_releasing_a_task_takes_its_disk_back(orchestrator, node):
    from looma_agent.proto import agent_pb2

    orchestrator.run_task("r1", [sys.executable, "-c",
                                 "import os, pathlib;"
                                 "pathlib.Path(os.environ['LOOMA_TASK_OUT'], 'x').write_text('y')"])
    assert orchestrator.wait_finished("r1").state == "done"
    directory = node.tasks.require("r1").directory.root
    orchestrator.send(agent_pb2.ServerMessage(
        release_task=agent_pb2.ReleaseTask(command_id="rel-1", task_id="r1")))
    deadline = time.time() + 20
    while time.time() < deadline:
        if node.tasks.get("r1") is None and not directory.exists():
            return
        time.sleep(0.1)
    pytest.fail("the task was never released")


def test_logs_can_be_fetched(orchestrator, node):
    from looma_agent.proto import agent_pb2

    orchestrator.run_task("l1", [sys.executable, "-c", "print('what the task said')"])
    assert orchestrator.wait_finished("l1").state == "done"
    orchestrator.send(agent_pb2.ServerMessage(fetch_logs=agent_pb2.FetchLogs(
        command_id="log-1", task_id="l1", tail_lines=10)))
    deadline = time.time() + 20
    while time.time() < deadline:
        if orchestrator.logs:
            assert "what the task said" in orchestrator.logs[0].text
            return
        time.sleep(0.1)
    pytest.fail("the logs never came back")


def test_collecting_a_file_the_task_did_not_produce_says_so(orchestrator, node):
    orchestrator.run_task("m1", [sys.executable, "-c", "pass"])
    assert orchestrator.wait_finished("m1").state == "done"
    orchestrator.collect("m1", "imaginary.txt")
    assert orchestrator.wait_collected("m1", "imaginary.txt")
    assert "no file" in orchestrator.result_errors["m1/imaginary.txt"]


# ------------------------------------------------------------------ refusals
def test_a_task_asking_for_cards_this_node_lacks_fails_with_the_count(orchestrator, node):
    orchestrator.run_task("g1", ["true"], resources={"gpus": 8})
    final = orchestrator.wait_finished("g1")
    assert final.state == "failed"
    assert "GPU" in final.error


def test_input_that_stops_arriving_does_not_hold_the_node_forever(orchestrator, node,
                                                                  monkeypatch):
    """A client that goes away mid-upload must give the resources back."""
    monkeypatch.setattr(tasks_mod, "INPUT_IDLE_TIMEOUT_S", 2.0)
    from looma_agent.proto import agent_pb2

    orchestrator.send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
        command_id="cmd-i1", task_id="i1", command=["true"], timeout_s=60,
        inputs=[agent_pb2.InputFile(name="never.bin", size_bytes=1000)],
    )))
    final = orchestrator.wait_finished("i1", timeout=60)
    assert final is not None, "the task neither started nor gave up"
    assert final.state == "failed"
    assert "stopped arriving" in final.error
    assert node.tasks.get("i1") is None


def test_an_input_name_that_escapes_the_directory_is_refused(orchestrator, node):
    from looma_agent.proto import agent_pb2

    orchestrator.send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
        command_id="cmd-i2", task_id="i2", command=["true"], timeout_s=60,
        inputs=[agent_pb2.InputFile(name="../../escaped", size_bytes=4)],
    )))
    final = orchestrator.wait_finished("i2", timeout=60)
    assert final.state == "failed"
    assert "directory" in final.error


def test_a_file_that_was_never_declared_is_not_written(orchestrator, node):
    """The agent writes what the task said it would need, and nothing else."""
    from looma_agent.proto import agent_pb2

    orchestrator.send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
        command_id="cmd-i3", task_id="i3", command=["true"], timeout_s=60,
        inputs=[agent_pb2.InputFile(name="declared.txt", size_bytes=2)],
    )))
    orchestrator.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
        task_id="i3", name="smuggled.sh", data=b"rm -rf /", last=True)))
    final = orchestrator.wait_finished("i3", timeout=60)
    assert final.state == "failed"
    assert "never declared" in final.error


# ----------------------------------------------------------------- telemetry
def test_telemetry_reports_what_the_node_has_left(orchestrator, node):
    orchestrator.run_task("tm1", [sys.executable, "-c", "import time; time.sleep(3)"])
    deadline = time.time() + 20
    while time.time() < deadline:
        recent = [t for t in orchestrator.telemetry if t.tasks_running >= 1]
        if recent:
            report = recent[-1]
            assert report.node_id == "node-1"
            assert report.gpus_free <= report.gpus_total
            assert any(t.task_id == "tm1" for t in report.tasks)
            return
        time.sleep(0.1)
    pytest.fail("telemetry never showed the running task")
