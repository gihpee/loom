"""Phase 3: input gets in, the result gets out, and neither escapes the directory.

Names in this phase are chosen by whoever submitted the task, on somebody
else's machine. Half of these tests are about the fact that a name is an
attack surface.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import time
from pathlib import Path

import pytest

from looma_agent.tasks.env import EnvironmentCache
from looma_agent.tasks.limits import resolve_isolation
from looma_agent.tasks.registry import TaskRegistry
from looma_agent.tasks.spec import TaskSpec
from looma_agent.transport.files import (
    Inbox,
    IncomingFile,
    Outbox,
    TransferRefused,
    safe_target,
)


@pytest.fixture
def work(tmp_path):
    directory = tmp_path / "work"
    directory.mkdir()
    return directory


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------- hostile names
@pytest.mark.parametrize("name", [
    "../escaped",
    "../../etc/passwd",
    "a/../../escaped",
    "/etc/passwd",
    "..",
    "",
])
def test_a_name_that_leaves_the_directory_is_refused(work, name):
    with pytest.raises(TransferRefused):
        safe_target(work, name)


def test_a_name_that_goes_through_a_symlink_is_refused(work, tmp_path):
    """`a/../../b` and a symlinked `a` are the same attack.

    Only one of them looks like one, which is why the check resolves rather
    than inspecting the string.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "innocent").symlink_to(outside)
    with pytest.raises(TransferRefused):
        safe_target(work, "innocent/planted")


def test_ordinary_names_are_allowed(work):
    assert safe_target(work, "run.py").parent == work.resolve()
    assert safe_target(work, "data/train.csv").name == "train.csv"


# ------------------------------------------------------------------- arriving
def test_a_file_arrives_whole(work):
    payload = b"x" * (3 * 1024 * 1024)
    inbox = Inbox(work)
    written = inbox.receive("data/big.bin", [payload[i:i + 8192]
                                             for i in range(0, len(payload), 8192)],
                            size_bytes=len(payload), digest=sha(payload))
    assert written.read_bytes() == payload


def test_a_transfer_cut_short_leaves_nothing(work):
    """A task must never see half a file: it would read it and fail elsewhere."""
    target = work / "dataset.bin"
    incoming = IncomingFile(target, expected_bytes=1000, digest=sha(b"y" * 1000))
    with incoming:
        incoming.write(b"y" * 400)
        with pytest.raises(TransferRefused) as exc:
            incoming.finish()
    assert "400 of 1000" in str(exc.value)
    assert not target.exists()
    assert list(work.iterdir()) == []


def test_a_connection_dropped_mid_transfer_leaves_nothing(work):
    """The realistic version: nobody calls finish() at all."""
    target = work / "half.bin"
    with pytest.raises(RuntimeError):
        with IncomingFile(target, expected_bytes=1000) as incoming:
            incoming.write(b"z" * 500)
            raise RuntimeError("the stream died")
    assert not target.exists()
    assert list(work.iterdir()) == []


def test_a_corrupted_file_is_rejected_not_delivered(work):
    target = work / "code.py"
    with pytest.raises(TransferRefused) as exc:
        with IncomingFile(target, digest=sha(b"what was sent")) as incoming:
            incoming.write(b"what arrived")
            incoming.finish()
    assert "corrupted" in str(exc.value)
    assert not target.exists()


def test_more_than_was_declared_is_refused(work):
    target = work / "liar.bin"
    with pytest.raises(TransferRefused) as exc:
        with IncomingFile(target, expected_bytes=10) as incoming:
            incoming.write(b"x" * 50)
    assert "declared" in str(exc.value)
    assert not target.exists()


def test_input_cannot_exceed_the_disk_the_task_was_given(work):
    inbox = Inbox(work, limit_bytes=1024)
    with pytest.raises(TransferRefused) as exc:
        inbox.receive("big.bin", [b"x" * 5000], size_bytes=5000)
    assert "disk" in str(exc.value)


# ------------------------------------------------------------------- archives
def make_archive(path: Path, members: dict, *, link: tuple = None) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        if link is not None:
            info = tarfile.TarInfo(link[0])
            info.type = tarfile.SYMTYPE
            info.linkname = link[1]
            tar.addfile(info)
    return path


def test_an_archive_of_input_is_unpacked(work, tmp_path):
    archive = make_archive(tmp_path / "in.tar.gz",
                           {"run.py": b"print(1)", "data/x.csv": b"a,b\n1,2\n"})
    written = Inbox(work).unpack(archive)
    assert (work / "run.py").read_bytes() == b"print(1)"
    assert (work / "data" / "x.csv").exists()
    assert len(written) == 2


def test_an_archive_that_writes_outside_is_refused_whole(work, tmp_path):
    """Thirty years old and still the first thing to try.

    Rejected whole rather than partially extracted: half an input is worse
    than none, because the task will run against it.
    """
    archive = make_archive(tmp_path / "evil.tar.gz",
                           {"innocent.txt": b"ok", "../../escaped": b"pwned"})
    with pytest.raises(TransferRefused):
        Inbox(work).unpack(archive)
    assert not (work / "innocent.txt").exists(), "it extracted part of a rejected archive"


def test_an_archive_containing_a_link_is_refused(work, tmp_path):
    archive = make_archive(tmp_path / "linky.tar.gz", {"ok.txt": b"fine"},
                           link=("shortcut", "/etc/passwd"))
    with pytest.raises(TransferRefused) as exc:
        Inbox(work).unpack(archive)
    assert "link" in str(exc.value)


def test_an_archive_over_the_disk_budget_is_refused_before_extracting(work, tmp_path):
    archive = make_archive(tmp_path / "fat.tar.gz", {"big.bin": b"x" * 100_000})
    with pytest.raises(TransferRefused):
        Inbox(work, limit_bytes=1024).unpack(archive)
    assert list(work.iterdir()) == []


# ------------------------------------------------------------------- leaving
def test_the_result_is_what_the_task_put_in_out(tmp_path):
    out = tmp_path / "out"
    (out / "nested").mkdir(parents=True)
    (out / "answer.txt").write_bytes(b"42")
    (out / "nested" / "model.bin").write_bytes(b"weights")
    manifest = Outbox(out).manifest()
    assert {f.name for f in manifest} == {"answer.txt", "nested/model.bin"}
    assert next(f for f in manifest if f.name == "answer.txt").digest == sha(b"42")


def test_scratch_is_not_part_of_the_result(tmp_path):
    """A task fills `work` with checkpoints nobody asked for."""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "huge.ckpt").write_bytes(b"x" * 10_000)
    (out / "answer.txt").write_bytes(b"42")
    assert [f.name for f in Outbox(out).manifest()] == ["answer.txt"]


def test_a_result_that_is_a_symlink_is_not_followed(tmp_path):
    """It would point at something the task did not produce."""
    out = tmp_path / "out"
    out.mkdir()
    secret = tmp_path / "secret"
    secret.write_bytes(b"not the task's to give")
    (out / "answer.txt").symlink_to(secret)
    assert Outbox(out).manifest() == []


def test_a_result_bigger_than_allowed_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "huge.bin").write_bytes(b"x" * 50_000)
    with pytest.raises(TransferRefused):
        Outbox(out, limit_bytes=1024).manifest()


def test_a_result_file_streams_back(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    payload = b"y" * (2 * 1024 * 1024 + 17)
    (out / "big.bin").write_bytes(payload)
    assert b"".join(Outbox(out).read("big.bin")) == payload


def test_reading_a_result_by_a_hostile_name_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(TransferRefused):
        list(Outbox(out).read("../../etc/passwd"))


# --------------------------------------------------------------- the round trip
def test_a_task_reads_its_input_and_its_result_comes_back(tmp_path, monkeypatch):
    """The whole point of the phase, end to end.

    This is what the old compute path could not do: a task whose answer is a
    file had nowhere to put it, so anything beyond a number in the logs was
    impossible.
    """
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    registry = TaskRegistry(
        root=tmp_path / "tasks",
        isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0,
        retention_s=60.0,
    )
    program = (
        "import os, pathlib;"
        "data = pathlib.Path('input.txt').read_text();"
        "out = pathlib.Path(os.environ['LOOMA_TASK_OUT']);"
        "(out / 'answer.txt').write_text(data.upper())"
    )
    payload = b"the client's data"

    def deliver(inbox):
        inbox.receive("input.txt", [payload], size_bytes=len(payload),
                      digest=sha(payload))

    task = registry.submit(
        TaskSpec.from_dict({"task_id": "rt1", "command": [sys.executable, "-c", program]}),
        deliver_input=deliver,
    )
    assert task.wait(timeout=60)
    assert task.state == "done", task.logs()

    manifest = task.results()
    assert [f.name for f in manifest] == ["answer.txt"]
    assert b"".join(task.read_result("answer.txt")) == payload.upper()
    registry.stop_all()


def test_input_that_will_not_arrive_means_the_task_never_starts(tmp_path, monkeypatch):
    """A task running against input that failed to arrive is worse than none."""
    monkeypatch.setenv("LOOMA_ALLOW_UNPRIVILEGED_TASKS", "1")
    registry = TaskRegistry(
        root=tmp_path / "tasks",
        isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=2,
        retention_s=60.0,
    )

    def deliver(inbox):
        inbox.receive("../escaped", [b"pwned"])

    with pytest.raises(Exception) as exc:
        registry.submit(
            TaskSpec.from_dict({"task_id": "bad", "command": ["true"],
                                "resources": {"gpus": 1}}),
            deliver_input=deliver,
        )
    assert "directory" in str(exc.value)
    # And nothing was left holding resources for a task that never ran.
    assert registry.get("bad") is None
    assert registry.free_devices() == [0, 1]
    assert not (tmp_path / "tasks" / "bad").exists()
    registry.stop_all()
