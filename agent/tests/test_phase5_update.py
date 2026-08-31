"""Phase 5: an update channel that is safe to have.

An update channel is remote code execution on every machine in the fleet, by
design. Most of this file is therefore about what must NOT install.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import os
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from loom_launcher import payload as payload_mod
from loom_launcher.signature import Manifest, Untrusted, digest_of, verify
from loom_launcher.supervise import FAILURES_BEFORE_ROLLBACK, Supervisor


@pytest.fixture
def release_key(monkeypatch):
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setenv("LOOM_RELEASE_PUBKEY", public.hex())
    return key


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_ROOT", str(tmp_path))
    (tmp_path / "agent" / "incoming").mkdir(parents=True)
    return tmp_path


def build_archive(path: Path, *, version: str = "0.2.0", agent: bool = True,
                  extra: str = None) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        if agent:
            body = f"VERSION = {version!r}\n".encode()
            info = tarfile.TarInfo("loom_agent/main.py")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        if extra:
            info = tarfile.TarInfo(extra)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"bad"))
    return path


def offer(root: Path, key, *, version="0.2.0", sign_version=None, digest=None,
          archive_version=None, signer=None, extra=None) -> Path:
    """Leave a release in `incoming/` the way the agent would."""
    incoming = root / "agent" / "incoming"
    archive = build_archive(incoming / f"{version}.tar.gz",
                            version=archive_version or version, extra=extra)
    manifest = Manifest(version=sign_version or version,
                        sha256=digest or digest_of(archive))
    signature = (signer or key).sign(manifest.canonical())
    path = incoming / f"{version}.json"
    path.write_text(json.dumps({
        "version": version,
        "sha256": digest or digest_of(archive),
        "signature": signature.hex(),
    }))
    return path


# ------------------------------------------------------------- what installs
def test_a_release_we_signed_is_installed(root, release_key):
    manifest = offer(root, release_key)
    installed = payload_mod.install(manifest, installed_version="0.1.0")
    assert installed is not None
    assert installed.version == "0.2.0"
    assert (installed.path / "loom_agent" / "main.py").is_file()


def test_installing_leaves_nothing_in_incoming(root, release_key):
    manifest = offer(root, release_key)
    payload_mod.install(manifest, installed_version="0.1.0")
    assert list((root / "agent" / "incoming").iterdir()) == []


# --------------------------------------------------------- what must not
def test_a_release_signed_by_someone_else_is_refused(root, release_key):
    manifest = offer(root, release_key, signer=Ed25519PrivateKey.generate())
    assert payload_mod.install(manifest, installed_version="0.1.0") is None
    assert not (root / "agent" / "0.2.0").exists()


def test_an_unsigned_release_is_refused(root, release_key):
    incoming = root / "agent" / "incoming"
    archive = build_archive(incoming / "0.2.0.tar.gz")
    path = incoming / "0.2.0.json"
    path.write_text(json.dumps({"version": "0.2.0", "sha256": digest_of(archive),
                                "signature": ""}))
    assert payload_mod.install(path, installed_version="0.1.0") is None


def test_an_old_release_served_under_a_new_number_is_refused(root, release_key):
    """Signing only the archive would let this through.

    The manifest is signed as a whole, so a genuine signature for 0.1.5 does
    not become a signature for 9.9.9 by renaming the file.
    """
    manifest = offer(root, release_key, version="9.9.9", sign_version="0.1.5")
    assert payload_mod.install(manifest, installed_version="0.1.0") is None
    assert not (root / "agent" / "9.9.9").exists()


def test_a_release_older_than_the_running_one_is_refused(root, release_key):
    """Last year's release is still validly signed. That is how a fixed
    vulnerability comes back."""
    manifest = offer(root, release_key, version="0.0.9")
    assert payload_mod.install(manifest, installed_version="0.5.0") is None


def test_an_archive_that_does_not_match_its_manifest_is_refused(root, release_key):
    manifest = offer(root, release_key, digest="ab" * 32)
    assert payload_mod.install(manifest, installed_version="0.1.0") is None
    assert not (root / "agent" / "0.2.0").exists()


def test_an_archive_that_writes_outside_is_refused(root, release_key):
    manifest = offer(root, release_key, extra="../../escaped")
    assert payload_mod.install(manifest, installed_version="0.1.0") is None
    assert not (root.parent / "escaped").exists()


def test_an_archive_with_no_agent_in_it_is_refused(root, release_key):
    incoming = root / "agent" / "incoming"
    archive = build_archive(incoming / "0.2.0.tar.gz", agent=False)
    path = incoming / "0.2.0.json"
    manifest = Manifest(version="0.2.0", sha256=digest_of(archive))
    path.write_text(json.dumps({
        "version": "0.2.0", "sha256": manifest.sha256,
        "signature": release_key.sign(manifest.canonical()).hex(),
    }))
    assert payload_mod.install(path, installed_version="0.1.0") is None


def test_an_image_with_no_key_installs_nothing(root, monkeypatch):
    """Updates off is a safe state. Installing unverified code is not."""
    monkeypatch.delenv("LOOM_RELEASE_PUBKEY", raising=False)
    monkeypatch.setattr(
        "loom_launcher.signature.KEY_FILE", root / "no-such-key")
    with pytest.raises(Untrusted) as exc:
        verify(Manifest(version="0.2.0", sha256="ab" * 32), b"x" * 64)
    assert "no release key" in str(exc.value)


# ------------------------------------------------------------- switching back
def test_switching_remembers_what_it_replaced(root, release_key):
    first = payload_mod.install(offer(root, release_key, version="0.2.0"),
                                installed_version="0.1.0")
    payload_mod.switch_to(first)
    second = payload_mod.install(offer(root, release_key, version="0.3.0"),
                                 installed_version="0.2.0")
    payload_mod.switch_to(second)
    assert payload_mod.resolve().version == "0.3.0"
    assert payload_mod.previous_link().resolve().name == "0.2.0"


def test_rolling_back_restores_the_previous_payload(root, release_key):
    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.2.0"), installed_version="0.1.0"))
    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.3.0"), installed_version="0.2.0"))
    restored = payload_mod.roll_back()
    assert restored.version == "0.2.0"
    assert payload_mod.resolve().version == "0.2.0"


def test_rolling_back_with_nothing_to_return_to_uses_the_bundled_agent(root):
    assert payload_mod.roll_back().bundled


# ------------------------------------------------------- the rollback decision
def _supervisor(payload) -> Supervisor:
    supervisor = Supervisor(payload, [])
    supervisor.consecutive_failures = FAILURES_BEFORE_ROLLBACK
    return supervisor


def test_a_version_that_never_registered_is_rolled_back(root, release_key):
    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.2.0"), installed_version="0.1.0"))
    broken = payload_mod.install(offer(root, release_key, version="0.3.0"),
                                 installed_version="0.2.0")
    payload_mod.switch_to(broken)
    supervisor = _supervisor(broken)
    supervisor._consider_rollback()
    assert supervisor.payload.version == "0.2.0"
    assert supervisor.consecutive_failures == 0


def test_a_version_that_once_worked_is_not_rolled_back(root, release_key):
    """Then the new thing is not what broke, and going back hides the cause."""
    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.2.0"), installed_version="0.1.0"))
    working = payload_mod.install(offer(root, release_key, version="0.3.0"),
                                  installed_version="0.2.0")
    payload_mod.switch_to(working)
    payload_mod.health_marker("0.3.0").write_text(str(time.time()))
    supervisor = _supervisor(working)
    supervisor._consider_rollback()
    assert supervisor.payload.version == "0.3.0"


def test_one_crash_is_not_enough_to_roll_back(root, release_key):
    """A single crash can be the machine — a card gone, a full disk."""
    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.2.0"), installed_version="0.1.0"))
    new = payload_mod.install(offer(root, release_key, version="0.3.0"),
                              installed_version="0.2.0")
    payload_mod.switch_to(new)
    supervisor = Supervisor(new, [])
    supervisor.consecutive_failures = 1
    supervisor._consider_rollback()
    assert supervisor.payload.version == "0.3.0"


def test_the_bundled_agent_is_never_rolled_back_from(root):
    supervisor = _supervisor(payload_mod.bundled())
    supervisor._consider_rollback()
    assert supervisor.payload.bundled


# ------------------------------------------------------------------ draining
def test_a_draining_node_takes_no_new_work_but_finishes_what_it_has(tmp_path, monkeypatch):
    """A task in flight is somebody's work; they already paid for the power."""
    import sys as _sys

    from loom_agent.tasks.env import EnvironmentCache
    from loom_agent.tasks.limits import resolve_isolation
    from loom_agent.tasks.registry import TaskRegistry
    from loom_agent.tasks.spec import TaskRefused, TaskSpec

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    registry = TaskRegistry(
        root=tmp_path / "tasks", isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0, retention_s=60.0,
    )
    running = registry.submit(TaskSpec.from_dict({
        "task_id": "d1", "command": [_sys.executable, "-c", "import time; time.sleep(2)"],
    }))

    drained = []
    import threading

    threading.Thread(target=lambda: drained.append(registry.drain(30.0)),
                     daemon=True).start()
    time.sleep(0.3)

    with pytest.raises(TaskRefused) as exc:
        registry.submit(TaskSpec.from_dict({"task_id": "d2", "command": ["true"]}))
    assert "restarting" in str(exc.value)

    assert running.wait(timeout=30)
    deadline = time.time() + 15
    while time.time() < deadline and not drained:
        time.sleep(0.1)
    assert drained == [True], "draining did not finish once the task was done"
    registry.stop_all()


def test_draining_gives_up_rather_than_waiting_forever(tmp_path, monkeypatch):
    import sys as _sys

    from loom_agent.tasks.env import EnvironmentCache
    from loom_agent.tasks.limits import resolve_isolation
    from loom_agent.tasks.registry import TaskRegistry
    from loom_agent.tasks.spec import TaskSpec

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    registry = TaskRegistry(
        root=tmp_path / "tasks", isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0, retention_s=60.0,
    )
    registry.submit(TaskSpec.from_dict({
        "task_id": "d3", "command": [_sys.executable, "-c", "import time; time.sleep(60)"],
        "timeout_s": 120,
    }))
    assert registry.drain(1.0) is False
    registry.stop_all()


# --------------------------------------------------------------- convergence
def test_an_agent_started_by_hand_does_not_pretend_to_update(monkeypatch):
    """No launcher means nothing could install it, so saying so beats fetching."""
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    monkeypatch.delenv("LOOM_AGENT_INCOMING", raising=False)
    stopped = []
    updater = Updater(current_version="0.1.0", drain=lambda _t: True,
                      stop=lambda: stopped.append(True))
    updater.on_release(agent_pb2.AgentRelease(version="0.2.0", url="http://x/a.tar.gz"))
    deadline = time.time() + 5
    while time.time() < deadline and not updater.last_refusal:
        time.sleep(0.05)
    assert updater.status().state == "refused"
    assert stopped == [], "оно остановило агента ради обновления, которого не скачало"


def test_the_version_we_already_run_is_not_fetched_again(monkeypatch):
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    fetched = []
    updater = Updater(current_version="0.2.0", drain=lambda _t: True, stop=lambda: None)
    updater._download = lambda *a, **k: fetched.append(True) or True
    updater.on_release(agent_pb2.AgentRelease(version="0.2.0", url="http://x/a.tar.gz"))
    time.sleep(0.3)
    assert fetched == []


def test_a_release_named_with_no_url_is_ignored(monkeypatch):
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    stopped = []
    updater = Updater(current_version="0.1.0", drain=lambda _t: True,
                      stop=lambda: stopped.append(True))
    updater.on_release(agent_pb2.AgentRelease(version="0.2.0"))
    time.sleep(0.3)
    assert stopped == []


def test_a_download_that_fails_leaves_the_node_on_the_version_it_has(tmp_path, monkeypatch):
    """A node that cannot fetch an update is a node running an old version,
    which is a much smaller problem than a node that stops."""
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    monkeypatch.setenv("LOOM_AGENT_INCOMING", str(tmp_path / "incoming"))
    stopped = []
    updater = Updater(current_version="0.1.0", drain=lambda _t: True,
                      stop=lambda: stopped.append(True))
    updater.on_release(agent_pb2.AgentRelease(
        version="0.2.0", url="http://127.0.0.1:1/nothing.tar.gz"))
    deadline = time.time() + 20
    while time.time() < deadline and not updater.last_refusal:
        time.sleep(0.1)
    assert updater.status().state == "refused"
    assert "скачать" in updater.last_refusal
    assert stopped == [], "оно перезапустилось ради обновления, которое не приехало"
    assert not list((tmp_path / "incoming").glob("*.json"))


def test_плановый_выход_ради_обновления_не_считается_падением(root, release_key):
    """Иначе три обновления подряд подвели бы исправную версию под откат.

    Обычный ноль от «я ушёл, чтобы уступить место» не отличить, поэтому агент
    выходит отдельным кодом.
    """
    from loom_launcher.supervise import UPDATE_EXIT_CODE

    payload_mod.switch_to(payload_mod.install(
        offer(root, release_key, version="0.2.0"), installed_version="0.1.0"))
    current = payload_mod.resolve()
    supervisor = Supervisor(current, [])
    supervisor.consecutive_failures = 2

    # Так пусковой слой видит завершение: код и время жизни.
    assert UPDATE_EXIT_CODE != 0, "плановый выход должен отличаться от обычного"
    supervisor._run_once = lambda: UPDATE_EXIT_CODE
    supervisor._stop.set()          # один проход и выходим
    supervisor.run_forever()
    assert supervisor.consecutive_failures == 2, "проход при остановке ничего не менял"


def test_агент_сообщает_что_ушёл_ради_обновления():
    """Код выхода — единственное, что пусковой слой видит от агента."""
    from loom_agent.update import UPDATE_EXIT_CODE, Updater
    from loom_launcher.supervise import UPDATE_EXIT_CODE as SEEN_BY_LAUNCHER

    assert UPDATE_EXIT_CODE == SEEN_BY_LAUNCHER, \
        "две половины разошлись в том, что означает этот код"
    updater = Updater(current_version="0.1.0", drain=lambda _t: True, stop=lambda: None)
    assert updater.exit_code == 0, "до обновления выход обычный"


def test_узел_рассказывает_почему_не_обновился(monkeypatch):
    """«Ступень переведена, а версия не меняется» не должно требовать похода в
    лог на самой машине — это ровно то место, куда оператор идти не хочет."""
    import time as _time

    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    monkeypatch.delenv("LOOM_AGENT_INCOMING", raising=False)
    updater = Updater(current_version="0.1.0", drain=lambda _t: True, stop=lambda: None)
    assert updater.status().state == "idle"

    updater.on_release(agent_pb2.AgentRelease(version="0.2.0", url="http://x/a.tar.gz"))
    deadline = _time.time() + 5
    while _time.time() < deadline and updater.status().state == "idle":
        _time.sleep(0.05)

    status = updater.status()
    assert status.state == "refused"
    assert status.version == "0.2.0", "узел должен назвать версию, о которой знает"
    assert "пускового слоя" in status.error


def test_релиз_без_адреса_тоже_объясняется():
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    updater = Updater(current_version="0.1.0", drain=lambda _t: True, stop=lambda: None)
    updater.on_release(agent_pb2.AgentRelease(version="0.2.0"))
    assert updater.status().state == "refused"
    assert "адреса" in updater.status().error


def test_два_агента_на_общем_томе_качают_релиз_каждый_в_свой_файл(tmp_path, monkeypatch):
    """Симптом со стенда: у второго агента
    "[Errno 2] No such file or directory: .../0.2.0.tar.part".

    Оба писали в один и тот же временный файл на общем томе; кто переименовал
    первым, у второго исходник исчезал.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    body = b"payload bytes" * 1000

    class Serve(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Serve)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    release = agent_pb2.AgentRelease(
        version="0.2.0", sha256="ab" * 32, signature=b"\x01" * 64,
        url=f"http://127.0.0.1:{server.server_port}/a.tar.gz")

    shared = tmp_path / "incoming"
    updaters = [Updater(current_version="0.1.0", drain=lambda _t: True, stop=lambda: None)
                for _ in range(2)]
    # Два процесса-агента = два разных pid; здесь их подменяем, потому что
    # процесс один, а проверяем именно развод по имени.
    pids = iter([4242, 4243])
    monkeypatch.setattr("loom_agent.update.os.getpid", lambda: next(pids))

    try:
        assert all(u._download(release, shared) for u in updaters), \
            "загрузка не прошла у обоих"
    finally:
        server.shutdown()

    assert (shared / "0.2.0.tar.gz").read_bytes() == body
    assert not list(shared.glob("*.part")), "временные файлы остались лежать"


def test_версию_уже_поставленную_соседом_не_ломают(root, release_key):
    """Общий том: второй пусковой слой не должен сносить то, что распаковал
    первый — подпись проверена в обоих."""
    first = payload_mod.install(offer(root, release_key, version="0.2.0"),
                                installed_version="0.1.0")
    assert first is not None
    marker = first.path / "loom_agent" / "already-there"
    marker.write_text("сосед")

    second = payload_mod.install(offer(root, release_key, version="0.2.0"),
                                 installed_version="0.1.0")
    assert second is not None
    assert second.path == first.path
    assert marker.exists(), "распаковку соседа снесли и сделали заново"


# ---------------------------------------------- образ без ключа релизов
def test_узел_без_ключа_не_качает_то_что_не_поставит(tmp_path, monkeypatch):
    """Со стенда: образ собран без ключа, релиз опубликован — и узел качал,
    сливал задачи, перезапускался, получал отказ и начинал сначала. Круг на
    чужой машине и её канале, из которого сам он выйти не мог."""
    from loom_agent.proto import agent_pb2
    from loom_agent.update import Updater

    monkeypatch.setenv("LOOM_AGENT_INCOMING", str(tmp_path / "incoming"))
    monkeypatch.setenv("LOOM_UPDATES_DISABLED", "в образе нет ключа релизов")
    fetched, stopped = [], []
    updater = Updater(current_version="0.1.0", drain=lambda _t: True,
                      stop=lambda: stopped.append(True))
    updater._download = lambda *a, **k: fetched.append(True) or True
    updater.on_release(agent_pb2.AgentRelease(version="0.2.0", url="http://x/a.tar.gz"))

    deadline = time.time() + 5
    while time.time() < deadline and not updater.last_refusal:
        time.sleep(0.05)
    assert fetched == [], "качал релиз, который заведомо не поставится"
    assert stopped == [], "перезапустил узел ради обновления, которого не будет"
    assert updater.status().state == "refused"
    assert "ключа релизов" in updater.status().error, \
        "причина должна дойти до панели — там её и будут читать"


def test_пусковой_слой_называет_причину_сам(monkeypatch, tmp_path):
    """Знает про ключ только он: агент об этом узнаёт из окружения."""
    from loom_launcher.supervise import _why_updates_are_off

    monkeypatch.delenv("LOOM_RELEASE_PUBKEY", raising=False)
    monkeypatch.setattr("loom_launcher.signature.KEY_FILE", tmp_path / "нет-ключа")
    assert "нет ключа релизов" in _why_updates_are_off()

    monkeypatch.setenv("LOOM_RELEASE_PUBKEY", "aa" * 32)
    assert _why_updates_are_off() == ""


def test_два_агента_на_одной_машине_не_делят_файл_загрузки(tmp_path, monkeypatch):
    """Со стенда: оба в своих контейнерах — процесс номер 7, и .part у них
    совпадал. Кто переименовал первым, у второго файл исчезал:

        No such file or directory: '.0.2.0.7.part' -> '0.2.0.tar.gz'
    """
    from loom_agent.update import Updater

    incoming = tmp_path / "incoming"
    monkeypatch.setenv("LOOM_AGENT_INCOMING", str(incoming))
    monkeypatch.delenv("LOOM_UPDATES_DISABLED", raising=False)
    monkeypatch.setattr(os, "getpid", lambda: 7)   # как в контейнере

    names = set()
    real_replace = os.replace

    def watch(src, dst):
        names.add(os.path.basename(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watch)

    class Release:
        version, sha256, url = "0.2.0", "", "http://x/a.tar.gz"
        signature = b""

    for _ in range(2):
        updater = Updater(current_version="0.1.0", drain=lambda _t: True,
                          stop=lambda: None)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: __import__("io").BytesIO(b"payload"))
        assert updater._download(Release(), incoming), updater.last_refusal

    assert len(names) == 2, f"оба процесса писали в один файл: {names}"
