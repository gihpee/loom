"""Phase 6: environments that are not Python.

`binary` is files. `oci` is a container image, fetched and unpacked without a
daemon — which is the point: images give universality, a daemon gives
privileges we refuse to ask for and nesting we cannot have.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from loom_agent.tasks.env import EnvironmentCache
from loom_agent.tasks.env.archive import BadArchive, apply_layer, extract
from loom_agent.tasks.env.oci import PullFailed, parse_reference, platform_pair, pull
from loom_agent.tasks.spec import EnvSpec, TaskRefused


def tar_of(members: dict, path: Path, *, links: dict = None,
           hardlinks: dict = None) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(content))
        for name, destination in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = destination
            tar.addfile(info)
        for name, destination in (hardlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = destination
            tar.addfile(info)
    return path


# ------------------------------------------------------------------ archives
def test_an_archive_unpacks(tmp_path):
    archive = tar_of({"bin/tool": b"#!/bin/sh\necho hi\n", "share/data": b"x"},
                     tmp_path / "a.tar.gz")
    target = tmp_path / "out"
    assert extract(archive, target) == 2
    assert (target / "bin" / "tool").read_bytes().startswith(b"#!")


def test_an_archive_that_writes_outside_is_refused(tmp_path):
    archive = tar_of({"../escaped": b"x"}, tmp_path / "a.tar.gz")
    with pytest.raises(BadArchive):
        extract(archive, tmp_path / "out")


def test_a_link_pointing_out_of_the_tree_is_refused(tmp_path):
    """A symlink to /etc/shadow inside an image would be followed by anything
    that later walked it."""
    archive = tar_of({"ok": b"x"}, tmp_path / "a.tar.gz",
                     links={"sneaky": "../../../../etc/shadow"})
    with pytest.raises(BadArchive) as exc:
        extract(archive, tmp_path / "out")
    assert "outside" in str(exc.value)


def test_an_absolute_link_means_inside_the_image(tmp_path):
    """`/bin/sh -> /bin/busybox` is the image's own /bin/busybox.

    Every real image is full of these — alpine alone has dozens — so reading an
    absolute link as an escape rejects almost everything on Docker Hub. Found
    by pulling alpine, not by thinking about it.
    """
    archive = tar_of({"bin/busybox": b"x"}, tmp_path / "a.tar.gz",
                     links={"bin/sh": "/bin/busybox", "bin/arch": "/bin/busybox"})
    target = tmp_path / "out"
    extract(archive, target)
    assert (target / "bin" / "sh").is_symlink()
    assert str((target / "bin" / "sh").readlink()) == "/bin/busybox"


def test_an_absolute_link_out_of_the_image_is_still_refused(tmp_path):
    archive = tar_of({"ok": b"x"}, tmp_path / "a.tar.gz",
                     links={"bin/sh": "/../../../etc/shadow"})
    with pytest.raises(BadArchive):
        extract(archive, tmp_path / "out")


def test_a_link_inside_the_tree_is_kept(tmp_path):
    """Real images are full of them: /bin/sh -> busybox and the like."""
    archive = tar_of({"busybox": b"x"}, tmp_path / "a.tar.gz", links={"sh": "busybox"})
    target = tmp_path / "out"
    extract(archive, target)
    assert (target / "sh").is_symlink()
    assert (target / "sh").readlink().name == "busybox"


# -------------------------------------------------------------- layer diffs
def test_a_later_layer_replaces_a_file(tmp_path):
    root = tmp_path / "rootfs"
    apply_layer(tar_of({"etc/conf": b"first"}, tmp_path / "1.tar.gz"), root)
    apply_layer(tar_of({"etc/conf": b"second"}, tmp_path / "2.tar.gz"), root)
    assert (root / "etc" / "conf").read_bytes() == b"second"


def test_a_whiteout_actually_deletes(tmp_path):
    """Ignoring these leaves files a later layer removed — most visibly a
    package that was uninstalled coming back."""
    root = tmp_path / "rootfs"
    apply_layer(tar_of({"usr/bin/gone": b"x", "usr/bin/kept": b"y"},
                       tmp_path / "1.tar.gz"), root)
    apply_layer(tar_of({"usr/bin/.wh.gone": b""}, tmp_path / "2.tar.gz"), root)
    assert not (root / "usr" / "bin" / "gone").exists()
    assert (root / "usr" / "bin" / "kept").exists()


def test_an_opaque_whiteout_empties_the_directory(tmp_path):
    root = tmp_path / "rootfs"
    apply_layer(tar_of({"var/cache/a": b"x", "var/cache/b": b"y"},
                       tmp_path / "1.tar.gz"), root)
    apply_layer(tar_of({"var/cache/.wh..wh..opq": b"", "var/cache/new": b"z"},
                       tmp_path / "2.tar.gz"), root)
    assert not (root / "var" / "cache" / "a").exists()
    assert (root / "var" / "cache" / "new").exists()


def test_a_whiteout_cannot_delete_outside_the_root(tmp_path):
    root = tmp_path / "rootfs"
    root.mkdir()
    victim = tmp_path / "important"
    victim.write_text("not the image's to remove")
    with pytest.raises(BadArchive):
        apply_layer(tar_of({"../.wh.important": b""}, tmp_path / "evil.tar.gz"), root)
    assert victim.exists()


# ------------------------------------------------------------------ registry
@pytest.mark.parametrize("given,expected", [
    ("alpine", "registry-1.docker.io/library/alpine:latest"),
    ("python:3.12-slim", "registry-1.docker.io/library/python:3.12-slim"),
    ("ghcr.io/owner/name:tag", "ghcr.io/owner/name:tag"),
    ("localhost:5000/mine:v1", "localhost:5000/mine:v1"),
])
def test_an_image_name_is_read_the_way_everyone_writes_it(given, expected):
    assert parse_reference(given).describe() == expected


def test_an_empty_image_name_says_what_one_looks_like():
    with pytest.raises(PullFailed) as exc:
        parse_reference("")
    assert "python:3.12-slim" in str(exc.value)


class FakeRegistry:
    """A registry with one image. Real HTTP, real manifest, real layers.

    Written out rather than mocked because the thing worth testing is that we
    speak the protocol, and a mock would only prove we can call our own code.
    """

    def __init__(self, tmp_path: Path, *, index: bool = False,
                 other_platform_only: bool = False) -> None:
        self.blobs: dict = {}
        self.tmp_path = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.other_platform_only = other_platform_only
        self.config = {"config": {
            "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8"],
            "Cmd": ["/bin/sh"], "WorkingDir": "/srv",
        }}
        self._build()

    def _add(self, payload: bytes) -> str:
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        self.blobs[digest] = payload
        return digest

    def _build(self) -> None:
        first = tar_of({"bin/hello": b"#!/bin/sh\necho hello\n", "etc/conf": b"old"},
                       self.tmp_path / "l1.tar.gz").read_bytes()
        second = tar_of({"etc/conf": b"new", "etc/.wh.removed": b""},
                        self.tmp_path / "l2.tar.gz").read_bytes()
        config_digest = self._add(json.dumps(self.config).encode())
        layers = [self._add(first), self._add(second)]
        self.manifest = {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": 1},
            "layers": [{"digest": d, "size": len(self.blobs[d])} for d in layers],
        }
        system, architecture = platform_pair()
        if self.other_platform_only:
            architecture = "s390x"
        self.index_document = {
            "schemaVersion": 2,
            "manifests": [{
                "digest": self._add(json.dumps(self.manifest).encode()),
                "platform": {"os": system, "architecture": architecture},
            }],
        }

    def start(self) -> str:
        registry = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if "/manifests/" in self.path:
                    reference = self.path.rsplit("/", 1)[1]
                    if reference.startswith("sha256:"):
                        body = registry.blobs[reference]
                    elif registry.index:
                        body = json.dumps(registry.index_document).encode()
                    else:
                        body = json.dumps(registry.manifest).encode()
                elif "/blobs/" in self.path:
                    body = registry.blobs.get(self.path.rsplit("/", 1)[1])
                    if body is None:
                        self.send_error(404)
                        return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"127.0.0.1:{self.server.server_port}/team/app:v1"

    def stop(self) -> None:
        self.server.shutdown()


@pytest.fixture
def registry(tmp_path):
    """A started registry; the fixture hands over its image reference."""
    fake = FakeRegistry(tmp_path / "reg")
    fake.reference = fake.start()
    yield fake
    fake.stop()


def test_an_image_is_pulled_and_its_layers_applied_in_order(tmp_path, registry):
    image = pull(registry.reference, tmp_path / "env")
    assert (image.root / "bin" / "hello").read_bytes().startswith(b"#!")
    # The second layer wins, and the whiteout in it took effect.
    assert (image.root / "etc" / "conf").read_bytes() == b"new"
    assert not (image.root / "etc" / "removed").exists()


def test_the_image_says_how_it_expects_to_be_run(tmp_path, registry):
    image = pull(registry.reference, tmp_path / "env")
    assert image.environment()["PATH"].startswith("/usr/local/bin")
    assert image.working_dir() == "/srv"
    assert image.default_command == ["/bin/sh"]


def test_a_multi_platform_image_picks_this_machine(tmp_path):
    fake = FakeRegistry(tmp_path / "reg", index=True)
    try:
        image = pull(fake.start(), tmp_path / "env")
        assert (image.root / "bin" / "hello").exists()
    finally:
        fake.stop()


def test_an_image_with_no_build_for_us_says_what_it_has(tmp_path):
    fake = FakeRegistry(tmp_path / "reg", index=True, other_platform_only=True)
    try:
        with pytest.raises(PullFailed) as exc:
            pull(fake.start(), tmp_path / "env")
        assert "s390x" in str(exc.value)
    finally:
        fake.stop()


def test_a_registry_that_is_not_there_fails_with_the_address(tmp_path):
    with pytest.raises(PullFailed) as exc:
        pull("127.0.0.1:1/nothing/here:v1", tmp_path / "env")
    assert "127.0.0.1:1" in str(exc.value)


# ------------------------------------------------------------------- caching
def test_an_image_is_pulled_once_and_reused(tmp_path, registry):
    cache = EnvironmentCache(tmp_path / "envs")
    spec = EnvSpec(kind="oci", source=registry.reference)
    first = cache.acquire(spec)
    assert first.rootfs is not None
    assert first.image_env["LANG"] == "C.UTF-8"
    assert first.image_workdir == "/srv"

    cache.release(first.fingerprint)
    registry.stop()  # gone: a second pull could not succeed
    second = cache.acquire(spec)
    assert second.path == first.path
    assert second.image_env["LANG"] == "C.UTF-8", "the image config was not cached"


def test_an_image_supplies_its_own_path_to_the_task(tmp_path, registry):
    cache = EnvironmentCache(tmp_path / "envs")
    image = cache.acquire(EnvSpec(kind="oci", source=registry.reference))
    overrides = image.overrides()
    assert overrides["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert "VIRTUAL_ENV" not in overrides


# -------------------------------------------------------------------- binary
def test_a_binary_environment_unpacks_and_goes_on_path(tmp_path):
    archive = tar_of({"bin/tool": b"#!/bin/sh\necho from-the-archive\n"},
                     tmp_path / "tool.tar.gz")
    cache = EnvironmentCache(tmp_path / "envs")
    built = cache.acquire(EnvSpec(kind="binary", source=str(archive)))
    assert (built.path / "bin" / "tool").exists()
    assert built.overrides()["PATH"].startswith(str(built.path / "bin"))


def test_a_binary_source_that_is_neither_url_nor_file_says_so(tmp_path):
    cache = EnvironmentCache(tmp_path / "envs")
    with pytest.raises(TaskRefused) as exc:
        cache.acquire(EnvSpec(kind="binary", source="tool-2.1"))
    assert "URL" in str(exc.value)


def test_a_binary_environment_needs_a_source(tmp_path):
    cache = EnvironmentCache(tmp_path / "envs")
    with pytest.raises(TaskRefused):
        cache.acquire(EnvSpec(kind="binary"))


def test_a_task_runs_a_program_from_a_binary_environment(tmp_path, monkeypatch):
    """The whole point of the kind: something that is not Python."""
    from loom_agent.tasks.limits import resolve_isolation
    from loom_agent.tasks.registry import TaskRegistry
    from loom_agent.tasks.spec import TaskSpec

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    archive = tar_of({"bin/greet": b"#!/bin/sh\necho from-the-archive\n"},
                     tmp_path / "tool.tar.gz")
    registry = TaskRegistry(
        root=tmp_path / "tasks", isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0, retention_s=60.0,
    )
    task = registry.submit(TaskSpec.from_dict({
        "task_id": "b1", "command": ["greet"],
        "environment": {"kind": "binary", "source": str(archive)},
    }))
    assert task.wait(timeout=60)
    assert task.state == "done", task.logs()
    assert "from-the-archive" in task.logs()
    registry.stop_all()


# ---------------------------------------------------------------- refusals
def test_running_an_image_without_root_is_refused_with_the_reason(tmp_path, registry,
                                                                  monkeypatch):
    """Entering an image needs root. Saying so beats running the task outside
    the image it asked for and letting it fail on a missing library."""
    from loom_agent.tasks.limits import resolve_isolation
    from loom_agent.tasks.registry import TaskRegistry
    from loom_agent.tasks.spec import TaskSpec

    monkeypatch.setenv("LOOM_ALLOW_UNPRIVILEGED_TASKS", "1")
    monkeypatch.setattr("loom_agent.tasks.runner.can_chroot", lambda: False)
    tasks = TaskRegistry(
        root=tmp_path / "tasks", isolation=resolve_isolation(),
        environments=EnvironmentCache(tmp_path / "envs"),
        total_gpus=0, retention_s=60.0,
    )
    with pytest.raises(TaskRefused) as exc:
        tasks.submit(TaskSpec.from_dict({
            "task_id": "o1", "command": ["/bin/hello"],
            "environment": {"kind": "oci", "source": registry.reference},
        }))
    assert "root" in str(exc.value)
    tasks.stop_all()
