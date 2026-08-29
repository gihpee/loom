"""Which agent to run.

Two places an agent can come from:

  bundled   installed in the image. Always present, never changes, and the
            floor we fall back to when everything else is broken.
  payload   unpacked under LOOM_ROOT/agent/<version>/, selected by the
            `current` symlink. This is what a network update replaces.

Switching versions is switching a symlink, which is atomic on POSIX: a reader
sees either the old target or the new one, never a half-written state.

The agent fetches; the launcher installs. That split is the point: the agent is
the part an update replaces, so it is not the part that decides what may be
installed. It drops an archive and a manifest into `incoming/` and stops; the
launcher verifies the signature, unpacks, and switches.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loom_launcher.signature import Manifest, Untrusted, check_archive, verify

logger = logging.getLogger("loom_launcher.payload")

BUILDING_PREFIX = ".building-"


def root() -> Path:
    return Path(os.environ.get("LOOM_ROOT", "/var/lib/loom"))


def agents_dir() -> Path:
    return root() / "agent"


def current_link() -> Path:
    return agents_dir() / "current"


def previous_link() -> Path:
    """The last payload known to have registered. The floor for a rollback."""
    return agents_dir() / "previous"


@dataclass(frozen=True)
class Payload:
    """How to start one agent."""

    version: str
    # Directory to put on PYTHONPATH, or None for the agent installed in the
    # image (already importable, nothing to prepend).
    path: Optional[Path]

    @property
    def bundled(self) -> bool:
        return self.path is None

    def describe(self) -> str:
        return f"{self.version} ({'bundled' if self.bundled else self.path})"


def bundled() -> Payload:
    """The agent shipped inside the image."""
    return Payload(version=_bundled_version(), path=None)


def _bundled_version() -> str:
    try:
        from importlib.metadata import version

        return version("loom-agent")
    except Exception:
        return "unknown"


def resolve(link: Path = None) -> Payload:
    """The payload to run now: whatever `current` points at, else the bundled one.

    A dangling or unreadable symlink is not an error worth stopping for — it
    means an update went wrong, and the right answer is to run the agent we
    know is intact rather than to leave the node dead.
    """
    link = link or current_link()
    try:
        target = link.resolve(strict=True)
    except (OSError, RuntimeError):
        return bundled()
    if not (target / "loom_agent" / "main.py").is_file():
        return bundled()
    return Payload(version=target.name, path=target)


def incoming_dir() -> Path:
    """Where the agent leaves what it downloaded."""
    return agents_dir() / "incoming"


def health_marker(version: str) -> Path:
    """Written by the agent once it has actually registered.

    "The process is alive" is not the same as "the agent works": a payload can
    start, fail to reach the orchestrator and sit there. This file is the
    difference, and it is what a rollback decision reads.
    """
    return agents_dir() / version / ".healthy"


def pending() -> List[Path]:
    """Manifests waiting to be installed, oldest first."""
    try:
        return sorted(incoming_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return []


def install(manifest_path: Path, *, installed_version: str = "") -> Optional[Payload]:
    """Verify and install one downloaded release. Returns it, or None.

    Never raises: a bad payload is a reason to keep running the agent we have,
    not a reason for the node to go down. The archive and manifest are removed
    either way, so a rejected release is not retried forever.
    """
    archive = manifest_path.with_suffix(".tar.gz")
    try:
        raw = json.loads(manifest_path.read_text())
        manifest = Manifest(version=raw["version"], sha256=raw["sha256"])
        signature = bytes.fromhex(raw.get("signature", ""))
    except (OSError, ValueError, KeyError) as exc:
        logger.error("ignoring an unreadable release manifest: %s", exc)
        _discard(manifest_path, archive)
        return None
    try:
        verify(manifest, signature, installed_version=installed_version)
        check_archive(archive, manifest)
        payload = _unpack(archive, manifest.version)
    except Untrusted as exc:
        logger.error("REFUSING release %s: %s", manifest.version, exc)
        _discard(manifest_path, archive)
        return None
    except (OSError, tarfile.TarError) as exc:
        logger.error("release %s could not be unpacked: %s", manifest.version, exc)
        _discard(manifest_path, archive)
        return None
    _discard(manifest_path, archive)
    logger.info("installed agent %s", manifest.version)
    return payload


def _discard(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


def _unpack(archive: Path, version: str) -> Payload:
    """Into a staging directory, then into place with a rename.

    Same reason as everywhere else in this codebase: a half-unpacked payload
    that something tries to run is a failure that looks nothing like its cause.
    """
    staging = agents_dir() / f"{BUILDING_PREFIX}{version}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk() or member.name.startswith(("/", "..")) \
                        or ".." in Path(member.name).parts:
                    raise Untrusted(
                        f"the release archive contains {member.name!r}, which would "
                        "write outside where it is unpacked"
                    )
            try:
                # Belt and braces: the members were already checked above, and
                # the stdlib filter catches anything that check did not think of.
                tar.extractall(staging, filter="data")
            except TypeError:
                tar.extractall(staging)  # older Python without the filter
        if not (staging / "loom_agent" / "main.py").is_file():
            raise Untrusted("the release archive holds no agent")
        final = agents_dir() / version
        shutil.rmtree(final, ignore_errors=True)
        os.rename(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Payload(version=version, path=agents_dir() / version)


def switch_to(payload: Payload) -> None:
    """Point `current` at this payload, remembering what it replaced.

    The symlink is replaced with a rename, which is atomic: nothing ever sees
    a moment with no `current` at all.
    """
    agents_dir().mkdir(parents=True, exist_ok=True)
    was = current_link()
    try:
        previous_target = was.resolve(strict=True)
    except (OSError, RuntimeError):
        previous_target = None
    if previous_target is not None and previous_target != payload.path:
        _point(previous_link(), previous_target)
    _point(current_link(), payload.path)


def roll_back() -> Optional[Payload]:
    """Go back to the last payload that worked, if there is one.

    Decided on the node, without asking anyone: the connection to the
    orchestrator may be exactly what the new version broke.
    """
    try:
        target = previous_link().resolve(strict=True)
    except (OSError, RuntimeError):
        logger.error("nothing to roll back to; falling back to the bundled agent")
        _discard(current_link())
        return bundled()
    _point(current_link(), target)
    logger.warning("rolled back to agent %s", target.name)
    return Payload(version=target.name, path=target)


def _point(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + ".swap")
    try:
        temporary.unlink()
    except OSError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)
