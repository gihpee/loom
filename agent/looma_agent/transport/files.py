"""Getting a task's input in and its result out.

This closes the gap that made the old compute path a demonstration rather than
a product: a task could only be handed environment variables and could only
answer through its logs, so anything that produced a file had nowhere to put
it.

Everything here treats names as HOSTILE. They are chosen by whoever submitted
the task, on somebody else's machine, and `../../.ssh/authorized_keys` is a
name. Path validation is therefore not a detail of this module — it is the
module's main job, and every path that reaches the filesystem goes through
`safe_target`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

logger = logging.getLogger("looma_agent.transport.files")

PARTIAL_SUFFIX = ".looma-partial"
# Read and write in pieces: a dataset must not have to fit in memory to move.
CHUNK_BYTES = 1024 * 1024


class TransferRefused(ValueError):
    """This node will not accept or return this, and says why."""


# --------------------------------------------------------------------- paths
def safe_target(base: Path, name: str) -> Path:
    """Where `name` may be written under `base`, or an error.

    Rejects absolute paths, parent traversal, and anything that resolves
    outside the base — including through a symlink somebody put there earlier.
    Resolving rather than string-checking is deliberate: `a/../../b` and a
    symlinked `a` are the same attack and only one of them looks like one.
    """
    if not name or name in (".", ".."):
        raise TransferRefused("a file needs a name")
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or name.startswith(("/", "\\")):
        raise TransferRefused(f"{name!r} is an absolute path; names must be relative")
    if any(part == ".." for part in candidate.parts):
        raise TransferRefused(f"{name!r} tries to leave the task's directory")
    base = base.resolve()
    target = (base / candidate)
    # The parent must resolve inside the base. The file itself may not exist
    # yet, so it is the directory that gets checked.
    parent = target.parent.resolve() if target.parent.exists() else _resolve_planned(base, candidate)
    if parent != base and base not in parent.parents:
        raise TransferRefused(f"{name!r} resolves outside the task's directory")
    return target


def _resolve_planned(base: Path, relative: Path) -> Path:
    """Resolve a parent that does not exist yet, following what does exist."""
    current = base
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise TransferRefused(
                f"{relative.as_posix()!r} goes through a symlink; that is how a "
                "file ends up somewhere it was not meant to"
            )
        if current.exists():
            current = current.resolve()
    return current


# ------------------------------------------------------------------ incoming
class IncomingFile:
    """One file arriving in pieces.

    Written under a partial name and renamed only when the whole thing has
    arrived and its digest matches. A task must never see an input that is
    half a file: it would read it, get nonsense, and fail somewhere unrelated.
    """

    def __init__(self, target: Path, *, expected_bytes: int = 0, digest: str = "") -> None:
        self.target = target
        self.expected_bytes = max(0, int(expected_bytes))
        self.expected_digest = (digest or "").lower()
        self.received = 0
        self._partial = target.with_name(target.name + PARTIAL_SUFFIX)
        self._digest = hashlib.sha256()
        self._handle = None

    def __enter__(self) -> "IncomingFile":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._partial, "wb")
        return self

    def __exit__(self, exc_type, *_rest) -> None:
        if exc_type is not None:
            self.abort()

    def write(self, chunk: bytes) -> None:
        if self._handle is None:
            raise TransferRefused("this transfer was already finished or aborted")
        self.received += len(chunk)
        if self.expected_bytes and self.received > self.expected_bytes:
            self.abort()
            raise TransferRefused(
                f"{self.target.name} sent more than the {self.expected_bytes} bytes "
                "it declared"
            )
        self._digest.update(chunk)
        self._handle.write(chunk)

    def finish(self) -> Path:
        """Make the file visible to the task, or refuse and leave nothing."""
        if self._handle is None:
            raise TransferRefused("this transfer was already finished or aborted")
        self._handle.close()
        self._handle = None
        if self.expected_bytes and self.received != self.expected_bytes:
            self.abort()
            raise TransferRefused(
                f"{self.target.name} stopped after {self.received} of "
                f"{self.expected_bytes} bytes"
            )
        actual = self._digest.hexdigest()
        if self.expected_digest and actual != self.expected_digest:
            self.abort()
            raise TransferRefused(f"{self.target.name} arrived corrupted")
        os.replace(self._partial, self.target)
        return self.target

    def abort(self) -> None:
        """Leave nothing behind. Safe to call more than once."""
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        try:
            self._partial.unlink()
        except FileNotFoundError:
            pass


class Inbox:
    """Where a task's input lands: its `work` directory and nowhere else."""

    def __init__(self, work: Path, *, limit_bytes: int = 0,
                 owner: Optional[tuple] = None) -> None:
        self.work = work
        self.limit_bytes = max(0, int(limit_bytes))
        # The agent writes these files, but the task has to be able to use
        # them. Without this an input arrives readable and unwritable, and a
        # task that edits its own input fails for a reason nobody would guess.
        self.owner = owner

    def _hand_over(self, path: Path) -> None:
        if self.owner is None:
            return
        try:
            os.chown(path, self.owner[0], self.owner[1])
        except (OSError, TypeError):
            logger.debug("could not hand %s to the task's user", path, exc_info=True)

    def receive(self, name: str, chunks: Iterable[bytes], *,
                size_bytes: int = 0, digest: str = "") -> Path:
        target = safe_target(self.work, name)
        self._check_budget(size_bytes)
        with IncomingFile(target, expected_bytes=size_bytes, digest=digest) as incoming:
            for chunk in chunks:
                incoming.write(chunk)
            written = incoming.finish()
        self._hand_over(written)
        return written

    def unpack(self, archive: Path, *, strip_components: int = 0) -> List[Path]:
        """Extract an archive of input files into the task's directory.

        Archives are the realistic case — a client sends code and data, not one
        file — and they are also where traversal has hidden for thirty years.
        Every member is checked before anything is written, and the archive is
        rejected whole rather than partially extracted: half an input is worse
        than none, because the task will run against it.
        """
        written: List[Path] = []
        with tarfile.open(archive, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile() or m.isdir()]
            for member in tar.getmembers():
                if member.issym() or member.islnk():
                    raise TransferRefused(
                        f"the archive contains a link ({member.name!r}); links are how "
                        "an archive writes outside where it was extracted"
                    )
            planned = []
            total = 0
            for member in members:
                name = _strip(member.name, strip_components)
                if not name:
                    continue
                planned.append((member, safe_target(self.work, name)))
                total += max(0, member.size)
            self._check_budget(total)
            for member, target in planned:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with open(target, "wb") as handle:
                    shutil.copyfileobj(source, handle, CHUNK_BYTES)
                self._hand_over(target)
                written.append(target)
        return written

    def _check_budget(self, incoming_bytes: int) -> None:
        if not self.limit_bytes:
            return
        already = _tree_bytes(self.work)
        if already + incoming_bytes > self.limit_bytes:
            raise TransferRefused(
                f"this task was given {self.limit_bytes / 1024**2:.0f} MB of disk and "
                f"its input needs more"
            )


def _strip(name: str, components: int) -> str:
    parts = Path(name).parts
    return str(Path(*parts[components:])) if len(parts) > components else ""


# ------------------------------------------------------------------ outgoing
@dataclass(frozen=True)
class ResultFile:
    name: str
    size_bytes: int
    digest: str


class Outbox:
    """What the task produced, and how it gets back to whoever asked.

    Only the `out` directory. A task writes scratch, checkpoints and logs all
    over its `work` directory, and shipping that back would send gigabytes
    nobody wanted — so what counts as the result is what the task deliberately
    put where it was told.
    """

    def __init__(self, out: Path, *, limit_bytes: int = 0) -> None:
        self.out = out
        self.limit_bytes = max(0, int(limit_bytes))

    def manifest(self) -> List[ResultFile]:
        if not self.out.is_dir():
            return []
        found: List[ResultFile] = []
        total = 0
        for path in sorted(self.out.rglob("*")):
            if path.is_symlink():
                # A result that is a link points at something the task did not
                # produce, which is the whole reason not to follow it.
                logger.warning("skipping %s in the result: it is a symlink", path.name)
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            if self.limit_bytes and total > self.limit_bytes:
                raise TransferRefused(
                    f"this task produced more than the "
                    f"{self.limit_bytes / 1024**2:.0f} MB it may return"
                )
            found.append(ResultFile(
                name=str(path.relative_to(self.out)),
                size_bytes=size,
                digest=_digest_of(path),
            ))
        return found

    def read(self, name: str) -> Iterator[bytes]:
        """Stream one result file back. Validated like anything else named."""
        target = safe_target(self.out, name)
        if not target.is_file():
            raise TransferRefused(f"this task produced no file called {name!r}")
        with open(target, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    return
                yield chunk

    def total_bytes(self) -> int:
        return _tree_bytes(self.out)


# --------------------------------------------------------------------- shared
def _digest_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
    return total
