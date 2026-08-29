"""Unpacking somebody else's archive without letting it write where it likes.

Both `binary` and `oci` environments are archives from elsewhere, so the rules
live in one place: no absolute paths, no traversal, no links out. A link is
refused rather than skipped — an image that expects `/bin/sh -> busybox` and
silently does not get it fails later in a way nobody will connect to this.
"""

from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger("loom_agent.tasks.env.archive")

# How an OCI layer says "this path is deleted in this layer".
WHITEOUT_PREFIX = ".wh."
OPAQUE_WHITEOUT = ".wh..wh..opq"


class BadArchive(ValueError):
    """This archive will not be unpacked, and why."""


def safe_member_path(root: Path, name: str) -> Optional[Path]:
    """Where a member may land, or None when it is not a real path at all."""
    cleaned = name.lstrip("/")
    if not cleaned or cleaned in (".", "./"):
        return None
    parts = Path(cleaned).parts
    if any(part == ".." for part in parts):
        raise BadArchive(f"{name!r} tries to write outside the archive root")
    return root.joinpath(*parts)


def extract(archive: Path, target: Path, *, allow_links: bool = True,
            strip_components: int = 0) -> int:
    """Unpack into `target`. Returns how many entries were written.

    Links are allowed by default because real images are full of them, but a
    link is only created when it stays inside the tree: a symlink to
    /etc/shadow inside an unpacked image would be followed by anything that
    later walked it.
    """
    written = 0
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = _strip(member.name, strip_components)
            if not name:
                continue
            where = safe_member_path(target, name)
            if where is None:
                continue
            if member.isdir():
                where.mkdir(parents=True, exist_ok=True)
                written += 1
            elif member.issym() or member.islnk():
                if not allow_links:
                    raise BadArchive(
                        f"the archive contains a link ({member.name!r}); that is how "
                        "an archive writes outside where it was unpacked"
                    )
                _link(target, where, member)
                written += 1
            elif member.isfile():
                where.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with open(where, "wb") as sink:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        sink.write(chunk)
                os.chmod(where, member.mode & 0o777)
                written += 1
            # Devices, fifos and sockets are dropped: a task has no business
            # being handed a device node it did not create.
    return written


def _link(root: Path, where: Path, member: tarfile.TarInfo) -> None:
    """Recreate a link only if it points inside the tree.

    An absolute link inside an image is relative to the image root, not to this
    machine: `/bin/sh -> /bin/busybox` means the image's own `/bin/busybox`,
    and it will once the image IS the root. Every real image is full of these,
    so reading them as escapes would reject almost everything on Docker Hub.
    """
    where.parent.mkdir(parents=True, exist_ok=True)
    if where.exists() or where.is_symlink():
        where.unlink()
    if member.issym():
        destination = member.linkname
        base = _real(root)
        target = base.joinpath(destination.lstrip("/")) if destination.startswith("/") \
            else _real(where.parent) / destination
        if not _inside(base, target):
            raise BadArchive(
                f"{member.name!r} links to {destination!r}, which is outside the archive"
            )
        where.symlink_to(destination)
        return
    source = safe_member_path(root, member.linkname)
    if source is None or not source.exists():
        # A hard link to something that was never unpacked. Skipped rather than
        # fatal: layers legitimately reference entries removed by a whiteout.
        return
    os.link(source, where)


def _real(path: Path) -> Path:
    """The path with symlinks followed as far as they exist.

    Both sides of a containment check must be normalised the same way or the
    check compares two spellings of the same directory and says no: on macOS
    /var and /private/var are one place, and one of them resolves.
    """
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(str(path)))


def _inside(root: Path, candidate: Path) -> bool:
    """Containment on normalised paths — the candidate need not exist yet."""
    normalised = Path(os.path.normpath(str(candidate)))
    return normalised == root or root in normalised.parents


def _strip(name: str, components: int) -> str:
    if components <= 0:
        return name
    parts = Path(name).parts
    return str(Path(*parts[components:])) if len(parts) > components else ""


def apply_layer(archive: Path, root: Path) -> int:
    """Unpack one OCI layer over what is already there.

    Layers are diffs, and a diff can delete: a `.wh.name` entry means "name is
    gone from here down", and `.wh..wh..opq` means "everything already in this
    directory is gone". Ignoring them leaves files a later layer deliberately
    removed — most visibly a package that was uninstalled coming back.
    """
    written = 0
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            base = os.path.basename(member.name)
            if base == OPAQUE_WHITEOUT:
                where = safe_member_path(root, os.path.dirname(member.name))
                if where is not None:
                    _empty(where)
                continue
            if base.startswith(WHITEOUT_PREFIX):
                target = os.path.join(os.path.dirname(member.name),
                                      base[len(WHITEOUT_PREFIX):])
                where = safe_member_path(root, target)
                if where is not None:
                    _remove(where)
                continue
            where = safe_member_path(root, member.name)
            if where is None:
                continue
            if member.isdir():
                where.mkdir(parents=True, exist_ok=True)
            elif member.issym() or member.islnk():
                _link(root, where, member)
            elif member.isfile():
                where.parent.mkdir(parents=True, exist_ok=True)
                if where.is_symlink():
                    where.unlink()
                source = tar.extractfile(member)
                if source is None:
                    continue
                with open(where, "wb") as sink:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        sink.write(chunk)
                os.chmod(where, member.mode & 0o777)
            else:
                continue
            written += 1
    return written


def _remove(path: Path) -> None:
    import shutil

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _empty(directory: Path) -> None:
    import shutil

    if not directory.is_dir():
        return
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
