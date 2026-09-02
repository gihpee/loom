"""An environment that is just files: a program and what it needs.

For everything that is not Python and does not want a whole image — a Go or
Rust binary, an ffmpeg build, a model runner someone compiled. The archive is
unpacked and its `bin` directory, if it has one, goes on PATH.

Cheap on purpose. An image is the general answer; this is the one that costs
nothing when the general answer is more than the task needs.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from looma_agent.tasks.env.archive import BadArchive, extract
from looma_agent.tasks.spec import TaskRefused

logger = logging.getLogger("looma_agent.tasks.env.binary")

FETCH_TIMEOUT_S = int(os.environ.get("LOOMA_BINARY_TIMEOUT_S", "600"))


def build(target: Path, source: str) -> None:
    """Fetch and unpack `source` into `target`.

    `source` is a URL, or a path already on this node — the second is how a
    payload that arrived with the task itself is used, and how tests work
    without a network.
    """
    source = (source or "").strip()
    if not source:
        raise TaskRefused(
            "a binary environment needs a source: a URL or a path to an archive"
        )
    target.mkdir(parents=True, exist_ok=True)
    archive = _obtain(source)
    try:
        written = extract(archive, target)
        if not written:
            raise TaskRefused(f"the archive at {source} is empty")
        logger.info("binary environment: %d entries from %s", written, source)
    except BadArchive as exc:
        raise TaskRefused(f"{source} cannot be unpacked here: {exc}") from None
    finally:
        if archive != Path(source):
            archive.unlink(missing_ok=True)


def _obtain(source: str) -> Path:
    local = Path(source)
    if local.is_file():
        return local
    if not source.startswith(("http://", "https://")):
        raise TaskRefused(
            f"{source!r} is neither a URL nor a file on this node"
        )
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with urllib.request.urlopen(source, timeout=FETCH_TIMEOUT_S) as answer, \
                open(staging, "wb") as sink:
            shutil.copyfileobj(answer, sink, 1024 * 1024)
    except (urllib.error.URLError, OSError) as exc:
        staging.unlink(missing_ok=True)
        raise TaskRefused(f"could not fetch {source}: {exc}") from None
    return staging
