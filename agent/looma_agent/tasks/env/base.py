"""One provisioned environment, and how a task uses it."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

MARKER = ".looma-env.json"


@dataclass(frozen=True)
class Environment:
    """A directory a task can run against.

    `path is None` is the real case of a command that needs nothing installed,
    not a missing value: it keeps the runner from having to ask whether an
    environment exists before using one.
    """

    fingerprint: str
    kind: str = "none"
    path: Optional[Path] = None
    size_bytes: int = 0
    # For `oci`: the environment variables and working directory the image
    # itself declares. Empty for every other kind.
    image_env: Dict[str, str] = field(default_factory=dict)
    image_workdir: str = ""

    @property
    def empty(self) -> bool:
        return self.path is None

    @property
    def rootfs(self) -> Optional[Path]:
        """An unpacked image's filesystem root, when this is one.

        Present only for `oci`: it is what a task gets chrooted into, and its
        absence is how everything else knows this environment is not one.
        """
        if self.kind != "oci" or self.path is None:
            return None
        root = self.path / "rootfs"
        return root if root.is_dir() else None

    def bin_dir(self) -> Optional[Path]:
        if self.path is None:
            return None
        for name in ("bin", "Scripts"):
            candidate = self.path / name
            if candidate.is_dir():
                return candidate
        return None

    def overrides(self) -> Dict[str, str]:
        """What the task's environment gains from this one.

        Prepending to PATH rather than replacing it is what makes
        `command: ["python", "run.py"]` mean the environment's interpreter
        without the task having to know where it lives.
        """
        if self.path is None:
            return {}
        if self.kind == "oci":
            # An image's own PATH and friends. Without them a task in a python
            # image cannot find `python`: nothing outside the image knows where
            # the image decided to put it.
            return dict(self.image_env)
        if self.kind == "binary":
            bin_dir = self.bin_dir()
            return {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"} \
                if bin_dir else {}
        values: Dict[str, str] = {"VIRTUAL_ENV": str(self.path)}
        bin_dir = self.bin_dir()
        if bin_dir is not None:
            values["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        return values


NO_ENVIRONMENT = Environment(fingerprint="none", kind="none")


def write_marker(path: Path, *, fingerprint: str, kind: str, size_bytes: int,
                 extra: Optional[dict] = None) -> None:
    """Record what this directory is.

    Its presence is also the proof that the build finished: a directory without
    one is a half-built environment that must never be handed to a task.
    """
    record = {
        "fingerprint": fingerprint,
        "kind": kind,
        "size_bytes": size_bytes,
        "built_at": time.time(),
    }
    record.update(extra or {})
    (path / MARKER).write_text(json.dumps(record))


def read_marker(path: Path) -> Optional[dict]:
    try:
        return json.loads((path / MARKER).read_text())
    except (OSError, ValueError):
        return None


def directory_size(path: Path) -> int:
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                continue
    return total
