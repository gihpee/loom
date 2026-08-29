"""Where a task lives while it runs.

    tasks/<task_id>/
      work/    what the task runs in: its code, its inputs, its scratch
      out/     what it means to give back (phase 3 ships this)

Two directories rather than one because they have different fates: `work` is
scratch that nobody will ever look at again, `out` is the point of the whole
exercise. Keeping them apart means the agent can hand back a result without
guessing which of a hundred files the task meant.

Nothing from the host is mounted or linked in. A task sees what it was given
and nothing else.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loom_agent.tasks.limits import Isolation
from loom_agent.transport.files import Inbox, Outbox

logger = logging.getLogger("loom_agent.tasks.directory")


@dataclass(frozen=True)
class TaskDirectory:
    root: Path
    # An unpacked image the task runs inside. When set, `work` and `out` live
    # within it so they are still there after the chroot — at /work and /out.
    rootfs: Optional[Path] = None

    @property
    def base(self) -> Path:
        return self.rootfs or self.root

    @property
    def work(self) -> Path:
        return self.base / "work"

    @property
    def out(self) -> Path:
        return self.base / "out"

    @property
    def inner_work(self) -> str:
        """Where the task will see its own directory once it is in the image."""
        return "/work" if self.rootfs else str(self.work)

    @property
    def inner_out(self) -> str:
        return "/out" if self.rootfs else str(self.out)

    def with_rootfs(self, rootfs: Path, isolation: "Isolation") -> "TaskDirectory":
        moved = TaskDirectory(root=self.root, rootfs=rootfs)
        for path in (moved.work, moved.out):
            path.mkdir(parents=True, exist_ok=True)
            if isolation.uid is not None and isolation.gid is not None:
                os.chown(path, isolation.uid, isolation.gid)
            path.chmod(stat.S_IRWXU)
        return moved

    @classmethod
    def create(cls, base: Path, task_id: str, isolation: Isolation) -> "TaskDirectory":
        """Make the directory and hand it to the task's user.

        Mode 0700 after chown: the task owns its directory outright, and no
        other task — nor anything else running as another user in this
        container — can read it.
        """
        directory = cls(base / task_id)
        if directory.root.exists():
            # A leftover from a crash. Its contents belong to a task that is
            # gone; reusing them would leak one tenant's files to the next.
            logger.warning("removing a leftover directory for task %s", task_id)
            directory.remove()
        for path in (directory.root, directory.work, directory.out):
            path.mkdir(parents=True, exist_ok=True)
            if isolation.uid is not None and isolation.gid is not None:
                os.chown(path, isolation.uid, isolation.gid)
            path.chmod(stat.S_IRWXU)
        return directory

    def inbox(self, isolation: Isolation, *, limit_bytes: int = 0) -> Inbox:
        owner = (isolation.uid, isolation.gid) if isolation.uid is not None else None
        return Inbox(self.work, limit_bytes=limit_bytes, owner=owner)

    def outbox(self, *, limit_bytes: int = 0) -> Outbox:
        return Outbox(self.out, limit_bytes=limit_bytes)

    def remove(self) -> None:
        """Take it all back. Never fails the task: cleanup is not the point.

        Files written by the task belong to the task's user, which the agent
        (running as root) can still remove. When the agent is not root the
        removal can fail, and a warning is the honest outcome — pretending it
        worked would hide a disk filling up.
        """
        try:
            shutil.rmtree(self.root, ignore_errors=False)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("could not remove %s; it will keep using disk", self.root,
                           exc_info=True)

    def size_bytes(self) -> int:
        total = 0
        for current, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += (Path(current) / name).stat().st_size
                except OSError:
                    continue
        return total
