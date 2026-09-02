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

import hashlib
import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from looma_agent.tasks.limits import Isolation
from looma_agent.transport.files import Inbox, Outbox

logger = logging.getLogger("looma_agent.tasks.directory")


def _scratch_name(task_id: str) -> str:
    """Короткое имя каталога, путь к которому обязан быть коротким.

    Unix-сокет не может лежать глубже 103 байт пути — это предел ядра, а не
    соглашение. Каталог задачи с полным task_id в имени съедает почти весь
    лимит: `/var/lib/looma/tasks/group-abcdef0123-r0/work` — уже 48 байт, а
    софту, который кладёт сокеты рядом с собой, остаётся 41.

    Так падает Ray (плазма-сокет), так падают многие СУБД и IPC-библиотеки, и
    сообщение при этом называет длину пути, а не то, что каталог задачи просто
    глубоко лежит.

    Имя детерминированное, чтобы уборка нашла его, ничего не запоминая.
    """
    return "looma-" + hashlib.sha256(task_id.encode()).hexdigest()[:8]


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

    @property
    def scratch(self) -> Path:
        """Короткий каталог задачи на диске агента.

        Внутри образа — его же /tmp, чтобы путь остался коротким и после
        chroot: снаружи он длинный, а видит задача одно и то же имя.
        """
        base = self.rootfs if self.rootfs else Path("/")
        return base / "tmp" / _scratch_name(self.root.name)

    @property
    def inner_scratch(self) -> str:
        """Как его увидит сама задача — одинаково с образом и без."""
        return f"/tmp/{_scratch_name(self.root.name)}"

    def with_rootfs(self, rootfs: Path, isolation: "Isolation") -> "TaskDirectory":
        moved = TaskDirectory(root=self.root, rootfs=rootfs)
        for path in (moved.work, moved.out, moved.scratch):
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
        if directory.root.exists() or directory.scratch.exists():
            # A leftover from a crash. Its contents belong to a task that is
            # gone; reusing them would leak one tenant's files to the next.
            #
            # Короткий каталог сюда же, и не только ради утечки: в нём остаются
            # unix-сокеты, а bind() по занятому пути отказывает — задача падала
            # бы на «адрес занят» из-за предшественника, которого уже нет.
            logger.warning("removing a leftover directory for task %s", task_id)
            directory.remove()
        for path in (directory.root, directory.work, directory.out, directory.scratch):
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
        # Короткий каталог лежит вне root (в этом и был смысл), так что
        # rmtree по root его не заденет и убрать его надо отдельно. Первым:
        # иначе ранний выход по FileNotFoundError оставил бы его навсегда.
        shutil.rmtree(self.scratch, ignore_errors=True)
        try:
            shutil.rmtree(self.root, ignore_errors=False)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("could not remove %s; it will keep using disk", self.root,
                           exc_info=True)

    def size_bytes(self) -> int:
        # Вместе с коротким каталогом: он вне root, и без этого задача обходила
        # бы свою дисковую квоту, просто записав туда.
        roots = [self.root]
        if self.rootfs is None:
            roots.append(self.scratch)
        total = 0
        for base in roots:
            for current, _dirs, files in os.walk(base):
                for name in files:
                    try:
                        total += (Path(current) / name).stat().st_size
                    except OSError:
                        continue
        return total
