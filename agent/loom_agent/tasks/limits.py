"""Keeping a task inside what it was given.

The node owner lent us a machine, not surrendered it. Every limit here answers
something a careless or greedy task would otherwise do to a stranger's
computer.

Two mechanisms, because one is not enough:

  rlimits    applied in the child before it execs. Cheap, immediate, and they
             cover the failures that kill a machine outright — fork bombs,
             runaway files, core dumps.
  watchdog   a thread that watches how much memory the task's process tree
             actually holds, and kills the tree when it goes over. Needed
             because the address-space rlimit cannot be used on GPU tasks (see
             below) and because RSS is what actually competes with the owner.
"""

from __future__ import annotations

import logging
import os
import pwd
import resource
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from loom_agent.tasks.spec import Resources, TaskRefused

logger = logging.getLogger("loom_agent.tasks.limits")


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default

# Потолок против форк-бомбы, и только против неё.
#
# На Linux RLIMIT_NPROC считает НЕ процессы, а потоки, и считает их на весь
# uid — а все задачи узла работают под одним. Поэтому прежние 512 были куда
# меньше, чем выглядели: Ray поднимает raylet, GCS и воркеры, у каждого свои
# потоки, и два ранга на одной машине упирались в общий бюджет. Падало это
# так, что причина не следует ни из чего:
#
#   RuntimeError: can't start new thread
#   RuntimeError: Resource temporarily unavailable      (EAGAIN от fork)
#   Failed to register worker to Raylet: End of file    (у соседа умер raylet)
#
# Число выбрано с запасом намеренно. Смысл лимита — не нормировать потоки, а
# не дать одной задаче исчерпать машину; ниже kernel.threads-max он остаётся
# на порядок, так что от исчерпания защищает по-прежнему. А каждый круг
# подбора значения поменьше стоил дня: узнать, что его не хватило, можно
# только на стенде, и выглядит это как поломка совсем в другом месте.
#
# Жёсткий предел агент поднять не может (нужен CAP_SYS_RESOURCE, которого у
# контейнера по умолчанию нет) — `_set` зажимается по нему. Если он окажется
# ниже, задача получит его, и об этом скажет строка limits в логе агента.
MAX_PROCESSES = _int("LOOM_TASK_MAX_PROCESSES", 65536)
# A task should not be able to fill the owner's disk with one file.
MAX_FILE_BYTES = 32 * 1024**3
DEFAULT_RAM_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class Isolation:
    """Who a task runs as. Resolved once, at agent start."""

    uid: Optional[int]
    gid: Optional[int]
    user: str
    # True when tasks run as the agent's own user because privileges could not
    # be dropped. The operator opted into this explicitly.
    unprivileged_fallback: bool = False

    @property
    def drops_privileges(self) -> bool:
        return self.uid is not None


def resolve_isolation() -> Isolation:
    """Work out which user tasks will run as, or refuse to run any.

    A task runs somebody else's code inside the agent's own container. Running
    it as the agent's user means it can read the join key, the other tasks'
    directories and the agent's own files. That is not a marketplace anyone
    sane joins, so the default is to refuse rather than to quietly do it.
    """
    name = os.environ.get("LOOM_TASK_USER", "loom-task").strip()
    if os.geteuid() != 0:
        return _cannot_drop(
            f"this agent does not run as root, so it cannot start tasks as {name!r}"
        )
    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        return _cannot_drop(
            f"there is no user {name!r} in this image to run tasks as"
        )
    return Isolation(uid=entry.pw_uid, gid=entry.pw_gid, user=name)


def _cannot_drop(why: str) -> Isolation:
    allowed = os.environ.get("LOOM_ALLOW_UNPRIVILEGED_TASKS") == "1"
    if not allowed:
        logger.warning(
            "%s. This node will refuse tasks. Set LOOM_ALLOW_UNPRIVILEGED_TASKS=1 "
            "to run them as this agent's own user instead — which lets a task "
            "read this node's join key and every other task's files",
            why,
        )
        return Isolation(uid=None, gid=None, user="", unprivileged_fallback=False)
    logger.warning(
        "%s; running tasks unisolated because LOOM_ALLOW_UNPRIVILEGED_TASKS=1. "
        "A task can read this node's credentials from here",
        why,
    )
    return Isolation(uid=None, gid=None, user=os.environ.get("USER", ""),
                     unprivileged_fallback=True)


def ensure_can_run(isolation: Isolation) -> None:
    if not isolation.drops_privileges and not isolation.unprivileged_fallback:
        raise TaskRefused(
            "this node runs no tasks: it cannot start them as a separate user. "
            "Run the agent as root in an image that has the task user, or set "
            "LOOM_ALLOW_UNPRIVILEGED_TASKS=1 to accept the weaker arrangement"
        )


def can_chroot() -> bool:
    return os.geteuid() == 0


def preexec(isolation: Isolation, resources: Resources,
            rootfs: Optional[str] = None, workdir: Optional[str] = None) -> Callable[[], None]:
    """What the child does between fork and exec.

    Order matters and is not stylistic: limits are set while still privileged,
    then the process enters the image, and only then are privileges dropped.
    Any other order either fails (chroot needs root) or lets the task raise its
    own ceilings back up.

    chroot is not a security boundary on its own — a root process inside one
    can leave. It is a boundary here because privileges are dropped immediately
    after, and because the whole thing is already inside the agent's own
    container. What it buys is that the image's own paths are correct: /usr/lib
    means the image's /usr/lib, which is the only way a binary built for that
    image can run at all.
    """

    def apply() -> None:
        # Its own process group, so a timeout kills the tree and not just
        # whatever shell happened to be on top of it.
        os.setsid()
        _set(resource.RLIMIT_NPROC, MAX_PROCESSES)
        _set(resource.RLIMIT_FSIZE, MAX_FILE_BYTES)
        _set(resource.RLIMIT_CORE, 0)
        if resources.gpus == 0 and resources.ram_bytes:
            # RLIMIT_AS is only safe without a GPU. CUDA reserves an enormous
            # virtual address space that it never backs with real memory, so an
            # address-space limit sized to real RAM makes a GPU task fail at
            # startup with an error that looks nothing like "out of memory".
            # For GPU tasks the watchdog below is what enforces the quota.
            _set(resource.RLIMIT_AS, resources.ram_bytes)
        if rootfs:
            os.chroot(rootfs)
            # Not Popen's `cwd`: that is applied before this runs, so it would
            # name a directory that no longer exists once we are inside.
            os.chdir(workdir or "/")
        if isolation.gid is not None:
            os.setgid(isolation.gid)
            try:
                os.setgroups([isolation.gid])
            except OSError:
                pass
        if isolation.uid is not None:
            os.setuid(isolation.uid)

    return apply


def _set(what: int, limit: int) -> None:
    try:
        soft, hard = resource.getrlimit(what)
        ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(what, (ceiling, hard))
    except (ValueError, OSError):
        # A limit we cannot set is worth knowing about but never worth
        # refusing to start over: the watchdog is still watching.
        logger.debug("could not set rlimit %s to %s", what, limit, exc_info=True)


class MemoryWatchdog:
    """Kill the task's process tree when it holds more memory than it was given.

    Watches RSS across the tree, because that is what actually competes with
    the owner's own work. Kills only the task — the agent stays up, keeps
    heartbeating, and reports the task as failed so the orchestrator can place
    it elsewhere.
    """

    def __init__(
        self,
        *,
        get_pid: Callable[[], Optional[int]],
        quota_bytes: int,
        on_exceeded: Callable[[str], None],
        poll_interval_s: float = 2.0,
    ) -> None:
        self.get_pid = get_pid
        self.quota_bytes = max(0, int(quota_bytes))
        self.on_exceeded = on_exceeded
        self.poll_interval_s = poll_interval_s
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.quota_bytes:
            return
        threading.Thread(target=self._watch, name="task-memory", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            pid = self.get_pid()
            if pid is None:
                return
            held = _tree_rss(pid)
            if held > self.quota_bytes:
                self.on_exceeded(
                    f"used {held / 1024**3:.1f} GB of its "
                    f"{self.quota_bytes / 1024**3:.1f} GB memory quota"
                )
                return


def _tree_rss(pid: int) -> int:
    """Resident memory of a process and everything it started.

    Returns 0 when it cannot be measured. A watchdog that guessed high would
    kill healthy work, which is worse than one that occasionally misses.
    """
    try:
        import psutil
    except ImportError:
        return 0
    try:
        proc = psutil.Process(pid)
        procs = [proc, *proc.children(recursive=True)]
    except Exception:
        return 0
    total = 0
    for one in procs:
        try:
            total += one.memory_info().rss
        except Exception:
            continue
    return total
