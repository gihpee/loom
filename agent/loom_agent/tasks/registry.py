"""What this node is running, and what it still has to give.

Two jobs that have to be one object, because they are the same question asked
twice: what is running determines what is free.

GPU accounting lives here and nowhere else. Cards are the scarce thing on a
node and the reason anyone rents it, so "which card is this task on" is not a
detail the runner may improvise. The old compute path handed every task
devices 0..N-1, which meant two tasks on one machine both sat on card 0 while
the others idled — the bug this module exists to make impossible.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loom_agent.tasks.directory import TaskDirectory
from loom_agent.tasks.env import EnvironmentCache
from loom_agent.tasks.limits import Isolation, ensure_can_run
from loom_agent.tasks.runner import Task
from loom_agent.tasks.spec import TaskRefused, TaskSpec


# Ниже этого порта биндить может только root. Агент — root, задача — нет.
PRIVILEGED_PORTS = 1024


def _free_port(hint: int) -> int:
    """Порт, который сможет занять ЗАДАЧА, а не только мы.

    The orchestrator asks for a task to serve; it does not get to say on which
    port. Two agents on one machine — which is how a multi-GPU host is split —
    would otherwise be handed the same number and the second would fail to
    bind, in a way that looks like the task never started.

    Привилегированные номера отбрасываются, не пробуя. Проверка идёт от имени
    агента, а он root: `bind(1)` у него проходит, номер уезжает задаче, и та
    падает на `Permission denied` — ошибке, которая не называет ни порт, ни
    того, кто его выбрал. Оркестратор шлёт сюда единицу как «да, служи»,
    так что это не редкий случай, а обычный.
    """
    import socket

    for candidate in ([hint, 0] if hint >= PRIVILEGED_PORTS else [0]):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1]
        except OSError:
            continue
    return 0

logger = logging.getLogger("loom_agent.tasks.registry")

# How long a finished task's directory survives so its result can still be
# read. Devices are freed immediately; only the disk waits.
RETENTION_S = float(os.environ.get("LOOM_TASK_RETENTION_S", "3600"))


class TaskRegistry:
    def __init__(
        self,
        *,
        root: Path,
        isolation: Isolation,
        environments: EnvironmentCache,
        total_gpus: int = 0,
        retention_s: float = RETENTION_S,
    ) -> None:
        self.root = root
        self.isolation = isolation
        self.environments = environments
        self.total_gpus = max(0, int(total_gpus))
        self.retention_s = retention_s
        self._tasks: Dict[str, Task] = {}
        self._held: Dict[str, Tuple[int, ...]] = {}
        # Ids claimed but not yet running: provisioning an environment can take
        # minutes, and the id has to be taken for all of it.
        self._claimed: set = set()
        self._lock = threading.RLock()
        # Why this node cannot take work, if it cannot. An unusable root is not
        # a reason to crash: a node whose volume was never mounted should still
        # register, report itself, and say what is wrong — a crash loop tells
        # its owner nothing and tells the orchestrator even less.
        # Where tasks post messages for the rest of their job. Empty on a node
        # that runs no groups.
        self.channel_url = ""
        self.unusable = ""
        # Set while the node is on its way out (an update, a shutdown). New
        # work is refused, running work is left alone to finish.
        self.draining = ""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.unusable = (
                f"this node cannot use {self.root} for task directories ({exc.strerror}). "
                "Mount a volume there, or point the agent elsewhere with --root"
            )
            logger.error("%s", self.unusable)

    # ------------------------------------------------------------- accounting
    def free_devices(self) -> List[int]:
        with self._lock:
            taken = {d for devices in self._held.values() for d in devices}
            return [d for d in range(self.total_gpus) if d not in taken]

    def _take_devices(self, task_id: str, wanted: int) -> Tuple[int, ...]:
        if wanted <= 0:
            return ()
        free = self.free_devices()
        if len(free) < wanted:
            raise TaskRefused(
                f"this node has {len(free)} of its {self.total_gpus} GPUs free and "
                f"the task asked for {wanted}"
            )
        devices = tuple(free[:wanted])
        self._held[task_id] = devices
        return devices

    def _release_devices(self, task_id: str) -> None:
        with self._lock:
            self._held.pop(task_id, None)

    # ----------------------------------------------------------------- submit
    def submit(self, spec: TaskSpec, deliver_input=None, group=None) -> Task:
        """Take a task on, or say plainly why not.

        Blocks while the environment is provisioned, which on a cache miss can
        be minutes. That is deliberate for now: the caller is a direct API call
        and wants to know whether the task started. When the task protocol
        lands (docs/AGENT_PLAN.md phase 4) this becomes an accepted-then-
        provisioning state, because a control stream must not wait on pip.

        `deliver_input` is called with the task's inbox after its directory
        exists and before it starts. A task must not be running while its own
        input is still arriving: it would read half a file and fail on
        something unrelated to the real cause.
        """
        if self.unusable:
            raise TaskRefused(self.unusable)
        if self.draining:
            raise TaskRefused(self.draining)
        ensure_can_run(self.isolation)
        with self._lock:
            if spec.task_id in self._tasks or spec.task_id in self._claimed:
                raise TaskRefused(f"task {spec.task_id} is already here")
            devices = self._take_devices(spec.task_id, spec.resources.gpus)
            self._claimed.add(spec.task_id)

        # Outside the lock on purpose: an install that takes half an hour must
        # not stop this node from answering anything else.
        directory = None
        environment = None
        try:
            if spec.serve_port:
                spec = replace(spec, serve_port=_free_port(spec.serve_port))
                if not spec.serve_port:
                    raise TaskRefused("this node has no free port for the task to serve on")
            environment = self.environments.acquire(spec.environment)
            directory = TaskDirectory.create(self.root, spec.task_id, self.isolation)
            if deliver_input is not None:
                deliver_input(directory.inbox(
                    self.isolation, limit_bytes=spec.resources.disk_bytes))
            task = Task(spec, directory, self.isolation, devices, environment,
                        group=group, channel_url=self.channel_url)
            task.start()
        except TaskRefused:
            self._undo(spec.task_id, directory, environment)
            raise
        except Exception as exc:
            self._undo(spec.task_id, directory, environment)
            raise TaskRefused(f"could not start task {spec.task_id}: {exc}") from exc

        with self._lock:
            self._tasks[spec.task_id] = task
            self._claimed.discard(spec.task_id)
        threading.Thread(target=self._reap, args=(task,), name=f"reap-{spec.task_id}",
                         daemon=True).start()
        return task

    def _undo(self, task_id: str, directory, environment) -> None:
        """Give back everything a failed submission took."""
        self._release_devices(task_id)
        with self._lock:
            self._claimed.discard(task_id)
        if environment is not None:
            self.environments.release(environment.fingerprint)
        if directory is not None:
            directory.remove()

    def _reap(self, task: Task) -> None:
        """Free the cards as soon as the task ends; free the disk later.

        Two different clocks on purpose. A card held by a finished task is a
        card nobody can rent, so it comes back at once. The directory still
        holds the result somebody is about to collect, so it waits.
        """
        task.wait()
        self._release_devices(task.spec.task_id)
        # The environment can be evicted again once nothing holds it.
        self.environments.release(task.environment.fingerprint)
        if self.retention_s <= 0:
            self.release(task.spec.task_id)
            return
        time.sleep(self.retention_s)
        if self.get(task.spec.task_id) is task:
            logger.info("task %s was never collected; reclaiming its disk",
                        task.spec.task_id)
            self.release(task.spec.task_id)

    # ------------------------------------------------------------------ reads
    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def require(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise TaskRefused(f"no task {task_id!r} on this node")
        return task

    def list(self) -> List[Task]:
        with self._lock:
            return list(self._tasks.values())

    def snapshot(self) -> dict:
        with self._lock:
            running = [t for t in self._tasks.values() if not t.finished]
            return {
                "unusable": self.unusable,
                "tasks": len(self._tasks),
                "running": len(running),
                "gpus_total": self.total_gpus,
                "gpus_free": len(self.free_devices()),
                "environments": self.environments.snapshot(),
            }

    # ----------------------------------------------------------------- writes
    def stop(self, task_id: str, *, reason: str = "cancelled") -> Task:
        task = self.require(task_id)
        task.stop(reason=reason)
        return task

    def release(self, task_id: str) -> None:
        """Forget a task and take its disk back. Stops it first if it is alive."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
        if task is None:
            return
        if not task.finished:
            task.stop(reason="released")
            task.wait(timeout=30)
        self._release_devices(task_id)
        task.directory.remove()

    def drain(self, timeout_s: float, *, reason: str = "this node is restarting") -> bool:
        """Take no new work and let what is running finish.

        Returns whether everything finished in time. A task in flight is
        somebody's work — they have already paid for the electricity — so it is
        waited for rather than killed, up to a point.
        """
        self.draining = reason
        deadline = time.time() + max(0.0, timeout_s)
        while time.time() < deadline:
            running = [t for t in self.list() if not t.finished]
            if not running:
                return True
            time.sleep(0.5)
        return not [t for t in self.list() if not t.finished]

    def resume(self) -> None:
        self.draining = ""

    def stop_all(self, *, reason: str = "agent shutting down") -> None:
        for task in self.list():
            try:
                task.stop(reason=reason)
            except Exception:
                logger.debug("stopping %s failed", task.spec.task_id, exc_info=True)
