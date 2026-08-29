"""One task: start it, watch it, stop it, say what happened.

The agent never runs a task in its own process. A separate process means a task
that crashes, hangs or eats the machine takes only itself down, while the agent
keeps its stream to the orchestrator open and can report what went wrong.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loom_agent.tasks.directory import TaskDirectory
from loom_agent.tasks.env.base import NO_ENVIRONMENT, Environment
from loom_agent.tasks.limits import Isolation, MemoryWatchdog, can_chroot, preexec
from loom_agent.tasks.spec import TaskRefused, TaskSpec
from loom_agent.transport.files import ResultFile

logger = logging.getLogger("loom_agent.tasks.runner")

# Enough output to diagnose a failure, bounded so a chatty task cannot fill the
# node's disk with its own logs.
LOG_LIMIT_BYTES = int(os.environ.get("LOOM_TASK_LOG_BYTES", str(1024 * 1024)))
# How long a task gets to stop politely before it is killed outright.
GRACE_S = 10.0

PENDING, RUNNING, DONE, FAILED, CANCELLED = "pending", "running", "done", "failed", "cancelled"


class Task:
    """A running task, its output, and how it ended."""

    def __init__(
        self,
        spec: TaskSpec,
        directory: TaskDirectory,
        isolation: Isolation,
        devices: Sequence[int] = (),
        environment: Environment = NO_ENVIRONMENT,
        group=None,
        channel_url: str = "",
    ) -> None:
        self.spec = spec
        self.directory = directory
        self.isolation = isolation
        self.devices = tuple(devices)
        self.environment = environment
        # Where this task sits in a job spread over several nodes, and how to
        # reach the agent. Both absent for an ordinary one-node task.
        self.group = group
        self.channel_url = channel_url
        # What the task was asked to serve on; replaced by what it says it
        # actually bound.
        self.serve_port = spec.serve_port
        self.state = PENDING
        self.exit_code: Optional[int] = None
        self.error = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._log = bytearray()
        self._lock = threading.RLock()
        self._watchdog: Optional[MemoryWatchdog] = None
        self._done = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._enter_image()
        logger.info(
            "task %s starting on %s: %s",
            self.spec.task_id,
            f"gpu {','.join(map(str, self.devices))}" if self.devices else "cpu",
            " ".join(self.spec.command[:8]) + (" ..." if len(self.spec.command) > 8 else ""),
        )
        self.started_at = time.time()
        self.state = RUNNING
        rootfs = str(self.directory.rootfs) if self.directory.rootfs else None
        self._proc = subprocess.Popen(
            list(self.spec.command),
            # Inside an image the working directory is set after the chroot,
            # so passing it here would name a path that stops existing.
            cwd=None if rootfs else str(self.directory.work),
            env=self.process_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec(self.isolation, self.spec.resources,
                               rootfs=rootfs, workdir="/work" if rootfs else None),
        )
        self._watchdog = MemoryWatchdog(
            get_pid=lambda: self._proc.pid if self._proc else None,
            quota_bytes=self.spec.resources.ram_bytes,
            on_exceeded=lambda why: self.stop(reason=why),
        )
        self._watchdog.start()
        threading.Thread(target=self._drain, name=f"task-{self.spec.task_id}-log",
                         daemon=True).start()
        threading.Thread(target=self._watch, name=f"task-{self.spec.task_id}",
                         daemon=True).start()

    def _enter_image(self) -> None:
        """Give this task its own copy of the image it asked for.

        A copy, not a link: hard links share an inode, so a task writing to one
        would edit the cached image every later task starts from. Copying costs
        seconds per gigabyte and is the honest price until we can mount an
        overlay — which needs privileges the whole design is built to avoid.
        """
        source = self.environment.rootfs
        if source is None:
            return
        if not can_chroot():
            raise TaskRefused(
                "this node cannot run an image: entering one needs root, and this "
                "agent is not running as root. Run the agent as root (the task "
                "itself still drops to an unprivileged user), or ask for a "
                "'python' or 'binary' environment instead"
            )
        destination = self.directory.root / "rootfs"
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
        self.directory = self.directory.with_rootfs(destination, self.isolation)

    def process_env(self) -> Dict[str, str]:
        """The variables the task's process starts with — deliberately bare.

        The agent's own environment holds the join key, which is a credential
        for this whole node. A task started with it inherited would be able to
        register as this node, or take its work. So the task gets what it needs
        and nothing that was not meant for it.
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            # As the task will see them: inside an image these are /work and
            # /out, and the paths outside it mean nothing to the task.
            "HOME": self.directory.inner_work,
            "TMPDIR": self.directory.inner_work,
            "LOOM_TASK_ID": self.spec.task_id,
            "LOOM_TASK_OUT": self.directory.inner_out,
        }
        if self.spec.serve_port:
            env["LOOM_SERVE_PORT"] = str(self.spec.serve_port)
        if self.channel_url:
            # Everything this task needs to reach the rest of its job: one URL
            # and its own rank. Where the other ranks actually are is the
            # agent's problem, and a task that knew would have to know about
            # NAT and relays too.
            env["LOOM_AGENT_URL"] = self.channel_url
        if self.group is not None:
            env["LOOM_GROUP_ID"] = self.group.group_id
            env["LOOM_RANK"] = str(self.group.rank)
            env["LOOM_GROUP_SIZE"] = str(self.group.size)
        if self.devices:
            # The cards this task was actually given — not 0..N-1. Two tasks on
            # one node must not both land on card 0 while the others idle.
            visible = ",".join(str(d) for d in self.devices)
            env["CUDA_VISIBLE_DEVICES"] = visible
            env["NVIDIA_VISIBLE_DEVICES"] = visible
        # The provisioned environment goes on before the task's own variables,
        # so a task can still override what it was given.
        env.update(self.environment.overrides())
        env.update(self.spec.env)
        return env

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def stop(self, *, reason: str = "cancelled") -> None:
        """End the task now. Safe to call twice, and from any thread."""
        with self._lock:
            if self.state not in (PENDING, RUNNING):
                return
            self.error = self.error or reason
            self.state = CANCELLED
        logger.info("stopping task %s: %s", self.spec.task_id, reason)
        self._kill()

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        group = _group_of(proc)
        if group is None:
            return
        # The whole group, not the process: a task that started children would
        # otherwise leave them running on the owner's machine after it "ended".
        _signal_group(group, signal.SIGTERM)
        try:
            proc.wait(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            logger.warning("task %s ignored SIGTERM; killing it", self.spec.task_id)
            _signal_group(group, signal.SIGKILL)

    # ---------------------------------------------------------------- threads
    def _drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            with self._lock:
                self._log.extend(line)
                if len(self._log) > LOG_LIMIT_BYTES:
                    # Keep the tail: the end of a failing run says more than
                    # its beginning.
                    del self._log[: len(self._log) - LOG_LIMIT_BYTES]

    def _watch(self) -> None:
        proc = self._proc
        if proc is None:
            return
        deadline = time.time() + max(1, self.spec.timeout_s)
        quota = self.spec.resources.disk_bytes
        while proc.poll() is None:
            if time.time() > deadline:
                self.stop(reason=f"exceeded its {self.spec.timeout_s}s limit")
                break
            if quota and self.directory.size_bytes() > quota:
                self.stop(reason=f"exceeded its {quota / 1024**3:.1f} GB disk quota")
                break
            time.sleep(0.5)
        proc.wait()
        self._finish(proc.poll())

    def _finish(self, code: Optional[int]) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()
        self.finished_at = time.time()
        self.exit_code = code
        with self._lock:
            if self.state != CANCELLED:
                self.state = DONE if code == 0 else FAILED
                if code not in (0, None) and not self.error:
                    self.error = f"exited with code {code}"
        logger.info("task %s %s (code %s)", self.spec.task_id, self.state, code)
        self._done.set()

    # ---------------------------------------------------------------- reading
    def logs(self, *, tail: int = 0) -> str:
        with self._lock:
            text = self._log.decode(errors="replace")
        if tail:
            return "\n".join(text.splitlines()[-tail:])
        return text

    def results(self) -> List["ResultFile"]:
        """What the task deliberately put where it was told.

        Its `work` directory is full of scratch and checkpoints nobody asked
        for; only `out` is the answer.
        """
        return self.directory.outbox(limit_bytes=self.spec.resources.disk_bytes).manifest()

    def read_result(self, name: str):
        return self.directory.outbox().read(name)

    def status(self) -> dict:
        return {
            "task_id": self.spec.task_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "devices": list(self.devices),
            "serve_port": self.serve_port,
            "seconds": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at else 0.0,
        }

    @property
    def finished(self) -> bool:
        return self.state in (DONE, FAILED, CANCELLED) and self._done.is_set()


def _group_of(proc: subprocess.Popen) -> Optional[int]:
    try:
        return os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return None


def _signal_group(group: int, sig: int) -> None:
    try:
        os.killpg(group, sig)
    except (ProcessLookupError, PermissionError):
        pass
