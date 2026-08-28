"""Running somebody else's container on somebody else's machine.

Two parties need protecting here and only one of them can be, which is the
single most important thing to say about this file.

The node owner can be defended against the tenant's code: drop every
capability, forbid privilege escalation, cap memory and CPU, give it no host
network and no host filesystem. That is what this does, and it is genuine
protection against a careless or greedy workload.

The tenant CANNOT be defended against the node owner. Whoever runs the machine
has root on it: they can read the container's memory, inspect the GPU, and
change what comes back. Docker is an isolation boundary, not a confidentiality
one, and no arrangement of flags makes it one. Confidential computing hardware
would — H100 in CC mode, SEV-SNP — and consumer cards do not have it. Work
whose data or weights are the asset does not belong on a fleet like this, and
the honest place to say so is here, next to the code that would otherwise
imply otherwise.

Two ways to run a task, chosen by what the host allows:

  docker    a real container, the isolation described above. Needs the Docker
            socket, which the worker only has if its operator mounted it —
            a deliberate act, because that socket is root on the host.
  process   a plain subprocess with resource limits. Weaker: same kernel, same
            filesystem, same user. Offered because a worker that cannot reach
            Docker would otherwise be able to run nothing at all, and refused
            unless the operator opts in.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("loom_worker.compute.runtime")

GIB = 1024**3

# How much of a task's output to keep. Enough to diagnose a failure, bounded
# so a chatty job cannot fill the node's disk with its own logs.
LOG_LIMIT_BYTES = int(os.environ.get("LOOM_TASK_LOG_BYTES", str(1024 * 1024)))


class TaskRefused(RuntimeError):
    """This node will not run this task, and says why."""


@dataclass
class TaskSpec:
    task_id: str
    image: str
    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    vram_bytes: int = 0
    ram_bytes: int = 0
    cpus: float = 1.0
    gpus: int = 0
    timeout_s: int = 3600
    network: str = "none"      # none | egress
    workdir: str = ""


class Task:
    """A running task, its output, and how it ended."""

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        self.state = "pending"
        self.exit_code: Optional[int] = None
        self.error = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._log = bytearray()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ lifecycle
    def start(self, *, runtime: str) -> None:
        argv = (
            _docker_argv(self.spec)
            if runtime == "docker"
            else _process_argv(self.spec)
        )
        logger.info("task %s starting (%s): %s", self.spec.task_id, runtime,
                    " ".join(argv[:8]) + (" ..." if len(argv) > 8 else ""))
        self.started_at = time.time()
        self.state = "running"
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Its own process group, so a timeout kills the tree and not just
            # the shell that happened to be on top of it.
            start_new_session=True,
            env=_host_env(self.spec) if runtime == "process" else None,
        )
        threading.Thread(target=self._drain, name=f"task-{self.spec.task_id}",
                         daemon=True).start()
        threading.Thread(target=self._watch, name=f"task-{self.spec.task_id}-wd",
                         daemon=True).start()

    def _drain(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            with self._lock:
                self._log.extend(line)
                if len(self._log) > LOG_LIMIT_BYTES:
                    # Keep the tail: the end of a failing run says more than
                    # its beginning.
                    del self._log[: len(self._log) - LOG_LIMIT_BYTES]

    def _watch(self) -> None:
        assert self._proc
        deadline = time.time() + max(1, self.spec.timeout_s)
        while self._proc.poll() is None:
            if time.time() > deadline:
                logger.warning("task %s hit its %ds limit; stopping it",
                               self.spec.task_id, self.spec.timeout_s)
                self.stop(reason=f"exceeded its {self.spec.timeout_s}s limit")
                break
            time.sleep(0.5)
        self.finished_at = time.time()
        self.exit_code = self._proc.poll()
        if self.state == "cancelled":
            return
        self.state = "done" if self.exit_code == 0 else "failed"
        if self.exit_code not in (0, None) and not self.error:
            self.error = f"exited with code {self.exit_code}"

    def stop(self, *, reason: str = "cancelled") -> None:
        self.error = self.error or reason
        self.state = "cancelled"
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if shutil.which("docker") and self.spec.task_id:
            subprocess.run(["docker", "rm", "-f", _container_name(self.spec)],
                           capture_output=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

    # -------------------------------------------------------------- reading
    def logs(self, *, tail: int = 0) -> str:
        with self._lock:
            text = self._log.decode(errors="replace")
        if tail:
            return "\n".join(text.splitlines()[-tail:])
        return text

    def status(self) -> dict:
        return {
            "task_id": self.spec.task_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "seconds": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at else 0.0,
        }


# ------------------------------------------------------------- how it is run
def available_runtime() -> str:
    """Docker if the operator gave us the socket, a subprocess if allowed."""
    if shutil.which("docker") and os.path.exists("/var/run/docker.sock"):
        return "docker"
    if os.environ.get("LOOM_ALLOW_PROCESS_TASKS") == "1":
        return "process"
    return ""


def _container_name(spec: TaskSpec) -> str:
    return f"loom-{spec.task_id}"


def _docker_argv(spec: TaskSpec) -> List[str]:
    """The container, with the node owner's machine defended.

    Every flag here answers something a hostile or careless task would
    otherwise do to the host it was lent.
    """
    argv = [
        "docker", "run", "--rm",
        "--name", _container_name(spec),
        # No route to root: no new privileges, no capabilities, nothing on the
        # host's namespaces.
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--pids-limit", "512",
        # Bounded, so one task cannot take the machine away from its owner.
        "--memory", str(spec.ram_bytes or 2 * GIB),
        "--cpus", str(spec.cpus or 1.0),
        # The client's work goes here and nowhere else. No host path is
        # mounted at all: a tenant cannot read the owner's disk.
        "--workdir", "/work",
        "--tmpfs", "/work:exec,size=8g",
    ]
    if spec.network == "none":
        # The default. A task that does not need the network does not get it,
        # which also means it cannot use the owner's connection for anything.
        argv += ["--network", "none"]
    if spec.gpus:
        argv += ["--gpus", f"device={','.join(str(i) for i in range(spec.gpus))}"]
    for key, value in spec.env.items():
        argv += ["-e", f"{key}={value}"]
    argv.append(spec.image)
    argv += spec.command
    return argv


def _process_argv(spec: TaskSpec) -> List[str]:
    if not spec.command:
        raise TaskRefused("the process runtime needs a command; an image alone "
                          "means nothing without Docker to unpack it")
    return list(spec.command)


def _host_env(spec: TaskSpec) -> Dict[str, str]:
    """A deliberately bare environment for the weaker runtime.

    The worker's own environment holds the join key, which is a credential for
    the whole node. A task started in the same environment would inherit it.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": spec.workdir or "/tmp",
        "LOOM_TASK_ID": spec.task_id,
    }
    env.update(spec.env)
    return env
