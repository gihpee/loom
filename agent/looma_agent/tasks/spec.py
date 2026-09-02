"""What the orchestrator asks for, and what this node will agree to.

A spec is data only. Nothing here touches the machine — deciding whether the
node can honour it belongs to the registry, and carrying it out belongs to the
runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

GIB = 1024**3


class TaskRefused(RuntimeError):
    """This node will not run this task, and says why in plain words."""


@dataclass(frozen=True)
class Resources:
    """What a task needs. GPUs are held, not consumed: a card is either free or
    it is not, while memory and CPU are shares of a whole."""

    gpus: int = 0
    ram_bytes: int = 0
    cpus: float = 1.0
    disk_bytes: int = 0

    @classmethod
    def from_dict(cls, raw: Dict) -> "Resources":
        raw = raw or {}
        return cls(
            gpus=max(0, int(raw.get("gpus") or 0)),
            ram_bytes=max(0, int(raw.get("ram_bytes") or 0)),
            cpus=max(0.1, float(raw.get("cpus") or 1.0)),
            disk_bytes=max(0, int(raw.get("disk_bytes") or 0)),
        )

    def as_dict(self) -> Dict:
        return {
            "gpus": self.gpus,
            "ram_bytes": self.ram_bytes,
            "cpus": self.cpus,
            "disk_bytes": self.disk_bytes,
        }


@dataclass(frozen=True)
class EnvSpec:
    """What has to end up in the task's directory before its command can run.

    `none` is not a placeholder for the unimplemented kinds — it is the real
    case of a command that needs nothing provisioned, and it stays useful after
    the others land. The rest arrive in phases 2 and 6 (docs/AGENT_PLAN.md).
    """

    kind: str = "none"  # none | python | binary | oci
    # `python`: packages to install. `binary`/`oci`: the source to unpack.
    requirements: Tuple[str, ...] = ()
    source: str = ""

    KINDS = ("none", "python", "binary", "oci")

    @classmethod
    def from_dict(cls, raw: Dict) -> "EnvSpec":
        raw = raw or {}
        kind = (raw.get("kind") or "none").strip().lower()
        if kind not in cls.KINDS:
            raise TaskRefused(
                f"unknown environment kind {kind!r}; this agent understands "
                f"{', '.join(cls.KINDS)}"
            )
        return cls(
            kind=kind,
            requirements=tuple(raw.get("requirements") or ()),
            source=(raw.get("source") or "").strip(),
        )

    def fingerprint(self) -> str:
        """Identity of the environment, not of the task that asked for it.

        Two tasks wanting the same thing must land on the same cached
        environment, so this covers everything that changes what gets
        installed and nothing that does not.
        """
        import hashlib

        material = "|".join([self.kind, self.source, *sorted(self.requirements)])
        digest = hashlib.sha256(material.encode()).hexdigest()[:12]
        return f"{self.kind}-{digest}"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    environment: EnvSpec = field(default_factory=EnvSpec)
    # A task with no deadline is a task that can hold a stranger's machine
    # forever, so there is always one.
    timeout_s: int = 3600
    # Loopback port the task serves on, when it serves anything. Chosen by the
    # orchestrator so a pipeline's stages agree on it without negotiating.
    serve_port: int = 0

    @classmethod
    def from_dict(cls, raw: Dict) -> "TaskSpec":
        task_id = (raw.get("task_id") or "").strip()
        if not task_id:
            raise TaskRefused("a task needs an id")
        command = list(raw.get("command") or ())
        if not command:
            raise TaskRefused(f"task {task_id} has no command to run")
        return cls(
            task_id=task_id,
            command=command,
            env=dict(raw.get("env") or {}),
            resources=Resources.from_dict(raw.get("resources")),
            environment=EnvSpec.from_dict(raw.get("environment")),
            timeout_s=max(1, int(raw.get("timeout_s") or 3600)),
            serve_port=max(0, int(raw.get("serve_port") or 0)),
        )
