"""Renting the fleet for work that is not a language model.

A client submits a job; the fleet runs it. What makes this different from the
model pipelines is that the work is opaque — Loom cannot look inside a
container and split it the way it splits layers.

So "combining the resources of several providers" means one of two things, and
which one it is has to be stated by whoever submits the job, because the two
cannot be told apart from outside:

  array   The job is many independent tasks. Loom spreads them over as many
          nodes as it takes. This is the shape that suits a fleet of other
          people's machines: nothing is exchanged between tasks, so a slow
          link, an odd card or a node that leaves costs one task and not the
          job. Batch rendering, per-item processing, a sweep over parameters.

  gang    The job is one computation that needs several machines at once and
          coordinates them itself. Loom allocates the group, hands every task
          its rank, the world size and the addresses of its peers, and starts
          them together. What they then do is the job's business — Loom is the
          landlord, not the framework.

Anything that needs more of a single machine than any single machine has is a
gang job, and its author has to have written it that way. There is no honest
way to make an unmodified program use two computers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GIB = 1024**3


class JobError(ValueError):
    """A job that cannot be run, with a message meant for the person who sent it."""


@dataclass
class Resources:
    """What one task needs. Everything is per task, never per job."""

    vram_bytes: int = 0
    ram_bytes: int = 0
    cpus: float = 1.0
    gpus: int = 0
    disk_bytes: int = 0

    @classmethod
    def from_request(cls, raw: Optional[dict]) -> "Resources":
        raw = raw or {}
        return cls(
            vram_bytes=int(float(raw.get("vram_gb", 0)) * GIB),
            ram_bytes=int(float(raw.get("ram_gb", 0)) * GIB),
            cpus=float(raw.get("cpus", 1.0)),
            gpus=int(raw.get("gpus", 1 if raw.get("vram_gb") else 0)),
            disk_bytes=int(float(raw.get("disk_gb", 0)) * GIB),
        )

    def as_dict(self) -> dict:
        return {
            "vram_gb": round(self.vram_bytes / GIB, 2),
            "ram_gb": round(self.ram_bytes / GIB, 2),
            "cpus": self.cpus,
            "gpus": self.gpus,
            "disk_gb": round(self.disk_bytes / GIB, 2),
        }


@dataclass
class Task:
    """One container on one node."""

    task_id: str
    index: int
    node_id: str = ""
    state: str = "pending"     # pending | starting | running | done | failed | cancelled
    exit_code: Optional[int] = None
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "index": self.index,
            "node_id": self.node_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "error": self.error,
            "seconds": round(
                (self.finished_at or time.time()) - self.started_at, 1
            ) if self.started_at else 0.0,
        }


@dataclass
class Job:
    job_id: str
    image: str
    command: List[str]
    env: Dict[str, str]
    resources: Resources
    tasks: List[Task]
    kind: str = "array"        # array | gang
    submitted_at: float = field(default_factory=time.time)
    state: str = "pending"
    timeout_s: int = 3600

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "image": self.image,
            "command": self.command,
            "kind": self.kind,
            "state": self.state,
            "resources": self.resources.as_dict(),
            "submitted_at": self.submitted_at,
            "timeout_s": self.timeout_s,
            "tasks": [t.as_dict() for t in self.tasks],
        }


def plan(
    *,
    image: str,
    command: List[str],
    nodes: Dict[str, dict],
    resources: Resources,
    task_count: int = 1,
    kind: str = "array",
    env: Optional[Dict[str, str]] = None,
    timeout_s: int = 3600,
    reserved: Optional[Dict[str, Resources]] = None,
) -> Job:
    """Choose which node runs each task, or say why none can.

    `nodes` is {node_id: {"vram_free_bytes": ..., "ram_bytes": ..., "cpus": ...}}
    and `reserved` is what earlier jobs already hold, so a second job does not
    hand out the same card twice.

    Packing is worst-fit — the emptiest node first. Best-fit would leave the
    fleet full of nodes with a little room each and no room for the next job's
    task, which is the failure mode that matters when the tasks are large and
    the nodes are not ours to defragment.

    A gang job is all-or-nothing: its tasks must start together, so a plan that
    places some of them is not a plan.
    """
    if not image:
        raise JobError("a job needs an image")
    if task_count < 1:
        raise JobError("a job needs at least one task")
    if kind not in ("array", "gang"):
        raise JobError(f"unknown job kind {kind!r}; use 'array' or 'gang'")

    capacity = {node_id: _free_of(info) for node_id, info in nodes.items()}
    # Two different questions, and conflating them produced a misleading
    # refusal: "could this machine ever take such a task" is about the machine,
    # "can it take one right now" is about what other jobs already hold.
    could = [n for n, avail in capacity.items() if _holds(avail, resources)]
    if not could:
        raise JobError(_why_nothing_fits(capacity, resources))

    free = dict(capacity)
    for node_id, held in (reserved or {}).items():
        if node_id in free:
            free[node_id] = _subtract(free[node_id], held)
    fits = [n for n, avail in free.items() if _holds(avail, resources)]
    if kind == "gang" and len(fits) < task_count:
        raise JobError(
            f"a gang job needs {task_count} nodes at once and only {len(fits)} "
            f"of {len(could)} can take a task of this size right now; gang "
            f"tasks start together or not at all"
        )

    assignments: List[str] = []
    remaining = dict(free)
    for _ in range(task_count):
        # Emptiest first, and never the same node twice in a gang: its tasks
        # are meant to be on different machines.
        pool = [
            n for n in remaining
            if _holds(remaining[n], resources)
            and not (kind == "gang" and n in assignments)
        ]
        if not pool:
            if kind == "gang":
                raise JobError(
                    f"the fleet ran out of room after {len(assignments)} of "
                    f"{task_count} gang tasks"
                )
            # An array job is happy to queue: the tasks are independent, so
            # the ones without a node now simply wait for one.
            assignments.append("")
            continue
        chosen = max(pool, key=lambda n: remaining[n].vram_bytes or remaining[n].ram_bytes)
        assignments.append(chosen)
        remaining[chosen] = _subtract(remaining[chosen], resources)

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    tasks = [
        Task(task_id=f"{job_id}-{i}", index=i, node_id=node_id)
        for i, node_id in enumerate(assignments)
    ]
    return Job(
        job_id=job_id,
        image=image,
        command=list(command or []),
        env=dict(env or {}),
        resources=resources,
        tasks=tasks,
        kind=kind,
        timeout_s=timeout_s,
    )


def gang_env(job: Job, task: Task, addresses: Dict[int, str]) -> Dict[str, str]:
    """What a gang task is told about its group.

    Deliberately the names distributed frameworks already read, so a program
    written for torchrun or MPI runs here without being rewritten for Loom.
    Loom allocates the group and gets out of the way; how the ranks talk to
    each other is the job's business.
    """
    peers = ",".join(addresses.get(i, "") for i in range(len(job.tasks)))
    return {
        "LOOM_JOB_ID": job.job_id,
        "LOOM_TASK_ID": task.task_id,
        "RANK": str(task.index),
        "WORLD_SIZE": str(len(job.tasks)),
        "LOOM_PEERS": peers,
        "MASTER_ADDR": addresses.get(0, ""),
        "MASTER_PORT": "29500",
    }


# --------------------------------------------------------------- the packing
def _free_of(info: dict) -> Resources:
    return Resources(
        vram_bytes=int(info.get("vram_free_bytes") or 0),
        ram_bytes=int(info.get("ram_bytes") or 0),
        cpus=float(info.get("cpus") or 0.0),
        gpus=int(info.get("num_gpus") or 0),
        disk_bytes=int(info.get("disk_bytes") or 0),
    )


def _holds(available: Resources, wanted: Resources) -> bool:
    return (
        available.vram_bytes >= wanted.vram_bytes
        and available.ram_bytes >= wanted.ram_bytes
        and available.cpus >= wanted.cpus
        and available.gpus >= wanted.gpus
        and available.disk_bytes >= wanted.disk_bytes
    )


def _subtract(available: Resources, taken: Resources) -> Resources:
    """What is left after a task takes its share.

    `gpus` is untouched on purpose: it is a requirement, not a consumable.
    Two tasks share one card perfectly well if the VRAM fits, and decrementing
    the count meant the second task on a node was refused for want of a GPU
    the first one had not used up.
    """
    return Resources(
        vram_bytes=max(0, available.vram_bytes - taken.vram_bytes),
        ram_bytes=max(0, available.ram_bytes - taken.ram_bytes),
        cpus=max(0.0, available.cpus - taken.cpus),
        gpus=available.gpus,
        disk_bytes=max(0, available.disk_bytes - taken.disk_bytes),
    )


def _why_nothing_fits(free: Dict[str, Resources], wanted: Resources) -> str:
    """Say what was short, not just that something was.

    "no node has room" sends the operator to look at every machine. Naming the
    dimension and the best node there was turns it into one decision: ask for
    less, or add a machine of this size.
    """
    if not free:
        return "no nodes are connected"
    best = max(free.values(), key=lambda r: r.vram_bytes)
    short = []
    if wanted.vram_bytes and best.vram_bytes < wanted.vram_bytes:
        short.append(
            f"VRAM (asked {wanted.vram_bytes / GIB:.1f} GB, the roomiest node "
            f"has {best.vram_bytes / GIB:.1f})"
        )
    if wanted.ram_bytes and best.ram_bytes < wanted.ram_bytes:
        short.append(
            f"RAM (asked {wanted.ram_bytes / GIB:.1f} GB, best has "
            f"{best.ram_bytes / GIB:.1f})"
        )
    if wanted.gpus and best.gpus < wanted.gpus:
        short.append(f"GPUs (asked {wanted.gpus}, best node has {best.gpus})")
    detail = "; ".join(short) or "no node satisfied every requirement at once"
    return (
        f"no node can take a task this size: {detail}. A task runs on ONE "
        f"machine — to use several, submit several tasks (kind 'array') or a "
        f"job that coordinates them itself (kind 'gang')"
    )
