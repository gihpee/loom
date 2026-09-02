"""What a task needs, and which node can give it.

One placement rule for the whole system. Everything above it — a single task, a
pipeline spread over four machines — asks the same question and gets the same
answer, so a task that was accepted here is never refused by the node that gets
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

GIB = 1024**3


@dataclass
class Resources:
    """What one task needs. Everything is per task, never per group."""

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

    def plus(self, other: "Resources") -> "Resources":
        return Resources(
            vram_bytes=self.vram_bytes + other.vram_bytes,
            ram_bytes=self.ram_bytes + other.ram_bytes,
            cpus=self.cpus + other.cpus,
            gpus=self.gpus + other.gpus,
            disk_bytes=self.disk_bytes + other.disk_bytes,
        )


def choose_node(
    *,
    nodes: Dict[str, dict],
    resources: Resources,
    reserved: Optional[Dict[str, Resources]] = None,
) -> Tuple[str, str]:
    """The emptiest node that fits, or "" and the reason none does.

    Worst fit — the emptiest first. Best fit would leave the fleet full of
    nodes with a little room each and no room for the next task, which is the
    failure that matters when tasks are large and the nodes are not ours to
    defragment.

    `nodes` is {node_id: {"vram_free_bytes": ..., "ram_bytes": ..., "cpus": ...,
    "num_gpus": ..., "disk_bytes": ...}}; `reserved` is what already-placed
    work holds, so a second task does not get a card the first has.
    """
    reserved = reserved or {}
    free: Dict[str, Resources] = {}
    for node_id, info in nodes.items():
        available = _free_of(info)
        taken = reserved.get(node_id)
        free[node_id] = _subtract(available, taken) if taken else available
    fitting = [n for n, available in free.items() if _holds(available, resources)]
    if not fitting:
        return "", _why_nothing_fits(free, resources)
    fitting.sort(key=lambda n: (free[n].gpus, free[n].vram_bytes, free[n].ram_bytes),
                 reverse=True)
    return fitting[0], ""


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

    GPUs included, unlike the shard scheduler this replaced: an agent hands a
    task its own devices and refuses when too few are free, so a placement that
    double-booked a card would be accepted here and rejected there — which
    looks like tasks failing at random on a busy node.
    """
    return Resources(
        vram_bytes=max(0, available.vram_bytes - taken.vram_bytes),
        ram_bytes=max(0, available.ram_bytes - taken.ram_bytes),
        cpus=max(0.0, available.cpus - taken.cpus),
        gpus=max(0, available.gpus - taken.gpus),
        disk_bytes=max(0, available.disk_bytes - taken.disk_bytes),
    )


def _why_nothing_fits(free: Dict[str, Resources], wanted: Resources) -> str:
    """Say what was short, not just that something was.

    "no node has room" sends the operator to look at every machine. Naming the
    dimension and the roomiest node there was turns it into one decision: ask
    for less, or add a machine of this size.
    """
    if not free:
        return "no nodes are connected"
    best = max(free.values(), key=lambda r: r.vram_bytes)
    roomiest_gpus = max((r.gpus for r in free.values()), default=0)
    short = []
    if wanted.vram_bytes and best.vram_bytes < wanted.vram_bytes:
        short.append(
            f"VRAM (asked {wanted.vram_bytes / GIB:.1f} GB, the roomiest node has "
            f"{best.vram_bytes / GIB:.1f})"
        )
    if wanted.ram_bytes and best.ram_bytes < wanted.ram_bytes:
        short.append(
            f"RAM (asked {wanted.ram_bytes / GIB:.1f} GB, best has "
            f"{best.ram_bytes / GIB:.1f})"
        )
    if wanted.gpus and roomiest_gpus < wanted.gpus:
        short.append(f"GPUs (asked {wanted.gpus}, the freest node has {roomiest_gpus})")
    detail = "; ".join(short) or "no node satisfied every requirement at once"
    return (
        f"no node can take a task this size: {detail}. A task runs on ONE machine "
        f"— to use several, submit a group, whose members are placed together"
    )
