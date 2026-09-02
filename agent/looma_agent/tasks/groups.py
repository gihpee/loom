"""Where the other members of a job are.

A pipeline stage sends its activations to the next stage on every token. That
makes "which node holds rank N" a question the agent has to answer from memory:
asking the orchestrator each time would add a network round trip per token to a
path whose whole cost is network round trips.

So the orchestrator tells every member about all of them when the group is
placed, and this holds what it said.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Member:
    rank: int
    node_id: str
    peer_id: str = ""


@dataclass(frozen=True)
class Group:
    group_id: str
    rank: int
    members: Dict[int, Member]

    def member(self, rank: int) -> Optional[Member]:
        return self.members.get(rank)

    @property
    def size(self) -> int:
        return len(self.members)


class GroupTable:
    """Which group each local task belongs to, and who else is in it."""

    def __init__(self) -> None:
        self._by_task: Dict[str, Group] = {}
        self._local_rank: Dict[str, Dict[int, str]] = {}
        self._lock = threading.RLock()

    def join(self, task_id: str, group: Group) -> None:
        with self._lock:
            self._by_task[task_id] = group
            self._local_rank.setdefault(group.group_id, {})[group.rank] = task_id

    def leave(self, task_id: str) -> None:
        with self._lock:
            group = self._by_task.pop(task_id, None)
            if group is None:
                return
            local = self._local_rank.get(group.group_id, {})
            local.pop(group.rank, None)
            if not local:
                self._local_rank.pop(group.group_id, None)

    def of(self, task_id: str) -> Optional[Group]:
        with self._lock:
            return self._by_task.get(task_id)

    def local_task(self, group_id: str, rank: int) -> Optional[str]:
        """The task on THIS node holding that rank, if it is here."""
        with self._lock:
            return self._local_rank.get(group_id, {}).get(rank)

    def member(self, group_id: str, rank: int) -> Optional[Member]:
        with self._lock:
            for group in self._by_task.values():
                if group.group_id == group_id:
                    return group.member(rank)
        return None

    def groups(self) -> List[str]:
        with self._lock:
            return list(self._local_rank)


def group_from_proto(message) -> Optional[Group]:
    if not message or not message.group_id:
        return None
    return Group(
        group_id=message.group_id,
        rank=message.rank,
        members={m.rank: Member(rank=m.rank, node_id=m.node_id, peer_id=m.peer_id)
                 for m in message.members},
    )
