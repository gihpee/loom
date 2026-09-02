"""Who this node is and what it may say.

Two things come out of the join key: where to call, and the secret that proves
a command really came from the orchestrator. Everything else about identity is
derived from the machine itself.

Wire format is kept in sync with src/looma/orchestrator/keys.py by contract, not
by shared code: the agent package must stay installable on its own.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
from dataclasses import dataclass
from typing import Optional

from looma_agent.hwinfo import gpu_fingerprint, sees_only_some_gpus

logger = logging.getLogger("looma_agent.identity")

KEY_PREFIX = "looma_"


class BadJoinKey(ValueError):
    """The key is not one of ours, or it is damaged."""


@dataclass(frozen=True)
class JoinKey:
    key_id: str
    secret: str
    address: str
    raw: str


def parse_join_key(key: str) -> JoinKey:
    key = (key or "").strip()
    if not key.startswith(KEY_PREFIX):
        raise BadJoinKey(
            "a join key starts with 'looma_'. Copy it whole from the admin page — "
            "it carries the orchestrator address, so there is nothing else to pass."
        )
    body = key[len(KEY_PREFIX):]
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return JoinKey(
            key_id=payload["i"],
            secret=payload["s"],
            address=payload.get("a", ""),
            raw=key,
        )
    except BadJoinKey:
        raise
    except Exception as exc:
        raise BadJoinKey(f"this join key is damaged and cannot be read ({exc})") from None


def default_node_id() -> str:
    """What this node calls itself when nobody said.

    The hostname — plus the cards it was given, when it was given only some of
    them. Two agents on one machine with `--gpus device=0` and `--gpus device=1`
    are two nodes and must say so: `--network host` makes them share a hostname,
    the orchestrator then treats them as one, and each registration evicts the
    other's session.

    An agent holding the whole machine keeps the bare hostname, so nothing
    turns into a new node just because it was upgraded.
    """
    hostname = socket.gethostname()
    if not sees_only_some_gpus():
        return hostname
    suffix = gpu_fingerprint()
    if not suffix:
        return hostname
    node_id = f"{hostname}-{suffix}"
    logger.info(
        "this agent was given only some of the machine's GPUs; calling itself %s "
        "so it does not collide with the others on %s",
        node_id, hostname,
    )
    return node_id
