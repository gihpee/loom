"""Join-key parsing on the worker side.

A key carries both the orchestrator address and the shared secret, so the node
owner passes exactly one opaque string and nothing else:

    docker run gihpee/loomworker --key loom_eyJpIjoi...

Kept in sync with src/loom/orchestrator/keys.py by wire format only (the worker
package must not import orchestrator code).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Optional

PREFIX = "loom_"


@dataclass
class ParsedKey:
    key_id: str
    secret: str
    address: str
    raw: str


def parse_join_key(key: str) -> Optional[ParsedKey]:
    key = (key or "").strip()
    if not key.startswith(PREFIX):
        return None
    body = key[len(PREFIX) :]
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return ParsedKey(
            key_id=payload["i"],
            secret=payload["s"],
            address=payload.get("a", ""),
            raw=key,
        )
    except Exception:
        return None
