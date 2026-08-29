"""Worker-side command verification: signature + freshness + replay protection.

Mirrors the orchestrator's scheme (kept in sync by contract, not shared code —
the worker package is self-contained): HMAC-SHA256 over the deterministic
serialization of the ControlMessage with meta.signature cleared, keyed by the
node's onboarding token.

Rejections:
- bad/absent signature           -> "signature mismatch"
- |now - issued_at| > max_skew   -> "stale command" (freshness window)
- command_id already executed    -> "replay rejected" (LRU dedupe cache)
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict
from typing import Optional, Tuple

from loom_agent.proto import gateway_pb2


class CommandVerifier:
    def __init__(self, key: str, *, max_skew_ms: int = 60_000, cache_size: int = 4096) -> None:
        self.key = key.encode()
        self.max_skew_ms = max_skew_ms
        self.cache_size = cache_size
        self._seen: "OrderedDict[str, None]" = OrderedDict()

    def verify(self, msg: gateway_pb2.ControlMessage) -> Tuple[bool, Optional[str]]:
        kind = msg.WhichOneof("cmd")
        if kind is None:
            return False, "empty control message"
        sub = getattr(msg, kind)
        if not hasattr(sub, "meta") or not sub.HasField("meta"):
            return False, "unsigned command (no meta)"
        meta = sub.meta

        # 1) Signature over the message with the signature field cleared.
        provided = bytes(meta.signature)
        clone = gateway_pb2.ControlMessage()
        clone.CopyFrom(msg)
        getattr(clone, kind).meta.signature = b""
        expected = hmac.new(
            self.key, clone.SerializeToString(deterministic=True), hashlib.sha256
        ).digest()
        if not provided or not hmac.compare_digest(provided, expected):
            return False, "signature mismatch"

        # 2) Freshness window.
        now_ms = int(time.time() * 1000)
        if abs(now_ms - meta.issued_at_unix_ms) > self.max_skew_ms:
            return False, "stale command"

        # 3) Replay: each command_id executes at most once.
        cid = meta.command_id
        if not cid:
            return False, "missing command_id"
        if cid in self._seen:
            return False, "replay rejected"
        self._seen[cid] = None
        while len(self._seen) > self.cache_size:
            self._seen.popitem(last=False)
        return True, None
