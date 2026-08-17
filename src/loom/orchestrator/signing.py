"""Control-plane command signing (v0).

HMAC-SHA256 over the deterministic serialization of the ControlMessage with
`meta.signature` cleared. The key is the node's onboarding token (shared
secret v0; per-node keypairs land with Phase-4 onboarding). Combined with
`command_id` dedupe and the `issued_at` freshness window on the worker side,
this gives authenticity + replay protection on top of transport security.
"""

from __future__ import annotations

import hashlib
import hmac

from loom.proto_gen import gateway_pb2


def sign_control_message(msg: gateway_pb2.ControlMessage, key: str) -> None:
    """Fill meta.signature in-place. No-op for commands without CommandMeta."""
    if not key:
        return
    kind = msg.WhichOneof("cmd")
    if kind is None:
        return
    sub = getattr(msg, kind)
    meta = getattr(sub, "meta", None)
    if meta is None or not sub.HasField("meta"):
        return
    meta.signature = b""
    payload = msg.SerializeToString(deterministic=True)
    meta.signature = hmac.new(key.encode(), payload, hashlib.sha256).digest()
