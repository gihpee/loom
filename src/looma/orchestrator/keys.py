"""Join keys: the only thing a GPU owner needs to attach a machine.

A key is a self-contained string that carries BOTH the orchestrator address and
a secret, so the node owner runs exactly one command:

    docker run gihpee/looma-worker --key looma_<payload>

The secret doubles as the per-node HMAC key for control-command signing, so a
revoked key cannot issue commands either.

Storage: in-memory + optional JSON file (LOOMA_KEYSTORE_PATH) so keys survive an
orchestrator restart. Postgres persistence lands with the Phase-4 marketplace.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PREFIX = "looma_"


@dataclass
class JoinKey:
    key_id: str
    secret: str
    address: str  # orchestrator gRPC address the worker should dial
    label: str = ""
    created_at: float = field(default_factory=time.time)
    max_nodes: int = 0  # 0 = unlimited
    # Ходить ли по этому адресу с TLS. Едет внутри ключа, а не выводится из
    # вида адреса: угадывание здесь означало бы, что узел молча уходит в
    # открытый канал, когда оркестратор ждал шифрованный.
    tls: bool = False
    revoked: bool = False
    nodes: List[str] = field(default_factory=list)  # node_ids seen with this key

    def encode(self) -> str:
        """Render the shareable key string."""
        payload = {"i": self.key_id, "s": self.secret, "a": self.address}
        if self.tls:
            # Только когда включено: ключи, выпущенные до шифрования, остаются
            # прежними строками, и старые узлы продолжают работать.
            payload["t"] = 1
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def public(self) -> dict:
        d = asdict(self)
        d.pop("secret")
        return d


def decode_key(key: str) -> Optional[dict]:
    """Parse a key string into {key_id, secret, address}; None if malformed."""
    if not key or not key.startswith(PREFIX):
        return None
    body = key[len(PREFIX) :]
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return {
            "key_id": payload["i"],
            "secret": payload["s"],
            "address": payload.get("a", ""),
            "tls": bool(payload.get("t")),
        }
    except Exception:
        return None


class KeyStore:
    """Issue / validate / revoke join keys."""

    def __init__(
        self,
        *,
        public_address: str,
        path: Optional[str | Path] = None,
        master_token: str = "",
        tls: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self.public_address = public_address
        # Чем ключи будут снабжаться при выпуске. Уже выпущенные не меняются:
        # строка ключа у владельца узла на руках, и переписать её мы не можем.
        self.tls = tls
        self.path = Path(path) if path else None
        # Legacy/dev escape hatch: a single shared token also authenticates.
        self.master_token = master_token
        self._keys: Dict[str, JoinKey] = {}
        self._load()

    # ------------------------------------------------------------- persistence
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            for item in raw.get("keys", []):
                key = JoinKey(**item)
                self._keys[key.key_id] = key
        except Exception:
            pass  # corrupt store must not block startup

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"keys": [asdict(k) for k in self._keys.values()]}, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------------- API
    def issue(self, *, label: str = "", max_nodes: int = 0) -> JoinKey:
        with self._lock:
            key = JoinKey(
                key_id=secrets.token_hex(6),
                secret=secrets.token_urlsafe(32),
                address=self.public_address,
                tls=self.tls,
                label=label,
                max_nodes=max_nodes,
            )
            self._keys[key.key_id] = key
            self._save()
            return key

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            key = self._keys.get(key_id)
            if key is None:
                return False
            key.revoked = True
            self._save()
            return True

    def list(self) -> List[JoinKey]:
        with self._lock:
            return list(self._keys.values())

    def validate(self, presented: str, *, node_id: str = "") -> Optional[str]:
        """Return the signing secret for an accepted key, else None.

        Accepts a full key string, or the master token when configured.
        """
        if self.master_token and presented == self.master_token:
            return self.master_token
        parsed = decode_key(presented)
        if parsed is None:
            return None
        with self._lock:
            key = self._keys.get(parsed["key_id"])
            if key is None or key.revoked:
                return None
            if not secrets.compare_digest(key.secret, parsed["secret"]):
                return None
            if node_id:
                if node_id not in key.nodes:
                    if key.max_nodes and len(key.nodes) >= key.max_nodes:
                        return None  # key already used by max_nodes machines
                    key.nodes.append(node_id)
                    self._save()
            return key.secret

    def open_registration(self) -> bool:
        """True if any worker may attach without a key (dev only)."""
        return not self.master_token and not self._keys
