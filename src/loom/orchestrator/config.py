"""Everything the orchestrator needs to start, and where it comes from.

Environment only. A config file would be one more thing to keep in sync with a
container's environment, and there is nothing here that a container cannot
express.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class OrchestratorConfig:
    grpc_port: int = 9000
    http_port: int = 8000
    # What a node dials. Detected when not set — see public_addr.py, which also
    # says how confident it is, because a wrong answer here means nobody can
    # join and nothing says why.
    public_address: str = ""
    # Where join keys, agent releases and anything else that must survive a
    # restart live.
    data_dir: str = "/data"
    admin_token: str = ""
    # How long a node may be silent before it is treated as gone.
    heartbeat_timeout_s: float = 30.0

    @property
    def keystore_path(self) -> str:
        return os.path.join(self.data_dir, "keys.json")

    @property
    def releases_dir(self) -> str:
        return os.path.join(self.data_dir, "releases")

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        return cls(
            grpc_port=_int("LOOM_GRPC_PORT", 9000),
            http_port=_int("LOOM_HTTP_PORT", 8000),
            public_address=os.environ.get("LOOM_PUBLIC_ADDRESS", "").strip(),
            data_dir=os.environ.get("LOOM_DATA_DIR", "/data"),
            admin_token=os.environ.get("LOOM_ADMIN_TOKEN", "").strip(),
            heartbeat_timeout_s=_float("LOOM_HEARTBEAT_TIMEOUT_S", 30.0),
        )
