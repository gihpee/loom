"""Everything the agent needs to know before it can do anything.

One rule: a node owner passes ONE opaque string and nothing else. The
orchestrator address, the node's secret and its identity all come out of the
join key. Every other setting has a working default, because a setting with no
default is a support ticket from someone who just wanted to lend us a GPU.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    join_key: str
    node_id: str
    region: str
    root: Path
    heartbeat_interval_s: float
    reconnect_delay_s: float

    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"

    @property
    def envs_dir(self) -> Path:
        return self.root / "envs"

    @property
    def models_dir(self) -> Path:
        """Веса. Рядом с окружениями и по той же причине: и то и другое
        переживает задачу, которая их запросила."""
        return self.root / "models"


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(prog="looma-agent")
    parser.add_argument(
        "--key",
        default=os.environ.get("LOOMA_JOIN_KEY", ""),
        help="join key issued by the orchestrator; carries its address and this node's secret",
    )
    parser.add_argument(
        "--node-id",
        default=os.environ.get("LOOMA_NODE_ID", ""),
        help="stable name for this node (default: hostname, plus a GPU suffix when this "
             "agent was given only some of the machine's cards)",
    )
    parser.add_argument("--region", default=os.environ.get("LOOMA_REGION", "default"))
    parser.add_argument(
        "--root",
        default=os.environ.get("LOOMA_ROOT", "/var/lib/looma"),
        help="where task directories and the environment cache live",
    )
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--reconnect-delay", type=float, default=3.0)
    args = parser.parse_args(argv)
    return Config(
        join_key=args.key.strip(),
        node_id=args.node_id.strip(),
        region=args.region,
        root=Path(args.root),
        heartbeat_interval_s=args.heartbeat_interval,
        reconnect_delay_s=args.reconnect_delay,
    )
