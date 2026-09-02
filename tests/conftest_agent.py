"""Starting a real agent against a test orchestrator.

Kept apart from conftest.py so importing the agent package — which lives in its
own tree and is not a dependency of the orchestrator — never happens for the
tests that do not need it.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))


def make_join_key(address: str) -> str:
    body = json.dumps({"i": "k1", "s": "secret", "a": address}).encode()
    return "looma_" + base64.urlsafe_b64encode(body).decode().rstrip("=")


def start_agent(port: int, root: Path, node_id: str = "test-agent"):
    from looma_agent.config import parse_args
    from looma_agent.main import Agent

    config = parse_args([
        "--key", make_join_key(f"127.0.0.1:{port}"),
        "--node-id", node_id,
        "--root", str(root),
        "--heartbeat-interval", "0.5",
        "--reconnect-delay", "0.2",
    ])
    agent = Agent(config)
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    return agent, thread
