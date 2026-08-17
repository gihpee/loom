"""Figure out the address workers should dial, without asking the operator.

Physics first: a worker behind NAT can only *dial out*, so **something** must be
publicly reachable. Parallax solves this with third-party relay servers baked
into its code (`relay-lattica.gradient.network`); a self-hosted Loom needs its
own reachable entry point. This module finds it automatically, in order:

1. `LOOM_PUBLIC_ADDR` — explicit override (a domain, a public IP, a VPN address).
2. A TCP tunnel running next to the orchestrator (ngrok's local API): gives a
   public `host:port` without any public IP or port forwarding.
3. The host's own routable address, if it happens to have one.
4. Fallback to `127.0.0.1:<port>` and a loud warning: keys issued this way only
   work for workers on the same machine/network.

The result is what gets embedded into every join key, so the worker needs
nothing but `--key`.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional

import httpx

from loom.logging_config import get_logger

logger = get_logger(__name__)

NGROK_API = "http://ngrok:4040/api/tunnels"
NGROK_API_LOCAL = "http://127.0.0.1:4040/api/tunnels"


@dataclass
class PublicAddress:
    address: str
    source: str  # "env" | "ngrok" | "host-ip" | "loopback"
    reachable_externally: bool

    @property
    def warning(self) -> Optional[str]:
        if self.reachable_externally:
            return None
        return (
            "Workers on other machines cannot reach this address. Either set "
            "LOOM_PUBLIC_ADDR to a reachable host:port, or run the bundled "
            "tunnel (docker compose --profile tunnel up -d) so a public "
            "endpoint is created for you."
        )


def _from_ngrok(grpc_port: int) -> Optional[str]:
    """Ask a co-located ngrok agent for its public TCP endpoint."""
    for api in (NGROK_API, NGROK_API_LOCAL):
        try:
            resp = httpx.get(api, timeout=3)
            if resp.status_code != 200:
                continue
            for tunnel in resp.json().get("tunnels", []):
                url = tunnel.get("public_url", "")
                if url.startswith("tcp://"):
                    return url[len("tcp://") :]
        except Exception:
            continue
    return None


def _host_ip() -> Optional[str]:
    """Best-effort address of this host as seen from the network it sits on.

    Returns private addresses too (LAN / docker network): workers on the same
    network can use them, and `reachable_externally` still reports the truth so
    the UI warns about the internet case.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
    except Exception:
        return None
    return None if ip.startswith("127.") else ip


def _is_private(host: str) -> bool:
    return (
        host.startswith("127.")
        or host.startswith("10.")
        or host.startswith("192.168.")
        or host.startswith("172.")
        or host in ("localhost", "orchestrator", "0.0.0.0")
    )


def resolve_public_address(grpc_port: int) -> PublicAddress:
    explicit = os.environ.get("LOOM_PUBLIC_ADDR", "").strip()
    if explicit:
        host = explicit.rsplit(":", 1)[0]
        return PublicAddress(explicit, "env", not _is_private(host))

    tunnel = _from_ngrok(grpc_port)
    if tunnel:
        logger.info("public address discovered via tunnel: %s", tunnel)
        return PublicAddress(tunnel, "ngrok", True)

    ip = _host_ip()
    if ip:
        addr = f"{ip}:{grpc_port}"
        private = _is_private(ip)
        logger.info(
            "using host address %s (%s)",
            addr,
            "same-network workers only" if private else "publicly reachable",
        )
        return PublicAddress(addr, "host-ip", not private)

    addr = f"127.0.0.1:{grpc_port}"
    logger.warning(
        "no public address found; join keys will point at %s and only work locally", addr
    )
    return PublicAddress(addr, "loopback", False)
