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
# Public-IP echo services. Needed because inside a container we only see the
# docker bridge address (172.x), never the host's public IP.
IP_SERVICES = ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com")


@dataclass
class PublicAddress:
    address: str
    source: str  # "env" | "ngrok" | "public-ip" | "host-ip" | "loopback"
    reachable_externally: bool  # routable from the internet (not RFC1918)
    self_check: Optional[bool] = None  # could WE open a TCP connection to it?

    @property
    def severity(self) -> str:
        """"ok" | "info" | "warn" — how loudly the UI should talk about it.

        A private address is NOT a problem per se: a LAN/VPN/datacenter address
        works perfectly for workers on that same network, which is a completely
        normal deployment. Only a loopback address (or a failed TCP check on an
        address we do control) actually blocks remote workers.
        """
        if self.address.startswith("127.") or self.address.startswith("localhost"):
            return "warn"
        if self.self_check is False and not self.reachable_externally:
            return "warn"
        if not self.reachable_externally or self.self_check is not True:
            return "info"
        return "ok"

    @property
    def note(self) -> Optional[str]:
        """Human-readable status of the dial address (None when all good)."""
        check = {True: "TCP-проверка прошла", False: "TCP-проверка НЕ прошла", None: "TCP-проверка не выполнялась"}[
            self.self_check
        ]
        if self.address.startswith("127.") or self.address.startswith("localhost"):
            return (
                "Это loopback-адрес: подключиться смогут только воркеры на этой же "
                "машине. Задайте LOOM_PUBLIC_ADDR (адрес в вашей сети или публичный) "
                "либо поднимите туннель: docker compose --profile tunnel up -d."
            )
        if not self.reachable_externally:
            base = (
                f"Адрес в приватной сети ({self.address}) — это нормально, если ваши "
                f"воркеры в этой же сети/VPN. {check}."
            )
            if self.self_check is False:
                return (
                    base
                    + " Проверьте, что порт опубликован и не закрыт фаерволом: "
                    "воркеры звонят именно на этот адрес."
                )
            return base
        if self.self_check is False:
            return (
                f"Публичный адрес {self.address}, но {check.lower()} изнутри контейнера. "
                "Для многих NAT это нормально (hairpin), однако убедитесь, что порт "
                "открыт снаружи — воркеры звонят именно сюда."
            )
        return None

    @property
    def warning(self) -> Optional[str]:
        """Kept for compatibility: only real problems."""
        return self.note if self.severity == "warn" else None


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


def _public_ip_via_service() -> Optional[str]:
    """Ask an echo service for the host's public IP (skippable, short timeout)."""
    if os.environ.get("LOOM_SKIP_IP_LOOKUP", "").lower() in ("1", "true", "yes"):
        return None
    for url in IP_SERVICES:
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                ip = resp.text.strip()
                if ip and not _is_private(ip) and len(ip) <= 45:
                    return ip
        except Exception:
            continue
    return None


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> Optional[bool]:
    """Best-effort: can we connect to the address we are about to advertise?"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
    except Exception:
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
        host, _, port_s = explicit.rpartition(":")
        host = host or explicit
        try:
            port = int(port_s)
        except ValueError:
            port = grpc_port
        return PublicAddress(
            explicit,
            "env",
            not _is_private(host),
            self_check=_tcp_reachable(host, port),
        )

    tunnel = _from_ngrok(grpc_port)
    if tunnel:
        logger.info("public address discovered via tunnel: %s", tunnel)
        return PublicAddress(tunnel, "ngrok", True)

    # A containerised orchestrator only sees the docker bridge IP, so ask the
    # outside world what our public address is.
    public_ip = _public_ip_via_service()
    if public_ip:
        addr = f"{public_ip}:{grpc_port}"
        check = _tcp_reachable(public_ip, grpc_port)
        logger.info(
            "public IP detected: %s (tcp self-check: %s)",
            addr,
            {True: "ok", False: "failed", None: "unknown"}[check],
        )
        return PublicAddress(addr, "public-ip", True, self_check=check)

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
