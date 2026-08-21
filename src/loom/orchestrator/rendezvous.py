"""The orchestrator's own p2p node: the single address every worker is given.

A worker cannot dial another worker out of nowhere. Bootstrap peers are fixed
when a libp2p node is built, so a peer discovered later cannot simply be
"connected to" — it has to be *found*, and finding requires an entry point both
sides already share. This node is that entry point.

It does two jobs, and neither of them is carrying activations:

  **Rendezvous.** Every worker bootstraps to this node and to nothing else.
  From then on a worker can reach any other by peer id alone, which is what
  makes the whole scheme work for machines behind NAT — they do not know their
  own address and could not announce it if they tried. Measured locally, two
  peers that knew only the rendezvous found each other in about 105 ms.

  **Relay.** Roughly a third of peer pairs cannot be connected directly at all;
  a symmetric NAT gives each conversation a different port, so the address the
  other side was told to aim at is never the one that arrives. Those pairs keep
  talking through this node — exactly as every pair does today.

Loom already trusts this machine with placement, keys and every control
command, so pointing workers at it for rendezvous adds no new trust. It does
add one requirement the rest of Loom does not have: **this port must be
reachable from the workers.** The orchestrator already accepts inbound gRPC, so
it is a firewall line, not a new class of problem — and it is still the only
host in the system that needs one.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger("loom.orchestrator.rendezvous")

DEFAULT_PORT = int(os.environ.get("LOOM_P2P_PORT", "47100"))
DEFAULT_KEY_DIR = os.environ.get("LOOM_P2P_KEY_DIR", "/data/p2p")

# Circuit-relay servers workers should reserve a slot on, comma separated.
# A separate process from this one on purpose: Lattica speaks the relay CLIENT
# protocol (/libp2p/circuit/relay/0.2.0/stop) and never the server half, so the
# rendezvous cannot be the relay however reachable it is. See relay/relay.mjs.
RELAY_ADDRS = [a.strip() for a in os.environ.get("LOOM_P2P_RELAY", "").split(",") if a.strip()]


def lattica_available() -> bool:
    try:
        import lattica  # noqa: F401
    except Exception:
        return False
    return True


class RendezvousNode:
    """The orchestrator's libp2p endpoint, or a well-behaved absence of one.

    Optional on purpose. An orchestrator without the p2p stack, or one that
    cannot bind the port, keeps relaying every message — which is what it did
    before this existed. Nothing about the control plane depends on it.
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_PORT,
        key_dir: str = DEFAULT_KEY_DIR,
        public_host: str = "",
    ) -> None:
        self.port = port
        self.key_dir = key_dir
        # The host workers should dial. Taken from the address they already use
        # to reach the control plane, because that address is known to work
        # from wherever they are — which a locally-detected interface is not.
        self.public_host = public_host
        self._lattica = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Bring the node up. Returns whether workers now have a rendezvous."""
        if os.environ.get("LOOM_P2P", "1").strip().lower() in ("0", "false", "no"):
            logger.info("p2p disabled by LOOM_P2P; workers will relay through us")
            return False
        if not lattica_available():
            logger.info(
                "no p2p stack installed; workers will relay through us "
                "(pip install lattica to enable direct worker links)"
            )
            return False
        try:
            self._lattica = self._build()
        except Exception:
            logger.warning(
                "the rendezvous node did not start; workers will relay through us",
                exc_info=True,
            )
            self._lattica = None
            return False
        logger.info(
            "rendezvous up: %s on port %d — workers will be told to bootstrap here",
            self._lattica.peer_id(),
            self.port,
        )
        return True

    def _build(self):
        from lattica import Lattica

        os.makedirs(self.key_dir, exist_ok=True)
        builder = (
            Lattica.builder()
            .with_listen_addrs(
                [
                    f"/ip4/0.0.0.0/tcp/{self.port}",
                    f"/ip4/0.0.0.0/udp/{self.port}/quic-v1",
                ]
            )
            # A stable identity: the rendezvous multiaddr contains this node's
            # peer id, and workers hold on to it. Regenerating it on restart
            # would invalidate every worker's entry point at once.
            .with_key_path(self.key_dir)
            .with_dcutr(True)
            .with_autonat(True)
            .with_mdns(False)
        )
        if self.public_host:
            # Tell the stack how the outside world reaches us. Without this it
            # would announce 0.0.0.0 and whatever private interfaces it found,
            # none of which help a worker on another continent.
            builder = builder.with_external_addrs(self._announced_addrs())
        return builder.build()

    def stop(self) -> None:
        if self._lattica is None:
            return
        try:
            self._lattica.close()
        except Exception:
            logger.exception("closing the rendezvous node failed")
        self._lattica = None

    # -------------------------------------------------------------- identity
    @property
    def running(self) -> bool:
        return self._lattica is not None

    @property
    def peer_id(self) -> str:
        return self._lattica.peer_id() if self._lattica is not None else ""

    def _announced_addrs(self) -> List[str]:
        return [
            f"/ip4/{self.public_host}/tcp/{self.port}",
            f"/ip4/{self.public_host}/udp/{self.port}/quic-v1",
        ]

    def multiaddrs(self) -> List[str]:
        """What a worker is told to bootstrap to. Empty when there is no node.

        Both transports are offered and the worker hands them to libp2p as-is;
        TCP is the one that works through the most middleboxes, QUIC the one
        that avoids head-of-line blocking, and which wins is not our call.
        """
        if self._lattica is None or not self.public_host:
            return []
        return [f"{addr}/p2p/{self.peer_id}" for addr in self._announced_addrs()]


def host_of(address: str) -> str:
    """The host part of the orchestrator's public "host:port" address.

    Workers reach the control plane at this address, so it is known-good from
    wherever they are — unlike anything the orchestrator could detect locally.
    Only the host carries over: the p2p port is its own.
    """
    if not address:
        return ""
    address = address.strip()
    if address.startswith("["):  # bracketed IPv6
        return address[1 : address.index("]")] if "]" in address else ""
    return address.rsplit(":", 1)[0] if ":" in address else address
