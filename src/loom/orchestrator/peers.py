"""The orchestrator's peer directory: who can reach whom, and at what address.

This is the rendezvous half of NAT traversal, and it is the only part of the
direct data path the orchestrator is involved in. It knows three things no
worker can know on its own:

  1. **Who the neighbours are.** Placement is the orchestrator's decision, so
     it alone knows that stage 2 of a pipeline moved to another machine.
  2. **What a node's public address is.** A machine behind NAT sees only its
     private address. The orchestrator sees the address its control connection
     arrives from — which is that node's public address, the same fact a STUN
     server exists to provide, obtained here for free.
  3. **Whether a direct path is worth attempting at all.** A node that detected
     a symmetric NAT cannot be hole-punched into; telling its peers to keep
     trying would cost a timeout per token.

Everything else — the handshake, the punching, the encryption, the upgrade from
a relayed connection to a direct one — happens between the workers, with no
orchestrator involvement and no orchestrator trust.

Note what is deliberately absent: a DHT. Loom has a coordinator that already
knows the answer, so looking it up in a distributed hash table would be slower,
weaker and pointless. libp2p is used here for the one thing it is uniquely good
at — getting two machines behind NAT to talk — not for decentralisation Loom
does not want.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Candidate ordering matters. A peer tries addresses in the order given, and a
# wrong guess costs a connection timeout, so the most likely one goes first.
# Private addresses lead: two nodes in the same datacentre reach each other
# directly on the LAN, and that path is both faster and always available. The
# observed public address follows, for peers actually separated by the internet.
_PRIVATE_PREFIXES = ("10.", "192.168.", "127.", "172.16.", "172.17.", "172.18.",
                     "172.19.", "172.2", "172.30.", "172.31.")


@dataclass
class PeerRecord:
    """What the orchestrator knows about one node's p2p endpoint."""

    node_id: str
    peer_id: str = ""
    listen_addrs: List[str] = field(default_factory=list)
    observed_host: str = ""
    symmetric_nat: bool = False
    # Addresses AutoNAT confirmed. Empty means nobody can dial in.
    visible_addrs: List[str] = field(default_factory=list)
    # Reported by the node itself, refreshed on every heartbeat.
    direct: int = 0
    relayed: int = 0
    fallbacks: int = 0
    direct_share: float = 0.0

    @property
    def dialable(self) -> bool:
        """Can a neighbour even try a direct connection to this node?

        A symmetric NAT is not a "maybe": hole punching through one cannot
        work, because the address a peer was told to aim at is not the one the
        NAT will use for that peer. Marking it here stops every neighbour from
        rediscovering that with a timeout.
        """
        return bool(self.peer_id) and not self.symmetric_nat

    def candidate_addrs(self, p2p_port: int = 0) -> List[str]:
        """Addresses a peer should try, best first.

        The node's own list first (it knows its port and its interfaces), then
        the address the orchestrator observed, rebuilt into a multiaddr with
        the same port the node reported. Duplicates are dropped: a node on a
        public IP reports the address the orchestrator also observes, and
        dialling it twice just doubles the failure timeout.
        """
        out: List[str] = []
        seen = set()

        def push(addr: str) -> None:
            if addr and addr not in seen:
                seen.add(addr)
                out.append(addr)

        for addr in sorted(self.listen_addrs, key=_private_first):
            push(addr)
        for addr in self._observed_multiaddrs(p2p_port):
            push(addr)
        return out

    def _observed_multiaddrs(self, fallback_port: int) -> List[str]:
        """The observed host, expressed on every transport the node listens on."""
        if not self.observed_host or ":" in self.observed_host:
            return []  # empty, or IPv6 which the node's own list already covers
        port = _port_from(self.listen_addrs) or fallback_port
        if not port:
            return []
        return [
            f"/ip4/{self.observed_host}/tcp/{port}",
            f"/ip4/{self.observed_host}/udp/{port}/quic-v1",
        ]


def _private_first(addr: str) -> Tuple[int, str]:
    """Sort key putting LAN addresses ahead of routable ones."""
    for prefix in _PRIVATE_PREFIXES:
        if f"/ip4/{prefix}" in addr:
            return (0, addr)
    return (1, addr)


def _port_from(addrs: Sequence[str]) -> int:
    """The port a node listens on, read back out of its own multiaddrs."""
    for addr in addrs:
        parts = addr.split("/")
        for i, part in enumerate(parts):
            if part in ("tcp", "udp") and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
    return 0


class PeerDirectory:
    """Every node's p2p endpoint, kept as its control connection reports it."""

    def __init__(self, *, default_port: int = 47100) -> None:
        self._records: Dict[str, PeerRecord] = {}
        self.default_port = default_port

    def remember(self, node_id: str, peer, observed_host: str = "") -> PeerRecord:
        """Record what a node declared at registration.

        `peer` is the PeerIdentity submessage, absent on workers that have no
        p2p stack — an old image, a CPU node, a build without lattica. Those
        get a record with no peer id, which simply never becomes dialable, and
        their traffic keeps flowing through the relay.
        """
        record = PeerRecord(
            node_id=node_id,
            peer_id=getattr(peer, "peer_id", "") or "",
            listen_addrs=list(getattr(peer, "listen_addrs", []) or []),
            observed_host=observed_host or "",
            symmetric_nat=bool(getattr(peer, "symmetric_nat", False)),
            visible_addrs=list(getattr(peer, "visible_addrs", []) or []),
        )
        self._records[node_id] = record
        return record

    def record_transport(self, node_id: str, status) -> None:
        """How this node's inter-stage traffic actually travelled.

        The direct path degrades to the relay without saying so, which is the
        right behaviour and a terrible property to debug blind: a deployment
        that fell back looks exactly like one that never tried. These counters
        are the only way to tell them apart from the outside.
        """
        record = self._records.get(node_id)
        if record is None:
            return
        record.direct = int(getattr(status, "direct", 0) or 0)
        record.relayed = int(getattr(status, "relayed", 0) or 0)
        record.fallbacks = int(getattr(status, "fallbacks", 0) or 0)
        record.direct_share = float(getattr(status, "direct_share", 0.0) or 0.0)

    def forget(self, node_id: str) -> None:
        self._records.pop(node_id, None)

    def get(self, node_id: str) -> Optional[PeerRecord]:
        return self._records.get(node_id)

    def routes_for(
        self, stages: Iterable[Tuple[int, str]]
    ) -> List[Tuple[int, str, str, List[str]]]:
        """Directory rows for one pipeline: (stage_index, node_id, peer_id, addrs).

        Every stage is described to every other one, including stages that
        cannot be dialled — a peer with an empty id is how a node learns to
        stop hoping and relay instead. The last stage sends its sampled token
        back to stage 0, so "the next stage" is not the only neighbour that
        matters and a full directory is simpler than predicting the topology.
        """
        rows = []
        for stage_index, node_id in stages:
            record = self._records.get(node_id)
            if record is None or not record.dialable:
                rows.append((stage_index, node_id, "", []))
                continue
            rows.append(
                (
                    stage_index,
                    node_id,
                    record.peer_id,
                    record.candidate_addrs(self.default_port),
                )
            )
        return rows

    def view(self) -> List[dict]:
        """Read-only state for the admin UI."""
        return [
            {
                "node_id": r.node_id,
                "peer_id": r.peer_id,
                "observed_host": r.observed_host,
                "listen_addrs": r.listen_addrs,
                "symmetric_nat": r.symmetric_nat,
                "dialable": r.dialable,
                "reachable": bool(r.visible_addrs),
                "visible_addrs": r.visible_addrs,
                "direct": r.direct,
                "relayed": r.relayed,
                "fallbacks": r.fallbacks,
                "direct_share": r.direct_share,
            }
            for r in sorted(self._records.values(), key=lambda r: r.node_id)
        ]
