"""Choosing how one stage's output reaches the next stage.

Two ways exist and both must keep working:

  relay   stage -> agent -> orchestrator -> agent -> stage
          Always available. The worker only dials out, so nothing needs an open
          port and no NAT has to be defeated. Costs two wide-area crossings.

  direct  stage -> agent =====> agent -> stage
          One crossing. Needs a libp2p route to the neighbour, which succeeds
          for roughly two pairs in three; symmetric NATs defeat it entirely.

This module owns the choice, per neighbour, at the moment a message is sent.
The rule is deliberately dull: use the direct link when one is known to be up,
otherwise relay, and never let a direct-path failure lose a token. A pipeline
that silently drops to relay is slower; a pipeline that drops a message is
broken, and the difference is the whole reason the fallback exists.

Nothing above this module knows which path was taken. The stage process still
POSTs to the agent's loopback relay exactly as before.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("loom_worker.p2p.links")

# How long a direct link is considered dead after it fails. Retrying a broken
# route on every token would add its timeout to every token; waiting a little
# costs one relayed hop and keeps the pipeline moving.
DIRECT_COOLDOWN_S = 30.0

# How often the worth of a link is re-examined. Long enough that the check is
# free, short enough that a link which becomes direct (hole punching succeeds
# some seconds after the first dial) starts being used within the same request.
WORTH_RECHECK_S = 30.0


@dataclass
class Neighbour:
    """A peer this node may need to send to, as the orchestrator described it."""

    stage_index: int
    node_id: str
    peer_id: str = ""
    addrs: List[str] = field(default_factory=list)

    @property
    def dialable(self) -> bool:
        """A peer with no identity can only ever be reached through the relay."""
        return bool(self.peer_id and self.addrs)


class LinkTable:
    """Neighbours of every pipeline this node takes part in.

    Keyed by (pipeline_id, stage_index), not by stage index alone. A node can
    host stages of several models at once, and "stage 1" means a different
    machine in each of them — a table keyed by index would send one pipeline's
    activations to the other's node, which is the kind of failure that produces
    plausible-looking wrong answers rather than an error.

    Each pipeline's entries are replaced wholesale when the orchestrator
    re-deploys it, and only that pipeline's: redeploying model A must not
    forget where model B's neighbours are.
    """

    def __init__(
        self,
        *,
        send_direct: Optional[Callable[[str, dict], dict]] = None,
        dial: Optional[Callable[[str, List[str]], None]] = None,
        cooldown_s: float = DIRECT_COOLDOWN_S,
        rtt: Optional[Callable[[str], Optional[float]]] = None,
        relay_rtt: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        self._lock = threading.RLock()
        # (pipeline_id, stage_index) -> Neighbour
        self._neighbours: Dict[Tuple[str, int], Neighbour] = {}
        self._blocked_until: Dict[str, float] = {}
        self._send_direct = send_direct
        self._dial = dial
        self._cooldown_s = cooldown_s
        self._rtt = rtt
        self._relay_rtt = relay_rtt
        # peer_id -> (verdict, when it expires, the two numbers behind it)
        self._worth: Dict[str, Tuple[bool, float, Optional[float], Optional[float]]] = {}
        self.stats = {"direct": 0, "relay": 0, "fallbacks": 0, "not_worth": 0}

    def attach(self, *, send_direct, dial, rtt=None, relay_rtt=None) -> None:
        """Hand over the p2p node once it exists.

        The table is created at process start, before the worker knows where
        the rendezvous is, and behaves as relay-only until this is called.
        Everything downstream holds the same object throughout, so nothing has
        to be re-wired when the direct path becomes available.
        """
        with self._lock:
            self._send_direct = send_direct
            self._dial = dial
            self._rtt = rtt
            self._relay_rtt = relay_rtt
        logger.info("direct path enabled")

    # ------------------------------------------------------------- directory
    def set_neighbours(self, pipeline_id: str, peers: List[Neighbour]) -> None:
        with self._lock:
            # Only this pipeline's rows: another model's stages on this node
            # keep their routes.
            self._neighbours = {
                key: value
                for key, value in self._neighbours.items()
                if key[0] != pipeline_id
            }
            for peer in peers:
                self._neighbours[(pipeline_id, peer.stage_index)] = peer
            self._blocked_until.clear()
        for peer in peers:
            if peer.dialable and self._dial is not None:
                # Start connecting now, not on the first token: hole punching
                # takes seconds, and the first request should not pay for it.
                try:
                    self._dial(peer.peer_id, peer.addrs)  # PeerNode.warm
                except Exception:
                    logger.debug("pre-dial of %s failed", peer.node_id, exc_info=True)
        logger.info(
            "peer directory: %s",
            ", ".join(
                f"stage {p.stage_index}={p.node_id}"
                f"{'' if p.dialable else ' (relay only)'}"
                for p in peers
            )
            or "empty",
        )

    def neighbour(self, pipeline_id: str, stage_index: int) -> Optional[Neighbour]:
        with self._lock:
            return self._neighbours.get((pipeline_id, stage_index))

    # ---------------------------------------------------------------- choice
    def direct_available(self, pipeline_id: str, stage_index: int) -> bool:
        """Would a message to this stage go directly, right now?"""
        with self._lock:
            peer = self._neighbours.get((pipeline_id, stage_index))
            if peer is None or not peer.dialable or self._send_direct is None:
                return False
            if time.monotonic() < self._blocked_until.get(peer.peer_id, 0.0):
                return False
        return self._worth_using(peer)

    def _worth_using(self, peer: Neighbour) -> bool:
        """Is reaching this peer over libp2p actually cheaper than relaying?

        The question sounds academic and is not. libp2p reports a link as
        established whether it is a real connection or a circuit through the
        relay, and Loom used to treat both as "direct" — so when hole punching
        failed, every activation went worker -> relay -> worker, which is the
        same two wide-area crossings as the orchestrator's tunnel plus a
        general-purpose relay in the middle. Measured on a two-stage pipeline
        across regions: transport went from 200 ms per token to 320 ms and the
        run halved in speed, while the admin page said "100% прямо".

        Worse, a circuit is not even stable: a relay applies the standard v2
        limits, 128 KB per connection, which for a 4B model is about 25 tokens
        before the connection is torn down and re-established mid-generation.

        The test is one comparison. A relay sits on the orchestrator's machine,
        so the trip to it is the same trip a relayed activation starts with; a
        peer that costs more to reach than the relay cannot be saving anything.
        And it settles the circuit question for free — a circuit runs THROUGH
        the relay, so it can never be cheaper than the relay.

        Deliberately conservative: when the numbers are not there yet the link
        is used, because that is what happened before this existed, and a
        missing measurement is not evidence of a bad link.
        """
        if self._relay_rtt is None or self._rtt is None:
            return True  # nothing to compare against: no relay, no circuits
        now = time.monotonic()
        with self._lock:
            cached = self._worth.get(peer.peer_id)
            if cached is not None and now < cached[1]:
                return cached[0]

        relay_rtt = self._safe(self._relay_rtt)
        peer_rtt = self._safe(lambda: self._rtt(peer.peer_id))
        verdict = True
        if relay_rtt is not None and peer_rtt is not None:
            verdict = peer_rtt <= relay_rtt

        with self._lock:
            previous = self._worth.get(peer.peer_id)
            self._worth[peer.peer_id] = (verdict, now + WORTH_RECHECK_S, peer_rtt, relay_rtt)
            if not verdict:
                self.stats["not_worth"] += 1
            changed = previous is None or previous[0] != verdict
        if changed and not verdict:
            logger.info(
                "link to %s costs %.0f ms against %.0f ms to the relay: it is a "
                "circuit or a detour, not a direct link. Relaying through the "
                "orchestrator instead (re-checked every %.0fs)",
                peer.node_id,
                peer_rtt,
                relay_rtt,
                WORTH_RECHECK_S,
            )
        elif changed and previous is not None:
            logger.info(
                "link to %s is now direct (%.0f ms against %.0f ms to the relay)",
                peer.node_id,
                peer_rtt if peer_rtt is not None else -1,
                relay_rtt if relay_rtt is not None else -1,
            )
        return verdict

    @staticmethod
    def _safe(call):
        """A measurement that fails is a measurement we do not have."""
        try:
            return call()
        except Exception:
            return None

    def send(
        self,
        pipeline_id: str,
        stage_index: int,
        message: dict,
        relay: Callable[[dict], None],
    ) -> str:
        """Deliver one message, and say which path carried it.

        `relay` is the caller's existing path through the orchestrator. It is
        passed in rather than imported so this module has no opinion about the
        control plane — and so the fallback is impossible to forget.
        """
        if self.direct_available(pipeline_id, stage_index):
            peer = self.neighbour(pipeline_id, stage_index)
            try:
                self._send_direct(peer.peer_id, message)
                with self._lock:
                    self.stats["direct"] += 1
                return "direct"
            except Exception as exc:
                # The token still has to arrive. Relay it, and stop choosing
                # this route until the cooldown expires.
                self._block(peer, exc)

        relay(message)
        with self._lock:
            self.stats["relay"] += 1
        return "relay"

    def _block(self, peer: Neighbour, exc: BaseException) -> None:
        with self._lock:
            self._blocked_until[peer.peer_id] = time.monotonic() + self._cooldown_s
            self.stats["fallbacks"] += 1
            count = self.stats["fallbacks"]
        logger.warning(
            "direct link to %s failed (%s); relaying for %.0fs (fallback #%d)",
            peer.node_id,
            exc,
            self._cooldown_s,
            count,
        )

    def snapshot(self) -> dict:
        """What the agent reports about its data path, for telemetry."""
        with self._lock:
            total = self.stats["direct"] + self.stats["relay"]
            rtts = [v for v in self._worth.values()]
            link_rtt = next((v[2] for v in rtts if v[2] is not None), None)
            relay_rtt = next((v[3] for v in rtts if v[3] is not None), None)
            return {
                "direct": self.stats["direct"],
                "relay": self.stats["relay"],
                "fallbacks": self.stats["fallbacks"],
                "not_worth": self.stats["not_worth"],
                "link_rtt_ms": round(link_rtt, 1) if link_rtt is not None else 0.0,
                "relay_rtt_ms": round(relay_rtt, 1) if relay_rtt is not None else 0.0,
                "direct_share": round(self.stats["direct"] / total, 3) if total else 0.0,
                "neighbours": [
                    {
                        "pipeline_id": pipeline_id,
                        "stage_index": peer.stage_index,
                        "node_id": peer.node_id,
                        "peer_id": peer.peer_id,
                        "direct": self.direct_available(pipeline_id, peer.stage_index),
                    }
                    for (pipeline_id, _index), peer in sorted(self._neighbours.items())
                ],
            }


def neighbours_from_topology(topology, *, self_node_id: str) -> List[Neighbour]:
    """Read the peer directory out of a LoadShard topology message.

    Tolerant on purpose: an orchestrator that predates the directory sends no
    `peers` at all, and the node must come up relaying rather than refuse to
    start. Same for a peer with no identity — it simply is not dialable.
    """
    out: List[Neighbour] = []
    for route in getattr(topology, "peers", []) or []:
        if route.node_id == self_node_id:
            continue  # a stage does not dial itself
        out.append(
            Neighbour(
                stage_index=int(route.stage_index),
                node_id=route.node_id,
                peer_id=route.peer_id or "",
                addrs=list(route.addrs or []),
            )
        )
    return out
