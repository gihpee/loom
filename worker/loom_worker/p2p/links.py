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
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("loom_worker.p2p.links")

# How long a direct link is considered dead after it fails. Retrying a broken
# route on every token would add its timeout to every token; waiting a little
# costs one relayed hop and keeps the pipeline moving.
DIRECT_COOLDOWN_S = 30.0

# How long to give an acknowledgement before treating the message as lost.
SEND_TIMEOUT_S = float(os.environ.get("LOOM_P2P_SEND_TIMEOUT_S", "2"))

# How many handed-over messages may be awaiting acknowledgement. A pipeline
# decoding one sequence has exactly one in flight; the rest is slack.
PENDING_ACKS = 64


@dataclass
class Neighbour:
    """A peer this node may need to send to, as the orchestrator described it."""

    stage_index: int
    node_id: str
    peer_id: str = ""
    addrs: List[str] = field(default_factory=list)
    # What the trip from THIS peer to the relay costs it. Reported for the
    # operator's benefit; it decides nothing.
    relay_rtt_ms: float = 0.0
    # Can anything open a connection TO this peer? A circuit address does not
    # count — see LinkTable._worth_using.
    reachable: bool = False

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
        timeout_s: float = SEND_TIMEOUT_S,
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
        # peer_id -> (verdict, when, direct rtt, MY trip to the relay)
        #
        # The last one is the near half alone, never the sum. It is reported to
        # the orchestrator and handed to this node's neighbours as their far
        # half — so storing the sum here fed each node's total back to the
        # other as an input, and the two inflated each other every round until
        # relaying looked infinitely expensive and every circuit looked worth
        # using. See test_a_node_reports_its_own_distance_to_the_relay.
        self._worth: Dict[str, Tuple[bool, float, Optional[float], Optional[float]]] = {}
        self._timeout_s = timeout_s
        # Whether peers can dial US. Either end being reachable is enough for
        # a single-hop connection, so this is half of the rule.
        self._self_reachable = False
        # Peers we have already explained ourselves about, so the reason is
        # logged once rather than on every token.
        self._told: Dict[str, bool] = {}
        self._pending: "queue.Queue" = queue.Queue(maxsize=PENDING_ACKS)
        self._watcher: Optional[threading.Thread] = None
        self.stats = {"direct": 0, "relay": 0, "fallbacks": 0}

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
            self._told.clear()
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
            worth = self._worth_using(peer)
            if not worth and not self._told.get(peer.peer_id):
                self._told[peer.peer_id] = True
                relayed = True
            else:
                relayed = False
        if relayed:
            logger.info(
                "neither this node nor %s can accept an incoming connection, "
                "so libp2p could only build a circuit through the relay — the "
                "same two hops as the orchestrator's tunnel. Relaying instead. "
                "Open a port on either side to get a real direct link",
                peer.node_id,
            )
        return worth

    def set_self_reachable(self, reachable: bool) -> None:
        """Tell the table whether the outside world can dial this node."""
        with self._lock:
            changed = reachable != self._self_reachable
            self._self_reachable = reachable
        if changed:
            logger.info(
                "this node %s reachable from outside; peers %s open a direct "
                "link to it",
                "is now" if reachable else "is no longer",
                "can" if reachable else "cannot",
            )

    def refresh(self) -> None:
        """Collect the numbers the operator sees. Sampler thread only.

        Purely observational since the routing rule became topological: asking
        the p2p stack anything means entering its runtime, which must never
        happen on the token path or the heartbeat path — both stalled outright
        when it was busy.
        """
        if self._rtt is None:
            return
        with self._lock:
            peers = list({p.peer_id: p for p in self._neighbours.values()}.values())
        near = self._safe(self._relay_rtt) if self._relay_rtt else None
        for peer in peers:
            if not peer.dialable:
                continue
            direct = self._safe(lambda: self._rtt(peer.peer_id))
            with self._lock:
                self._worth[peer.peer_id] = (True, time.monotonic(), direct, near)

    def _worth_using(self, peer: Neighbour) -> bool:
        """Is there a real connection to be had, or only a detour?

        One question, answered from the topology rather than from a stopwatch:
        can either end accept an incoming connection? If yes, libp2p opens ONE
        hop between the two workers and it is unambiguously shorter than going
        through the orchestrator. If neither can, the only thing libp2p can
        build is a circuit through the relay — and Loom runs that relay on the
        orchestrator's own machine, so the circuit is the same two hops as the
        tunnel, minus the tunnel's advantages.

        This replaces a latency comparison that could not work. It measured
        the round trip to the peer and weighed it against "my trip to the relay
        plus the peer's" — but when the connection IS a circuit, those two
        quantities are the same journey, so the rule was comparing a path
        against a formula describing that same path. The winner was decided by
        jitter, the route flapped every 30 s, and no arrangement of the
        arithmetic could have fixed it.

        Latency is still measured, and still reported. It just does not decide
        anything: what matters here is topology, and topology is known.
        """
        return bool(peer.reachable or self._self_reachable)

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
                # Handed over, not awaited. The acknowledgement costs a return
                # trip that nothing here needs; watching for it happens on
                # another thread, so a failure still quarantines the link and
                # still relays what was lost.
                pending = self._send_direct(peer.peer_id, message)
                self._watch(peer, pending, message, relay)
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

    def _watch(self, peer, pending, message, relay) -> None:
        """Follow a handed-over message to its acknowledgement, off the token path."""
        if pending is None or not hasattr(pending, "result"):
            return  # a transport that does not acknowledge; nothing to watch
        self._start_watcher()
        try:
            self._pending.put_nowait((peer, pending, message, relay))
        except queue.Full:
            # More in flight than the watcher can follow. Waiting here is the
            # backpressure: better a slow token than an unbounded queue.
            self._settle(peer, pending, message, relay)

    def _start_watcher(self) -> None:
        with self._lock:
            if self._watcher is not None:
                return
            self._watcher = threading.Thread(
                target=self._watch_loop, name="loom-p2p-acks", daemon=True
            )
            self._watcher.start()

    def _watch_loop(self) -> None:
        while True:
            peer, pending, message, relay = self._pending.get()
            try:
                self._settle(peer, pending, message, relay)
            except Exception:
                logger.exception("following up a direct send failed")

    def _settle(self, peer, pending, message, relay) -> None:
        """Wait for one acknowledgement and repair the damage if it never comes."""
        try:
            pending.result(timeout=max(1, int(self._timeout_s)))
        except Exception as exc:  # noqa: BLE001 - every failure is handled alike
            self._block(peer, exc)
            # The message was never delivered. Sending it now is late, but a
            # late token is a token; a lost one ends the request.
            try:
                relay(message)
                with self._lock:
                    self.stats["direct"] = max(0, self.stats["direct"] - 1)
                    self.stats["relay"] += 1
            except Exception:
                logger.exception("relaying after a failed direct send also failed")

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
            measured = list(self._worth.values())
            link_rtt = next((v[2] for v in measured if v[2] is not None), None)
            # This node's own distance to the relay, and nothing else: it
            # travels to the neighbours as the half they cannot measure.
            relay_rtt = next((v[3] for v in measured if v[3]), None)
            return {
                "direct": self.stats["direct"],
                "relay": self.stats["relay"],
                "fallbacks": self.stats["fallbacks"],
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
                relay_rtt_ms=float(getattr(route, "relay_rtt_ms", 0.0) or 0.0),
                reachable=bool(getattr(route, "reachable", False)),
            )
        )
    return out
