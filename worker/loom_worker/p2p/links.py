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

# How often the worth of a link is re-examined, on the sampler thread. Long
# enough that the measuring is free, short enough that a link which becomes
# direct (hole punching succeeds some seconds after the first dial) starts
# being used within the same request.
WORTH_RECHECK_S = 30.0

# How much better the other path must be before the route is changed. Without
# it a pair whose two paths cost about the same flips on every re-check, and
# each flip costs a dial.
HYSTERESIS = 0.15

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
    # What the trip from THIS peer to the relay costs it. The far half of the
    # relayed path, which cannot be measured from here.
    relay_rtt_ms: float = 0.0

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
        self._pending: "queue.Queue" = queue.Queue(maxsize=PENDING_ACKS)
        self._watcher: Optional[threading.Thread] = None
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
            cached = self._worth.get(peer.peer_id)
        # Cache only, never a measurement. Asking the p2p stack anything means
        # calling into its runtime, and this runs on the token path and on the
        # heartbeat path — both of which stalled outright when the runtime was
        # busy. `refresh()` does the asking, on a thread of its own.
        return True if cached is None else cached[0]

    def refresh(self) -> None:
        """Re-measure every link. Called from the sampler thread, only."""
        with self._lock:
            peers = list({p.peer_id: p for p in self._neighbours.values()}.values())
        for peer in peers:
            if peer.dialable:
                self._worth_using(peer)

    def _worth_using(self, peer: Neighbour) -> bool:
        """Is reaching this peer over libp2p cheaper than relaying to it?

        libp2p reports a link as established whether it is a real connection
        or a circuit through the relay, and Loom used to treat both as direct.
        When hole punching failed, every activation went worker -> relay ->
        worker — the same two wide-area crossings as the orchestrator's tunnel
        plus a general-purpose relay in the middle — while the admin page said
        "100% прямо". Measured across regions: 200 ms per token became 320 ms.

        The comparison is between two whole paths:

            direct   = my trip to the peer
            relayed  = my trip to the relay + the peer's trip to the relay

        The second half cannot be measured from here, so the peer reports it
        and the orchestrator passes it on with the rest of the directory.
        Leaving it out is not a small simplification — it was wrong in a way
        that showed up immediately on a real stand. A node 8 ms from the relay
        rejected a genuinely direct 94 ms link to a peer that was itself 90 ms
        from the relay, because 94 > 8. The relayed path was 98 ms; direct was
        the better route and got refused.

        A circuit is rejected by the same arithmetic without a special case:
        it runs through the relay, so it costs both halves by construction.
        """
        if self._relay_rtt is None or self._rtt is None:
            return True  # nothing to compare against: no relay, no circuits
        now = time.monotonic()
        with self._lock:
            cached = self._worth.get(peer.peer_id)
            previous = cached[0] if cached is not None else None

        direct = self._safe(lambda: self._rtt(peer.peer_id))
        near = self._safe(self._relay_rtt)
        far = peer.relay_rtt_ms or 0.0
        verdict = self._verdict(direct, near, far, previous)
        relayed = (near or 0.0) + far

        with self._lock:
            self._worth[peer.peer_id] = (verdict, now, direct, near)
            if not verdict:
                self.stats["not_worth"] += 1
        if previous is not None and previous == verdict:
            return verdict
        if not verdict:
            logger.info(
                "link to %s costs %.0f ms against %.0f ms relayed (%.0f + %.0f): "
                "a circuit or a detour, not a direct link. Relaying through the "
                "orchestrator instead",
                peer.node_id,
                direct if direct is not None else -1,
                relayed,
                near or 0.0,
                far,
            )
        elif previous is not None:
            logger.info(
                "link to %s is direct and cheaper: %.0f ms against %.0f ms relayed",
                peer.node_id,
                direct if direct is not None else -1,
                relayed,
            )
        return verdict

    def _verdict(self, direct, near, far, previous) -> bool:
        """Which path wins, with enough hysteresis to stop it oscillating.

        Both paths often cost nearly the same — on a stand where the relay sat
        almost exactly between two nodes they measured 94 ms and 98 ms — and a
        plain comparison then flips the route every time it is re-examined,
        purely on jitter. Switching costs a dial and a cooldown, so a tie must
        resolve to "leave it alone" rather than to whichever number won this
        second.
        """
        if direct is None or near is None:
            return True  # a missing measurement is not evidence of a bad link
        if not far:
            # The peer never reported its distance to the relay. The far half
            # is unknown, and guessing it as zero is what rejected good links,
            # so trust the link instead.
            return True
        relayed = near + far
        if previous is True:
            return direct <= relayed * (1.0 + HYSTERESIS)
        if previous is False:
            return direct < relayed * (1.0 - HYSTERESIS)
        return direct <= relayed

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
                relay_rtt_ms=float(getattr(route, "relay_rtt_ms", 0.0) or 0.0),
            )
        )
    return out
