"""The node's peer-to-peer identity and its link to other workers.

Loom's control plane is hub-and-spoke and stays that way: the orchestrator is
the only trusted party, it decides placement, and every worker reaches it with
one outbound connection and no open ports. What this module adds is a second,
UNTRUSTED path used only for bulk data — activations flowing from one pipeline
stage to the next.

Why bother. On a geographically spread pipeline the relayed path costs two wide
-area crossings per hop (worker -> orchestrator -> worker) where a direct one
costs one. Measured on a three-stage pipeline across regions: 80 ms of the
133 ms spent per token was transport, over six crossings. Halving the crossings
is worth more than any amount of tuning inside the stages.

What this is NOT: a decentralised system. There is no DHT lookup, no gossip and
no peer discovery here. The orchestrator tells each worker exactly who its
neighbours are and where to try reaching them — the "rendezvous service" role
that Lattica's own design assigns to a coordinator. We use libp2p (through
Lattica) for the one thing it is uniquely good at: getting two machines behind
NAT to talk to each other.

Degradation is the design, not an afterthought. Roughly a third of peer pairs
cannot be connected directly — symmetric NATs make hole punching impossible —
so every path here has the orchestrator's relay behind it. A node with no p2p
stack at all keeps working exactly as it did before.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("loom_worker.p2p")

# Default port for both transports. libp2p is happy to share a number between
# TCP and QUIC because they are different protocols; keeping them equal makes
# a firewall rule (for the operators who do open one) a single line.
DEFAULT_P2P_PORT = int(os.environ.get("LOOM_P2P_PORT", "47100"))

# Where the node keeps its keypair. The peer id must survive restarts: it is
# how the orchestrator and the neighbours refer to this machine, and a node
# that regenerates its identity on every boot looks like a stranger each time.
# Expanded from HOME rather than hardcoded. In the worker image HOME is /root,
# so this resolves to exactly the path it always did; on a Mac running the
# agent natively it lands in the user's cache instead of /root, which does not
# exist there and is not writable — a hardcoded container path is how a node
# that is otherwise fine ends up relaying every message.
DEFAULT_KEY_DIR = os.environ.get(
    "LOOM_P2P_KEY_DIR", os.path.join(os.path.expanduser("~"), ".cache", "loom", "p2p")
)

# How long one direct send may take before the relay is used instead. A token
# budget is a couple of hundred milliseconds, so anything here is expensive —
# but the cooldown means it is paid once per failure, not once per token.
SEND_TIMEOUT_S = float(os.environ.get("LOOM_P2P_SEND_TIMEOUT_S", "5"))


class P2PUnavailable(RuntimeError):
    """No usable p2p stack on this node; the caller must fall back to relay."""


def _address_in_use(exc: BaseException) -> bool:
    """Is this the Rust core telling us the port is taken?

    By string, because it arrives as a chain of anonymous Either wrappers with
    no type to match on:

        Transport(Left(Left(Left(Os { code: 98, kind: AddrInUse, ... }))))
    """
    text = str(exc)
    return "AddrInUse" in text or "Address already in use" in text


@dataclass
class PeerIdentity:
    """What this node tells the orchestrator about how to reach it."""

    peer_id: str
    listen_addrs: List[str] = field(default_factory=list)
    symmetric_nat: bool = False
    # Addresses AutoNAT confirmed the outside world can actually reach. This
    # is the difference between "probably fine" and "reachable": a node with
    # none of these cannot accept an inbound connection, so no peer can open a
    # direct link TO it however hard it tries.
    visible_addrs: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "listen_addrs": list(self.listen_addrs),
            "symmetric_nat": self.symmetric_nat,
        }


def lattica_available() -> bool:
    """Is the p2p stack importable at all?

    The worker image ships it, but the module must keep working without it:
    the transformers-only image, a CPU node, and every test on a machine with
    no networking all import this package.
    """
    try:
        import lattica  # noqa: F401
    except Exception:
        return False
    return True


def local_candidate_addrs(port: int) -> List[str]:
    """Multiaddrs this node believes it listens on.

    Candidates, not truth. A node behind NAT sees only its private address and
    has no way to learn its public one on its own — that is what the observed
    address from the orchestrator is for. Both are offered to the neighbour,
    which tries them in order, because either can be the one that works: the
    private one when the peers share a LAN, the public one otherwise.
    """
    addrs: List[str] = []
    for ip in _local_ips():
        addrs.append(f"/ip4/{ip}/tcp/{port}")
        addrs.append(f"/ip4/{ip}/udp/{port}/quic-v1")
    return addrs


# Docker's default address pools. A container on a bridge network sees only
# these, and that is the whole problem — see behind_container_nat().
_DOCKER_POOL = ("172.1", "172.2", "172.3")


def behind_container_nat() -> bool:
    """Is this node inside a container whose ports the host does not share?

    It matters more than anything else about NAT here. Hole punching works by
    dialling OUT from the very port you listen on, so the mapping the router
    creates is the one a peer can aim at. A bridge network breaks that in two
    independent ways: the container's addresses are its own private ones, and
    Docker re-translates every outgoing packet, so the port the peer is told
    to aim at is not the port anything arrives on. And with no published port
    there is no socket on the host at all — a punched TCP connection is met
    with a reset.

    No amount of relay or DCUtR recovers from this: the two peers can be
    perfectly willing and the packets still have nowhere to land. On Linux the
    fix is one flag, `--network host`.
    """
    if not os.path.exists("/.dockerenv"):
        return False
    ips = [ip for ip in _local_ips() if not ip.startswith("127.")]
    return bool(ips) and all(ip.startswith(_DOCKER_POOL) for ip in ips)


def _local_ips() -> List[str]:
    """Non-loopback IPv4 addresses of this host, best effort."""
    found: List[str] = []
    # The usual trick: ask the routing table which source address it would use
    # to reach the outside world. No packet is sent (UDP connect is local).
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        found.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found or ["127.0.0.1"]


class PeerNode:
    """This worker's libp2p node: identity, dialling, and inbound RPC.

    One per agent process, shared by every stage it hosts. Built lazily and
    never fatally: if it cannot start, the agent logs it once and keeps using
    the orchestrator's relay for everything.
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_P2P_PORT,
        key_dir: str = DEFAULT_KEY_DIR,
        bootstraps: Optional[List[str]] = None,
        relay_servers: Optional[List[str]] = None,
    ) -> None:
        # Bootstraps are fixed at build time — this is how a node enters the
        # network, and it is why the orchestrator runs a rendezvous node that
        # every worker is pointed at. Peers are found by id afterwards.
        self.port = port
        self.key_dir = key_dir
        self.bootstraps = list(bootstraps or [])
        self.relay_servers = list(relay_servers or [])
        self._lattica = None
        self._handler = None
        self._lock = threading.RLock()
        self._stubs: Dict[str, object] = {}
        self._on_message: Optional[Callable[[dict], None]] = None

    # How long to wait for the node to actually join the network through its
    # bootstrap peer. Building a Lattica node returns immediately, but until it
    # has joined, a peer cannot be resolved and a send fails with "failed to
    # reconnect". Announcing a direct path before that would make the first
    # tokens of every deployment fall back to the relay for no reason.
    JOIN_TIMEOUT_S = 15.0

    # ------------------------------------------------------------- lifecycle
    def start(self, on_message: Callable[[dict], None]) -> PeerIdentity:
        """Bring the node up and serve `on_message` to peers that dial in.

        Raises P2PUnavailable rather than returning a broken node, so a caller
        that forgets to handle failure fails loudly here instead of silently
        losing the direct path later.
        """
        if not lattica_available():
            raise P2PUnavailable("the lattica package is not installed")
        with self._lock:
            if self._lattica is not None:
                return self.identity()
            self._on_message = on_message
            self._lattica = self._build()
            self._handler = _make_handler(self._lattica, self._deliver)
            self._lattica.register_service(self._handler)
            joined = self._await_join()
            identity = self.identity()
            if self.bootstraps and not joined:
                logger.warning(
                    "p2p node did not reach its rendezvous within %.0fs; peers "
                    "may be unreachable until it does",
                    self.JOIN_TIMEOUT_S,
                )
            logger.info(
                "p2p node up: %s on port %d (%d bootstrap, %d relay)",
                identity.peer_id,
                self.port,
                len(self.bootstraps),
                len(self.relay_servers),
            )
            return identity

    def _await_join(self) -> bool:
        """Block until this node is part of the network, or give up.

        `get_all_peers()` lists ESTABLISHED connections, so it is not a
        readiness signal for a specific peer — bootstrapping does not connect
        two workers to each other, and each is resolved on first use. It is a
        perfectly good signal for "am I in the network at all", which is what
        has to be true before any resolution can succeed.
        """
        if not self.bootstraps:
            return True
        import time

        deadline = time.monotonic() + self.JOIN_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.connected_peers():
                return True
            time.sleep(0.1)
        return False

    # How many ports to try before giving up. One machine can host several
    # workers, and with --network host they share the host's ports — so the
    # second one to start finds 47100 taken. The port number itself does not
    # have to be the configured one: peers are found by id, and the address
    # a peer dials is announced, not assumed.
    PORT_ATTEMPTS = 10

    def _build(self):
        """The node, on the configured port or the next free one.

        A busy port used to surface as a Rust error nested eight levels deep —
        `Transport(Left(Left(Left(Os { code: 98 ... }))))` — and cost the node
        its direct path entirely.
        """
        last: Optional[BaseException] = None
        for offset in range(self.PORT_ATTEMPTS):
            port = self.port + offset
            try:
                node = self._build_on(port)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not _address_in_use(exc):
                    raise
                last = exc
                continue
            if offset:
                logger.warning(
                    "port %d was already in use (another worker on this host?); "
                    "this node took %d instead. Open %d, not %d, if you forward "
                    "ports for it",
                    self.port,
                    port,
                    port,
                    self.port,
                )
                self.port = port
            return node
        raise P2PUnavailable(
            f"ports {self.port}-{self.port + self.PORT_ATTEMPTS - 1} are all in "
            f"use on this host. With --network host every worker shares the "
            f"host's ports, so give each one its own LOOM_P2P_PORT"
        ) from last

    def _build_on(self, port: int):
        from lattica import Lattica

        key_dir = self._usable_key_dir()
        builder = (
            Lattica.builder()
            .with_listen_addrs(
                [
                    f"/ip4/0.0.0.0/tcp/{port}",
                    f"/ip4/0.0.0.0/udp/{port}/quic-v1",
                ]
            )
            # A stable identity across restarts (see DEFAULT_KEY_DIR).
            .with_key_path(key_dir)
            # Hole punching, and the reachability probe that tells us whether
            # it is even worth attempting.
            .with_dcutr(True)
            .with_autonat(True)
            # Off: peers are named by the orchestrator, never found by shouting
            # on the local network. On a rented host mDNS is noise at best.
            .with_mdns(False)
            # Ask the router to forward our port. Free when it works (a lot of
            # home routers support it), silent when it does not, and every node
            # it makes reachable is one more pair that can connect directly.
            .with_upnp(True)
        )
        if self.bootstraps:
            builder = builder.with_bootstraps(self.bootstraps)
        if self.relay_servers:
            # Deliberately rare. `with_relay_servers` makes the node reserve a
            # circuit-relay slot on those hosts and WAIT for it; pointing it at
            # a node that offers no relay service leaves it stuck outside the
            # network entirely (measured: 15 s timeout and zero peers, against
            # 112 ms to join without it).
            #
            # Loom does not need libp2p's relay: it already has one. Traffic
            # that cannot go directly falls back to the orchestrator's tunnel,
            # which is built, authenticated and carrying every message today.
            builder = builder.with_relay_servers(self.relay_servers)
        return builder.build()

    def _usable_key_dir(self) -> str:
        """A directory this process can actually write to.

        A node that cannot store its keypair should still join the network:
        losing the identity across restarts costs nothing much, because the
        orchestrator relearns every peer id from the heartbeats anyway. Losing
        the direct path costs a wide-area round trip on every token.
        """
        import tempfile

        try:
            os.makedirs(self.key_dir, exist_ok=True)
            probe = os.path.join(self.key_dir, ".writable")
            with open(probe, "w"):
                pass
            os.unlink(probe)
            return self.key_dir
        except OSError as exc:
            fallback = tempfile.mkdtemp(prefix="loom-p2p-")
            logger.warning(
                "cannot use %s for the p2p identity (%s); falling back to %s, "
                "so this node's peer id will change when it restarts",
                self.key_dir,
                exc,
                fallback,
            )
            return fallback

    def close(self) -> None:
        with self._lock:
            if self._lattica is None:
                return
            try:
                self._lattica.close()
            except Exception:
                logger.exception("closing the p2p node failed")
            self._lattica = None
            self._handler = None
            self._stubs.clear()

    @property
    def running(self) -> bool:
        return self._lattica is not None

    # -------------------------------------------------------------- identity
    def identity(self) -> PeerIdentity:
        if self._lattica is None:
            raise P2PUnavailable("the p2p node is not running")
        return PeerIdentity(
            peer_id=self._lattica.peer_id(),
            listen_addrs=local_candidate_addrs(self.port),
            symmetric_nat=bool(self._lattica.is_symmetric_nat()),
            visible_addrs=self.visible_addrs(),
        )

    def visible_addrs(self) -> List[str]:
        """What AutoNAT says the outside world can reach, if anything."""
        if self._lattica is None:
            return []
        try:
            return list(self._lattica.get_visible_maddrs() or [])
        except Exception:
            return []

    def rtt_ms(self, peer_id: str) -> Optional[float]:
        """Round trip to a peer, in milliseconds, or None if not connected.

        Lattica reports seconds; the rest of Loom speaks milliseconds, and the
        perf map that decides placement is in milliseconds too.
        """
        if self._lattica is None:
            return None
        try:
            seconds = self._lattica.get_peer_rtt(peer_id)
        except Exception:
            return None
        return None if seconds is None else float(seconds) * 1000.0

    def relay_rtt_ms(self) -> Optional[float]:
        """Round trip to the nearest relay, in milliseconds.

        The yardstick for whether a direct link is worth using at all. The
        relay sits on the orchestrator's machine, so the trip to it is the
        same trip a relayed activation makes on its way out — and a peer that
        costs more to reach than the orchestrator saves nothing.

        It also settles a question the Python API cannot answer otherwise:
        whether a libp2p link is really direct or a circuit through the relay.
        A circuit passes through the relay by construction, so it can never be
        cheaper than the relay itself.
        """
        best: Optional[float] = None
        for addr in self.relay_servers:
            peer_id = addr.rsplit("/p2p/", 1)[-1] if "/p2p/" in addr else ""
            rtt = self.rtt_ms(peer_id) if peer_id else None
            if rtt is not None and (best is None or rtt < best):
                best = rtt
        return best

    def connected_peers(self) -> List[str]:
        if self._lattica is None:
            return []
        try:
            return list(self._lattica.get_all_peers())
        except Exception:
            return []

    # ------------------------------------------------------------- messaging
    def warm(self, peer_id: str, addrs: Optional[List[str]] = None) -> bool:
        """Get the route to a peer ready before the first token needs it.

        Note what this does NOT do: hand libp2p an address. Bootstrap peers are
        fixed when the node is built, and a peer is found afterwards by its id
        alone — both ends share the orchestrator's rendezvous node, so the
        lookup resolves through it and the connection is then upgraded to a
        direct one by hole punching. Measured locally, two peers that knew only
        the rendezvous found each other in 0.2 s.

        The addresses the orchestrator collected are still worth having: they
        are what a future version will use to short-circuit the lookup on a LAN.
        They are accepted here and deliberately unused, so the caller's contract
        does not change when that lands.

        Returns whether the peer is reachable right now. Called ahead of time
        because the first attempt can take seconds, and the first request of a
        deployment should not be the one that pays for it.
        """
        if self._lattica is None:
            raise P2PUnavailable("the p2p node is not running")
        if peer_id in self.connected_peers():
            return True
        try:
            # Asking for a peer's addresses is what makes the stack go looking
            # for it; the answer matters less than the lookup it triggers.
            return bool(self._lattica.get_peer_addresses(peer_id))
        except Exception as exc:
            logger.debug("warming a route to %s failed: %s", peer_id, exc)
            return False

    def send(self, peer_id: str, message: dict, timeout_s: float = 0.0) -> dict:
        """One inter-stage message, straight to the peer that must handle it.

        Bounded, and tightly. This runs once per token: a send that hangs does
        not just fail, it holds up the whole pipeline while it does. Lattica's
        own default is 180 seconds — three minutes of a stalled generation to
        learn something the fallback could have handled immediately.

        The bound is generous enough for a cold start (the first send to a peer
        also resolves it through the rendezvous, about 100 ms) and short enough
        that paying it is survivable. And it is paid at most once per cooldown:
        after a failure the link is quarantined and the relay takes over.
        """
        if self._handler is None:
            raise P2PUnavailable("the p2p node is not running")
        stub = self._stub_for(peer_id)
        future = stub.stage_forward(message)
        if not hasattr(future, "result"):
            return {"ok": True}
        # Whole seconds: the binding rejects a float outright, and a mock that
        # accepts one hides that until it reaches a real peer.
        seconds = max(1, int(timeout_s or SEND_TIMEOUT_S))
        result = future.result(timeout=seconds)
        return result if isinstance(result, dict) else {"ok": True}

    def _stub_for(self, peer_id: str):
        with self._lock:
            stub = self._stubs.get(peer_id)
            if stub is None:
                stub = self._handler.get_stub(peer_id)
                self._stubs[peer_id] = stub
            return stub

    def _deliver(self, message: dict) -> dict:
        """An inbound message from a peer, on the p2p stack's own thread."""
        callback = self._on_message
        if callback is None:
            return {"ok": False, "error": "node is not serving"}
        try:
            callback(message)
        except Exception as exc:
            logger.exception("handling a peer message failed")
            return {"ok": False, "error": str(exc)}
        return {"ok": True}


def _make_handler(lattica, deliver: Callable[[dict], dict]):
    """Build the RPC service this node exposes to its peers.

    Defined here rather than at module scope because the decorators need the
    lattica package, which is optional. The service name is the class name, so
    both ends agree on `LoomStage` without configuring anything.
    """
    from lattica import ConnectionHandler, rpc_method

    class LoomStage(ConnectionHandler):
        def __init__(self, node) -> None:
            super().__init__(node)

        @rpc_method
        def stage_forward(self, message):
            return deliver(message)

    return LoomStage(lattica)
