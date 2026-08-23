"""Choosing between the direct peer path and the orchestrator's relay.

The direct path is an optimisation; the relay is the contract. Every test here
exists to pin down one half of that: a message must go directly when it can,
and must still arrive when it cannot. A pipeline that quietly relays is slower.
A pipeline that drops a token is broken, and the fallback is the whole reason
the direct path is allowed to fail at all.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.p2p import (  # noqa: E402
    LinkTable,
    Neighbour,
    local_candidate_addrs,
    neighbours_from_topology,
)


def table(**kwargs):
    """A link table wired to recorders instead of a real p2p node."""
    sent, dialled = [], []
    links = LinkTable(
        send_direct=kwargs.pop("send_direct", lambda pid, msg: sent.append((pid, msg))),
        dial=lambda pid, addrs: dialled.append((pid, addrs)),
        **kwargs,
    )
    return links, sent, dialled


def peer(stage, node="n2", pid="12D3KooWPeer", addrs=("/ip4/10.0.0.2/tcp/47100",),
         reachable=True):
    """A neighbour something can actually dial, unless a test says otherwise.

    Reachability is what decides whether the direct path is used at all: with
    neither end able to accept a connection, libp2p can only build a circuit
    through the relay, which is the relay path by another name.
    """
    return Neighbour(
        stage_index=stage,
        node_id=node,
        peer_id=pid,
        addrs=list(addrs),
        reachable=reachable,
    )


# ------------------------------------------------------------- the happy path
def test_a_known_peer_gets_the_message_directly():
    links, sent, _ = table()
    links.set_neighbours("p#0", [peer(1)])
    relayed = []

    path = links.send("p#0", 1, {"kind": "activations", "step": 7}, relay=relayed.append)

    assert path == "direct"
    assert sent and sent[0][0] == "12D3KooWPeer"
    assert not relayed, "the orchestrator was involved in a direct hop"


def test_neighbours_are_dialled_before_the_first_token():
    """Hole punching takes seconds; the first request must not pay for it."""
    links, _, dialled = table()
    links.set_neighbours("p#0", [peer(1), peer(2, node="n3", pid="12D3KooWThird")])
    assert [pid for pid, _ in dialled] == ["12D3KooWPeer", "12D3KooWThird"]


# ------------------------------------------------------------- the fallbacks
def test_a_peer_with_no_identity_is_relayed():
    """An old worker, a CPU node, an image without the p2p stack."""
    links, sent, _ = table()
    links.set_neighbours("p#0", [Neighbour(stage_index=1, node_id="n2")])
    relayed = []

    assert links.send("p#0", 1, {"step": 1}, relay=relayed.append) == "relay"
    assert not sent and len(relayed) == 1


def test_an_unknown_stage_is_relayed():
    links, sent, _ = table()
    links.set_neighbours("p#0", [peer(1)])
    relayed = []
    assert links.send("p#0", 5, {"step": 1}, relay=relayed.append) == "relay"
    assert not sent and len(relayed) == 1


def test_a_failing_direct_link_still_delivers_the_message():
    """The decisive property: a broken route costs speed, never a token."""

    def explode(peer_id, message):
        raise ConnectionError("hole punching lost the race")

    links, _, _ = table(send_direct=explode)
    links.set_neighbours("p#0", [peer(1)])
    relayed = []

    assert links.send("p#0", 1, {"step": 1}, relay=relayed.append) == "relay"
    assert relayed == [{"step": 1}], "the token vanished when the link failed"


def test_a_failed_link_is_not_retried_on_every_token():
    """Retrying a dead route would add its timeout to each token."""
    attempts = []

    def explode(peer_id, message):
        attempts.append(message)
        raise ConnectionError("down")

    links, _, _ = table(send_direct=explode, cooldown_s=60.0)
    links.set_neighbours("p#0", [peer(1)])
    relayed = []
    for step in range(5):
        links.send("p#0", 1, {"step": step}, relay=relayed.append)

    assert len(attempts) == 1, "the dead link was dialled again"
    assert len(relayed) == 5, "every token must still arrive"


def test_the_cooldown_lets_a_recovered_link_back_in():
    """A network hiccup must not exile a peer for the rest of the run."""
    state = {"fail": True}

    def flaky(peer_id, message):
        if state["fail"]:
            raise ConnectionError("down")

    links, _, _ = table(send_direct=flaky, cooldown_s=0.0)
    links.set_neighbours("p#0", [peer(1)])
    relayed = []

    assert links.send("p#0", 1, {"step": 0}, relay=relayed.append) == "relay"
    state["fail"] = False
    assert links.send("p#0", 1, {"step": 1}, relay=relayed.append) == "direct"


def test_re_deploying_replaces_the_directory_outright():
    """Stages move. A stale route sends activations to a node that has no KV."""
    links, sent, _ = table()
    links.set_neighbours("p#0", [peer(1, node="old", pid="12D3KooWOld")])
    links.set_neighbours("p#0", [peer(1, node="new", pid="12D3KooWNew")])
    links.send("p#0", 1, {"step": 1}, relay=lambda m: None)
    assert sent[0][0] == "12D3KooWNew"


# ------------------------------------------------------------------ reporting
def test_the_split_between_paths_is_reported():
    """Otherwise there is no way to know whether p2p is doing anything."""
    calls = {"n": 0}

    def half_broken(peer_id, message):
        calls["n"] += 1
        if calls["n"] > 2:
            raise ConnectionError("down")

    links, _, _ = table(send_direct=half_broken, cooldown_s=60.0)
    links.set_neighbours("p#0", [peer(1)])
    for step in range(5):
        links.send("p#0", 1, {"step": step}, relay=lambda m: None)

    snap = links.snapshot()
    assert snap["direct"] == 2 and snap["relay"] == 3
    assert snap["fallbacks"] == 1
    assert snap["direct_share"] == 0.4
    assert snap["neighbours"][0]["node_id"] == "n2"


# ------------------------------------------------------- reading the topology
def test_the_directory_comes_out_of_the_topology_message():
    topology = SimpleNamespace(
        peers=[
            SimpleNamespace(stage_index=0, node_id="me", peer_id="a", addrs=["/x"]),
            SimpleNamespace(stage_index=1, node_id="n2", peer_id="b", addrs=["/y"]),
        ]
    )
    peers = neighbours_from_topology(topology, self_node_id="me")
    assert [p.node_id for p in peers] == ["n2"], "a stage must not dial itself"
    assert peers[0].peer_id == "b"


def test_a_topology_without_peers_is_not_an_error():
    """An orchestrator that predates the directory must still be usable."""
    assert neighbours_from_topology(SimpleNamespace(), self_node_id="me") == []
    assert neighbours_from_topology(SimpleNamespace(peers=[]), self_node_id="me") == []


# ----------------------------------------------------------------- addressing
def test_a_node_offers_both_transports_for_every_interface():
    addrs = local_candidate_addrs(47100)
    assert addrs, "a node with no candidate address can never be dialled"
    assert any(a.endswith("/tcp/47100") for a in addrs)
    assert any(a.endswith("/udp/47100/quic-v1") for a in addrs)


# ------------------------------------------------- the agent's own decision
def layer(monkeypatch, **env):
    """A PeerLayer with a stub data plane and a controlled environment."""
    from loom_worker.main import PeerLayer

    monkeypatch.delenv("LOOM_P2P_RENDEZVOUS", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return PeerLayer(SimpleNamespace(deliver_direct=lambda m: None))


def test_everything_relays_until_the_rendezvous_is_known(monkeypatch):
    """The node cannot start earlier: the address arrives in the register ack.

    So the table exists from process start and relays, and the p2p node is
    attached to it later. Everything downstream holds the same object and is
    never re-wired.
    """
    peers = layer(monkeypatch)
    relayed = []
    peers.links.set_neighbours("p#0", [peer(1)])

    assert peers.links.send("p#0", 1, {"step": 1}, relay=relayed.append) == "relay"
    assert len(relayed) == 1
    assert peers.node is None


def test_an_orchestrator_without_a_rendezvous_leaves_us_relaying(monkeypatch):
    """Nothing to bootstrap to means nobody can find this node."""
    peers = layer(monkeypatch)
    peers.on_rendezvous([])
    assert peers.node is None
    peers.on_rendezvous(["  "])
    assert peers.node is None


def test_p2p_can_be_turned_off_outright(monkeypatch):
    peers = layer(monkeypatch, LOOM_P2P="0")
    peers.on_rendezvous(["/ip4/1.2.3.4/tcp/47100/p2p/12D3KooWx"])
    assert peers.node is None


def test_the_heartbeat_reports_a_relay_only_node_honestly(monkeypatch):
    """An empty peer id is how the orchestrator learns to keep relaying."""
    peers = layer(monkeypatch)
    peers.links.set_neighbours("p#0", [peer(1)])
    peers.links.send("p#0", 1, {"step": 1}, relay=lambda m: None)

    status = peers.status()
    assert status.peer_id == ""
    assert status.relayed == 1 and status.direct == 0



# ------------------------------------------------- a link worth using at all
def one_link(peer_reachable, self_reachable):
    """A table with a single neighbour and a stated topology."""
    sent = []
    table = LinkTable(
        send_direct=lambda pid, msg: sent.append((pid, msg)),
        dial=lambda pid, addrs: None,
    )
    table.set_self_reachable(self_reachable)
    table.set_neighbours("p#0", [peer(1, reachable=peer_reachable)])
    return table, sent


def test_with_neither_end_reachable_there_is_no_direct_path_to_use():
    """The failure this rule ends, stated as plainly as it can be.

    Two workers that both sit behind NAT cannot be connected by libp2p except
    through a circuit — and Loom runs its relay on the orchestrator's own
    machine, so that circuit is the same two hops as the tunnel, over the same
    wire, with the tunnel's advantages removed. Using it is strictly worse,
    and on a real stand it was: 6 tok/s over the circuit against 8 through the
    orchestrator, same hardware, same minute.

    libp2p reports such a circuit as an ordinary connection, which is how it
    got counted as "direct" for so long.
    """
    table, sent = one_link(peer_reachable=False, self_reachable=False)
    relayed = []
    assert table.send("p#0", 1, {"s": 1}, relay=relayed.append) == "relay"
    assert not sent and len(relayed) == 1


def test_a_reachable_peer_is_dialled():
    """One hop instead of two, and no measurement needed to know it."""
    table, sent = one_link(peer_reachable=True, self_reachable=False)
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"
    assert len(sent) == 1


def test_being_reachable_ourselves_is_enough():
    """A connection has two ends and either one may be the one that opens it.

    If peers can dial US, the connection they open is a real one, and messages
    in both directions travel over it. Requiring the far end to be reachable
    too would throw away half the cases that work.
    """
    table, sent = one_link(peer_reachable=False, self_reachable=True)
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"
    assert len(sent) == 1


def test_a_peer_reachable_only_through_the_relay_does_not_count():
    """"Reachable through the relay" is the relay path wearing a disguise."""
    from loom.orchestrator.peers import PeerRecord

    circuit_only = PeerRecord(
        node_id="nv3",
        peer_id="12D3KooWNv3",
        visible_addrs=[
            "/ip4/198.51.100.1/tcp/47200/p2p/12D3KooWRelay/p2p-circuit/p2p/12D3KooWNv3"
        ],
    )
    open_port = PeerRecord(
        node_id="open",
        peer_id="12D3KooWOpen",
        visible_addrs=["/ip4/203.0.113.7/tcp/47100"],
    )
    nothing = PeerRecord(node_id="dark", peer_id="12D3KooWDark")

    assert not circuit_only.reachable
    assert open_port.reachable
    assert not nothing.reachable


# ----------------------------------------- several pipelines on one node
def test_two_pipelines_on_one_node_do_not_share_routes():
    """"Stage 1" is a different machine in every pipeline.

    A node can host stages of several models at once. Keying routes by stage
    index alone would send one model's activations to the other model's node —
    a failure that produces plausible-looking wrong answers instead of an
    error, because the receiving stage happily runs its layers on whatever
    tensor arrives.
    """
    links, sent, _ = table()
    links.set_neighbours("alpha#0", [peer(1, node="a2", pid="12D3KooWAlpha")])
    links.set_neighbours("beta#0", [peer(1, node="b2", pid="12D3KooWBeta")])

    links.send("alpha#0", 1, {"step": 1}, relay=lambda m: None)
    links.send("beta#0", 1, {"step": 1}, relay=lambda m: None)

    assert [pid for pid, _ in sent] == ["12D3KooWAlpha", "12D3KooWBeta"]


def test_redeploying_one_model_leaves_the_other_alone():
    """The scenario: reshuffle model A's nodes while model B keeps serving."""
    links, sent, _ = table()
    links.set_neighbours("alpha#0", [peer(1, node="a2", pid="12D3KooWAlpha")])
    links.set_neighbours("beta#0", [peer(1, node="b2", pid="12D3KooWBeta")])

    # Model A is re-placed onto a different node.
    links.set_neighbours("alpha#0", [peer(1, node="a9", pid="12D3KooWNewAlpha")])

    links.send("alpha#0", 1, {"step": 1}, relay=lambda m: None)
    links.send("beta#0", 1, {"step": 1}, relay=lambda m: None)
    assert [pid for pid, _ in sent] == ["12D3KooWNewAlpha", "12D3KooWBeta"]


def test_a_pipeline_that_was_torn_down_stops_being_routed():
    """After an undeploy the stale route must not resolve to the old node."""
    links, sent, _ = table()
    links.set_neighbours("alpha#0", [peer(1, node="a2", pid="12D3KooWAlpha")])
    links.set_neighbours("alpha#0", [])  # torn down

    relayed = []
    assert links.send("alpha#0", 1, {"step": 1}, relay=relayed.append) == "relay"
    assert not sent and len(relayed) == 1


def test_the_snapshot_names_the_pipeline_each_route_belongs_to():
    links, _, _ = table()
    links.set_neighbours("alpha#0", [peer(1, node="a2", pid="12D3KooWAlpha")])
    links.set_neighbours("beta#0", [peer(2, node="b3", pid="12D3KooWBeta")])
    rows = links.snapshot()["neighbours"]
    assert {(r["pipeline_id"], r["stage_index"]) for r in rows} == {
        ("alpha#0", 1),
        ("beta#0", 2),
    }


def test_a_direct_send_is_bounded(monkeypatch):
    """A hung peer must cost one bounded wait, not a stalled generation.

    This runs once per token. Lattica's own default is 180 seconds; inheriting
    it would mean a single unreachable neighbour freezing the pipeline for
    three minutes to learn what the relay could have done immediately.
    """
    import loom_worker.p2p.peer as peer_module

    captured = {}

    class Future:
        def result(self, timeout=None):
            # Lattica's binding takes whole seconds and rejects a float. A mock
            # that quietly accepts one lets the mistake through to a real peer,
            # which is exactly what happened once.
            assert isinstance(timeout, int), f"timeout must be an int, got {timeout!r}"
            captured["timeout"] = timeout
            return {"ok": True}

    node = peer_module.PeerNode.__new__(peer_module.PeerNode)
    node._handler = object()
    node._stub_for = lambda pid: SimpleNamespace(stage_forward=lambda msg: Future())

    monkeypatch.setattr(peer_module, "SEND_TIMEOUT_S", 5.0)
    node.send("12D3KooWPeer", {"step": 1})
    assert captured["timeout"] == 5, "the send inherited Lattica's 180s default"

    node.send("12D3KooWPeer", {"step": 2}, 30)
    assert captured["timeout"] == 30, "an explicit bound must win"

    # Sub-second bounds round up rather than becoming zero, which would mean
    # "no timeout at all" to some bindings.
    node.send("12D3KooWPeer", {"step": 3}, 0.25)
    assert captured["timeout"] == 1


def test_the_identity_directory_follows_home_not_a_container_path(monkeypatch):
    """A hardcoded /root is how a perfectly good Mac ends up relaying.

    The worker image runs as root, so HOME is /root and the path is unchanged
    there. A native agent on macOS gets the user's cache instead of a directory
    that does not exist and cannot be created.
    """
    import importlib

    import loom_worker.p2p.peer as peer_module

    monkeypatch.delenv("LOOM_P2P_KEY_DIR", raising=False)
    monkeypatch.setenv("HOME", "/root")
    assert importlib.reload(peer_module).DEFAULT_KEY_DIR == "/root/.cache/loom/p2p"

    monkeypatch.setenv("HOME", "/Users/someone")
    assert (
        importlib.reload(peer_module).DEFAULT_KEY_DIR
        == "/Users/someone/.cache/loom/p2p"
    )
    monkeypatch.delenv("HOME", raising=False)
    importlib.reload(peer_module)


def test_an_unwritable_identity_directory_does_not_cost_the_direct_path(tmp_path):
    """Losing the keypair is cheap; losing p2p is a round trip per token.

    The orchestrator relearns every peer id from the heartbeats, so an identity
    that changes on restart costs almost nothing — while falling back to the
    relay costs a wide-area crossing on every single token.
    """
    from loom_worker.p2p.peer import PeerNode

    node = PeerNode(key_dir="/proc/nonexistent/loom")
    resolved = node._usable_key_dir()
    assert resolved != "/proc/nonexistent/loom"
    assert Path(resolved).is_dir(), "the fallback must be usable, not just different"


def test_a_writable_directory_is_used_as_given(tmp_path):
    from loom_worker.p2p.peer import PeerNode

    target = tmp_path / "keys"
    assert PeerNode(key_dir=str(target))._usable_key_dir() == str(target)
    assert target.is_dir()
    assert not (target / ".writable").exists(), "the probe file must be cleaned up"


# --------------------------------------------------------- a port already taken
def test_a_busy_port_does_not_cost_the_node_its_direct_path():
    """One machine, several workers, and --network host: they share the ports.

    The failure used to arrive as a Rust error nested eight levels deep —
    `Transport(Left(Left(Left(Os { code: 98, kind: AddrInUse }))))` — and the
    node gave up on the direct path entirely because of a number it could
    simply have picked differently. The port is not meaningful: peers are
    found by id, and the address a peer dials is announced, not assumed.
    """
    import socket
    import tempfile

    from loom_worker.p2p.peer import PeerNode, _address_in_use

    assert _address_in_use(
        RuntimeError('Transport(Left(Left(Left(Os { code: 98, kind: AddrInUse }))))')
    )
    assert not _address_in_use(RuntimeError("something else entirely"))

    hog = socket.socket()
    hog.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    hog.bind(("127.0.0.1", 0))
    taken = hog.getsockname()[1]
    hog.listen(1)
    try:
        built = []

        class Probe(PeerNode):
            def _build_on(self, port):
                built.append(port)
                if port == taken:
                    raise RuntimeError("Os { code: 98, kind: AddrInUse }")
                return object()

        node = Probe(port=taken, key_dir=tempfile.mkdtemp())
        assert node._build() is not None
        assert built == [taken, taken + 1]
        assert node.port == taken + 1  # and it reports where it actually is
    finally:
        hog.close()
# ------------------------------------------- nothing may wait on the p2p stack
def test_neither_the_token_path_nor_the_heartbeat_enters_the_p2p_runtime():
    """The failure this prevents took a whole run down.

    Measuring means calling into the p2p runtime. Both the send path and the
    heartbeat used to do it, so when the runtime got busy — which it does, at
    exactly the moment the link is in trouble — sends stalled at 100 ms of
    real latency and one agent went four minutes without a heartbeat while its
    GPU was serving normally.

    The sampler measures. Everything else reads what it left behind.
    """
    calls = {"rtt": 0, "relay": 0}

    def measured_rtt(_peer_id):
        calls["rtt"] += 1
        return 40.0

    def measured_relay():
        calls["relay"] += 1
        return 30.0

    table = LinkTable(
        send_direct=lambda pid, msg: None,
        dial=lambda pid, addrs: None,
        rtt=measured_rtt,
        relay_rtt=measured_relay,
    )
    neighbour = peer(1)
    neighbour.relay_rtt_ms = 30.0
    table.set_neighbours("p#0", [neighbour])
    calls["rtt"] = calls["relay"] = 0

    for step in range(20):
        table.send("p#0", 1, {"s": step}, relay=lambda m: None)
    table.snapshot()          # what every heartbeat asks for
    table.direct_available("p#0", 1)
    assert calls == {"rtt": 0, "relay": 0}, "the hot path measured something"

    table.refresh()           # the sampler, and only the sampler
    assert calls["rtt"] == 1 and calls["relay"] == 1


def test_an_inbound_message_is_accepted_without_waiting_for_the_stage():
    """The handler runs on the p2p runtime's thread. It must not linger there.

    Delivery is a blocking HTTP POST to the local stage with a 60 s timeout,
    and it used to happen inline — occupying a runtime thread while the sender
    waited for a reply that could not come until it finished. Two nodes doing
    that to each other at every token wedged both.
    """
    import threading
    import time

    from loom_worker.p2p.peer import _make_handler

    started = threading.Event()
    release = threading.Event()
    delivered = []

    def slow_deliver(message):
        started.set()
        release.wait(timeout=5)
        delivered.append(message)

    class FakeLattica:
        """Enough of the stack for the handler to register itself against."""

        def register_service(self, service):
            self.service = service

    handler = _make_handler(FakeLattica(), slow_deliver)

    began = time.monotonic()
    assert handler.stage_forward({"step": 1}) == {"ok": True}
    took = time.monotonic() - began
    assert took < 0.5, f"the handler blocked for {took:.2f}s waiting on delivery"
    assert started.wait(timeout=5), "the message was never picked up"
    assert not delivered, "it should still be in flight while we hold it"

    release.set()
    for _ in range(50):
        if delivered:
            break
        time.sleep(0.02)
    assert delivered == [{"step": 1}], "the message was accepted but never delivered"


# ------------------------------------- handing over versus waiting to be heard
class SlowAck:
    """A future whose acknowledgement takes as long as a network round trip."""

    def __init__(self, delay=0.3, fail=False):
        self.delay, self.fail = delay, fail

    def result(self, timeout=None):
        import time

        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("operation timeout")
        return {"ok": True}


def test_the_token_path_does_not_wait_to_be_acknowledged():
    """Waiting for the reply is what made the direct path the slower one.

    An activation is one-way: the next stage needs it, and nothing in this
    step depends on hearing back. Waiting anyway cost a full round trip, while
    the relay — a queue — cost half of one. Measured across regions: 219 ms
    per token direct against 116 ms relayed, over a link whose actual round
    trip was 100 ms. The direct path was beating itself.
    """
    import time

    table = LinkTable(
        send_direct=lambda pid, msg: SlowAck(delay=0.3),
        dial=lambda pid, addrs: None,
    )
    table.set_neighbours("p#0", [peer(1)])

    began = time.monotonic()
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"
    took = time.monotonic() - began
    assert took < 0.1, f"the send waited {took:.2f}s for an acknowledgement"


def test_a_message_that_was_never_acknowledged_is_relayed_after_all():
    """Not waiting must not mean not noticing.

    The guarantee that survives every rewrite: a direct path may be slower, it
    may be unavailable, it may fail — but it may not lose a token. A lost one
    ends the request.
    """
    relayed = []
    table = LinkTable(
        send_direct=lambda pid, msg: SlowAck(delay=0.0, fail=True),
        dial=lambda pid, addrs: None,
        timeout_s=1.0,
    )
    table.set_neighbours("p#0", [peer(1)])
    assert table.send("p#0", 1, {"s": 7}, relay=relayed.append) == "direct"

    for _ in range(100):
        if relayed:
            break
        time.sleep(0.02)
    assert relayed == [{"s": 7}], "a message nobody acknowledged was simply lost"
    # ...and the link is quarantined, so the next token does not repeat it.
    assert not table.direct_available("p#0", 1)
    assert table.snapshot()["fallbacks"] == 1


def test_the_counters_do_not_credit_a_send_that_failed():
    """A message counted as direct and then relayed is one message, not two."""
    table = LinkTable(
        send_direct=lambda pid, msg: SlowAck(delay=0.0, fail=True),
        dial=lambda pid, addrs: None,
        timeout_s=1.0,
    )
    table.set_neighbours("p#0", [peer(1)])
    table.send("p#0", 1, {"s": 1}, relay=lambda m: None)
    for _ in range(100):
        if table.snapshot()["relay"]:
            break
        time.sleep(0.02)
    stats = table.snapshot()
    assert stats["direct"] == 0 and stats["relay"] == 1


# ------------------------------------------------ the number two nodes exchange
def test_a_node_reports_its_own_distance_to_the_relay_and_not_the_whole_path():
    """A feedback loop that made every circuit look worth using.

    Each node measures its own trip to the relay and reports it; the
    orchestrator hands that number to its neighbours as the half they cannot
    measure themselves. Reporting the SUM instead feeds each node's total back
    to the other as an input, and the two inflate each other on every
    heartbeat:

        round 1:  nv3 says 8,   mac says 90
        round 2:  nv3 says 98,  mac says 98      (each added the other's)
        round 3:  nv3 says 188, mac says 106
        ...

    Within minutes "relayed" looks arbitrarily expensive, every link wins the
    comparison, and both nodes route their activations down a circuit through
    the relay — which is the slowest path available. Observed on a stand as
    100% direct on both sides at 2.4 tok/s.
    """
    table = LinkTable(
        send_direct=lambda pid, msg: None,
        dial=lambda pid, addrs: None,
        rtt=lambda pid: 94.0,
        relay_rtt=lambda: 8.0,      # what THIS node measures
    )
    neighbour = peer(1)
    neighbour.relay_rtt_ms = 90.0   # what the neighbour reported
    table.set_neighbours("p#0", [neighbour])

    for _ in range(5):              # several heartbeats worth of sampling
        table.refresh()
        reported = table.snapshot()["relay_rtt_ms"]
        assert reported == 8.0, f"reported {reported}, not its own 8 ms"


# --------------------------------------------------------------------- IPv6
def test_a_node_with_ipv6_offers_it_alongside_ipv4(monkeypatch):
    """The address family where none of the NAT machinery is needed.

    An IPv6 host knows its own globally routable address: nothing translates
    it, so there is no mapping to guess and no hole to punch. Two nodes that
    both have IPv6 reach each other directly, and the routing rule takes them
    without a special case — a global IPv6 address is simply a non-circuit
    address, which is all it asks about.
    """
    from loom_worker.p2p import peer as peer_mod

    monkeypatch.setattr(peer_mod, "_local_ips", lambda: ["10.0.0.4"])
    monkeypatch.setattr(peer_mod, "_local_ipv6", lambda: ["2001:db8::4"])

    addrs = peer_mod.local_candidate_addrs(47100)
    assert "/ip4/10.0.0.4/tcp/47100" in addrs
    assert "/ip6/2001:db8::4/tcp/47100" in addrs
    assert "/ip6/2001:db8::4/udp/47100/quic-v1" in addrs


def test_a_host_without_ipv6_does_not_try_to_listen_on_it(monkeypatch):
    """Binding :: on a kernel with IPv6 off fails, and takes the node with it.

    Losing the direct path over an address family the machine was never going
    to use would be a poor trade, so the listener is conditional.
    """
    from loom_worker.p2p import peer as peer_mod

    monkeypatch.setattr(peer_mod, "ipv6_supported", lambda: False)
    assert peer_mod._listen_addrs(47100) == [
        "/ip4/0.0.0.0/tcp/47100",
        "/ip4/0.0.0.0/udp/47100/quic-v1",
    ]

    monkeypatch.setattr(peer_mod, "ipv6_supported", lambda: True)
    assert "/ip6/::/tcp/47100" in peer_mod._listen_addrs(47100)
    assert "/ip6/::/udp/47100/quic-v1" in peer_mod._listen_addrs(47100)


def test_link_local_ipv6_is_never_advertised(monkeypatch):
    """fe80:: means nothing without the interface it belongs to.

    A multiaddr cannot carry a zone id, so a link-local address handed to a
    neighbour is an address it can only fail to dial.
    """
    from loom_worker.p2p import peer as peer_mod

    class FakeSocket:
        def __init__(self, *a, **kw):
            pass

        def connect(self, *a):
            raise OSError("no route")

        def close(self):
            pass

    monkeypatch.setattr(peer_mod.socket, "socket", FakeSocket)
    monkeypatch.setattr(
        peer_mod.socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (None, None, None, None, ("fe80::1%en0", 0, 0, 4)),
            (None, None, None, None, ("::1", 0, 0, 0)),
            (None, None, None, None, ("2001:db8::9", 0, 0, 0)),
        ],
    )
    assert peer_mod._local_ipv6() == ["2001:db8::9"]
