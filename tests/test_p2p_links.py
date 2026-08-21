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


def peer(stage, node="n2", pid="12D3KooWPeer", addrs=("/ip4/10.0.0.2/tcp/47100",)):
    return Neighbour(stage_index=stage, node_id=node, peer_id=pid, addrs=list(addrs))


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
def worth_table(peer_rtt, relay_rtt, peer_relay_rtt=1.0, **kw):
    """A table whose only interesting property is what the two paths cost."""
    sent = []
    table = LinkTable(
        send_direct=lambda pid, msg: sent.append((pid, msg)),
        dial=lambda pid, addrs: None,
        rtt=lambda pid: peer_rtt,
        relay_rtt=lambda: relay_rtt,
        **kw,
    )
    neighbour = peer(1)
    neighbour.relay_rtt_ms = peer_relay_rtt
    table.set_neighbours("p#0", [neighbour])
    # The sampler measures; the send path only reads what it left behind.
    table.refresh()
    return table, sent


def test_a_link_costlier_than_the_relay_is_not_used():
    """libp2p says "connected" for a circuit through the relay too.

    That is how this went wrong on a real stand: hole punching failed, every
    activation went worker -> relay -> worker, transport per token rose from
    200 ms to 320 ms, the run halved in speed — and the admin page reported
    "100% прямо" the whole time, because a circuit is a connection like any
    other as far as the API is concerned.

    The relay sits on the orchestrator's machine, so reaching a peer must cost
    less than reaching the relay for the direct path to be saving anything at
    all. A circuit runs THROUGH the relay and so can never pass this test.
    """
    table, sent = worth_table(peer_rtt=180.0, relay_rtt=60.0, peer_relay_rtt=20.0)
    relayed = []
    assert table.send("p#0", 1, {"step": 1}, relay=relayed.append) == "relay"
    assert not sent and len(relayed) == 1
    assert table.snapshot()["not_worth"] == 1


def test_a_genuinely_direct_link_is_used():
    table, sent = worth_table(peer_rtt=12.0, relay_rtt=60.0, peer_relay_rtt=20.0)
    assert table.send("p#0", 1, {"step": 1}, relay=lambda m: None) == "direct"
    assert len(sent) == 1


def test_a_link_is_re_examined_so_hole_punching_can_win_later():
    """A circuit becomes a direct connection seconds later, or never.

    Deciding once at deployment would settle the question before the answer
    exists: DCUtR upgrades the connection after the first dial, and a verdict
    taken at that moment is a verdict about the circuit.
    """
    rtt = {"value": 180.0}
    table = LinkTable(
        send_direct=lambda pid, msg: None,
        dial=lambda pid, addrs: None,
        rtt=lambda pid: rtt["value"],
        relay_rtt=lambda: 60.0,
    )
    neighbour = peer(1)
    neighbour.relay_rtt_ms = 20.0
    table.set_neighbours("p#0", [neighbour])
    table.refresh()
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "relay"

    rtt["value"] = 15.0  # hole punching succeeded
    assert table.send("p#0", 1, {"s": 2}, relay=lambda m: None) == "relay"  # not sampled yet

    table.refresh()  # what the sampler thread does on its own
    assert table.send("p#0", 1, {"s": 3}, relay=lambda m: None) == "direct"


def test_without_measurements_the_link_is_used_as_before():
    """A missing number is not evidence of a bad link.

    Nodes with no relay configured have nothing to compare against, and there
    are no circuits for them to be fooled by either. They must behave exactly
    as they did before any of this existed.
    """
    table, sent = worth_table(peer_rtt=None, relay_rtt=None)
    assert table.send("p#0", 1, {"step": 1}, relay=lambda m: None) == "direct"
    assert len(sent) == 1

    plain = LinkTable(send_direct=lambda pid, msg: None, dial=lambda pid, a: None)
    plain.set_neighbours("p#0", [peer(1)])
    assert plain.send("p#0", 1, {"step": 1}, relay=lambda m: None) == "direct"


def test_a_measurement_that_raises_does_not_lose_the_token():
    """The transport may be mid-reconnect when we ask it anything."""

    def explode(*_):
        raise RuntimeError("no route")

    table = LinkTable(
        send_direct=lambda pid, msg: None,
        dial=lambda pid, addrs: None,
        rtt=explode,
        relay_rtt=explode,
    )
    table.set_neighbours("p#0", [peer(1)])
    assert table.send("p#0", 1, {"step": 1}, relay=lambda m: None) == "direct"

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


def test_the_near_half_of_the_relay_path_is_not_the_whole_of_it():
    """The numbers are from a real stand, and the old rule got them wrong.

    nv3 sat 8 ms from the relay; the Mac it talked to sat 90 ms from the same
    relay. Comparing the direct link (94 ms) against only nv3's own trip to
    the relay (8 ms) rejected it — while the actual relayed path cost 98 ms
    and was the slower of the two. A node cannot measure the far half, so the
    peer reports it and the orchestrator passes it on.
    """
    table, sent = worth_table(peer_rtt=94.0, relay_rtt=8.0, peer_relay_rtt=90.0)
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"
    assert len(sent) == 1


def test_two_paths_of_the_same_cost_do_not_make_the_route_flap():
    """94 ms against 98 ms, re-examined every 30 s, flipped every time.

    Each flip costs a dial, and jitter alone decided the winner. A tie has to
    resolve to "leave it as it is".
    """
    rtt = {"value": 94.0}
    table = LinkTable(
        send_direct=lambda pid, msg: None,
        dial=lambda pid, addrs: None,
        rtt=lambda pid: rtt["value"],
        relay_rtt=lambda: 8.0,
    )
    neighbour = peer(1)
    neighbour.relay_rtt_ms = 90.0
    table.set_neighbours("p#0", [neighbour])
    table.refresh()
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"

    for jitter in (99.0, 105.0, 96.0, 102.0):  # both sides of 98 ms
        rtt["value"] = jitter
        table.refresh()  # what the sampler thread does on its own
        assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct", jitter

    rtt["value"] = 160.0  # a genuine change, well past the band
    table.refresh()
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "relay"


def test_a_peer_that_never_reported_its_distance_is_trusted():
    """Guessing the unknown half as zero is exactly what rejected good links."""
    table, sent = worth_table(peer_rtt=94.0, relay_rtt=8.0, peer_relay_rtt=0.0)
    assert table.send("p#0", 1, {"s": 1}, relay=lambda m: None) == "direct"


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
