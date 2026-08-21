"""Choosing between the direct peer path and the orchestrator's relay.

The direct path is an optimisation; the relay is the contract. Every test here
exists to pin down one half of that: a message must go directly when it can,
and must still arrive when it cannot. A pipeline that quietly relays is slower.
A pipeline that drops a token is broken, and the fallback is the whole reason
the direct path is allowed to fail at all.
"""

import sys
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
