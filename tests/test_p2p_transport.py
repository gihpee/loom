"""Two real libp2p nodes, moving real activation payloads between them.

Everything else about p2p can be tested with fakes; this cannot. Whether the
Rust core starts, whether two nodes find each other from an address the
orchestrator handed over, and whether a 16 KB tensor survives the round trip
are facts about the stack, not about our glue — and they are exactly the facts
that decide whether any of this works on a real stand.

Skipped when the p2p stack is not installed, so the suite still runs in an
image that ships without it.
"""

import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.p2p import LinkTable, Neighbour, PeerNode, lattica_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not lattica_available(), reason="the lattica p2p stack is not installed"
)

# Qwen3-8B's hidden width in float32 — one decode step between two stages.
ACTIVATION_BYTES = 4096 * 4


# Ports are never reused inside a run: a closed Lattica node does not release
# its listener instantly, and the next node silently fails to bind — which
# looks exactly like "the network is broken" from the test's point of view.
_next_port = [47320]


def free_port() -> int:
    _next_port[0] += 2
    return _next_port[0]


@pytest.fixture(scope="module")
def rendezvous(tmp_path_factory):
    """The orchestrator's node: the one address every worker is given.

    Peers find each other by id through it and never need to be told where the
    other one lives — which is what makes this work for machines behind NAT
    that cannot know their own address, let alone announce it.
    """
    port = free_port()
    node = PeerNode(port=port, key_dir=str(tmp_path_factory.mktemp("rdv")))
    identity = node.start(on_message=lambda msg: None)
    try:
        yield f"/ip4/127.0.0.1/tcp/{port}/p2p/{identity.peer_id}"
    finally:
        node.close()


@pytest.fixture
def two_nodes(tmp_path_factory, rendezvous):
    """Two peers that know the rendezvous and nothing about each other.

    Separate key directories on purpose: a Lattica node keeps more than its
    keypair there, and two nodes pointed at the same one interfere.
    """
    received = []
    port_a, port_b = free_port(), free_port()
    a = PeerNode(port=port_a, key_dir=str(tmp_path_factory.mktemp("a")), bootstraps=[rendezvous])
    b = PeerNode(port=port_b, key_dir=str(tmp_path_factory.mktemp("b")), bootstraps=[rendezvous])
    try:
        id_a = a.start(on_message=lambda msg: received.append(msg))
        id_b = b.start(on_message=lambda msg: received.append(msg))
        yield a, b, id_a, id_b, received
    finally:
        a.close()
        b.close()


def wait_for(predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_a_node_has_a_stable_identity_and_reachable_addresses(two_nodes):
    a, _b, id_a, id_b, _ = two_nodes
    assert id_a.peer_id.startswith("12D3KooW"), id_a.peer_id
    assert id_a.peer_id != id_b.peer_id, "two nodes must not share an identity"
    assert any("/tcp/" in addr for addr in id_a.listen_addrs)
    assert any("quic-v1" in addr for addr in id_a.listen_addrs)


def test_an_identity_survives_a_restart(tmp_path, rendezvous):
    """The peer id is how the orchestrator and the neighbours name this node.

    Regenerating it on every boot would make a restarted worker look like a
    stranger to a pipeline that is still pointing at the old one.
    """
    key_dir = str(tmp_path / "stable")
    first = PeerNode(port=free_port(), key_dir=key_dir, bootstraps=[rendezvous])
    peer_id = first.start(on_message=lambda msg: None).peer_id
    first.close()

    second = PeerNode(port=free_port(), key_dir=key_dir, bootstraps=[rendezvous])
    try:
        assert second.start(on_message=lambda msg: None).peer_id == peer_id
    finally:
        second.close()


def test_activations_reach_the_peer_without_an_orchestrator(two_nodes):
    """The whole point: one wide-area crossing instead of two.

    Note there is no wait for a connection first. `connected_peers()` lists
    links that are already established, and bootstrapping to the rendezvous
    does not by itself connect two workers to each other — the peer is resolved
    when a message is actually sent to it. Measured locally, that first send
    costs about 100 ms and every one after it is sub-millisecond.
    """
    a, b, id_a, _id_b, received = two_nodes

    payload = {
        "kind": "activations",
        "target_stage": 1,
        "step": 3,
        "pipeline_id": "m#0",
        "model_id": "m",
        "tensor_b64": b"\x00" * ACTIVATION_BYTES,
    }
    b.send(id_a.peer_id, payload)

    assert wait_for(lambda: received), "the peer never got the message"
    got = received[0]
    assert got["step"] == 3 and got["kind"] == "activations"
    assert len(got["tensor_b64"]) == ACTIVATION_BYTES, "the tensor did not survive"


def test_the_link_table_drives_a_real_peer(two_nodes):
    """The same object the agent uses, over a real connection rather than a stub."""
    a, b, id_a, _id_b, received = two_nodes
    links = LinkTable(send_direct=b.send, dial=b.warm)
    links.set_neighbours(
        "p#0",
        [Neighbour(stage_index=1, node_id="node-a", peer_id=id_a.peer_id,
                   addrs=list(id_a.listen_addrs))]
    )
    relayed = []
    path = links.send("p#0", 1, {"kind": "activations", "step": 1}, relay=relayed.append)
    assert path == "direct" and not relayed
    assert wait_for(lambda: received)


def test_the_measured_rtt_is_available_for_placement(two_nodes):
    """The scheduler splits layers by node speed; the network belongs there too.

    A connection has to exist before there is a round trip to report, so this
    sends first — the same order a real pipeline uses.
    """
    a, b, id_a, _id_b, _ = two_nodes
    b.send(id_a.peer_id, {"kind": "activations", "step": 0})
    assert wait_for(lambda: b.rtt_ms(id_a.peer_id) is not None)

    rtt = b.rtt_ms(id_a.peer_id)
    assert 0.0 <= rtt < 1000.0, f"implausible loopback RTT: {rtt} ms"


def test_an_unreachable_peer_fails_instead_of_hanging(two_nodes):
    """A dead route must surface as an error the link table can fall back from."""
    _a, b, _id_a, _id_b, _ = two_nodes
    with pytest.raises(Exception):
        b.send("12D3KooWNobodyHomeAtThisAddressAtAllEver", {"kind": "activations"})


# ------------------------------------------------- the whole loop, end to end
def test_two_workers_meet_through_the_orchestrator_and_then_bypass_it(tmp_path_factory):
    """The complete Phase-2 handshake, with nothing faked but the placement.

    An orchestrator brings up a rendezvous; two agents are handed its address
    exactly as a registration ack would; each starts its peer layer, reports an
    identity, and the orchestrator builds a directory from those reports. The
    directory is then handed to one agent as a LoadShard topology would, and a
    stage message goes to the other WITHOUT the orchestrator carrying it.
    """
    from loom.orchestrator.peers import PeerDirectory
    from loom.orchestrator.rendezvous import RendezvousNode
    from loom_worker.main import PeerLayer
    from loom_worker.p2p import neighbours_from_topology

    orchestrator = RendezvousNode(
        port=free_port(), key_dir=str(tmp_path_factory.mktemp("orch")), public_host="127.0.0.1"
    )
    assert orchestrator.start(), "the rendezvous did not come up"
    delivered = {"a": [], "b": []}

    def agent(name):
        # Stands in for DataPlaneClient: the only thing PeerLayer asks of it is
        # somewhere to put a message that arrived from a peer.
        return SimpleNamespace(deliver_direct=lambda msg: delivered[name].append(msg))

    peers_a = PeerLayer(
        agent("a"), port=free_port(), key_dir=str(tmp_path_factory.mktemp("wa"))
    )
    peers_b = PeerLayer(
        agent("b"), port=free_port(), key_dir=str(tmp_path_factory.mktemp("wb"))
    )
    try:
        for layer in (peers_a, peers_b):
            layer.on_rendezvous(orchestrator.multiaddrs())
            assert layer.node is not None, "the agent did not join the rendezvous"

        # The orchestrator learns both identities from their heartbeats.
        directory = PeerDirectory()
        for node_id, layer in (("node-a", peers_a), ("node-b", peers_b)):
            directory.remember(node_id, layer.status(), observed_host="127.0.0.1")
        rows = directory.routes_for([(0, "node-a"), (1, "node-b")])
        assert all(row[2] for row in rows), f"a node was not dialable: {rows}"

        # ...and hands node-a the directory, as a LoadShard topology would.
        topology = SimpleNamespace(
            peers=[
                SimpleNamespace(stage_index=i, node_id=n, peer_id=p, addrs=a)
                for i, n, p, a in rows
            ]
        )
        peers_a.links.set_neighbours(
        "p#0",
            neighbours_from_topology(topology, self_node_id="node-a")
        )

        relayed = []
        path = peers_a.links.send(
            "p#0",
            1,
            {"kind": "activations", "step": 1, "tensor_b64": b"\x00" * ACTIVATION_BYTES},
            relay=relayed.append,
        )
        assert path == "direct", "the message went through the orchestrator"
        assert not relayed
        assert wait_for(lambda: delivered["b"]), "stage 1 never received the activations"
        assert len(delivered["b"][0]["tensor_b64"]) == ACTIVATION_BYTES

        # And the split is reported back, so a fallback would be visible.
        assert peers_a.status().direct == 1
        assert peers_a.status().relayed == 0
    finally:
        for layer in (peers_a, peers_b):
            if layer.node is not None:
                layer.node.close()
        orchestrator.stop()
