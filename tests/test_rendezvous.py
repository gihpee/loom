"""The orchestrator's rendezvous node, and the loop that makes peers findable.

A worker cannot be told "connect to node X at address Y" after the fact:
bootstrap peers are fixed when a libp2p node is built. It can only be given one
entry point at startup and then find everyone else through it. That entry point
is the orchestrator, and these tests pin down the handshake that delivers it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom.orchestrator.peers import PeerDirectory  # noqa: E402
from loom.orchestrator.rendezvous import RendezvousNode, host_of  # noqa: E402
from loom.orchestrator.rendezvous import lattica_available  # noqa: E402


# --------------------------------------------------------------- the address
@pytest.mark.parametrize(
    "address, host",
    [
        ("203.0.113.7:9000", "203.0.113.7"),
        ("orch.example.com:19090", "orch.example.com"),
        ("[2001:db8::1]:9000", "2001:db8::1"),
        ("203.0.113.7", "203.0.113.7"),
        ("", ""),
    ],
)
def test_the_rendezvous_host_comes_from_the_address_workers_already_use(address, host):
    """Known-good from wherever the workers are, unlike a local interface.

    The orchestrator could detect its own IPs, but a private one is useless to
    a worker on another continent. The address they already dial for the
    control plane is the one address proven to work from their side.
    """
    assert host_of(address) == host


def test_a_node_that_is_not_running_offers_nothing():
    """Never a half-address: a worker either gets a rendezvous or relays."""
    node = RendezvousNode(public_host="203.0.113.7")
    assert node.running is False
    assert node.multiaddrs() == []
    assert node.peer_id == ""


def test_p2p_can_be_switched_off_on_the_orchestrator(monkeypatch):
    monkeypatch.setenv("LOOM_P2P", "0")
    node = RendezvousNode(public_host="203.0.113.7")
    assert node.start() is False
    assert node.multiaddrs() == []


@pytest.mark.skipif(not lattica_available(), reason="no p2p stack installed")
def test_a_running_rendezvous_publishes_a_dialable_address(tmp_path):
    """What a worker receives must contain everything it needs to bootstrap.

    Transport, host, port and the peer id it should expect to find there — a
    multiaddr missing the /p2p/ part cannot authenticate the far end.
    """
    node = RendezvousNode(port=47280, key_dir=str(tmp_path), public_host="203.0.113.7")
    assert node.start() is True
    try:
        addrs = node.multiaddrs()
        assert len(addrs) == 2, "both TCP and QUIC should be offered"
        assert f"/ip4/203.0.113.7/tcp/47280/p2p/{node.peer_id}" in addrs
        assert f"/ip4/203.0.113.7/udp/47280/quic-v1/p2p/{node.peer_id}" in addrs
        assert node.peer_id.startswith("12D3KooW")
    finally:
        node.stop()


@pytest.mark.skipif(not lattica_available(), reason="no p2p stack installed")
def test_without_a_public_host_there_is_nothing_to_hand_out(tmp_path):
    """A rendezvous nobody can reach is worse than none: it wastes attempts."""
    node = RendezvousNode(port=47282, key_dir=str(tmp_path), public_host="")
    assert node.start() is True
    try:
        assert node.multiaddrs() == []
    finally:
        node.stop()


# ------------------------------------------------- identity over the heartbeat
def status(peer_id="12D3KooWA", direct=0, relayed=0, fallbacks=0, share=0.0, symmetric=False):
    return SimpleNamespace(
        peer_id=peer_id,
        listen_addrs=["/ip4/10.0.0.5/tcp/47100"],
        symmetric_nat=symmetric,
        direct=direct,
        relayed=relayed,
        fallbacks=fallbacks,
        direct_share=share,
    )


def test_identity_arrives_on_the_heartbeat_not_at_registration():
    """The order is forced: the worker learns the rendezvous FROM the ack.

    At the moment it registers it has no p2p node and therefore no peer id, so
    the directory is filled in from the beats that follow.
    """
    directory = PeerDirectory()
    directory.remember("n1", None, observed_host="203.0.113.7")
    assert directory.get("n1").dialable is False

    directory.remember("n1", status(), observed_host="203.0.113.7")
    assert directory.get("n1").dialable is True
    assert "/ip4/203.0.113.7/tcp/47100" in directory.get("n1").candidate_addrs()


def test_the_path_split_is_recorded_for_every_node():
    directory = PeerDirectory()
    directory.remember("n1", status())
    directory.record_transport("n1", status(direct=90, relayed=10, fallbacks=1, share=0.9))

    record = directory.get("n1")
    assert (record.direct, record.relayed, record.fallbacks) == (90, 10, 1)
    assert record.direct_share == 0.9
    assert directory.view()[0]["direct_share"] == 0.9


def test_transport_counters_for_an_unknown_node_are_ignored():
    """A beat can outrace the registration it belongs to; that is not an error."""
    directory = PeerDirectory()
    directory.record_transport("ghost", status(direct=5))
    assert directory.get("ghost") is None


def test_reachability_is_reported_separately_from_nat_type():
    """"peer" must not mean "probably fine".

    A node that is merely not behind a symmetric NAT still cannot accept an
    inbound connection unless something out there can reach it. Reporting only
    the NAT type made a pipeline look ready to go direct while every single
    message fell back to the relay.
    """
    directory = PeerDirectory()
    directory.remember("open", status(), observed_host="203.0.113.7")
    directory.remember("closed", status(), observed_host="203.0.113.8")
    directory._records["open"].visible_addrs = ["/ip4/203.0.113.7/tcp/47100"]

    view = {row["node_id"]: row for row in directory.view()}
    assert view["open"]["reachable"] is True
    assert view["closed"]["reachable"] is False
    # Both are still attempted: an unreachable node can dial OUT, and a LAN
    # pair works without AutoNAT ever confirming anything.
    assert view["open"]["dialable"] and view["closed"]["dialable"]



def test_a_node_reachable_only_through_the_relay_is_shown_as_such():
    """The one state that proves the relay is working, and it needs a name.

    A node holding a circuit-relay reservation is reachable — peers can open a
    connection to it and try to punch through from there — but it accepts
    nothing on its own. Lumping it in with a plainly dialable node hides the
    only signal that says the relay did anything at all; lumping it in with
    "unreachable" hides it just as well in the other direction.
    """
    directory = PeerDirectory()
    directory.remember("relayed", status(), observed_host="203.0.113.9")
    directory.remember("open", status(), observed_host="203.0.113.7")
    directory._records["relayed"].visible_addrs = [
        "/ip4/198.51.100.1/tcp/47200/p2p/12D3KooWRelay/p2p-circuit/p2p/12D3KooWMe"
    ]
    directory._records["open"].visible_addrs = ["/ip4/203.0.113.7/tcp/47100"]

    view = {row["node_id"]: row for row in directory.view()}
    assert view["relayed"]["reachable"] and view["relayed"]["via_relay"]
    assert view["open"]["reachable"] and not view["open"]["via_relay"]


# --------------------------------------------------------------- relay wiring
def test_relay_addresses_are_configured_not_discovered(monkeypatch):
    """A relay is infrastructure somebody stood up, not something to find.

    It is also a SEPARATE process from the rendezvous: Lattica announces
    /libp2p/circuit/relay/0.2.0/stop and never the /hop half, so the
    orchestrator's own node cannot be the relay however reachable it is.
    """
    import importlib

    import loom.orchestrator.rendezvous as rdv

    monkeypatch.setenv("LOOM_P2P_RELAY", "/ip4/1.2.3.4/tcp/47200/p2p/12D3KooWa, ")
    assert importlib.reload(rdv).RELAY_ADDRS == ["/ip4/1.2.3.4/tcp/47200/p2p/12D3KooWa"]

    monkeypatch.delenv("LOOM_P2P_RELAY", raising=False)
    assert importlib.reload(rdv).RELAY_ADDRS == []


def test_a_worker_takes_the_relay_from_the_registration_ack(monkeypatch):
    """The same message that says where the rendezvous is says where to reserve."""
    from loom_worker.main import PeerLayer

    monkeypatch.delenv("LOOM_P2P_RELAY", raising=False)
    monkeypatch.setenv("LOOM_P2P", "1")
    captured = {}

    class FakeNode:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self, on_message):
            raise RuntimeError("not starting a real node in this test")

    monkeypatch.setattr("loom_worker.main.PeerNode", FakeNode)
    monkeypatch.setattr("loom_worker.main.lattica_available", lambda: True)

    layer = PeerLayer(SimpleNamespace(deliver_direct=lambda m: None))
    layer.on_rendezvous(["/ip4/1.1.1.1/tcp/47100/p2p/12D3KooWr"], ["/ip4/2.2.2.2/tcp/47200/p2p/12D3KooWl"])

    assert captured["bootstraps"] == ["/ip4/1.1.1.1/tcp/47100/p2p/12D3KooWr"]
    assert captured["relay_servers"] == ["/ip4/2.2.2.2/tcp/47200/p2p/12D3KooWl"]
    assert layer.node is None, "a node that failed to start must not be kept"


def test_a_worker_can_be_given_a_relay_by_hand(monkeypatch):
    """For a node configured outside the orchestrator's reach."""
    from loom_worker.main import _env_relays

    monkeypatch.setenv("LOOM_P2P_RELAY", "/ip4/9.9.9.9/tcp/47200/p2p/12D3KooWz")
    assert _env_relays() == ["/ip4/9.9.9.9/tcp/47200/p2p/12D3KooWz"]
