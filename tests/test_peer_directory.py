"""The orchestrator's half of NAT traversal: who to dial, and at what address.

The orchestrator never carries a direct message. It contributes the three facts
neither worker can produce alone — who the neighbours are, what a node's public
address is, and whether a direct attempt is worth making at all — and then gets
out of the way.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loom.orchestrator.peers import PeerDirectory, PeerRecord  # noqa: E402


def identity(peer_id="12D3KooWA", addrs=("/ip4/10.0.0.5/tcp/47100",), symmetric=False):
    return SimpleNamespace(peer_id=peer_id, listen_addrs=list(addrs), symmetric_nat=symmetric)


# ------------------------------------------------------------ what is recorded
def test_the_observed_address_becomes_a_dialable_candidate():
    """The one fact a NATed node cannot learn about itself.

    A machine behind NAT sees only its private address. The orchestrator sees
    the address the control connection arrives from — the node's public one —
    and that is exactly what a STUN server exists to provide.
    """
    directory = PeerDirectory()
    directory.remember("n1", identity(), observed_host="203.0.113.7")

    addrs = directory.get("n1").candidate_addrs()
    assert "/ip4/203.0.113.7/tcp/47100" in addrs
    assert "/ip4/203.0.113.7/udp/47100/quic-v1" in addrs, "QUIC must be offered too"


def test_the_port_is_taken_from_what_the_node_reported():
    """The orchestrator sees a NAT-translated source port, never the listener."""
    directory = PeerDirectory()
    directory.remember("n1", identity(addrs=["/ip4/10.0.0.5/tcp/51999"]), observed_host="1.2.3.4")
    assert "/ip4/1.2.3.4/tcp/51999" in directory.get("n1").candidate_addrs()


def test_lan_addresses_are_tried_before_public_ones():
    """Two nodes in one datacentre should meet on the LAN, not via the internet.

    Order is not cosmetic: a peer tries candidates in sequence and a wrong
    guess costs a connection timeout on the critical path.
    """
    directory = PeerDirectory()
    directory.remember("n1", identity(addrs=["/ip4/10.0.0.5/tcp/47100"]), observed_host="203.0.113.7")
    addrs = directory.get("n1").candidate_addrs()
    assert addrs[0].startswith("/ip4/10.0.0.5"), f"public address came first: {addrs}"


def test_a_public_node_is_not_dialled_twice_at_the_same_address():
    """Duplicates only double the failure timeout."""
    directory = PeerDirectory()
    directory.remember(
        "n1", identity(addrs=["/ip4/203.0.113.7/tcp/47100"]), observed_host="203.0.113.7"
    )
    addrs = directory.get("n1").candidate_addrs()
    assert len(addrs) == len(set(addrs))
    assert addrs.count("/ip4/203.0.113.7/tcp/47100") == 1


# --------------------------------------------------------- who cannot be dialled
def test_a_symmetric_nat_node_is_never_offered_for_direct_dialling():
    """Hole punching through a symmetric NAT cannot work — not "usually fails".

    The address the peer was told to aim at is not the one that NAT will use
    for that peer, so every attempt buys a timeout. Marking it here stops each
    neighbour from rediscovering it the expensive way.
    """
    directory = PeerDirectory()
    directory.remember("n1", identity(symmetric=True), observed_host="203.0.113.7")
    assert directory.get("n1").dialable is False

    rows = directory.routes_for([(0, "n1")])
    assert rows == [(0, "n1", "", [], 0.0, False)], "a symmetric-NAT node must be relay-only"


def test_a_worker_with_no_p2p_stack_is_recorded_but_not_dialable():
    """An old image or a CPU node: still a node, just always relayed."""
    directory = PeerDirectory()
    directory.remember("old", None, observed_host="203.0.113.9")
    record = directory.get("old")
    assert record is not None and record.dialable is False
    assert directory.routes_for([(1, "old")]) == [(1, "old", "", [], 0.0, False)]


def test_a_node_nobody_registered_is_relay_only():
    assert PeerDirectory().routes_for([(2, "ghost")]) == [(2, "ghost", "", [], 0.0, False)]


# ------------------------------------------------------------------ directory
def test_every_stage_is_described_to_the_others():
    """The last stage sends its token back to stage 0, so "next" is not enough."""
    directory = PeerDirectory()
    for i in range(3):
        directory.remember(
            f"n{i}",
            identity(peer_id=f"12D3KooW{i}", addrs=[f"/ip4/10.0.0.{i}/tcp/47100"]),
            observed_host=f"203.0.113.{i}",
        )
    rows = directory.routes_for([(0, "n0"), (1, "n1"), (2, "n2")])
    assert [r[0] for r in rows] == [0, 1, 2]
    assert all(r[2] for r in rows), "every stage should be dialable here"


def test_a_disconnected_node_is_forgotten():
    directory = PeerDirectory()
    directory.remember("n1", identity())
    directory.forget("n1")
    assert directory.get("n1") is None


def test_the_view_reports_why_a_node_is_relay_only():
    directory = PeerDirectory()
    directory.remember("n1", identity(symmetric=True), observed_host="1.2.3.4")
    row = directory.view()[0]
    assert row["symmetric_nat"] is True and row["dialable"] is False
    assert row["observed_host"] == "1.2.3.4"


# --------------------------------------------------------------------- IPv6
def test_an_ipv6_node_is_described_to_its_neighbours():
    """The address family that makes all the NAT machinery unnecessary.

    IPv6 used to be discarded here with the comment "the node's own list
    already covers it" — and that list was built from AF_INET only, so a node
    reaching the orchestrator over IPv6 offered its neighbours nothing at all.

    It is also the one observed address certain to be usable as seen: with no
    translation in the way, the address we observe IS the address the node
    listens on.
    """
    record = PeerRecord(
        node_id="v6",
        peer_id="12D3KooWV6",
        listen_addrs=["/ip6/2001:db8::5/tcp/47100"],
        observed_host="2001:db8::5",
    )
    addrs = record.candidate_addrs(47100)
    assert "/ip6/2001:db8::5/tcp/47100" in addrs
    assert "/ip6/2001:db8::5/udp/47100/quic-v1" in addrs
    assert not any("/ip4/" in a for a in addrs)


def test_a_global_ipv6_address_counts_as_reachable():
    """No circuit, no relay — the routing rule needs no IPv6 special case."""
    record = PeerRecord(
        node_id="v6",
        peer_id="12D3KooWV6",
        visible_addrs=["/ip6/2001:db8::5/tcp/47100"],
    )
    assert record.reachable


def test_a_lan_address_is_still_offered_first_in_either_family():
    """A pair on one network should try the local address before the routable one."""
    record = PeerRecord(
        node_id="both",
        peer_id="12D3KooWBoth",
        listen_addrs=[
            "/ip6/2001:db8::5/tcp/47100",   # global
            "/ip4/10.0.0.4/tcp/47100",      # LAN
            "/ip6/fd00::4/tcp/47100",       # unique-local: a LAN address too
        ],
    )
    order = record.candidate_addrs(47100)
    assert order.index("/ip4/10.0.0.4/tcp/47100") < order.index(
        "/ip6/2001:db8::5/tcp/47100"
    )
    assert order.index("/ip6/fd00::4/tcp/47100") < order.index(
        "/ip6/2001:db8::5/tcp/47100"
    )
