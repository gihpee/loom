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

from looma.orchestrator.rendezvous import RendezvousNode, host_of  # noqa: E402
from looma.orchestrator.rendezvous import lattica_available  # noqa: E402

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
    monkeypatch.setenv("LOOMA_P2P", "0")
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


# --------------------------------------------------------------- relay wiring
def test_relay_addresses_are_configured_not_discovered(monkeypatch):
    """A relay is infrastructure somebody stood up, not something to find.

    It is also a SEPARATE process from the rendezvous: Lattica announces
    /libp2p/circuit/relay/0.2.0/stop and never the /hop half, so the
    orchestrator's own node cannot be the relay however reachable it is.
    """
    from looma.orchestrator.rendezvous import relay_addrs

    monkeypatch.setenv("LOOMA_P2P_RELAY", "/ip4/1.2.3.4/tcp/47200/p2p/12D3KooWa, ")
    assert relay_addrs() == ["/ip4/1.2.3.4/tcp/47200/p2p/12D3KooWa"]

    monkeypatch.delenv("LOOMA_P2P_RELAY", raising=False)
    monkeypatch.setenv("LOOMA_P2P_RELAY_FILE", "/nonexistent/relay/address")
    assert relay_addrs() == []


def test_the_relay_hands_its_address_over_without_a_human(monkeypatch, tmp_path):
    """The step that was lost on a live stand, removed rather than documented.

    Enabling the relay used to mean copying a multiaddr from its log into .env
    and restarting the orchestrator. Skipping the copy leaves every part of the
    system looking healthy — relay up, workers up, model serving — and the only
    evidence is "(2 bootstrap, 0 relay)" in a worker's startup log.

    Both processes share a volume, so the relay writes the address there and
    the orchestrator reads it. Read per registration, not cached: a relay
    started after the orchestrator must still be picked up.
    """
    from looma.orchestrator.rendezvous import relay_addrs

    published = tmp_path / "relay" / "address"
    monkeypatch.delenv("LOOMA_P2P_RELAY", raising=False)
    monkeypatch.setenv("LOOMA_P2P_RELAY_FILE", str(published))

    assert relay_addrs() == []  # no relay running yet

    published.parent.mkdir()
    published.write_text("/ip4/203.0.113.7/tcp/47200/p2p/12D3KooWRelay\n")
    assert relay_addrs() == ["/ip4/203.0.113.7/tcp/47200/p2p/12D3KooWRelay"]

    # An explicit setting still wins: someone running the relay elsewhere must
    # not be overruled by a stale file from a relay that used to be local.
    monkeypatch.setenv("LOOMA_P2P_RELAY", "/ip4/198.51.100.9/tcp/47200/p2p/12D3KooWOther")
    assert relay_addrs() == ["/ip4/198.51.100.9/tcp/47200/p2p/12D3KooWOther"]



# --------------------------------------------------- имя против адреса
# Со стенда: узлы не находили друг друга, Ray не собирал кластер, и искать
# причину пошли в Ray, в порты и в файрвол. Она была здесь.
def test_домен_объявляется_как_dns4_а_не_ip4():
    """`/ip4/<имя>` — невалидный мультиадрес: после /ip4 обязан идти literal
    IPv4. Работник получает такой адрес, не может его набрать, остаётся вне
    DHT — и дальше не находит соседей по peer id. Снаружи это выглядит как
    «кластер не собирается», и до настоящей причины идти три слоя."""
    from looma.orchestrator.rendezvous import host_proto

    assert host_proto("loomafloat.ru") == "dns4"
    assert host_proto("looma.example.com") == "dns4"


def test_адрес_объявляется_как_ip4():
    from looma.orchestrator.rendezvous import host_proto

    assert host_proto("203.0.113.7") == "ip4"
    assert host_proto("2001:db8::1") == "ip6"


def test_объявленные_адреса_домена_можно_набрать():
    """Проверяется целиком то, что уезжает работнику."""
    from looma.orchestrator.rendezvous import RendezvousNode

    node = RendezvousNode(public_host="loomafloat.ru", port=47100)
    addrs = node._announced_addrs()
    assert addrs == ["/dns4/loomafloat.ru/tcp/47100",
                     "/dns4/loomafloat.ru/udp/47100/quic-v1"]
    for addr in addrs:
        assert not addr.startswith("/ip4/"), "имя под /ip4 не разбирается"


def test_объявленные_адреса_ip_остаются_ip4():
    from looma.orchestrator.rendezvous import RendezvousNode

    node = RendezvousNode(public_host="203.0.113.7", port=47100)
    assert node._announced_addrs()[0] == "/ip4/203.0.113.7/tcp/47100"
