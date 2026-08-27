"""A worker whose network path changes must come back on its own.

Node owners turn VPNs on and off, move laptops between networks, and lose
Wi-Fi. Each of those replaces the route to the orchestrator without closing
anything: the old TCP connection is not reset, it simply stops delivering.

That is the failure this file is about. The control channel notices within
seconds because it pings; the data-plane tunnel used to have no keepalive at
all, so it sat reading a stream whose far end was gone — forever. The node
kept heartbeating and looked healthy, its tunnel showed red, and no inference
could be routed to it until someone restarted the worker by hand.
"""

import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.dataplane_client import DataPlaneClient  # noqa: E402
from loom_worker.gateway_client import GatewayClient  # noqa: E402


def client():
    return DataPlaneClient(orchestrator_addr="127.0.0.1:1", join_key="k", state=None)


def test_the_tunnel_pings_so_a_dead_path_becomes_an_error():
    """Without this the stream never fails, so it never reconnects."""
    options = dict(client().channel_options())

    assert options["grpc.keepalive_time_ms"] > 0, "the tunnel sends no keepalive"
    assert options["grpc.keepalive_timeout_ms"] > 0
    # The tunnel is silent between requests, which is exactly when a route
    # disappears unnoticed. Pinging only during calls would miss it.
    assert options["grpc.keepalive_permit_without_calls"] == 1


def test_the_tunnel_pings_on_the_same_terms_as_the_control_channel():
    """One of the two noticing and the other not is the bug, precisely.

    A node whose control channel reconnects while its tunnel stays wedged is
    worse than one that drops entirely: it reports itself healthy and accepts
    placements it cannot serve.
    """
    tunnel = dict(client().channel_options())
    control = dict(GatewayClient.CHANNEL_OPTIONS)

    for option in (
        "grpc.keepalive_time_ms",
        "grpc.keepalive_timeout_ms",
        "grpc.keepalive_permit_without_calls",
    ):
        assert tunnel[option] == control[option], option


def test_a_broken_tunnel_is_retried_rather_than_given_up_on():
    """Reconnecting is the other half: noticing is no use without it."""
    assert client().reconnect_delay_s > 0
