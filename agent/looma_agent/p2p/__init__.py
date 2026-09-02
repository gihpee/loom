"""Direct worker-to-worker data path, with the orchestrator as rendezvous.

The control plane does not change: the orchestrator remains the single trusted
node and the only party a worker authenticates against. What changes is where
activations travel — straight to the next stage when a route can be made, and
through the orchestrator's relay whenever it cannot.

See docs/P2P_TRANSPORT.md for the design and the measurements behind it.
"""

from looma_agent.p2p.links import LinkTable, Neighbour, neighbours_from_topology
from looma_agent.p2p.peer import (
    behind_container_nat,
    DEFAULT_P2P_PORT,
    P2PUnavailable,
    PeerIdentity,
    PeerNode,
    lattica_available,
    local_candidate_addrs,
)

__all__ = [
    "DEFAULT_P2P_PORT",
    "behind_container_nat",
    "LinkTable",
    "Neighbour",
    "P2PUnavailable",
    "PeerIdentity",
    "PeerNode",
    "lattica_available",
    "local_candidate_addrs",
    "neighbours_from_topology",
]
