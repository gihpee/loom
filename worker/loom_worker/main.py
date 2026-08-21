"""Worker entrypoint.

The node owner runs one command and does nothing else:

    docker run gihpee/loomworker --key loom_<...>

Everything else is automatic: hardware is detected locally, the orchestrator
address comes from the key, and all inference traffic is tunnelled back through
the worker's own outbound connection (no ports to open, no address to declare).

Optional env (mostly for testing/ops):
- LOOM_KEY            same as --key
- LOOM_ORCH_ADDR      override the address embedded in the key
- LOOM_NODE_ID        stable node id (default: hostname)
- LOOM_REGION         region label used by the Resource Broker
- LOOM_DEVICE / LOOM_MEMORY_GB / LOOM_TFLOPS_FP16 / ...  hardware overrides
- LOOM_VERIFY_COMMANDS=0  disable command-signature verification (dev only)
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from typing import List, Optional

from loom_worker import __version__
from loom_worker.dataplane_client import DataPlaneClient
from loom_worker.gateway_client import GatewayClient
from loom_worker.handlers import CommandHandlers
from loom_worker.hwinfo import detect_hardware
from loom_worker.p2p import (
    DEFAULT_P2P_PORT,
    LinkTable,
    PeerNode,
    behind_container_nat,
    lattica_available,
)
from loom_worker.proto import worker_control_pb2
from loom_worker.joinkey import parse_join_key
from loom_worker.proto import gateway_pb2
from loom_worker.security import CommandVerifier
from loom_worker.state import WorkerState

logger = logging.getLogger("loom_worker.main")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="loom-worker", description="Loom worker agent (executes orchestrator commands)"
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("LOOM_KEY", ""),
        help="join key issued by the orchestrator (loom_...)",
    )
    parser.add_argument(
        "--orchestrator",
        default=os.environ.get("LOOM_ORCH_ADDR", ""),
        help="override orchestrator gRPC address (host:port)",
    )
    parser.add_argument("--node-id", default=os.environ.get("LOOM_NODE_ID", ""))
    parser.add_argument("--region", default=os.environ.get("LOOM_REGION", "default"))
    return parser.parse_args(argv)


def build_hardware_message() -> gateway_pb2.HardwareInfo:
    hw = detect_hardware()
    logger.info(
        "detected hardware: device=%s gpu=%s x%d vram_free=%.1fGB tflops=%.1f (%s)",
        hw.device,
        hw.gpu_name,
        hw.num_gpus,
        hw.vram_free_bytes / 1024**3,
        hw.tflops_fp16,
        hw.detection_source,
    )
    return gateway_pb2.HardwareInfo(
        num_gpus=hw.num_gpus,
        tflops_fp16=hw.tflops_fp16,
        gpu_name=hw.gpu_name,
        memory_gb=hw.memory_gb,
        memory_bandwidth_gbps=hw.memory_bandwidth_gbps,
        device=hw.device,
        vram_free_bytes=hw.vram_free_bytes,
        vram_total_bytes=hw.vram_total_bytes,
        host_ram_gb=hw.host_ram_gb,
        detection_source=hw.detection_source,
    )


def _env_relays() -> List[str]:
    """Relay servers from the environment, for a worker configured by hand."""
    return [a.strip() for a in os.environ.get("LOOM_P2P_RELAY", "").split(",") if a.strip()]


class PeerLayer:
    """This node's direct path: brought up when the rendezvous becomes known.

    A worker cannot start its p2p node at process start, because the address it
    must bootstrap to arrives in the registration ack. So the LinkTable exists
    from the beginning and relays everything, and the node is attached to it
    later — everything downstream holds the same object and never has to be
    re-wired.

    Started on the ack rather than on the first multi-stage LoadShard on
    purpose: joining the network takes seconds, and a deployment should not be
    the thing that waits for it.
    """

    def __init__(self, dataplane, *, port: Optional[int] = None, key_dir: str = "") -> None:
        self.dataplane = dataplane
        self.links = LinkTable()
        self.node: Optional[PeerNode] = None
        self.identity = None
        # Overrides for the node's port and key directory. Defaults come from
        # the environment; both are explicit here so a test can run several
        # agents in one process — two nodes sharing a key directory interfere,
        # and a closed node does not free its port instantly.
        self._port = port
        self._key_dir = key_dir
        self._enabled = os.environ.get("LOOM_P2P", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    def on_rendezvous(self, addrs: List[str], relays: Optional[List[str]] = None) -> None:
        """Bring the node up against the orchestrator's rendezvous.

        Never fatal, and never retried in a loop: a worker that cannot join the
        peer network is slower, not broken, and taking a working GPU out of the
        pool over a networking nicety would be the wrong trade.
        """
        if self.node is not None or not self._enabled:
            return
        addrs = [a for a in (addrs or []) if a.strip()]
        if not addrs:
            logger.info(
                "the orchestrator offers no rendezvous; stage messages go through it"
            )
            return
        if not lattica_available():
            logger.info(
                "no p2p stack installed; stage messages go through the "
                "orchestrator. Install the extra to let this node talk to its "
                "neighbours directly: pip install 'loom-worker[p2p]'"
            )
            return
        # The rendezvous is a DHT entry point and nothing else — pointing the
        # relay client at it leaves the node waiting for a reservation that
        # never comes (measured: 15 s and zero peers). A real circuit-relay
        # server is a separate process; see relay/relay.mjs.
        #
        # The relay is what makes hole punching possible at all for a node
        # nothing can dial into: it needs a working channel to agree on the
        # moment to punch through. Without one such a node can only be reached
        # by relaying its activations through the orchestrator.
        relay_addrs = [a for a in (relays or []) if a.strip()] or _env_relays()
        options = {"bootstraps": addrs, "relay_servers": relay_addrs}
        if self._port is not None:
            options["port"] = self._port
        if self._key_dir:
            options["key_dir"] = self._key_dir
        node = PeerNode(**options)
        try:
            self.identity = node.start(on_message=self.dataplane.deliver_direct)
        except Exception:
            logger.warning(
                "the p2p node did not start; relaying every stage message",
                exc_info=True,
            )
            return
        self.node = node
        self.links.attach(
            send_direct=node.send,
            dial=node.warm,
            rtt=node.rtt_ms,
            relay_rtt=node.relay_rtt_ms,
        )
        if self.identity.symmetric_nat:
            logger.warning(
                "this node is behind a symmetric NAT: peers cannot open a direct "
                "link to it and will relay"
            )
        elif any("/p2p-circuit" in a for a in self.identity.visible_addrs):
            # A reservation is held: nothing can dial this node directly, but
            # peers can now reach it through the relay and try to punch through
            # from there. This is the state the relay exists to produce.
            logger.info(
                "this node is reachable through the relay: %s",
                self.identity.visible_addrs[0],
            )
        elif not self.identity.visible_addrs:
            # The quiet case, and the one that wastes an afternoon: the node
            # looks fine, reports a peer id, and nobody can reach it. Naming
            # which of the two reasons it is saves the whole investigation —
            # they look identical from here and have nothing in common.
            if behind_container_nat():
                logger.warning(
                    "this worker runs on a Docker bridge network, so no peer "
                    "can ever open a direct link to it: the container's port "
                    "is not the host's, and outgoing packets are translated "
                    "again on the way out. Hole punching cannot work from "
                    "here. Restart the container with --network host (and "
                    "open port %d) to make the direct path possible",
                    self.node.port if self.node else DEFAULT_P2P_PORT,
                )
            elif relay_addrs:
                logger.warning(
                    "no address of this node is reachable from outside, and "
                    "the relay at %s gave it no reservation either. Check that "
                    "the relay is running and its port is open from here",
                    relay_addrs[0],
                )
            else:
                logger.warning(
                    "no address of this node is reachable from outside (AutoNAT "
                    "confirmed none) and no relay was offered. Peers will relay "
                    "TO it through the orchestrator; it can still send "
                    "directly. Forward port %d (TCP and UDP), or run a relay "
                    "(docs/P2P_RELAY.md), to make it dialable",
                    self.node.port if self.node else DEFAULT_P2P_PORT,
                )

    def status(self):
        """What this node reports about its p2p state on every heartbeat.

        Reachability is re-read from the node rather than taken from the
        identity captured at startup. It is not a constant: a relay reservation
        can arrive a second after the node joins, and AutoNAT needs a few
        probes before it confirms anything. Reporting the startup snapshot
        forever showed nodes as unreachable long after they were not.
        """
        stats = self.links.snapshot()
        identity = self.identity
        # The counters are reported even with no identity at all: a node with
        # no p2p stack still relays every message, and that is exactly what the
        # orchestrator needs to see.
        visible = self.node.visible_addrs() if self.node is not None else []
        return worker_control_pb2.PeerStatus(
            peer_id=identity.peer_id if identity else "",
            listen_addrs=identity.listen_addrs if identity else [],
            symmetric_nat=bool(identity.symmetric_nat) if identity else False,
            direct=stats["direct"],
            relayed=stats["relay"],
            fallbacks=stats["fallbacks"],
            direct_share=stats["direct_share"],
            visible_addrs=visible or (identity.visible_addrs if identity else []),
            not_worth=stats["not_worth"],
            link_rtt_ms=stats["link_rtt_ms"],
            relay_rtt_ms=stats["relay_rtt_ms"],
        )


def main(argv=None) -> None:
    logging.basicConfig(
        level=os.environ.get("LOOM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args(argv)

    parsed = parse_join_key(args.key)
    if args.key and parsed is None and args.key.startswith("loom_"):
        logger.error("the provided --key is malformed (expected a loom_... string)")
        sys.exit(2)

    if parsed is not None:
        # Normal path: the key carries the address AND the signing secret.
        address = args.orchestrator or parsed.address
        secret = parsed.secret
        presented_key = parsed.raw
    else:
        # Dev/legacy path: a plain shared token; the address must be given.
        address = args.orchestrator
        secret = args.key or os.environ.get("LOOM_NODE_TOKEN", "")
        presented_key = secret

    if not address:
        logger.error(
            "no orchestrator address: pass a valid --key (or LOOM_KEY), "
            "or set --orchestrator/LOOM_ORCH_ADDR"
        )
        sys.exit(2)

    hostname = socket.gethostname()
    state = WorkerState(node_id=args.node_id or hostname, advertise_host="127.0.0.1")
    hardware = build_hardware_message()

    verifier = None
    if secret and os.environ.get("LOOM_VERIFY_COMMANDS", "1") != "0":
        verifier = CommandVerifier(
            secret, max_skew_ms=int(os.environ.get("LOOM_CMD_MAX_SKEW_MS", "60000"))
        )

    holder: dict = {}
    # Data plane first: the stage relay URL must exist before any LoadShard.
    dataplane = DataPlaneClient(
        orchestrator_addr=address, join_key=presented_key, state=state
    )
    relay_url = dataplane.start_stage_relay()

    # Direct worker-to-worker path. Relay-only until the orchestrator tells us
    # where its rendezvous is (see PeerLayer). Optional by construction: a node
    # that cannot join keeps relaying, which is what every node did before this
    # existed. See docs/P2P_TRANSPORT.md.
    peers = PeerLayer(dataplane)
    dataplane.links = peers.links

    handlers = CommandHandlers(
        state,
        send=lambda m: holder["client"].send(m),
        device=hardware.device,
        watchdog_poll_s=float(os.environ.get("LOOM_WATCHDOG_POLL_S", "2")),
        relay_url=relay_url,
        links=peers.links,
        peer_status=peers.status,
        backend_kwargs={
            "vllm": {"total_vram_bytes": hardware.vram_total_bytes},
            "sglang": {"total_vram_bytes": hardware.vram_total_bytes},
        },
    )
    client = GatewayClient(
        orchestrator_addr=address,
        join_key=presented_key,
        state=state,
        hardware=hardware,
        handlers=handlers,
        region=args.region,
        verifier=verifier,
        agent_version=__version__,
        heartbeat_interval_s=float(os.environ.get("LOOM_HEARTBEAT_S", "5")),
        on_rendezvous=peers.on_rendezvous,
    )
    holder["client"] = client

    # Inference and inter-stage traffic are relayed over this outbound stream.
    dataplane.start()

    logger.info("worker %s attaching to %s", state.node_id, address)
    client.start_heartbeats()
    try:
        client.run_forever()
    finally:
        dataplane.stop()


if __name__ == "__main__":
    main()
