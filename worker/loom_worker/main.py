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

from loom_worker import __version__
from loom_worker.dataplane_client import DataPlaneClient
from loom_worker.gateway_client import GatewayClient
from loom_worker.handlers import CommandHandlers
from loom_worker.hwinfo import detect_hardware
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

    handlers = CommandHandlers(
        state,
        send=lambda m: holder["client"].send(m),
        device=hardware.device,
        watchdog_poll_s=float(os.environ.get("LOOM_WATCHDOG_POLL_S", "2")),
        relay_url=relay_url,
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
