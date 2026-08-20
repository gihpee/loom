"""Orchestrator entrypoint: control plane + data plane + client API in one loop.

Run: python -m loom.orchestrator.server  (configuration via env, see config.py)

Two gRPC services share port LOOM_GRPC_PORT:
- ControlGateway — commands/telemetry
- DataPlane      — inference tunnelled back over the worker's own connection
"""

from __future__ import annotations

import asyncio

import grpc
import uvicorn

from loom.api.app import create_app
from loom.logging_config import get_logger
from loom.orchestrator.config import OrchestratorConfig
from loom.orchestrator.controller import MultiModelController
from loom.orchestrator.gateway import ControlGatewayServicer
from loom.orchestrator.keys import KeyStore
from loom.orchestrator.public_addr import resolve_public_address
from loom.orchestrator.rendezvous import RendezvousNode, host_of
from loom.orchestrator.tunnel import DataPlaneServicer, TunnelHub, add_dataplane_to_server
from loom.proto_gen import gateway_pb2_grpc

logger = get_logger(__name__)

MAX_MESSAGE_BYTES = 64 * 1024 * 1024


async def serve_grpc(
    *, control: ControlGatewayServicer, data: DataPlaneServicer, port: int
) -> grpc.aio.Server:
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            # Both control streams are long-lived and mostly silent, which is
            # exactly the traffic pattern NATs and tunnels drop without telling
            # anyone. Without keepalive a worker can sit reading a stream whose
            # other end is long gone — it never errors, so it never reconnects,
            # and the node simply disappears from the orchestrator's view.
            ("grpc.keepalive_time_ms", 20000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
            # The server must not scold clients for pinging at that rate.
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ("grpc.http2.max_pings_without_data", 0),
        ]
    )
    gateway_pb2_grpc.add_ControlGatewayServicer_to_server(control, server)
    add_dataplane_to_server(server, data)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC (ControlGateway + DataPlane) listening on :%d", port)
    return server


async def run(config: OrchestratorConfig) -> None:
    # Workers must dial SOMETHING reachable; find it ourselves instead of
    # making the operator hunt for an IP (see public_addr.py).
    public = resolve_public_address(config.grpc_port)
    config.public_address = public.address
    keystore = KeyStore(
        public_address=public.address,
        path=config.keystore_path or None,
        master_token=config.node_token,
    )
    logger.info("workers will be told to dial %s (source: %s)", public.address, public.source)
    if public.warning:
        logger.warning("%s", public.warning)
    if keystore.open_registration():
        logger.warning(
            "OPEN REGISTRATION: no join keys issued and no master token set — "
            "any worker reaching :%d will be accepted. Issue a key via "
            "POST /admin/keys before exposing this port.",
            config.grpc_port,
        )
    tunnel = TunnelHub()
    controller = MultiModelController(config, tunnel=tunnel)
    controller.keystore = keystore
    controller.public_address = public

    # The single address workers bootstrap to, so they can then reach each
    # other by peer id. Optional: without it every stage message keeps going
    # through this process, which is what Loom did before direct links existed.
    rendezvous = RendezvousNode(public_host=host_of(public.address))
    rendezvous.start()
    controller.rendezvous = rendezvous
    if rendezvous.running:
        logger.info("rendezvous multiaddrs: %s", ", ".join(rendezvous.multiaddrs()))

    grpc_server = await serve_grpc(
        control=ControlGatewayServicer(keystore=keystore, controller=controller),
        data=DataPlaneServicer(tunnel, keystore),
        port=config.grpc_port,
    )

    app = create_app(controller)
    http_config = uvicorn.Config(
        app, host="0.0.0.0", port=config.http_port, log_level="info", loop="asyncio"
    )
    http_server = uvicorn.Server(http_config)

    background = [
        asyncio.create_task(controller.perfmap_sync_loop()),
        asyncio.create_task(controller.rebalance_timer_loop()),
        asyncio.create_task(controller.slo_check_loop()),
    ]
    try:
        await http_server.serve()
    finally:
        for task in background:
            task.cancel()
        await grpc_server.stop(grace=2)
        rendezvous.stop()


def main() -> None:
    asyncio.run(run(OrchestratorConfig.from_env()))


if __name__ == "__main__":
    main()
