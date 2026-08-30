"""Orchestrator entrypoint: one gRPC service, one HTTP app, one loop.

    python -m loom.orchestrator.server        (configuration via env, see config.py)

Everything a node says arrives on the stream it opened — commands, task input,
results, telemetry. Nothing dials a node, so no node needs an address, an open
port, or anything configured on its owner's router.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import grpc
import uvicorn

from loom.api.app import create_app
from loom.logging_config import get_logger
from loom.orchestrator.agents import AgentHub, add_agent_gateway_to_server
from loom.orchestrator.config import OrchestratorConfig
from loom.orchestrator.keys import KeyStore
from loom.orchestrator.public_addr import resolve_public_address
from loom.orchestrator.releases import ReleaseStore
from loom.orchestrator.rendezvous import RendezvousNode, host_of
from loom.orchestrator.state import StateStore

logger = get_logger(__name__)

# A result file or a model's input can be large; the default 4 MB would make
# anything real fail in a way that looks like a network problem.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


async def serve_grpc(*, hub: AgentHub, port: int) -> grpc.aio.Server:
    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        # The stream is long-lived and mostly silent, which is exactly the
        # traffic pattern NATs and tunnels drop without telling anyone. Without
        # keepalive a node sits reading a stream whose other end is long gone:
        # it never errors, so it never reconnects, and simply disappears.
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
        # And the server must not scold agents for pinging at that rate.
        ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
    ])
    add_agent_gateway_to_server(server, hub)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("AgentGateway listening on :%d", port)
    return server


async def run() -> None:
    config = OrchestratorConfig.from_env()
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    public = resolve_public_address(config.grpc_port)
    logger.info("nodes will dial %s (%s)", public.address, public.source)
    if public.warning:
        logger.warning("%s", public.warning)

    keystore = KeyStore(public_address=public.address, path=config.keystore_path)
    releases = ReleaseStore(Path(config.releases_dir))

    # The p2p entry point handed to every node at registration. One address is
    # all a node needs to reach every other: peers are found by id through a
    # shared entry point, and bootstrap addresses are fixed when a libp2p node
    # is built.
    rendezvous = RendezvousNode(public_host=host_of(public.address))
    rendezvous.start()

    # Where a node fetches an agent payload from: the same host it dials, on
    # the HTTP port. Empty when we do not know our own address, in which case
    # no release is offered rather than one nobody can download.
    dial_host = host_of(public.address)
    hub = AgentHub(
        keystore=keystore,
        rendezvous=rendezvous,
        releases=releases,
        release_base_url=f"http://{dial_host}:{config.http_port}" if dial_host else "",
        store=StateStore(config.state_path),
    )
    # До того, как узлы начнут дозваниваться: иначе первый доклад придёт в
    # пустой хаб, его задачи окажутся незнакомыми, и мы заведём их заново —
    # уже без имени модели и без группы, которые лежат в файле.
    hub.restore()

    grpc_server = await serve_grpc(hub=hub, port=config.grpc_port)
    app = create_app(agents=hub, releases=releases, keystore=keystore,
                     config=config, public_address=public)
    http = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.http_port,
                                         log_level="info", loop="asyncio"))
    logger.info("HTTP on :%d  (dashboard at /admin)", config.http_port)
    flusher = asyncio.create_task(hub.flush_loop())
    try:
        await http.serve()
    finally:
        flusher.cancel()
        try:
            await flusher
        except asyncio.CancelledError:
            pass
        hub.flush()
        await grpc_server.stop(5)
        rendezvous.stop()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("stopped")


if __name__ == "__main__":
    main()
