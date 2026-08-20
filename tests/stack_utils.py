"""Helpers to run an in-process orchestrator + real workers for e2e tests.

Mirrors production wiring: workers authenticate with a join key, hardware is
auto-detected (overridden via env in tests), and inference flows through the
data-plane tunnel.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.dataplane_client import DataPlaneClient  # noqa: E402
from loom_worker.gateway_client import GatewayClient  # noqa: E402
from loom_worker.handlers import CommandHandlers  # noqa: E402
from loom_worker.main import PeerLayer  # noqa: E402
from loom_worker.proto import gateway_pb2 as w_gateway_pb2  # noqa: E402
from loom_worker.security import CommandVerifier  # noqa: E402
from loom_worker.state import WorkerState  # noqa: E402

from loom.orchestrator.config import OrchestratorConfig  # noqa: E402
from loom.orchestrator.controller import MultiModelController  # noqa: E402
from loom.orchestrator.gateway import ControlGatewayServicer  # noqa: E402
from loom.orchestrator.keys import KeyStore  # noqa: E402
from loom.orchestrator.server import serve_grpc  # noqa: E402
from loom.orchestrator.placement import Placement  # noqa: E402
from loom.orchestrator.tunnel import DataPlaneServicer, TunnelHub  # noqa: E402

GIB = 1024**3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class OrchestratorHarness:
    """Orchestrator gRPC server (control + data plane) in an asyncio thread."""

    def __init__(self, registry, *, keystore_path=None, auto_deploy=True) -> None:
        """`auto_deploy` mirrors what production does NOT do.

        A catalog entry is an offer, not an order: nothing runs until someone
        deploys it, which is why restarting the orchestrator no longer brings
        models up on its own. Most tests here predate that and are about the
        broker, self-healing or the SLO loop, so they opt every catalog model
        into brokered placement and carry on. Tests about placement itself pass
        auto_deploy=False and deploy explicitly.
        """
        self.auto_deploy = auto_deploy
        self.grpc_port = free_port()
        self.config = OrchestratorConfig(
            registry=registry,
            grpc_port=self.grpc_port,
            public_address=f"127.0.0.1:{self.grpc_port}",
            keystore_path=str(keystore_path) if keystore_path else "",
            perfmap_sync_interval_s=1.0,
            rebalance_interval_s=3600.0,  # explicit triggers only in tests
            # Production waits minutes before retrying a failed placement (a
            # retry costs a checkpoint download); tests must not.
            deploy_retry_s=5.0,
        )
        self.keystore = KeyStore(
            public_address=self.config.public_address, path=keystore_path or None
        )
        # One key for the whole test pool; production issues one per provider.
        self.join_key = self.keystore.issue(label="test-pool").encode()
        self.tunnel = TunnelHub()
        self.loop = asyncio.new_event_loop()
        self.controller = MultiModelController(self.config, tunnel=self.tunnel)
        self.controller.keystore = self.keystore
        if auto_deploy:
            for spec in registry.list():
                self.controller.placements[spec.model_id] = Placement.auto(spec.model_id)
        self.servicer = ControlGatewayServicer(
            keystore=self.keystore, controller=self.controller
        )
        self._started = threading.Event()
        self._stop_event: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        async def main():
            server = await serve_grpc(
                control=self.servicer,
                data=DataPlaneServicer(self.tunnel, self.keystore),
                port=self.grpc_port,
            )
            self._stop_event = asyncio.Event()
            self._started.set()
            await self._stop_event.wait()
            await server.stop(grace=None)

        self.loop.run_until_complete(main())

    def start(self) -> "OrchestratorHarness":
        self._thread.start()
        assert self._started.wait(10)
        return self

    def stop(self) -> None:
        if self._stop_event is not None:
            self.loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)

    def submit(self, coro):
        """Run a coroutine on the orchestrator loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=60)

    def call_api(self, fn, *, timeout: float = 60):
        """Run an API call on the orchestrator's loop.

        Mandatory for anything that touches the data-plane tunnel: its queues
        belong to that loop. Production runs the API and gRPC in one loop
        (see loom.orchestrator.server), so this mirrors real wiring.
        """
        import httpx

        from loom.api.app import create_app

        async def run():
            app = create_app(self.controller)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://api", timeout=timeout
            ) as client:
                return await fn(client)

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout=timeout)


class WorkerHarness:
    def __init__(
        self,
        orch_port: int,
        *,
        node_id: str,
        memory_gb: float,
        join_key: str,
        region: str = "default",
        device: str = "cpu",
        verify_commands: bool = True,
        rss_overhead_bytes: int = 4 * GIB,
        p2p_port: int | None = None,
        p2p_key_dir: str = "",
    ) -> None:
        from loom_worker.joinkey import parse_join_key

        parsed = parse_join_key(join_key)
        secret = parsed.secret if parsed else ""
        self.state = WorkerState(node_id=node_id, advertise_host="127.0.0.1")
        holder: dict = {}
        # Data plane first: the stage relay URL must exist before any LoadShard
        # (mirrors loom_worker.main wiring).
        self.dataplane = DataPlaneClient(
            orchestrator_addr=f"127.0.0.1:{orch_port}",
            join_key=join_key,
            state=self.state,
        )
        relay_url = self.dataplane.start_stage_relay()
        # Same wiring as loom_worker.main: the direct path is relay-only until
        # the orchestrator's registration ack says where its rendezvous is.
        # Built here rather than skipped so the e2e tests exercise the real
        # data path, not a version of it that only ever relays.
        self.peers = PeerLayer(self.dataplane, port=p2p_port, key_dir=p2p_key_dir)
        self.dataplane.links = self.peers.links
        self.handlers = CommandHandlers(
            self.state,
            send=lambda m: holder["client"].send(m),
            device=device,
            watchdog_poll_s=0.3,
            relay_url=relay_url,
            links=self.peers.links,
            peer_status=self.peers.status,
            # Tiny test quotas: RSS of a torch process dwarfs them, so give the
            # runtime room instead of having the watchdog kill every stage.
            # Watchdog tests pass 0 to exercise strict enforcement.
            rss_overhead_bytes=rss_overhead_bytes,
            # Bounded so a broken start does not wait out the production-sized
            # timeout, but generous: several torch stages warming up in
            # parallel on a busy machine legitimately take minutes.
            ready_timeout_s=300.0,
        )
        self.client = GatewayClient(
            orchestrator_addr=f"127.0.0.1:{orch_port}",
            join_key=join_key,
            state=self.state,
            hardware=w_gateway_pb2.HardwareInfo(
                num_gpus=1,
                tflops_fp16=10,
                gpu_name="test-gpu",
                memory_gb=memory_gb,
                memory_bandwidth_gbps=100,
                device=device,
                vram_free_bytes=int(memory_gb * GIB),
                vram_total_bytes=int(memory_gb * GIB),
                detection_source="test",
            ),
            handlers=self.handlers,
            region=region,
            on_rendezvous=self.peers.on_rendezvous,
            verifier=CommandVerifier(secret) if (verify_commands and secret) else None,
            agent_version="test",
            heartbeat_interval_s=1.0,
        )
        holder["client"] = self.client
        self._thread = threading.Thread(target=self.client.run_forever, daemon=True)

    def start(self) -> "WorkerHarness":
        self.client.start_heartbeats()
        self._thread.start()
        self.dataplane.start()
        return self

    def stop(self) -> None:
        self.client.stop()
        self.dataplane.stop()
        for shard in self.state.snapshot().values():
            if shard.backend is not None:
                shard.backend.stop()


def wait_until(predicate, timeout_s=30.0, poll_s=0.1):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False
