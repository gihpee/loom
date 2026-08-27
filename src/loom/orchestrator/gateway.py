"""gRPC ControlGateway server: accepts worker dial-ins, one bidi stream each.

The servicer owns per-worker sessions (outgoing command queue + pending-ack
futures) and forwards worker events to the controller callbacks.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict, Optional

import grpc

from loom.logging_config import get_logger
from loom.orchestrator.signing import sign_control_message
from loom.proto_gen import gateway_pb2, gateway_pb2_grpc, worker_control_pb2

logger = get_logger(__name__)


def _observed_host(context) -> str:
    """The worker's address as seen from here, e.g. "ipv4:203.0.113.7:51234".

    gRPC hands it over as a scheme-prefixed string; only the host matters,
    because the p2p port is the worker's own and travels in its PeerIdentity.
    An IPv6 peer is reported bracketed, hence the rsplit on the last colon.
    """
    try:
        raw = context.peer() or ""
    except Exception:
        return ""
    for prefix in ("ipv4:", "ipv6:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    else:
        return ""  # unix sockets and other transports have nothing to offer
    host = raw.rsplit(":", 1)[0]
    return host.strip("[]")


class WorkerSession:
    """Server-side view of one connected worker."""

    def __init__(
        self,
        node_id: str,
        register: gateway_pb2.RegisterRequest,
        sign_key: str = "",
        observed_host: str = "",
    ) -> None:
        self.node_id = node_id
        self.register = register
        self.sign_key = sign_key
        # The address this worker's control connection arrives from, as the
        # orchestrator sees it. For a node behind NAT this is its PUBLIC
        # address — the one fact it cannot learn about itself, and exactly what
        # a STUN server would tell it. We get it for free from a connection the
        # worker had to open anyway, and pass it to that node's peers so they
        # have something dialable to aim at.
        self.observed_host = observed_host
        self.outbox: asyncio.Queue[Optional[gateway_pb2.ControlMessage]] = asyncio.Queue()
        self.pending_acks: Dict[str, asyncio.Future] = {}
        self.endpoints: Dict[str, str] = {}  # model_id -> serving url

    async def send(self, msg: gateway_pb2.ControlMessage) -> None:
        await self.outbox.put(msg)

    async def send_command(
        self, msg: gateway_pb2.ControlMessage, command_id: str, timeout_s: float = 600.0
    ) -> worker_control_pb2.Ack:
        """Sign the command and send it, awaiting its Ack."""
        sign_control_message(msg, self.sign_key)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending_acks[command_id] = fut
        await self.send(msg)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        finally:
            self.pending_acks.pop(command_id, None)

    def resolve_ack(self, ack: worker_control_pb2.Ack) -> None:
        fut = self.pending_acks.get(ack.command_id)
        if fut is not None and not fut.done():
            fut.set_result(ack)

    def close(self) -> None:
        self.outbox.put_nowait(None)
        for fut in self.pending_acks.values():
            if not fut.done():
                fut.set_exception(ConnectionError("worker disconnected"))


def new_meta() -> worker_control_pb2.CommandMeta:
    import time as _time

    return worker_control_pb2.CommandMeta(
        command_id=str(uuid.uuid4()), issued_at_unix_ms=int(_time.time() * 1000)
    )


class ControlGatewayServicer(gateway_pb2_grpc.ControlGatewayServicer):
    def __init__(self, *, keystore, controller) -> None:
        self.keystore = keystore
        self.controller = controller  # duck-typed: on_register/on_telemetry/on_endpoint/on_disconnect
        self.sessions: Dict[str, WorkerSession] = {}
        # When each node id last registered, to tell a reconnect apart from
        # two workers fighting over one name.
        self._registered_at: Dict[str, float] = {}

    async def Attach(self, request_iterator, context):
        session: Optional[WorkerSession] = None
        holder = {"session": None, "ready": asyncio.Event()}
        observed = _observed_host(context)
        recv_task = asyncio.create_task(
            self._recv_loop(request_iterator, holder, observed_host=observed)
        )
        try:
            # Wait until registration produced a session (or the recv loop died).
            ready_task = asyncio.create_task(holder["ready"].wait())
            done, _ = await asyncio.wait(
                {recv_task, ready_task}, return_when=asyncio.FIRST_COMPLETED
            )
            ready_task.cancel()
            session = holder["session"]
            if session is None:
                if recv_task in done:
                    recv_task.result()  # surface exceptions
                return
            # Pump the outbox into the response stream, ending as soon as the
            # worker's half of the stream closes (half-open connections must
            # not look like healthy sessions).
            while True:
                get_task = asyncio.create_task(session.outbox.get())
                done, _ = await asyncio.wait(
                    {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_task not in done:
                    get_task.cancel()
                    break
                msg = get_task.result()
                if msg is None:
                    break
                yield msg
        finally:
            recv_task.cancel()
            if session is not None:
                self.sessions.pop(session.node_id, None)
                session.close()
                await self.controller.on_disconnect(session)
                logger.info("worker %s disconnected", session.node_id)

    async def _recv_loop(
        self, request_iterator, holder: dict, observed_host: str = ""
    ) -> None:
        session: Optional[WorkerSession] = None
        async for msg in request_iterator:
            kind = msg.WhichOneof("msg")
            if kind == "register":
                reg = msg.register
                # Authenticate with a join key; its secret becomes this node's
                # command-signing key. Open registration only when no key/token
                # is configured at all (dev stacks).
                secret = self.keystore.validate(reg.join_key, node_id=reg.node_id)
                if secret is None and not self.keystore.open_registration():
                    logger.warning("rejecting %s: invalid join key", reg.node_id)
                    tmp = WorkerSession(reg.node_id, reg)
                    holder["session"] = tmp
                    holder["ready"].set()
                    await tmp.send(
                        gateway_pb2.ControlMessage(
                            register_ack=gateway_pb2.RegisterAck(
                                ok=False, error="invalid join key"
                            )
                        )
                    )
                    tmp.outbox.put_nowait(None)
                    return
                session = WorkerSession(
                    reg.node_id, reg, sign_key=secret or "", observed_host=observed_host
                )
                # Replace a stale session for the same node id, if any.
                old = self.sessions.pop(reg.node_id, None)
                if old is not None:
                    old.close()
                    # A session replaced seconds after it opened is not a
                    # reconnect — it is two different workers answering to one
                    # name, each evicting the other. The node then blinks in
                    # the table and serves nothing, which looks like a network
                    # fault and is not one.
                    previous = self._registered_at.get(reg.node_id)
                    if previous is not None and time.time() - previous < 30:
                        logger.warning(
                            "node %s registered again %.1fs after the last one: "
                            "two workers are probably sharing this node id. With "
                            "--network host they share a hostname too — give each "
                            "its own LOOM_NODE_ID",
                            reg.node_id,
                            time.time() - previous,
                        )
                self._registered_at[reg.node_id] = time.time()
                self.sessions[reg.node_id] = session
                holder["session"] = session
                holder["ready"].set()
                await session.send(
                    gateway_pb2.ControlMessage(
                        register_ack=gateway_pb2.RegisterAck(
                            ok=True,
                            node_id=reg.node_id,
                            # The one address this worker needs to reach every
                            # other worker. Empty when we run no p2p node, in
                            # which case it keeps relaying through us.
                            rendezvous=self.controller.rendezvous_addrs(),
                            relays=self.controller.relay_addrs(),
                        )
                    )
                )
                logger.info(
                    "worker %s registered: %s %s x%d, %.1f GB free, %.0f TFLOPs (%s)",
                    reg.node_id,
                    reg.hardware.device,
                    reg.hardware.gpu_name,
                    reg.hardware.num_gpus or 1,
                    reg.hardware.vram_free_bytes / 1024**3,
                    reg.hardware.tflops_fp16,
                    reg.hardware.detection_source,
                )
                # A controller failure must not silently kill the stream.
                try:
                    await self.controller.on_register(session)
                except Exception:
                    logger.exception("on_register failed for %s", reg.node_id)
            elif session is None:
                logger.warning("message %s before registration; closing", kind)
                return
            else:
                try:
                    if kind == "ack":
                        session.resolve_ack(msg.ack)
                        await self.controller.on_ack(session, msg.ack)
                    elif kind == "telemetry":
                        await self.controller.on_telemetry(session, msg.telemetry)
                    elif kind == "serving_endpoint":
                        ep = msg.serving_endpoint
                        session.endpoints[ep.model_id] = ep.local_port
                        await self.controller.on_endpoint(session, ep)
                except Exception:
                    logger.exception(
                        "handling %s from %s failed", kind, session.node_id
                    )


async def serve_gateway(servicer: ControlGatewayServicer, port: int) -> grpc.aio.Server:
    server = grpc.aio.server()
    gateway_pb2_grpc.add_ControlGatewayServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("ControlGateway listening on :%d", port)
    return server
