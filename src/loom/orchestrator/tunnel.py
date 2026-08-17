"""Data-plane tunnel hub: inference over the worker's outbound stream.

The worker opens a second bidi stream (DataPlane.Tunnel) and keeps it open. To
serve a request the orchestrator pushes an HttpRequest into that stream; the
worker relays it to its local backend and streams the response back as chunks.

Consequence: a GPU owner needs no public address, no port forwarding and no
advertise host — exactly like a private node joining Parallax over relayed P2P,
expressed in our hub-and-spoke model.

Concurrency: one stream carries many requests, correlated by request_id. Each
in-flight request has its own asyncio.Queue of events.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Tuple

import grpc

from loom.logging_config import get_logger
from loom.proto_gen import dataplane_pb2, dataplane_pb2_grpc

logger = get_logger(__name__)

# Sentinel pushed into a request queue when the stream dies mid-flight.
_BROKEN = object()


@dataclass
class ResponseHead:
    status: int
    headers: Dict[str, str]


class TunnelError(RuntimeError):
    """Raised when the tunnel cannot deliver a request/response."""


class TunnelSession:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.outbox: asyncio.Queue[Optional[dataplane_pb2.TunnelMessage]] = asyncio.Queue()
        self.pending: Dict[str, asyncio.Queue] = {}

    async def send(self, msg: dataplane_pb2.TunnelMessage) -> None:
        await self.outbox.put(msg)

    def close(self) -> None:
        for queue in self.pending.values():
            queue.put_nowait(_BROKEN)
        self.pending.clear()
        self.outbox.put_nowait(None)


class TunnelHub:
    """Registry of live tunnels + request/response plumbing.

    Also routes inter-stage traffic for models split across nodes: a stage
    sends a StageEnvelope up its own tunnel, the hub looks up which node owns
    the target stage of that pipeline and pushes it down that node's tunnel.
    Workers therefore never need to reach each other.
    """

    def __init__(self) -> None:
        self.sessions: Dict[str, TunnelSession] = {}
        # (pipeline_id, stage_index) -> node_id
        self.stage_routes: Dict[Tuple[str, int], str] = {}

    def is_connected(self, node_id: str) -> bool:
        return node_id in self.sessions

    def attach(self, node_id: str) -> TunnelSession:
        old = self.sessions.pop(node_id, None)
        if old is not None:
            old.close()
        session = TunnelSession(node_id)
        self.sessions[node_id] = session
        return session

    def detach(self, session: TunnelSession) -> None:
        if self.sessions.get(session.node_id) is session:
            self.sessions.pop(session.node_id, None)
        session.close()

    # ------------------------------------------------------- pipeline routing
    def register_stage_routes(self, pipeline_id: str, stages: Dict[int, str]) -> None:
        """Publish which node serves each stage of a pipeline."""
        for stage_index, node_id in stages.items():
            self.stage_routes[(pipeline_id, stage_index)] = node_id

    def clear_stage_routes(self, pipeline_id: str) -> None:
        for key in [k for k in self.stage_routes if k[0] == pipeline_id]:
            self.stage_routes.pop(key, None)

    def pipeline_stages(self, pipeline_id: str) -> Dict[int, str]:
        return {
            stage: node
            for (pid, stage), node in self.stage_routes.items()
            if pid == pipeline_id
        }

    async def route_stage(self, env: dataplane_pb2.StageEnvelope) -> None:
        """Forward an inter-stage message to the node owning the target stage.

        `target_stage == -1` means broadcast (used by FREE to release the KV
        state of a finished request on every stage of the pipeline).
        """
        stages = self.pipeline_stages(env.pipeline_id)
        if not stages:
            logger.warning("no stage routes for pipeline %s", env.pipeline_id)
            return
        targets = (
            list(stages.values())
            if env.target_stage < 0
            else [stages.get(env.target_stage)]
        )
        for node_id in targets:
            if node_id is None:
                logger.warning(
                    "pipeline %s has no node for stage %d", env.pipeline_id, env.target_stage
                )
                continue
            session = self.sessions.get(node_id)
            if session is None:
                logger.warning("stage %s@%s has no tunnel", env.target_stage, node_id)
                continue
            await session.send(dataplane_pb2.TunnelMessage(stage=env))

    def dispatch(self, msg: dataplane_pb2.TunnelMessage, session: TunnelSession) -> None:
        """Route a worker->orchestrator message to its request queue."""
        kind = msg.WhichOneof("msg")
        if kind == "stage":
            # Inter-stage hop: forward to the target stage's node.
            asyncio.create_task(self.route_stage(msg.stage))
            return
        request_id = getattr(getattr(msg, kind), "request_id", "") if kind else ""
        queue = session.pending.get(request_id)
        if queue is None:
            return  # request already finished/cancelled
        queue.put_nowait(msg)

    async def request(
        self,
        node_id: str,
        *,
        model_id: str,
        path: str,
        body: bytes,
        stream: bool,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: int = 600,
    ) -> Tuple[ResponseHead, AsyncIterator[bytes]]:
        """Send an HTTP request through the tunnel.

        Returns the response head and an async iterator of body chunks. The
        iterator must be consumed (or closed) to release the request slot.
        """
        session = self.sessions.get(node_id)
        if session is None:
            raise TunnelError(f"node {node_id} has no data-plane tunnel")

        request_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        session.pending[request_id] = queue
        await session.send(
            dataplane_pb2.TunnelMessage(
                request=dataplane_pb2.HttpRequest(
                    request_id=request_id,
                    model_id=model_id,
                    method="POST",
                    path=path,
                    body=body,
                    headers=headers or {"Content-Type": "application/json"},
                    stream=stream,
                    timeout_s=timeout_s,
                )
            )
        )

        # Wait for the head (or an error) before returning to the caller.
        head: Optional[ResponseHead] = None
        while head is None:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                session.pending.pop(request_id, None)
                raise TunnelError("tunnel request timed out waiting for response head")
            if item is _BROKEN:
                session.pending.pop(request_id, None)
                raise TunnelError("tunnel closed while waiting for response")
            kind = item.WhichOneof("msg")
            if kind == "head":
                head = ResponseHead(status=item.head.status, headers=dict(item.head.headers))
            elif kind == "error":
                session.pending.pop(request_id, None)
                raise TunnelError(item.error.error)
            elif kind == "end":
                session.pending.pop(request_id, None)
                raise TunnelError("tunnel returned no response head")

        async def chunks() -> AsyncIterator[bytes]:
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        raise TunnelError("tunnel request timed out mid-body")
                    if item is _BROKEN:
                        raise TunnelError("tunnel closed mid-response")
                    kind = item.WhichOneof("msg")
                    if kind == "chunk":
                        yield item.chunk.data
                    elif kind == "end":
                        return
                    elif kind == "error":
                        raise TunnelError(item.error.error)
            finally:
                session.pending.pop(request_id, None)
                # Best-effort cancel so the worker stops relaying.
                if session.node_id in self.sessions:
                    await session.send(
                        dataplane_pb2.TunnelMessage(
                            cancel=dataplane_pb2.HttpCancel(request_id=request_id)
                        )
                    )

        return head, chunks()

    async def request_bytes(self, node_id: str, **kwargs) -> Tuple[ResponseHead, bytes]:
        """Convenience wrapper: collect the whole body."""
        head, stream = await self.request(node_id, stream=False, **kwargs)
        buf = bytearray()
        async for chunk in stream:
            buf.extend(chunk)
        return head, bytes(buf)


class DataPlaneServicer(dataplane_pb2_grpc.DataPlaneServicer):
    def __init__(self, hub: TunnelHub, keystore) -> None:
        self.hub = hub
        self.keystore = keystore

    async def Tunnel(self, request_iterator, context):
        holder: dict = {"session": None, "ready": asyncio.Event()}
        recv_task = asyncio.create_task(self._recv_loop(request_iterator, holder))
        session: Optional[TunnelSession] = None
        try:
            ready_task = asyncio.create_task(holder["ready"].wait())
            done, _ = await asyncio.wait(
                {recv_task, ready_task}, return_when=asyncio.FIRST_COMPLETED
            )
            ready_task.cancel()
            session = holder["session"]
            if session is None:
                if recv_task in done:
                    recv_task.result()
                return
            # Pump the outbox, but stop as soon as the worker's half of the
            # stream ends — otherwise a half-closed connection would look like
            # a healthy tunnel and requests routed into it would hang.
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
                self.hub.detach(session)
                logger.info("data-plane tunnel closed for %s", session.node_id)

    async def _recv_loop(self, request_iterator, holder: dict) -> None:
        session: Optional[TunnelSession] = None
        async for msg in request_iterator:
            kind = msg.WhichOneof("msg")
            if kind == "hello":
                hello = msg.hello
                accepted = self.keystore.open_registration() or bool(
                    self.keystore.validate(hello.join_key, node_id=hello.node_id)
                )
                if not accepted:
                    tmp = TunnelSession(hello.node_id)
                    holder["session"] = tmp
                    holder["ready"].set()
                    await tmp.send(
                        dataplane_pb2.TunnelMessage(
                            hello_ack=dataplane_pb2.TunnelHelloAck(
                                ok=False, error="invalid join key"
                            )
                        )
                    )
                    tmp.outbox.put_nowait(None)
                    return
                session = self.hub.attach(hello.node_id)
                holder["session"] = session
                holder["ready"].set()
                await session.send(
                    dataplane_pb2.TunnelMessage(
                        hello_ack=dataplane_pb2.TunnelHelloAck(ok=True)
                    )
                )
                logger.info("data-plane tunnel open for %s", hello.node_id)
            elif session is None:
                return  # traffic before hello
            else:
                self.hub.dispatch(msg, session)


def add_dataplane_to_server(server: grpc.aio.Server, servicer: DataPlaneServicer) -> None:
    dataplane_pb2_grpc.add_DataPlaneServicer_to_server(servicer, server)
