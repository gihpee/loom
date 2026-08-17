"""Outbound control channel: the worker dials the orchestrator (hub-and-spoke).

One bidirectional gRPC stream carries everything:
- worker -> orchestrator: register, acks, telemetry/heartbeats
- orchestrator -> worker: WorkerControl commands (see gateway.proto)

The worker never accepts inbound connections; NAT/firewall-friendly.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import grpc

from loom_worker.handlers import CommandHandlers
from loom_worker.proto import gateway_pb2, gateway_pb2_grpc, worker_control_pb2
from loom_worker.security import CommandVerifier
from loom_worker.state import WorkerState

logger = logging.getLogger("loom_worker.gateway")

_CLOSE = object()


class GatewayClient:
    def __init__(
        self,
        *,
        orchestrator_addr: str,
        join_key: str,
        state: WorkerState,
        hardware: gateway_pb2.HardwareInfo,
        handlers: CommandHandlers,
        region: str = "default",
        verifier: Optional[CommandVerifier] = None,
        agent_version: str = "",
        heartbeat_interval_s: float = 5.0,
        reconnect_delay_s: float = 3.0,
    ) -> None:
        self.region = region
        self.verifier = verifier
        self.agent_version = agent_version
        self.orchestrator_addr = orchestrator_addr
        self.join_key = join_key
        self.state = state
        self.hardware = hardware
        self.handlers = handlers
        self.heartbeat_interval_s = heartbeat_interval_s
        self.reconnect_delay_s = reconnect_delay_s
        # Per-connection outbox: after a reconnect a stale generator must not
        # consume messages meant for the live stream.
        self._outbox: Optional["queue.Queue[object]"] = None
        self._stop = threading.Event()
        self._registered = threading.Event()

    # Used by handlers to push async messages (acks, endpoints).
    def send(self, msg: gateway_pb2.WorkerMessage) -> None:
        outbox = self._outbox
        if outbox is None:
            return  # not connected; the orchestrator re-issues commands on reconnect
        outbox.put(msg)

    def stop(self) -> None:
        self._stop.set()
        outbox = self._outbox
        if outbox is not None:
            outbox.put(_CLOSE)

    def run_forever(self) -> None:
        """Connect, serve the stream, reconnect on failure until stopped."""
        while not self._stop.is_set():
            try:
                self._run_once()
            except grpc.RpcError as exc:
                logger.warning("control stream broken: %s; reconnecting", exc)
            except Exception:
                logger.exception("unexpected control-channel error; reconnecting")
            self._registered.clear()
            if not self._stop.is_set():
                time.sleep(self.reconnect_delay_s)

    def wait_registered(self, timeout_s: float) -> bool:
        return self._registered.wait(timeout_s)

    def _request_iter(self, outbox: "queue.Queue[object]"):
        # First message on every (re)connect is registration.
        yield gateway_pb2.WorkerMessage(
            register=gateway_pb2.RegisterRequest(
                node_id=self.state.node_id,
                join_key=self.join_key,
                hardware=self.hardware,
                region=self.region,
                agent_version=self.agent_version,
            )
        )
        while True:
            msg = outbox.get()
            if msg is _CLOSE:
                return
            yield msg

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if self._registered.is_set():
                self.send(self.handlers.telemetry_report())
            time.sleep(self.heartbeat_interval_s)

    def _run_once(self) -> None:
        outbox: "queue.Queue[object]" = queue.Queue()
        self._outbox = outbox
        try:
            self._serve_stream(outbox)
        finally:
            if self._outbox is outbox:
                self._outbox = None
            outbox.put(_CLOSE)  # release the generator if it is still blocked

    def _serve_stream(self, outbox: "queue.Queue[object]") -> None:
        with grpc.insecure_channel(self.orchestrator_addr) as channel:
            stub = gateway_pb2_grpc.ControlGatewayStub(channel)
            stream = stub.Attach(self._request_iter(outbox))
            for control_msg in stream:
                self._dispatch(control_msg)
                if self._stop.is_set():
                    return

    def _dispatch(self, msg: gateway_pb2.ControlMessage) -> None:
        kind = msg.WhichOneof("cmd")
        logger.debug("control message: %s", kind)
        reply: Optional[gateway_pb2.WorkerMessage] = None
        # Everything except the registration ack is a signed command.
        if kind != "register_ack" and self.verifier is not None:
            ok, error = self.verifier.verify(msg)
            if not ok:
                logger.error("rejecting %s command: %s", kind, error)
                sub = getattr(msg, kind, None)
                command_id = (
                    sub.meta.command_id if sub is not None and sub.HasField("meta") else ""
                )
                self.send(
                    gateway_pb2.WorkerMessage(
                        ack=worker_control_pb2.Ack(
                            command_id=command_id, ok=False, error=f"rejected: {error}"
                        )
                    )
                )
                return
        if kind == "register_ack":
            if msg.register_ack.ok:
                self._registered.set()
                logger.info("registered with orchestrator as %s", self.state.node_id)
            else:
                logger.error("registration rejected: %s", msg.register_ack.error)
                self.stop()
        elif kind == "load_shard":
            reply = self.handlers.load_shard(msg.load_shard)
        elif kind == "start_serving":
            reply = self.handlers.start_serving(msg.start_serving)
        elif kind == "stop_serving":
            reply = self.handlers.stop_serving(msg.stop_serving)
        elif kind == "unload_shard":
            reply = self.handlers.unload_shard(msg.unload_shard)
        elif kind == "set_quota":
            reply = self.handlers.set_quota(msg.set_quota)
        elif kind == "report_telemetry":
            reply = self.handlers.telemetry_report()
        elif kind == "heartbeat_probe":
            reply = self.handlers.telemetry_report()
        else:
            logger.warning("unknown control message: %s", kind)
        if reply is not None:
            self.send(reply)

    def start_heartbeats(self) -> None:
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
