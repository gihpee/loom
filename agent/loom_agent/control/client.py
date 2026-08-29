"""One outbound stream, and everything that travels on it.

The node dials the orchestrator and keeps a single bidirectional gRPC stream
open. Nothing ever connects TO a node: it has no reachable address, opens no
port, and needs nothing configured on the owner's router. Commands arrive on
the same connection the node opened.

Two flows share the stream and neither blocks the other:

  outgoing   a generator gRPC pulls from. Its first item is always the
             registration, so a reconnect re-announces the node without anyone
             asking. Later messages are handed to it through a queue.
  incoming   a plain loop over the stream.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import grpc

from loom_agent.proto import agent_pb2, agent_pb2_grpc

logger = logging.getLogger("loom_agent.control")

_CLOSE = object()

# A dead stream and an idle one look identical from inside a `for` loop: both
# are silence. Keepalive pings are the only thing that tells them apart —
# without them the agent blocks forever, never raises, never reconnects, and
# the node just disappears from the orchestrator. Most visibly after an
# orchestrator restart, where no clean close ever reaches us.
CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 20000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    # Come back quickly when the orchestrator returns, and never let the
    # backoff grow into minutes of an idle node.
    ("grpc.initial_reconnect_backoff_ms", 500),
    ("grpc.max_reconnect_backoff_ms", 5000),
]


class ControlClient:
    def __init__(
        self,
        *,
        address: str,
        register_message: Callable[[], agent_pb2.Register],
        on_message: Callable[[agent_pb2.ServerMessage], None],
        on_registered: Optional[Callable[[agent_pb2.RegisterAck], None]] = None,
        reconnect_delay_s: float = 3.0,
    ) -> None:
        self.address = address
        self.register_message = register_message
        self.on_message = on_message
        self.on_registered = on_registered
        self.reconnect_delay_s = reconnect_delay_s
        # Rebuilt per connection: after a reconnect a stale generator must not
        # take messages meant for the live stream.
        self._outbox: Optional["queue.Queue[object]"] = None
        self._stop = threading.Event()
        self._registered = threading.Event()

    # ---------------------------------------------------------------- public
    def send(self, msg: agent_pb2.AgentMessage) -> None:
        """Queue a message for the orchestrator. Silently dropped when offline.

        Dropping is correct: the orchestrator re-issues commands after a
        reconnect, so a queued ack for a stream that no longer exists would be
        answering a question nobody is still asking.
        """
        outbox = self._outbox
        if outbox is not None:
            outbox.put(msg)

    def wait_registered(self, timeout_s: float) -> bool:
        return self._registered.wait(timeout_s)

    @property
    def registered(self) -> bool:
        return self._registered.is_set()

    def stop(self) -> None:
        self._stop.set()
        outbox = self._outbox
        if outbox is not None:
            outbox.put(_CLOSE)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self._serve()
            except grpc.RpcError as exc:
                logger.warning("control stream broken: %s; reconnecting", exc)
            except Exception:
                logger.exception("unexpected control-channel error; reconnecting")
            self._registered.clear()
            if not self._stop.is_set():
                time.sleep(self.reconnect_delay_s)

    # --------------------------------------------------------------- private
    def _serve(self) -> None:
        outbox: "queue.Queue[object]" = queue.Queue()
        self._outbox = outbox
        try:
            with grpc.insecure_channel(self.address, options=CHANNEL_OPTIONS) as channel:
                stub = agent_pb2_grpc.AgentGatewayStub(channel)
                for message in stub.Attach(self._outgoing(outbox)):
                    self._receive(message)
                    if self._stop.is_set():
                        return
        finally:
            if self._outbox is outbox:
                self._outbox = None
            outbox.put(_CLOSE)  # free the generator if it is still blocked

    def _outgoing(self, outbox: "queue.Queue[object]"):
        yield agent_pb2.AgentMessage(register=self.register_message())
        while True:
            msg = outbox.get()
            if msg is _CLOSE:
                return
            yield msg

    def _receive(self, message: agent_pb2.ServerMessage) -> None:
        if message.WhichOneof("msg") == "register_ack":
            ack = message.register_ack
            if not ack.ok:
                logger.error("orchestrator refused this node: %s", ack.error)
                return
            self._registered.set()
            logger.info("registered as %s", ack.node_id)
            if self.on_registered is not None:
                self.on_registered(ack)
            return
        self.on_message(message)
