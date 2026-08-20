"""Data-plane tunnel client: relays orchestrator requests to the local backend.

The worker opens an outbound bidi stream and answers HttpRequest messages by
calling its own backend on 127.0.0.1 (the backend never listens publicly).
Responses are streamed back chunk by chunk, so SSE token streaming works.

No inbound connectivity, no port forwarding, no advertised address.
"""

from __future__ import annotations

import http.client
import logging
import queue
import threading
import time
from typing import Dict, Optional

import grpc

from loom_worker.proto import dataplane_pb2, dataplane_pb2_grpc
from loom_worker.stage_relay import (
    StageRelayServer,
    envelope_from_json,
    envelope_to_json,
    post_to_stage,
)
from loom_worker.state import ShardStatus, WorkerState

logger = logging.getLogger("loom_worker.dataplane")

_CLOSE = object()
CHUNK_SIZE = 16 * 1024


class DataPlaneClient:
    def __init__(
        self,
        *,
        orchestrator_addr: str,
        join_key: str,
        state: WorkerState,
        reconnect_delay_s: float = 3.0,
        max_message_mb: int = 64,
    ) -> None:
        self.orchestrator_addr = orchestrator_addr
        self.join_key = join_key
        self.state = state
        self.reconnect_delay_s = reconnect_delay_s
        self.max_message_mb = max_message_mb
        # Per-connection outbox: a stale generator from a dropped stream must
        # not steal messages belonging to the new one after a reconnect.
        self._outbox: Optional["queue.Queue[object]"] = None
        self._stop = threading.Event()
        self._cancelled: Dict[str, bool] = {}
        # Loopback relay: the stage subprocess posts outgoing inter-stage
        # messages here, and we push them into the tunnel.
        self.stage_relay = StageRelayServer(on_message=self._on_stage_outgoing)
        # Set by the agent once the p2p node is up. None means every message
        # goes through the orchestrator, which is the behaviour Loom had before
        # direct links existed and the behaviour it falls back to.
        self.links = None

    # ------------------------------------------------------------ stage relay
    def start_stage_relay(self) -> str:
        self.stage_relay.start()
        return self.stage_relay.url()

    def _shard_for_pipeline(self, pipeline_id: str, model_id: str = ""):
        """Find the local shard belonging to a pipeline (or the only one)."""
        shards = self.state.snapshot()
        if model_id and model_id in shards:
            return model_id, shards[model_id]
        for mid, shard in shards.items():
            if shard.spec.role.pipeline_id and shard.spec.role.pipeline_id == pipeline_id:
                return mid, shard
        if len(shards) == 1:
            mid = next(iter(shards))
            return mid, shards[mid]
        return None, None

    def _on_stage_outgoing(self, payload: dict) -> None:
        """Stage -> next stage, directly when possible and relayed when not.

        The stage process posted this to our loopback relay and is done with
        it; which of the two paths carries it is decided here and nowhere else.
        """
        # The stage process does not track pipeline ids, so they are recovered
        # from local state and STAMPED INTO the payload here. The relayed path
        # could recover them again on the way out, but a peer receiving this
        # directly has only what the message carries.
        self._stamp_ids(payload)
        target = int(payload.get("target_stage", -1))
        # A broadcast (target -1, used by FREE to release a finished request on
        # every stage) keeps going through the orchestrator on purpose: it is
        # tiny, rare, off the critical path, and the relay already fans it out
        # to the whole pipeline in one message.
        if self.links is not None and target >= 0:
            self.links.send(
                payload.get("pipeline_id", ""), target, payload, relay=self._relay_stage
            )
            return
        self._relay_stage(payload)

    def deliver_direct(self, payload: dict) -> None:
        """A message that arrived straight from a peer, bypassing the relay.

        Same destination as a relayed one: the local stage process. Arriving by
        a different road changes nothing about what happens next.
        """
        pipeline_id = payload.get("pipeline_id", "")
        model_id = payload.get("model_id", "")
        _, shard = self._shard_for_pipeline(pipeline_id, model_id)
        if shard is None or shard.backend is None:
            logger.warning(
                "direct stage message for unknown pipeline %s / model %s",
                pipeline_id,
                model_id,
            )
            return
        post_to_stage(shard.backend.port, payload)

    def _stamp_ids(self, payload: dict) -> None:
        """Fill in the pipeline this message belongs to, if the stage did not.

        A current stage always stamps its own ids (it is the only party that
        knows them for certain). This guess remains for a stage process from an
        older image, and it is only ever right when this node hosts a single
        multi-stage model — which is why the stage does it now instead.
        """
        if payload.get("pipeline_id") and payload.get("model_id"):
            return
        candidates = [
            (mid, shard.spec.role)
            for mid, shard in self.state.snapshot().items()
            if shard.spec.role.is_multi_stage
        ]
        if len(candidates) != 1:
            if candidates:
                logger.warning(
                    "a stage message arrived with no pipeline id and this node "
                    "hosts %d pipelines; it cannot be routed reliably",
                    len(candidates),
                )
            return
        model_id, role = candidates[0]
        payload.setdefault("pipeline_id", role.pipeline_id)
        payload.setdefault("model_id", model_id)

    def _relay_stage(self, payload: dict) -> None:
        """Stage -> tunnel: wrap the JSON message into a StageEnvelope."""
        self._stamp_ids(payload)
        pipeline_id = payload.get("pipeline_id", "")
        model_id = payload.get("model_id", "")
        from_stage = 0
        _, shard = self._shard_for_pipeline(pipeline_id, model_id)
        if shard is not None:
            from_stage = shard.spec.role.stage_index
        env = envelope_from_json(
            payload, pipeline_id=pipeline_id, model_id=model_id, from_stage=from_stage
        )
        self._send(dataplane_pb2.TunnelMessage(stage=env))

    def _on_stage_incoming(self, env) -> None:
        """Tunnel -> stage: deliver to the local stage server."""
        model_id, shard = self._shard_for_pipeline(env.pipeline_id, env.model_id)
        if shard is None or shard.backend is None:
            logger.warning(
                "stage message for unknown pipeline %s / model %s", env.pipeline_id, env.model_id
            )
            return
        post_to_stage(shard.backend.port, envelope_to_json(env))

    # -------------------------------------------------------------- lifecycle
    def stop(self) -> None:
        self._stop.set()
        outbox = self._outbox
        if outbox is not None:
            outbox.put(_CLOSE)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="dataplane", daemon=True)
        thread.start()
        return thread

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_once()
            except grpc.RpcError as exc:
                logger.warning("data-plane stream broken: %s; reconnecting", exc.code())
            except Exception:
                logger.exception("unexpected data-plane error; reconnecting")
            if not self._stop.is_set():
                time.sleep(self.reconnect_delay_s)

    def _request_iter(self, outbox: "queue.Queue[object]"):
        yield dataplane_pb2.TunnelMessage(
            hello=dataplane_pb2.TunnelHello(
                node_id=self.state.node_id, join_key=self.join_key
            )
        )
        while True:
            msg = outbox.get()
            if msg is _CLOSE:
                return
            yield msg

    def _run_once(self) -> None:
        limit = self.max_message_mb * 1024 * 1024
        options = [
            ("grpc.max_send_message_length", limit),
            ("grpc.max_receive_message_length", limit),
        ]
        outbox: "queue.Queue[object]" = queue.Queue()
        self._outbox = outbox
        try:
            self._serve_stream(outbox, options)
        finally:
            if self._outbox is outbox:
                self._outbox = None
            outbox.put(_CLOSE)  # release the generator if it is still blocked

    def _serve_stream(self, outbox, options) -> None:
        with grpc.insecure_channel(self.orchestrator_addr, options=options) as channel:
            stub = dataplane_pb2_grpc.DataPlaneStub(channel)
            stream = stub.Tunnel(self._request_iter(outbox))
            for msg in stream:
                kind = msg.WhichOneof("msg")
                if kind == "hello_ack":
                    if msg.hello_ack.ok:
                        logger.info("data-plane tunnel established")
                    else:
                        logger.error("data-plane rejected: %s", msg.hello_ack.error)
                        return
                elif kind == "request":
                    # One thread per request: a slow backend must not block others.
                    threading.Thread(
                        target=self._handle_request,
                        args=(msg.request,),
                        name=f"relay-{msg.request.request_id[:8]}",
                        daemon=True,
                    ).start()
                elif kind == "cancel":
                    self._cancelled[msg.cancel.request_id] = True
                elif kind == "stage":
                    # Inter-stage traffic: hand to the local stage process.
                    threading.Thread(
                        target=self._on_stage_incoming,
                        args=(msg.stage,),
                        name="stage-in",
                        daemon=True,
                    ).start()
                if self._stop.is_set():
                    return

    # ---------------------------------------------------------------- relaying
    def _local_port(self, model_id: str) -> Optional[int]:
        shard = self.state.get(model_id)
        if shard is None or shard.status != ShardStatus.SERVING or shard.backend is None:
            return None
        return shard.backend.port

    def _send(self, msg: dataplane_pb2.TunnelMessage) -> None:
        outbox = self._outbox
        if outbox is None:
            return  # stream is down; the orchestrator will retry the request
        outbox.put(msg)

    def _fail(self, request_id: str, error: str) -> None:
        self._send(
            dataplane_pb2.TunnelMessage(
                error=dataplane_pb2.HttpError(request_id=request_id, error=error)
            )
        )

    def _handle_request(self, req: dataplane_pb2.HttpRequest) -> None:
        rid = req.request_id
        port = self._local_port(req.model_id)
        if port is None:
            self._fail(rid, f"model {req.model_id} is not serving on this node")
            return
        conn = None
        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=req.timeout_s or 600
            )
            headers = dict(req.headers) or {"Content-Type": "application/json"}
            conn.request(req.method or "POST", req.path, body=req.body, headers=headers)
            resp = conn.getresponse()
            self._send(
                dataplane_pb2.TunnelMessage(
                    head=dataplane_pb2.HttpResponseHead(
                        request_id=rid,
                        status=resp.status,
                        headers={k: v for k, v in resp.getheaders()},
                    )
                )
            )
            while True:
                if self._cancelled.pop(rid, False):
                    logger.debug("request %s cancelled by orchestrator", rid[:8])
                    return
                # read1() hands over whatever has already arrived; plain
                # read(n) blocks until n bytes accumulate, which held every SSE
                # token back until the answer was complete — "streaming" then
                # delivered the whole reply in one burst at the end.
                data = (
                    resp.read1(CHUNK_SIZE)
                    if hasattr(resp, "read1")
                    else resp.read(CHUNK_SIZE)
                )
                if not data:
                    break
                self._send(
                    dataplane_pb2.TunnelMessage(
                        chunk=dataplane_pb2.HttpBodyChunk(request_id=rid, data=data)
                    )
                )
            self._send(dataplane_pb2.TunnelMessage(end=dataplane_pb2.HttpEnd(request_id=rid)))
        except Exception as exc:
            logger.warning("relay of %s failed: %s", rid[:8], exc)
            self._fail(rid, str(exc))
        finally:
            self._cancelled.pop(rid, None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
