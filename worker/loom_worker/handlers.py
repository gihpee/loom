"""Command handlers: pure executors of orchestrator commands.

Each handler returns an Ack (and optionally extra WorkerMessages to send).
No placement/routing decisions are made here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from typing import Callable, List, Optional

from loom_worker.backends import make_backend
from loom_worker.proto import gateway_pb2, worker_control_pb2
from loom_worker.state import PipelineRole, ShardSpec, ShardState, ShardStatus, WorkerState
from loom_worker.watchdog import QuotaWatchdog

logger = logging.getLogger("loom_worker.handlers")


class CommandHandlers:
    def __init__(
        self,
        state: WorkerState,
        *,
        send: Callable[[gateway_pb2.WorkerMessage], None],
        backend_kwargs: Optional[dict] = None,
        device: str = "cpu",
        watchdog_poll_s: float = 2.0,
        relay_url: str = "",
        rss_overhead_bytes: Optional[int] = None,
        vram_overhead_bytes: Optional[int] = None,
        ready_timeout_s: Optional[float] = None,
    ) -> None:
        self.state = state
        self.send = send
        self.backend_kwargs = backend_kwargs or {}
        self.device = device
        # Loopback URL the stage subprocess posts inter-stage messages to.
        self.relay_url = relay_url
        # Host-memory headroom for the runtime when enforcing an RSS limit.
        self.rss_overhead_bytes = (
            rss_overhead_bytes
            if rss_overhead_bytes is not None
            else int(float(os.environ.get("LOOM_RSS_OVERHEAD_MB", "2048")) * 1024 * 1024)
        )
        # CUDA context + kernels on top of the quota (see QuotaWatchdog).
        self.vram_overhead_bytes = (
            vram_overhead_bytes
            if vram_overhead_bytes is not None
            else int(float(os.environ.get("LOOM_VRAM_OVERHEAD_MB", "1024")) * 1024 * 1024)
        )
        # How long a backend may take to answer /health. None = the adapter's
        # own default (LOOM_BACKEND_READY_TIMEOUT_S); tests pass seconds.
        self.ready_timeout_s = ready_timeout_s
        self.watchdog_poll_s = watchdog_poll_s
        self._watchdogs: dict[str, QuotaWatchdog] = {}

    # --- helpers -----------------------------------------------------------
    def _ack(self, command_id: str, ok: bool, error: str = "") -> gateway_pb2.WorkerMessage:
        return gateway_pb2.WorkerMessage(
            ack=worker_control_pb2.Ack(command_id=command_id, ok=ok, error=error)
        )

    def _is_live(self, shard: ShardState) -> bool:
        """Is this shard really up, not just recorded as up?

        Idempotency must be checked against the process, not the bookkeeping: a
        backend killed behind our back (watchdog, OOM killer, crash) leaves the
        status saying SERVING, and answering "already serving" to the next
        command would advertise an endpoint with nothing behind it.
        """
        if shard.backend is None:
            return False
        if shard.status == ShardStatus.LOADED:
            return True  # nothing spawned yet, by design
        if shard.status == ShardStatus.STARTING:
            # A start is in flight: between claiming the shard and spawning the
            # process there is a moment with no pid, and calling that "dead"
            # would let a retry launch a second engine — the exact failure this
            # whole guard exists to prevent.
            return True
        if shard.status != ShardStatus.SERVING:
            return False
        if shard.backend.is_running():
            return True
        logger.warning(
            "shard %s is recorded as %s but its process is gone; treating it as failed",
            shard.spec.model_id,
            shard.status.value,
        )
        return False

    # --- command handlers --------------------------------------------------
    def load_shard(self, req: worker_control_pb2.LoadShardRequest) -> gateway_pb2.WorkerMessage:
        command_id = req.meta.command_id
        topo = req.topology
        role = PipelineRole(
            pipeline_id=topo.pipeline_id,
            stage_index=topo.stage_index,
            num_stages=max(1, topo.num_stages),
            is_first=topo.is_first if topo.num_stages else True,
            is_last=topo.is_last if topo.num_stages else True,
            num_model_layers=int(topo.num_model_layers or 0),
        )
        spec = ShardSpec(
            model_id=req.model_id,
            start_layer=req.start_layer,
            end_layer=req.end_layer,
            backend_type=req.backend_type,
            weights_uri=req.weights_uri,
            vram_quota_bytes=req.vram_quota_bytes,
            role=role,
        )
        logger.info(
            "LoadShard %s layers [%d,%d) backend=%s quota=%.1fGB stage %d/%d",
            req.model_id,
            req.start_layer,
            req.end_layer,
            req.backend_type,
            req.vram_quota_bytes / (1024**3),
            role.stage_index,
            role.num_stages,
        )
        existing = self.state.get(req.model_id)
        if existing is not None and self._is_live(existing):
            # Idempotent: the same shard is already here. Never rebuild the
            # backend under a running process — that would orphan it.
            logger.info(
                "LoadShard %s: already %s, nothing to do", req.model_id, existing.status.value
            )
            return self._ack(command_id, True)
        if existing is not None:
            # Replacing a FAILED/STOPPED shard: release its leftovers first.
            self._teardown(req.model_id, existing, to_status=ShardStatus.STOPPED)
        try:
            backend_kwargs = dict(self.backend_kwargs.get(spec.backend_type, {}))
            if spec.backend_type in ("shard", "vllm_shard"):
                # The stage needs to know its place in the pipeline and where to
                # hand off activations (the agent's loopback relay).
                backend_kwargs.setdefault("device", self.device)
                backend_kwargs["topology"] = {
                    "pipeline_id": role.pipeline_id,
                    "stage_index": role.stage_index,
                    "num_stages": role.num_stages,
                    "is_first": role.is_first,
                    "is_last": role.is_last,
                    "num_model_layers": role.num_model_layers,
                }
                backend_kwargs["relay_url"] = self.relay_url or ""
            backend = make_backend(
                spec.backend_type,
                model_id=spec.model_id,
                weights_uri=spec.weights_uri,
                start_layer=spec.start_layer,
                end_layer=spec.end_layer,
                vram_quota_bytes=spec.vram_quota_bytes,
                **backend_kwargs,
            )
            shard = ShardState(spec=spec, backend=backend, status=ShardStatus.LOADING)
            self.state.put(spec.model_id, shard)
            backend.prepare()
            shard.status = ShardStatus.LOADED
            return self._ack(command_id, True)
        except Exception as exc:  # report, never crash the agent
            logger.exception("LoadShard failed for %s", req.model_id)
            self.state.put(
                spec.model_id,
                ShardState(spec=spec, status=ShardStatus.FAILED, error=str(exc)),
            )
            return self._ack(command_id, False, str(exc))

    def start_serving(self, req: worker_control_pb2.ModelRequest) -> gateway_pb2.WorkerMessage:
        command_id = req.meta.command_id
        shard = self.state.get(req.model_id)
        if shard is None or shard.backend is None:
            return self._ack(command_id, False, f"model {req.model_id} is not loaded")
        if not self._is_live(shard) and shard.status in (
            ShardStatus.STARTING,
            ShardStatus.SERVING,
        ):
            # Recorded as up, but the process died. Fall through and start it
            # again instead of re-announcing a dead endpoint.
            shard.status = ShardStatus.LOADED
            shard.endpoint_url = None
        if shard.status == ShardStatus.STARTING:
            # A start is already in flight. Launching another engine here would
            # orphan the first one — it keeps the GPU while nobody holds its
            # handle, and every later attempt then dies with OOM.
            logger.info(
                "StartServing %s: start already in progress on port %d; ignoring",
                req.model_id,
                shard.backend.port,
            )
            return self._ack(command_id, True)
        if shard.status == ShardStatus.SERVING:
            # Idempotent — but re-announce the endpoint: the orchestrator may
            # have restarted and lost its endpoint table.
            self.send(
                gateway_pb2.WorkerMessage(
                    serving_endpoint=gateway_pb2.ServingEndpoint(
                        model_id=req.model_id,
                        local_port=shard.backend.port,
                        command_id=command_id,
                    )
                )
            )
            return self._ack(command_id, True)

        def _serve() -> None:
            backend = shard.backend
            try:
                logger.info(
                    "StartServing %s: launching %s backend on port %d",
                    req.model_id,
                    shard.spec.backend_type,
                    backend.port,
                )
                backend.start()
                watchdog = QuotaWatchdog(
                    get_pid=backend.pid,
                    quota_bytes=shard.spec.vram_quota_bytes,
                    on_kill=lambda reason: self._on_watchdog_kill(req.model_id, reason),
                    device=self.device,
                    poll_interval_s=self.watchdog_poll_s,
                    rss_overhead_bytes=self.rss_overhead_bytes,
                    vram_overhead_bytes=self.vram_overhead_bytes,
                )
                watchdog.start()
                self._watchdogs[req.model_id] = watchdog
                healthy = backend.wait_healthy(timeout_s=self.ready_timeout_s)
                if shard.status != ShardStatus.STARTING:
                    # Stopped, unloaded or watchdog-killed while we were
                    # starting: that verdict stands, but the orchestrator is
                    # still waiting on this command — never leave it hanging.
                    reason = shard.error or f"start aborted ({shard.status.value})"
                    logger.info("StartServing %s: %s", req.model_id, reason)
                    self.send(self._ack(command_id, False, reason))
                    return
                if not healthy:
                    # Release the process before reporting failure: a backend
                    # that is stuck (or still downloading) holds the whole VRAM
                    # quota, and the re-placement would hit an OOM cascade.
                    error = "backend failed health check"
                    self._teardown(req.model_id, shard, to_status=ShardStatus.FAILED)
                    shard.error = error
                    self.send(self._ack(command_id, False, error))
                    return
                shard.status = ShardStatus.SERVING
                # Loopback only: the orchestrator reaches the backend through
                # the data-plane tunnel, so no host is advertised.
                shard.endpoint_url = f"http://127.0.0.1:{backend.port}"
                self.send(self._ack(command_id, True))
                self.send(
                    gateway_pb2.WorkerMessage(
                        serving_endpoint=gateway_pb2.ServingEndpoint(
                            model_id=req.model_id,
                            local_port=backend.port,
                            command_id=command_id,
                        )
                    )
                )
            except Exception as exc:
                logger.exception("StartServing failed for %s", req.model_id)
                self._teardown(req.model_id, shard, to_status=ShardStatus.FAILED)
                shard.error = str(exc)
                self.send(self._ack(command_id, False, str(exc)))

        # Claim the shard before the thread exists: the next StartServing (a
        # retry, a rebalance race) must see STARTING, not LOADED.
        shard.status = ShardStatus.STARTING
        # Serve-start can be slow (model load); run it off the dispatch thread.
        threading.Thread(target=_serve, name=f"serve-{req.model_id}", daemon=True).start()
        return None  # ack is sent asynchronously by _serve

    def stop_serving(self, req: worker_control_pb2.ModelRequest) -> gateway_pb2.WorkerMessage:
        command_id = req.meta.command_id
        shard = self.state.get(req.model_id)
        if shard is None:
            return self._ack(command_id, True)  # idempotent
        self._teardown(req.model_id, shard, to_status=ShardStatus.STOPPED)
        return self._ack(command_id, True)

    def unload_shard(self, req: worker_control_pb2.UnloadShardRequest) -> gateway_pb2.WorkerMessage:
        command_id = req.meta.command_id
        shard = self.state.pop(req.model_id)
        if shard is not None:
            self._teardown(req.model_id, shard, to_status=ShardStatus.STOPPED)
        return self._ack(command_id, True)

    def set_quota(self, req: worker_control_pb2.QuotaRequest) -> gateway_pb2.WorkerMessage:
        command_id = req.meta.command_id
        shard = self.state.get(req.model_id)
        if shard is None:
            return self._ack(command_id, False, f"model {req.model_id} is not loaded")
        shard.spec.vram_quota_bytes = req.vram_quota_bytes
        watchdog = self._watchdogs.get(req.model_id)
        if watchdog is not None:
            watchdog.set_quota(req.vram_quota_bytes)
        return self._ack(command_id, True)

    def _stage_stats(self, shard) -> dict:
        """Ask the stage what it has measured about itself.

        Cheap (loopback, one small GET) and best-effort: telemetry must never
        be the thing that breaks, so a stage that does not answer simply
        reports nothing and the planner keeps its own estimate.
        """
        backend = shard.backend
        if backend is None or shard.status != ShardStatus.SERVING:
            return {}
        try:
            with urllib.request.urlopen(
                backend.local_url() + backend.health_path(), timeout=1.0
            ) as response:
                return json.loads(response.read() or b"{}") or {}
        except Exception:
            return {}

    def telemetry_report(self) -> gateway_pb2.WorkerMessage:
        shards = []
        for model_id, shard in self.state.snapshot().items():
            port = shard.backend.port if shard.backend is not None else 0
            stats = self._stage_stats(shard)
            shards.append(
                worker_control_pb2.ShardTelemetry(
                    model_id=model_id,
                    start_layer=shard.spec.start_layer,
                    end_layer=shard.spec.end_layer,
                    # What this node actually costs per layer, measured on the
                    # real model. The scheduler splits layers in proportion to
                    # node speed, and without this it has only a spec table —
                    # which does not know every card and cannot know that a
                    # given one is throttled, shared or on a slow link.
                    avg_layer_latency_ms=float(stats.get("layer_latency_ms") or 0.0),
                    current_requests=int(stats.get("active_requests") or 0),
                    healthy=shard.status == ShardStatus.SERVING,
                    status=shard.status.value,
                    local_port=port if shard.status == ShardStatus.SERVING else 0,
                    pipeline_id=shard.spec.role.pipeline_id,
                    stage_index=shard.spec.role.stage_index,
                    num_stages=shard.spec.role.num_stages,
                )
            )
        return gateway_pb2.WorkerMessage(
            telemetry=worker_control_pb2.TelemetryReport(
                node_id=self.state.node_id, shards=shards
            )
        )

    # --- internals ---------------------------------------------------------
    def _teardown(self, model_id: str, shard: ShardState, *, to_status: ShardStatus) -> None:
        watchdog = self._watchdogs.pop(model_id, None)
        if watchdog is not None:
            watchdog.stop()
        if shard.backend is not None:
            try:
                shard.backend.stop()
            except Exception:
                logger.exception("backend stop failed for %s", model_id)
        shard.status = to_status
        shard.endpoint_url = None

    def _on_watchdog_kill(self, model_id: str, reason: str) -> None:
        logger.error("watchdog: %s", reason)
        shard = self.state.get(model_id)
        if shard is not None:
            shard.status = ShardStatus.FAILED
            shard.error = reason
            shard.endpoint_url = None
