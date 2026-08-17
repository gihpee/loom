"""Multi-model controller (Phase 2).

Flow:
- worker registers            -> node pool entry -> Resource Broker pass
- broker plan                 -> per-model Scheduler (Phase-1 on granted
                                 sub-pool) -> diff -> worker commands
- telemetry                   -> Perf-map Store (model_id-namespaced) -> Phase-2
- model added/removed (admin) -> broker pass (high-priority models evict
                                 lower-score ones when VRAM is short)
- periodic timer              -> broker pass (safety net)

Deployment diff semantics v0: any change in (quota, layer range) for a
(model, node) is a full reload (Stop/Unload + Load/Start). SetQuota-without-
restart is kept for Phase 3. Removals are awaited BEFORE additions are issued
so a node's VRAM is never oversubscribed.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from loom.api.endpoints import Endpoint, EndpointRegistry
from loom.logging_config import get_logger
from loom.orchestrator.broker import BrokerPlan, PoolNode, ResourceBroker
from loom.orchestrator.config import OrchestratorConfig
from loom.orchestrator.gateway import WorkerSession, new_meta
from loom.orchestrator.pool import ModelInstance, NodeDescriptor, SchedulerPool
from loom.orchestrator.registry import ModelSpec
from loom.perfmap import InMemoryPerfMapStore, PerfMapStore, ShardPerf, sync_perfmap_to_scheduler
from loom.orchestrator.tunnel import TunnelHub
from loom.planning import NodeHardwareInfo
from loom.proto_gen import gateway_pb2, worker_control_pb2

logger = get_logger(__name__)

GIB = 1024**3

# (model_id, node_id) -> (vram_quota, start_layer, end_layer, pipeline_id, stage_index, num_stages)
Deployment = Dict[Tuple[str, str], Tuple[int, int, int, str, int, int]]


def build_pipelines(
    allocation: List[Tuple[str, int, int]], num_model_layers: int
) -> List[List[Tuple[str, int, int]]]:
    """Group a Phase-1 allocation into ordered pipelines (stage chains).

    Phase-1 returns per-node layer ranges; several nodes whose ranges tile
    `[0, L)` form one pipeline (one replica of the model). Nodes are ordered by
    start layer, and a new pipeline begins whenever a range starts at 0.
    """
    pipelines: List[List[Tuple[str, int, int]]] = []
    current: List[Tuple[str, int, int]] = []
    for node_id, start, end in sorted(allocation, key=lambda a: (a[1], a[2], a[0])):
        if start == 0 and current:
            pipelines.append(current)
            current = []
        if current and current[-1][2] != start:
            # Gap or overlap: this range cannot continue the chain.
            pipelines.append(current)
            current = []
        current.append((node_id, start, end))
        if end >= num_model_layers:
            pipelines.append(current)
            current = []
    if current:
        pipelines.append(current)
    # Keep only chains that actually cover the whole model.
    return [
        p
        for p in pipelines
        if p and p[0][1] == 0 and p[-1][2] >= num_model_layers
    ]


class MultiModelController:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        store: Optional[PerfMapStore] = None,
        tunnel: Optional[TunnelHub] = None,
    ) -> None:
        self.config = config
        self.registry = config.registry
        self.store = store or InMemoryPerfMapStore(ttl_seconds=config.heartbeat_timeout_s * 2)
        # Data-plane hub: inference is relayed over workers' outbound streams.
        self.tunnel = tunnel or TunnelHub()
        self.broker = ResourceBroker(
            qps_per_pipeline=config.qps_per_pipeline,
            param_mem_ratio=config.param_mem_ratio,
        )
        self.pool = SchedulerPool(
            param_mem_ratio=config.param_mem_ratio,
            kvcache_mem_ratio=config.kvcache_mem_ratio,
        )
        self.endpoints = EndpointRegistry()
        self.nodes: Dict[str, NodeDescriptor] = {}
        self.sessions: Dict[str, WorkerSession] = {}
        self.deployed: Deployment = {}
        # (model, node) -> when the current deploy attempt was issued. A worker
        # heartbeat still describing the PREVIOUS attempt must not be mistaken
        # for a fresh failure — that turns one bad start into a launch storm.
        self.deploy_started: Dict[Tuple[str, str], float] = {}
        # (model, node) -> (when it failed, why). Re-placement waits out a
        # cooldown so a shard that cannot start (bad model, too little VRAM) is
        # retried on a human timescale instead of every rebalance pass.
        self.deploy_failures: Dict[Tuple[str, str], Tuple[float, str]] = {}
        self.last_plan: Optional[BrokerPlan] = None
        self._lock = asyncio.Lock()
        # SLO tracking: model_id -> deque[(ts, ttft_ms, error)]
        self._slo_samples: Dict[str, Deque[Tuple[float, float, bool]]] = {}
        self.slo_boosts: Dict[str, float] = {}
        # Read-only observability state for the admin UI (no control logic).
        self.node_last_seen: Dict[str, float] = {}
        self.shard_status: Dict[Tuple[str, str], Tuple[str, float]] = {}  # -> (status, ts)
        # model_id -> [pipeline_id, ...] (one per replica)
        self.model_pipelines: Dict[str, List[str]] = {}
        # Set by the server once the dial address is resolved (admin UI shows it).
        self.public_address = None

    # ------------------------------------------------------------- gateway events
    async def on_register(self, session: WorkerSession) -> None:
        reg = session.register
        hw = reg.hardware
        self.nodes[session.node_id] = NodeDescriptor(
            node_id=session.node_id,
            region=reg.region or "default",
            hardware=NodeHardwareInfo(
                node_id=session.node_id,
                num_gpus=hw.num_gpus or 1,
                tflops_fp16=hw.tflops_fp16 or 1.0,
                gpu_name=hw.gpu_name,
                memory_gb=hw.memory_gb,
                memory_bandwidth_gbps=hw.memory_bandwidth_gbps or 1.0,
                device=hw.device or "cpu",
            ),
            vram_free_bytes=int(hw.vram_free_bytes or hw.memory_gb * GIB),
            vram_total_bytes=int(hw.vram_total_bytes or hw.vram_free_bytes or hw.memory_gb * GIB),
            detection_source=hw.detection_source,
            agent_version=reg.agent_version,
        )
        self.sessions[session.node_id] = session
        # A re-registering worker may be a restarted container with nothing
        # loaded, so drop our bookkeeping and let it be re-placed; if it is the
        # same agent with shards still running, its LoadShard/StartServing are
        # idempotent and its telemetry re-syncs the routing table.
        for key in [k for k in self.deployed if k[1] == session.node_id]:
            self.deployed.pop(key, None)
            self.deploy_started.pop(key, None)
        for key in [k for k in self.deploy_failures if k[1] == session.node_id]:
            self.deploy_failures.pop(key, None)  # fresh agent, fresh attempt
        await self.rebalance(reason=f"node-join {session.node_id}")

    async def on_disconnect(self, session: WorkerSession) -> None:
        node_id = session.node_id
        if self.sessions.get(node_id) is not session:
            return  # replaced by a newer session (reconnect)
        self.sessions.pop(node_id, None)
        self.nodes.pop(node_id, None)
        for model_id in self.registry.ids():
            self.endpoints.unregister(model_id=model_id, node_id=node_id)
            self.store.delete_shard_perf(model_id, node_id)
        for key in [k for k in self.deployed if k[1] == node_id]:
            self.deployed.pop(key, None)
            self.deploy_started.pop(key, None)
        self.node_last_seen.pop(node_id, None)
        for key in [k for k in self.shard_status if k[1] == node_id]:
            self.shard_status.pop(key, None)
        await self.rebalance(reason=f"node-leave {node_id}")

    async def on_ack(self, session: WorkerSession, ack: worker_control_pb2.Ack) -> None:
        if not ack.ok:
            logger.warning("worker %s NACK %s: %s", session.node_id, ack.command_id, ack.error)

    async def on_endpoint(self, session: WorkerSession, ep: gateway_pb2.ServingEndpoint) -> None:
        if self.registry.get(ep.model_id) is None:
            return
        # Only stage 0 accepts client requests; middle/tail stages serve
        # activations over the tunnel and must never be routed to directly.
        entry = self.deployed.get((ep.model_id, session.node_id))
        if entry is not None and entry[4] != 0:
            logger.info(
                "model %s stage %d ready on %s (no client endpoint)",
                ep.model_id,
                entry[4],
                session.node_id,
            )
            return
        # The backend listens on loopback inside the worker; we address it
        # through the data-plane tunnel, so the "url" is a routing handle, not
        # a dialable address.
        handle = f"tunnel://{session.node_id}:{ep.local_port}"
        self.endpoints.register(
            model_id=ep.model_id, base_url=handle, node_id=session.node_id
        )
        logger.info(
            "model %s serving on %s (local port %d, via tunnel)",
            ep.model_id,
            session.node_id,
            ep.local_port,
        )

    async def on_telemetry(self, session: WorkerSession, report) -> None:
        self.node_last_seen[session.node_id] = time.time()
        for shard in report.shards:
            self.shard_status[(shard.model_id, session.node_id)] = (
                shard.status or ("serving" if shard.healthy else "unknown"),
                time.time(),
            )
            if self.registry.get(shard.model_id) is None:
                continue
            self.store.upsert_shard_perf(
                ShardPerf(
                    model_id=shard.model_id,
                    node_id=session.node_id,
                    start_layer=shard.start_layer,
                    end_layer=shard.end_layer,
                    latency_ms=shard.avg_layer_latency_ms or None,
                    current_requests=shard.current_requests,
                    is_healthy=shard.healthy,
                )
            )
            # Endpoint re-sync: heartbeats carry the serving port, so an
            # orchestrator restart (or any desync) recovers the routing table
            # without waiting for a new StartServing.
            key = (shard.model_id, session.node_id)
            if shard.status == "serving" and shard.local_port:
                stage_index = int(shard.stage_index)
                num_stages = max(1, int(shard.num_stages))
                pipeline_id = shard.pipeline_id or f"{shard.model_id}#0"
                if key not in self.deployed:
                    # Rebuild deployment + stage routing after an orchestrator
                    # restart: the worker is the source of truth about what it
                    # actually runs, including its place in the pipeline.
                    self.deployed[key] = (
                        0,
                        shard.start_layer,
                        shard.end_layer,
                        pipeline_id,
                        stage_index,
                        num_stages,
                    )
                    self.tunnel.register_stage_routes(pipeline_id, {stage_index: session.node_id})
                    self.model_pipelines.setdefault(shard.model_id, [])
                    if pipeline_id not in self.model_pipelines[shard.model_id]:
                        self.model_pipelines[shard.model_id].append(pipeline_id)
                    logger.info(
                        "re-synced %s stage %d/%d on %s from telemetry",
                        shard.model_id,
                        stage_index,
                        num_stages,
                        session.node_id,
                    )
                if stage_index == 0:
                    handle = f"tunnel://{session.node_id}:{shard.local_port}"
                    known = {ep.base_url for ep in self.endpoints.candidates(shard.model_id)}
                    if handle not in known:
                        self.endpoints.register(
                            model_id=shard.model_id, base_url=handle, node_id=session.node_id
                        )

            # Self-healing: a shard the worker reports as FAILED (e.g. the
            # watchdog killed the backend) is forgotten and re-placed on the
            # next broker pass. "loading"/"loaded"/"starting" are NOT failures.
            key = (shard.model_id, session.node_id)
            issued_at = self.deploy_started.get(key, 0.0)
            if time.time() - issued_at < self.config.deploy_grace_s:
                # Too soon to judge: this heartbeat may still describe the
                # previous attempt, and re-placing now would stack a second
                # engine on top of the one that is starting.
                continue
            if shard.status == "failed" and key in self.deployed:
                logger.warning(
                    "shard %s on %s reported failed; scheduling re-placement",
                    shard.model_id,
                    session.node_id,
                )
                self.deployed.pop(key, None)
                self.endpoints.unregister(model_id=shard.model_id, node_id=session.node_id)
                self.store.delete_shard_perf(shard.model_id, session.node_id)
                asyncio.create_task(
                    self.rebalance(reason=f"shard-failed {shard.model_id}@{session.node_id}")
                )

    # ------------------------------------------------------------- admin (catalog)
    async def add_model(self, spec: ModelSpec) -> None:
        self.registry.add(spec)
        await self.rebalance(reason=f"model-added {spec.model_id}")

    async def remove_model(self, model_id: str) -> bool:
        spec = self.registry.remove(model_id)
        if spec is None:
            return False
        await self.rebalance(reason=f"model-removed {model_id}")
        return True

    # ------------------------------------------------------------- rebalance core
    async def rebalance(self, reason: str) -> None:
        async with self._lock:
            previous: Dict[str, Dict[str, int]] = {}
            for (model_id, node_id), entry in self.deployed.items():
                previous.setdefault(model_id, {})[node_id] = entry[0]
            plan = self.broker.plan(
                [
                    PoolNode(
                        node_id=d.node_id,
                        region=d.region,
                        vram_free_bytes=d.vram_free_bytes,
                        tflops_fp16=d.hardware.tflops_fp16,
                    )
                    for d in self.nodes.values()
                ],
                self.registry.list(),
                previous=previous,
                score_boosts=dict(self.slo_boosts),
            )
            self.last_plan = plan
            logger.info(
                "[Rebalance] reason=%s allocations=%s unscheduled=%s",
                reason,
                {m: {n: q // GIB for n, q in g.items()} for m, g in plan.allocations.items()},
                plan.unscheduled,
            )

            # Rebuild per-model schedulers on their granted sub-pools (Phase-1).
            desired: Deployment = {}
            # pipeline_id -> {stage_index: node_id}, published to the tunnel hub
            # so inter-stage activations can be routed.
            pipeline_routes: Dict[str, Dict[int, str]] = {}
            self.model_pipelines = {}
            for spec in self.registry.list():
                grants = plan.allocations.get(spec.model_id) or {}
                if not grants:
                    self.pool.drop(spec.model_id)
                    continue
                self.pool.rebuild(spec, grants, self.nodes)
                allocation = self.pool.shard_plan(spec.model_id)
                pipelines = build_pipelines(allocation, spec.model_info.num_layers)
                if not pipelines and allocation:
                    logger.warning(
                        "[Rebalance] %s: allocation %s forms no complete pipeline",
                        spec.model_id,
                        allocation,
                    )
                self.model_pipelines[spec.model_id] = []
                for idx, stages in enumerate(pipelines):
                    pipeline_id = f"{spec.model_id}#{idx}"
                    self.model_pipelines[spec.model_id].append(pipeline_id)
                    pipeline_routes[pipeline_id] = {}
                    for stage_index, (node_id, start, end) in enumerate(stages):
                        desired[(spec.model_id, node_id)] = (
                            grants.get(node_id, 0),
                            start,
                            end,
                            pipeline_id,
                            stage_index,
                            len(stages),
                        )
                        pipeline_routes[pipeline_id][stage_index] = node_id
            for model_id in self.pool.model_ids():
                if self.registry.get(model_id) is None:
                    self.pool.drop(model_id)

            # Refresh routing tables before any stage starts talking.
            known = {pid for (pid, _) in self.tunnel.stage_routes.keys()}
            for stale in known - set(pipeline_routes):
                self.tunnel.clear_stage_routes(stale)
            for pipeline_id, stages in pipeline_routes.items():
                self.tunnel.register_stage_routes(pipeline_id, stages)

            removals = [k for k, v in self.deployed.items() if desired.get(k) != v]
            additions = [k for k, v in desired.items() if self.deployed.get(k) != v]

            # 1) Evict first (await), so VRAM is freed before new loads.
            for model_id, node_id in removals:
                await self._teardown_on_worker(model_id, node_id)
                self.deployed.pop((model_id, node_id), None)
                self.deploy_started.pop((model_id, node_id), None)

            # 2) Deploy additions without holding the lock on slow backend starts.
            now = time.time()
            for model_id, node_id in additions:
                key = (model_id, node_id)
                failed_at, error = self.deploy_failures.get(key, (0.0, ""))
                if now - failed_at < self.config.deploy_retry_s:
                    # Same placement failed moments ago; a backend start costs
                    # minutes and a full checkpoint download, so retrying every
                    # pass would just hammer the GPU.
                    logger.info(
                        "[Rebalance] skipping %s on %s for %.0fs more (last failure: %s)",
                        model_id,
                        node_id,
                        self.config.deploy_retry_s - (now - failed_at),
                        error,
                    )
                    continue
                entry = desired[key]
                self.deployed[key] = entry
                self.deploy_started[key] = now
                asyncio.create_task(self._deploy_on_worker(model_id, node_id, entry))

    async def _teardown_on_worker(self, model_id: str, node_id: str) -> None:
        self.endpoints.unregister(model_id=model_id, node_id=node_id)
        self.store.delete_shard_perf(model_id, node_id)
        session = self.sessions.get(node_id)
        if session is None:
            return
        try:
            meta = new_meta()
            await session.send_command(
                gateway_pb2.ControlMessage(
                    stop_serving=worker_control_pb2.ModelRequest(model_id=model_id, meta=meta)
                ),
                meta.command_id,
                timeout_s=30,
            )
            meta = new_meta()
            await session.send_command(
                gateway_pb2.ControlMessage(
                    unload_shard=worker_control_pb2.UnloadShardRequest(
                        model_id=model_id, meta=meta
                    )
                ),
                meta.command_id,
                timeout_s=30,
            )
            logger.info("[Rebalance] evicted %s from %s", model_id, node_id)
        except (ConnectionError, asyncio.TimeoutError) as exc:
            logger.warning("teardown of %s on %s failed: %s", model_id, node_id, exc)

    async def _deploy_on_worker(self, model_id: str, node_id: str, entry: tuple) -> None:
        quota, start, end, pipeline_id, stage_index, num_stages = entry
        session = self.sessions.get(node_id)
        spec = self.registry.get(model_id)
        if session is None or spec is None:
            return
        try:
            meta = new_meta()
            ack = await session.send_command(
                gateway_pb2.ControlMessage(
                    load_shard=worker_control_pb2.LoadShardRequest(
                        model_id=model_id,
                        start_layer=start,
                        end_layer=end,
                        backend_type=spec.backend_type,
                        weights_uri=spec.weights_uri,
                        vram_quota_bytes=quota,
                        meta=meta,
                        topology=worker_control_pb2.PipelineTopology(
                            pipeline_id=pipeline_id,
                            stage_index=stage_index,
                            num_stages=num_stages,
                            is_first=stage_index == 0,
                            is_last=stage_index == num_stages - 1,
                            num_model_layers=spec.model_info.num_layers,
                        ),
                    )
                ),
                meta.command_id,
            )
            if not ack.ok:
                logger.error("LoadShard %s on %s failed: %s", model_id, node_id, ack.error)
                self._mark_deploy_failed(model_id, node_id, ack.error)
                return
            meta = new_meta()
            ack = await session.send_command(
                gateway_pb2.ControlMessage(
                    start_serving=worker_control_pb2.ModelRequest(model_id=model_id, meta=meta)
                ),
                meta.command_id,
                timeout_s=self.config.start_timeout_s,
            )
            if not ack.ok:
                logger.error("StartServing %s on %s failed: %s", model_id, node_id, ack.error)
                self._mark_deploy_failed(model_id, node_id, ack.error)
                return
            # Started for real: forget any earlier failure on this placement.
            self.deploy_failures.pop((model_id, node_id), None)
            logger.info(
                "[Rebalance] %s stage %d/%d serving on %s",
                model_id,
                stage_index,
                num_stages,
                node_id,
            )
        except (ConnectionError, asyncio.TimeoutError) as exc:
            logger.warning("deploy of %s on %s failed: %s", model_id, node_id, exc)
            self._mark_deploy_failed(model_id, node_id, str(exc))

    def _mark_deploy_failed(self, model_id: str, node_id: str, error: str) -> None:
        """Forget the placement and start its retry cooldown.

        The failure is kept (not just dropped) so the admin UI can say WHY a
        model is not running, instead of showing an endless silent retry."""
        key = (model_id, node_id)
        self.deployed.pop(key, None)
        self.deploy_started.pop(key, None)
        self.deploy_failures[key] = (time.time(), error or "unknown error")

    # ------------------------------------------------------------- quota override
    async def set_quota(self, model_id: str, node_id: str, vram_quota_bytes: int):
        """Push a SetQuota command to a worker (ops tool / watchdog demo)."""
        session = self.sessions.get(node_id)
        if session is None:
            raise KeyError(f"node {node_id} is not connected")
        meta = new_meta()
        return await session.send_command(
            gateway_pb2.ControlMessage(
                set_quota=worker_control_pb2.QuotaRequest(
                    model_id=model_id, vram_quota_bytes=vram_quota_bytes, meta=meta
                )
            ),
            meta.command_id,
            timeout_s=30,
        )

    # ------------------------------------------------------------- SLO monitoring
    def record_request(self, model_id: str, *, ttft_ms: float, error: bool) -> None:
        """Called by the API proxy after every request (Phase-2 signal source)."""
        samples = self._slo_samples.setdefault(model_id, deque(maxlen=512))
        samples.append((time.time(), ttft_ms, error))

    def slo_snapshot(self, model_id: str) -> Optional[dict]:
        spec = self.registry.get(model_id)
        samples = self._slo_samples.get(model_id)
        if samples is None:
            return None
        horizon = time.time() - self.config.slo_window_s
        window = [(t, l, e) for (t, l, e) in samples if t >= horizon]
        if not window:
            return None
        latencies = sorted(l for _, l, _ in window)
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        return {
            "samples": len(window),
            "p95_ttft_ms": round(p95, 1),
            "error_rate": round(sum(1 for _, _, e in window if e) / len(window), 3),
            "slo_p95_ttft_ms": spec.slo_p95_ttft_ms if spec else None,
            "boost": self.slo_boosts.get(model_id, 1.0),
        }

    def slo_evaluate(self) -> bool:
        """Update SLO boosts from current windows; True if anything changed.

        p95 TTFT above the model's SLO -> boost its score for the next broker
        pass (+1 target pipeline). Hysteresis: the boost is lifted only when
        p95 drops below 70% of the SLO, to avoid flapping.
        """
        changed = False
        for spec in self.registry.list():
            if spec.slo_p95_ttft_ms is None:
                continue
            snap = self.slo_snapshot(spec.model_id)
            if snap is None or snap["samples"] < self.config.slo_min_samples:
                continue
            boosted = spec.model_id in self.slo_boosts
            if not boosted and snap["p95_ttft_ms"] > spec.slo_p95_ttft_ms:
                self.slo_boosts[spec.model_id] = self.config.slo_boost_factor
                logger.warning(
                    "[SLO] %s violated (p95=%.0fms > %.0fms); boosting for next pass",
                    spec.model_id,
                    snap["p95_ttft_ms"],
                    spec.slo_p95_ttft_ms,
                )
                changed = True
            elif boosted and snap["p95_ttft_ms"] < 0.7 * spec.slo_p95_ttft_ms:
                self.slo_boosts.pop(spec.model_id, None)
                logger.info("[SLO] %s recovered; boost lifted", spec.model_id)
                changed = True
        return changed

    async def slo_check_loop(self) -> None:
        """SLO-based rebalance trigger (generalizes membership-only triggers)."""
        while True:
            await asyncio.sleep(self.config.slo_check_interval_s)
            try:
                if self.slo_evaluate():
                    await self.rebalance(reason="slo")
            except Exception:
                logger.exception("SLO check failed")

    # ------------------------------------------------------------- periodic loops
    async def perfmap_sync_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.perfmap_sync_interval_s)
            for model_id in self.pool.model_ids():
                instance = self.pool.get(model_id)
                if instance is None:
                    continue
                try:
                    n = sync_perfmap_to_scheduler(self.store, instance.scheduler, model_id)
                    if n:
                        await asyncio.to_thread(instance.scheduler._process_node_updates)
                except Exception:
                    logger.exception("perfmap sync failed for %s", model_id)

    async def rebalance_timer_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.rebalance_interval_s)
            try:
                await self.rebalance(reason="timer")
            except Exception:
                logger.exception("periodic rebalance failed")

    # ------------------------------------------------------------- request routing
    def head_nodes(self, model_id: str) -> Dict[str, str]:
        """pipeline_id -> node_id of stage 0 for each replica of the model."""
        heads: Dict[str, str] = {}
        for (mid, node_id), entry in self.deployed.items():
            if mid != model_id:
                continue
            _, _, _, pipeline_id, stage_index, _ = entry
            if stage_index == 0:
                heads[pipeline_id] = node_id
        return heads

    def pick_endpoint(self, model_id: str) -> Optional[Endpoint]:
        """Choose which replica serves a request.

        A request always enters at the HEAD of a pipeline (stage 0): that stage
        owns the client request and drives generation, pulling the remaining
        stages in over the tunnel. Phase-2 picks the chain; we then map its
        first hop to that pipeline's head endpoint.
        """
        heads = set(self.head_nodes(model_id).values())
        instance: Optional[ModelInstance] = self.pool.get(model_id)
        if instance is not None:
            path, latency = instance.scheduler.request_router.find_optimal_path()
            if path and latency != float("inf"):
                for ep in self.endpoints.candidates(model_id):
                    if ep.node_id == path[0]:
                        return ep
        # Fall back to load balancing, but only across pipeline heads.
        candidates = [
            ep for ep in self.endpoints.candidates(model_id) if not heads or ep.node_id in heads
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda ep: ep.metrics.inflight)

    # ---------------------------------------------------- admin UI (read-only)
    def nodes_view(self) -> dict:
        """Per-node view: hardware, liveness, hosted shards. Pure state read."""
        now = time.time()
        out = {}
        for nid, d in self.nodes.items():
            shards = []
            for (model_id, node_id), entry in self.deployed.items():
                if node_id != nid:
                    continue
                quota, start, end = entry[0], entry[1], entry[2]
                status, status_ts = self.shard_status.get((model_id, nid), ("unknown", 0.0))
                perf = next(
                    (p for p in self.store.get_shard_perf(model_id) if p.node_id == nid), None
                )
                shards.append(
                    {
                        "model_id": model_id,
                        "layers": [start, end],
                        "vram_quota_gb": round(quota / GIB, 2),
                        "status": status,
                        "current_requests": perf.current_requests if perf else 0,
                        "latency_ms": perf.latency_ms if perf else None,
                    }
                )
            last_seen = self.node_last_seen.get(nid)
            out[nid] = {
                "device": d.hardware.device,
                "region": d.region,
                "gpu_name": d.hardware.gpu_name,
                "num_gpus": d.hardware.num_gpus,
                "vram_declared_gb": round(d.vram_free_bytes / GIB, 2),
                "vram_total_gb": round(d.vram_total_bytes / GIB, 2),
                "tflops_fp16": d.hardware.tflops_fp16,
                "detected_by": d.detection_source,
                "agent_version": d.agent_version,
                "price": None,  # pricing lands with Phase-4 onboarding
                "connected": nid in self.sessions,
                "tunnel": self.tunnel.is_connected(nid),
                "last_heartbeat_s_ago": round(now - last_seen, 1) if last_seen else None,
                "shards": shards,
            }
        return {"nodes": out}

    def models_view(self) -> dict:
        """Catalog + actual Phase-1 placement per model. Pure state read."""
        out = {}
        for spec in self.registry.list():
            instance = self.pool.get(spec.model_id)
            pipelines_actual = 0
            if instance is not None:
                pipelines_actual = instance.scheduler.node_manager.num_full_pipelines(
                    spec.model_info.num_layers
                )
            placement = [
                {
                    "node_id": node_id,
                    "layers": [start, end],
                    "vram_quota_gb": round(
                        self.deployed.get((spec.model_id, node_id), (0, 0, 0))[0] / GIB, 2
                    ),
                    "status": self.shard_status.get((spec.model_id, node_id), ("unknown", 0))[0],
                }
                for node_id, start, end in self.pool.shard_plan(spec.model_id)
            ]
            out[spec.model_id] = {
                "priority": spec.priority,
                "demand_qps": spec.demand_qps,
                "price_willing": spec.price_willing,
                "score": spec.score(),
                "slo_p95_ttft_ms": spec.slo_p95_ttft_ms,
                "k_target": self.broker.target_pipelines(spec),
                "k_actual": pipelines_actual,
                "num_layers": spec.model_info.num_layers,
                "backend_type": spec.backend_type,
                "placement": placement,
                "endpoints": [
                    {"node_id": ep.node_id, "url": ep.base_url, "inflight": ep.metrics.inflight}
                    for ep in self.endpoints.candidates(spec.model_id)
                ],
                "unscheduled": bool(
                    self.last_plan and spec.model_id in self.last_plan.unscheduled
                ),
                # Why a model is not running, in the operator's own words.
                "failures": [
                    {
                        "node_id": node_id,
                        "error": error,
                        "age_s": round(time.time() - failed_at, 1),
                        "retry_in_s": max(
                            0.0,
                            round(self.config.deploy_retry_s - (time.time() - failed_at), 1),
                        ),
                    }
                    for (model_id_, node_id), (failed_at, error) in sorted(
                        self.deploy_failures.items()
                    )
                    if model_id_ == spec.model_id
                ],
            }
        return {"models": out}

    def perfmap_view(self, model_id: str) -> Optional[dict]:
        """What Phase-2 sees for a model: τ (per-shard latency) and ρ (RTT)."""
        if self.registry.get(model_id) is None:
            return None
        measured = {
            p.node_id: {
                "layers": [p.start_layer, p.end_layer],
                "latency_ms": p.latency_ms,
                "current_requests": p.current_requests,
                "healthy": p.is_healthy,
                "age_s": round(time.time() - p.updated_at, 1),
            }
            for p in self.store.get_shard_perf(model_id)
        }
        effective = {}
        route = None
        instance = self.pool.get(model_id)
        if instance is not None:
            for node in instance.scheduler.node_manager.nodes:
                lat = node.layer_latency_ms
                effective[node.node_id] = {
                    "layers": [node.start_layer, node.end_layer],
                    "layer_latency_ms": None if lat == float("inf") else round(lat, 3),
                    "source": "measured" if node.avg_layer_latency_ms is not None else "roofline",
                    "rtt_to": dict(node.rtt_to_nodes or {}),
                }
            path, latency = instance.scheduler.request_router.find_optimal_path()
            route = {
                "path": path,
                "est_latency_ms": None if latency == float("inf") else round(latency, 3),
            }
        return {
            "model_id": model_id,
            "tau_measured": measured,  # from Perf-map Store (worker telemetry)
            "tau_effective": effective,  # what the Phase-2 DP actually consumes
            "route_preview": route,
        }

    def status(self) -> dict:
        return {
            "nodes": {
                nid: {
                    "region": d.region,
                    "vram_free_gb": round(d.vram_free_bytes / GIB, 2),
                    "connected": nid in self.sessions,
                }
                for nid, d in self.nodes.items()
            },
            "models": {
                spec.model_id: {
                    "score": spec.score(),
                    "priority": spec.priority,
                    "grants_gb": {
                        n: round(q / GIB, 2)
                        for (m, n), (q, *_rest) in self.deployed.items()
                        if m == spec.model_id
                    },
                    "endpoints": [
                        ep.base_url for ep in self.endpoints.candidates(spec.model_id)
                    ],
                    "slo": self.slo_snapshot(spec.model_id),
                }
                for spec in self.registry.list()
            },
            "unscheduled": list(self.last_plan.unscheduled) if self.last_plan else [],
        }
