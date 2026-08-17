"""Centralized Perf-map Store — Loom's replacement for the Parallax DHT.

All keys are namespaced by ``model_id``. Live performance data
(per-shard latency, health, load) and pairwise RTTs are written by the
orchestrator's telemetry ingest (from ``ReportTelemetry``/``Heartbeat`` RPCs)
and read by each model's scheduler instance via :mod:`loom.perfmap.sync`.
"""

from loom.perfmap.store import (
    InMemoryPerfMapStore,
    PerfMapStore,
    RedisPerfMapStore,
    ShardPerf,
)
from loom.perfmap.sync import sync_perfmap_to_scheduler

__all__ = [
    "PerfMapStore",
    "InMemoryPerfMapStore",
    "RedisPerfMapStore",
    "ShardPerf",
    "sync_perfmap_to_scheduler",
]
