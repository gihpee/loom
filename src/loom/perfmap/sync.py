"""Feed Perf-map Store data into a model's scheduler instance.

Own code (no borrowed Parallax logic), but it deliberately reuses the
*existing* update path of the ported scheduler: in the original Parallax the
Phase-2 DP reads ``Node.layer_latency_ms`` / ``Node.rtt_to_nodes``, which were
populated from DHT broadcasts via ``Scheduler.enqueue_node_update``. Loom
keeps the DP math and the update path untouched and only swaps the data
source: the centralized Perf-map Store instead of the Lattica DHT.
"""

from __future__ import annotations

from loom.logging_config import get_logger
from loom.perfmap.store import PerfMapStore
from loom.planning.scheduler import Scheduler

logger = get_logger(__name__)


def sync_perfmap_to_scheduler(store: PerfMapStore, scheduler: Scheduler, model_id: str) -> int:
    """Push all fresh perf entries for ``model_id`` into the scheduler.

    Returns the number of node updates enqueued. Intended to be called
    periodically (or on telemetry ingest) by the orchestrator.
    """
    entries = store.get_shard_perf(model_id)
    count = 0
    for perf in entries:
        if scheduler.node_manager.get(perf.node_id) is None:
            logger.debug(
                "Perf entry for unknown node %s (model %s); skipping", perf.node_id, model_id
            )
            continue
        scheduler.enqueue_node_update(
            perf.node_id,
            current_requests=perf.current_requests,
            layer_latency_ms=perf.latency_ms,
            new_rtt_to_nodes=store.get_rtt_map(perf.node_id) or None,
            is_active=perf.is_healthy,
        )
        count += 1
    return count
