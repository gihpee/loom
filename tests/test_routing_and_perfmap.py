"""Phase-2 routing fed from the Perf-map Store (DHT replacement) + store tests."""

import time

from conftest import make_model_info_kwargs
from test_layer_allocation import make_node

from loom.perfmap import InMemoryPerfMapStore, ShardPerf, sync_perfmap_to_scheduler
from loom.planning import (
    DynamicProgrammingRouting,
    GreedyLayerAllocator,
    ModelInfo,
    NodeManager,
    Scheduler,
)

MODEL_ID = "test-model"


def test_inmemory_store_ttl_and_rtt():
    store = InMemoryPerfMapStore(ttl_seconds=0.05)
    store.upsert_shard_perf(
        ShardPerf(model_id=MODEL_ID, node_id="a", start_layer=0, end_layer=12, latency_ms=3.0)
    )
    store.upsert_rtt("a", "b", 12.5)
    assert len(store.get_shard_perf(MODEL_ID)) == 1
    assert store.get_rtt_map("a") == {"b": 12.5}
    # Other models are namespaced away.
    assert store.get_shard_perf("other-model") == []
    time.sleep(0.06)
    assert store.get_shard_perf(MODEL_ID) == []
    assert store.get_rtt_map("a") == {}


def test_dp_routing_picks_low_latency_chain():
    mi = ModelInfo(**make_model_info_kwargs(num_layers=24))
    head = make_node("head", mi, quota_layers=30)
    tail_fast = make_node("tail-fast", mi, quota_layers=30)
    tail_slow = make_node("tail-slow", mi, quota_layers=30)
    head.set_layer_allocation(0, 12)
    tail_fast.set_layer_allocation(12, 24)
    tail_slow.set_layer_allocation(12, 24)
    head.set_layer_latency_ms(1.0)
    tail_fast.set_layer_latency_ms(1.0)
    tail_slow.set_layer_latency_ms(50.0)
    head.update_rtt("tail-fast", 5.0)
    head.update_rtt("tail-slow", 1.0)

    manager = NodeManager()
    for n in (head, tail_fast, tail_slow):
        manager.upsert(n)
        manager.activate([n.node_id])

    router = DynamicProgrammingRouting(manager, total_layers=24)
    path, latency = router.find_optimal_path()
    assert path == ["head", "tail-fast"]
    assert latency < float("inf")


def test_perfmap_feeds_scheduler_routing():
    """End-to-end (in-process): allocate, then Phase-2 consumes Perf-map data."""
    mi = ModelInfo(**make_model_info_kwargs(num_layers=24))
    nodes = [make_node(f"w{i}", mi, quota_layers=16) for i in range(2)]
    scheduler = Scheduler(mi, nodes, min_nodes_bootstrapping=2, routing_strategy="dp")
    assert scheduler.bootstrap()
    assert scheduler.has_full_pipeline()

    # Worker telemetry lands in the store (keyed by model_id), then is synced
    # into the scheduler — replacing the original DHT broadcast path.
    store = InMemoryPerfMapStore(ttl_seconds=60)
    allocs = scheduler.list_node_allocations()
    for node_id, s, e in allocs:
        store.upsert_shard_perf(
            ShardPerf(
                model_id=MODEL_ID,
                node_id=node_id,
                start_layer=s,
                end_layer=e,
                latency_ms=2.0,
                current_requests=0,
                is_healthy=True,
            )
        )
    ids = [a[0] for a in allocs]
    for src in ids:
        for dst in ids:
            if src != dst:
                store.upsert_rtt(src, dst, 3.0)

    assert sync_perfmap_to_scheduler(store, scheduler, MODEL_ID) == len(ids)
    scheduler._process_node_updates()

    for node_id, _, _ in allocs:
        node = scheduler.get_node(node_id)
        assert node.avg_layer_latency_ms == 2.0

    path, latency = scheduler.request_router.find_optimal_path()
    assert path and latency < float("inf")


def test_greedy_allocator_end_to_end_with_scheduler():
    mi = ModelInfo(**make_model_info_kwargs(num_layers=24))
    nodes = [make_node(f"g{i}", mi, quota_layers=10) for i in range(3)]
    manager = NodeManager(initial_nodes=nodes)
    allocator = GreedyLayerAllocator(model_info=mi, node_management=manager)
    assert allocator.allocate_from_standby()
    covered = set()
    for _, s, e in manager.list_node_allocations(24):
        covered.update(range(s, e))
    assert covered == set(range(24))
