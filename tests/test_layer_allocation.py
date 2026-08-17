"""Phase-1 DP / greedy allocation on the ported (capacity-parametrized) core."""

from conftest import GIB, make_model_info_kwargs

from loom.planning import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
    ModelInfo,
    Node,
    NodeHardwareInfo,
    NodeManager,
    ShardCapacity,
)


def make_node(node_id: str, mi: ModelInfo, *, quota_layers: int, tflops: float = 100.0) -> Node:
    """Build a node whose broker-granted quota fits ~quota_layers decoder layers."""
    per_layer = mi.decoder_layer_io_bytes(roofline=False)
    # Reserve headroom for embedding + lm_head so endpoint variants stay positive.
    quota = int((quota_layers * per_layer + 2 * mi.embedding_io_bytes) / 0.5) + 1
    hw = NodeHardwareInfo(
        node_id=node_id,
        num_gpus=1,
        tflops_fp16=tflops,
        gpu_name="test-gpu",
        memory_gb=quota / GIB,
        memory_bandwidth_gbps=1000.0,
        device="cuda",
    )
    cap = ShardCapacity.from_model_info(mi, vram_quota_bytes=quota)
    return Node(node_id=node_id, hardware=hw, model_info=mi, capacity=cap)


def _allocate(allocator_cls, mi, nodes, **kwargs):
    manager = NodeManager(initial_nodes=nodes)
    allocator = allocator_cls(model_info=mi, node_management=manager, **kwargs)
    ok = allocator.allocate_from_standby()
    return ok, manager


def test_dp_prefers_more_pipelines():
    """Docstring case: caps ~(40,40,20,20,10,10), L=70 -> two 3-stage pipelines."""
    mi = ModelInfo(**make_model_info_kwargs(num_layers=70))
    nodes = [
        make_node("a", mi, quota_layers=40),
        make_node("b", mi, quota_layers=40),
        make_node("c", mi, quota_layers=20),
        make_node("d", mi, quota_layers=20),
        make_node("e", mi, quota_layers=10),
        make_node("f", mi, quota_layers=10),
    ]
    ok, manager = _allocate(DynamicProgrammingLayerAllocator, mi, nodes)
    assert ok
    assert manager.num_full_pipelines(70) >= 2
    # Every layer covered.
    covered = set()
    for _, s, e in manager.list_node_allocations(70):
        covered.update(range(s, e))
    assert covered == set(range(70))


def test_greedy_allocates_full_pipeline():
    mi = ModelInfo(**make_model_info_kwargs(num_layers=24))
    nodes = [
        make_node("g1", mi, quota_layers=16),
        make_node("g2", mi, quota_layers=16),
    ]
    ok, manager = _allocate(GreedyLayerAllocator, mi, nodes)
    assert ok
    assert manager.has_full_pipeline(24)


def test_insufficient_capacity_fails_cleanly():
    mi = ModelInfo(**make_model_info_kwargs(num_layers=48))
    nodes = [make_node("tiny", mi, quota_layers=8)]
    ok, manager = _allocate(DynamicProgrammingLayerAllocator, mi, nodes)
    assert not ok
    assert not manager.has_full_pipeline(48)


def test_quota_cut_shrinks_allocation():
    """Same physical node, smaller broker quota -> fewer hosted layers."""
    mi = ModelInfo(**make_model_info_kwargs(num_layers=24))
    big = make_node("n-big", mi, quota_layers=30)
    small = make_node("n-small", mi, quota_layers=12)
    assert small.get_decoder_layer_capacity() < big.get_decoder_layer_capacity()
    # A single small node cannot host the whole model even though the same
    # hardware with a full quota could.
    ok_big, _ = _allocate(GreedyLayerAllocator, mi, [big])
    ok_small, _ = _allocate(GreedyLayerAllocator, mi, [small])
    assert ok_big and not ok_small


def test_water_filling_respects_compute_proportionality():
    mi = ModelInfo(**make_model_info_kwargs(num_layers=30))
    fast = make_node("fast", mi, quota_layers=40, tflops=300.0)
    slow = make_node("slow", mi, quota_layers=40, tflops=100.0)
    manager = NodeManager(initial_nodes=[fast, slow])
    allocator = GreedyLayerAllocator(model_info=mi, node_management=manager)
    allocator.adjust_pipeline_layers([fast, slow])
    assert fast.num_current_layers + slow.num_current_layers == 30
    assert fast.num_current_layers > slow.num_current_layers
