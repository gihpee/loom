"""Regression parity: Loom's ported allocators vs the ORIGINAL Parallax code.

Same inputs (hardware, model, ratios), where the Loom node receives its
capacity as an explicit ShardCapacity computed from a quota equal to the full
device memory — the case in which the two implementations must agree exactly.
"""

import pytest
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

NODE_SPECS = [
    # (node_id, memory_gb, tflops)
    ("n0", 30.0, 200.0),
    ("n1", 30.0, 150.0),
    ("n2", 16.0, 120.0),
    ("n3", 16.0, 100.0),
    ("n4", 8.0, 80.0),
    ("n5", 8.0, 60.0),
]


def build_pools(px, num_layers):
    kwargs = make_model_info_kwargs(num_layers=num_layers)
    loom_mi = ModelInfo(**kwargs)
    px_mi = px.ModelInfo(**kwargs)

    loom_nodes, px_nodes = [], []
    for node_id, mem_gb, tflops in NODE_SPECS:
        hw_kwargs = dict(
            node_id=node_id,
            num_gpus=1,
            tflops_fp16=tflops,
            gpu_name="test-gpu",
            memory_gb=mem_gb,
            memory_bandwidth_gbps=1000.0,
            device="cuda",
        )
        loom_hw = NodeHardwareInfo(**hw_kwargs)
        px_hw = px.NodeHardwareInfo(**hw_kwargs)
        cap = ShardCapacity.from_model_info(
            loom_mi, vram_quota_bytes=int(mem_gb * GIB), device="cuda"
        )
        loom_nodes.append(
            Node(node_id=node_id, hardware=loom_hw, model_info=loom_mi, capacity=cap)
        )
        px_nodes.append(px.Node(node_id=node_id, hardware=px_hw, model_info=px_mi))
    return loom_mi, px_mi, loom_nodes, px_nodes


def allocations(manager, num_layers):
    return sorted(manager.list_node_allocations(num_layers))


@pytest.mark.parametrize("num_layers", [24, 48])
@pytest.mark.parametrize("strategy", ["dp", "greedy"])
def test_allocation_parity(parallax_modules, num_layers, strategy):
    px = parallax_modules
    loom_mi, px_mi, loom_nodes, px_nodes = build_pools(px, num_layers)

    # Capacities must agree node-by-node before any allocation runs.
    for ln, pn in zip(loom_nodes, px_nodes):
        for embed in (False, True):
            for head in (False, True):
                assert ln.get_decoder_layer_capacity(
                    include_input_embed=embed, include_lm_head=head
                ) == pn.get_decoder_layer_capacity(
                    include_input_embed=embed, include_lm_head=head
                ), f"capacity mismatch on {ln.node_id} (embed={embed}, head={head})"

    loom_manager = NodeManager(initial_nodes=loom_nodes)
    px_manager = px.NodeManager(initial_nodes=px_nodes)

    if strategy == "dp":
        loom_alloc = DynamicProgrammingLayerAllocator(
            model_info=loom_mi, node_management=loom_manager
        )
        px_alloc = px.DPAllocator(model_info=px_mi, node_management=px_manager)
    else:
        loom_alloc = GreedyLayerAllocator(model_info=loom_mi, node_management=loom_manager)
        px_alloc = px.GreedyAllocator(model_info=px_mi, node_management=px_manager)

    loom_ok = loom_alloc.allocate_from_standby()
    px_ok = px_alloc.allocate_from_standby()

    assert loom_ok == px_ok
    assert allocations(loom_manager, num_layers) == allocations(px_manager, num_layers)


def test_kv_load_parity(parallax_modules):
    """LayerLoad inputs (per-layer KV memory) match after identical allocation."""
    px = parallax_modules
    loom_mi, px_mi, loom_nodes, px_nodes = build_pools(px, 24)
    for ln, pn in zip(loom_nodes, px_nodes):
        ln.set_layer_allocation(0, 12)
        pn.set_layer_allocation(0, 12)
        assert ln.per_decoder_layer_kv_cache_memory == pn.per_decoder_layer_kv_cache_memory
        assert ln.max_requests == pn.max_requests
