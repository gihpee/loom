"""Resource Broker unit tests: greedy FFD per the specification."""

from conftest import make_model_info_kwargs

from loom.orchestrator.broker import PoolNode, ResourceBroker, pipeline_vram_bytes
from loom.orchestrator.registry import ModelRegistry, ModelSpec
from loom.planning import ModelInfo

GIB = 1024**3


def make_spec(model_id: str, *, priority=1.0, demand=1.0, price=1.0, k=1, layers=28):
    return ModelSpec(
        model_id=model_id,
        weights_uri="hf://test",
        backend_type="echo",
        model_info=ModelInfo(**make_model_info_kwargs(num_layers=layers)),
        demand_qps=demand,
        priority=priority,
        price_willing=price,
        target_pipelines=k,
    )


def node(node_id, vram_gb, region="default"):
    return PoolNode(node_id=node_id, region=region, vram_free_bytes=int(vram_gb * GIB))


def test_score_ordering_and_eviction_of_low_priority():
    """Pool fits 2 of 3 models -> lowest score is left unscheduled."""
    broker = ResourceBroker()
    spec = make_spec("a", priority=2)
    need = pipeline_vram_bytes(spec.model_info)
    pool = [PoolNode("w0", "default", int(need * 2.5))]  # room for 2 pipelines + change
    models = [
        make_spec("low", priority=1),
        make_spec("high", priority=10),
        make_spec("mid", priority=2),
    ]
    plan = broker.plan(pool, models)
    assert set(plan.allocations) == {"high", "mid"}
    assert plan.unscheduled == ["low"]


def test_partial_node_consumption_two_models_one_gpu():
    """Two models co-located on one physical node with separate quotas."""
    broker = ResourceBroker()
    spec_a, spec_b = make_spec("a", priority=2), make_spec("b", priority=1)
    need = pipeline_vram_bytes(spec_a.model_info)
    pool = [PoolNode("w0", "default", int(need * 2.2))]
    plan = broker.plan(pool, [spec_a, spec_b])
    assert plan.quota("a", "w0") >= need
    assert plan.quota("b", "w0") >= need
    total = plan.quota("a", "w0") + plan.quota("b", "w0")
    assert total <= int(need * 2.2)


def test_pipeline_spans_nodes_within_region():
    broker = ResourceBroker()
    spec = make_spec("a")
    need = pipeline_vram_bytes(spec.model_info)
    half = int(need * 0.6)
    plan = broker.plan([PoolNode("w0", "r1", half), PoolNode("w1", "r1", half)], [spec])
    grants = plan.allocations["a"]
    assert set(grants) == {"w0", "w1"}
    assert sum(grants.values()) >= need


def test_no_cross_region_pipeline():
    """v0: a pipeline never spans regions (bridging is a later, penalized step)."""
    broker = ResourceBroker()
    spec = make_spec("a")
    need = pipeline_vram_bytes(spec.model_info)
    half = int(need * 0.6)
    plan = broker.plan([PoolNode("w0", "us", half), PoolNode("w1", "eu", half)], [spec])
    assert plan.unscheduled == ["a"]


def test_k_degradation():
    """Wants 3 pipelines, VRAM fits 2 -> degraded, not unscheduled."""
    broker = ResourceBroker()
    spec = make_spec("a", k=3)
    need = pipeline_vram_bytes(spec.model_info)
    plan = broker.plan([PoolNode("w0", "default", int(need * 2.4))], [spec])
    granted = sum(plan.allocations["a"].values())
    assert need * 2 <= granted < need * 3


def test_target_pipelines_from_demand():
    broker = ResourceBroker(qps_per_pipeline=10)
    assert broker.target_pipelines(make_spec("a", demand=5, k=0)) == 1
    assert broker.target_pipelines(make_spec("a", demand=25, k=0)) == 3
    assert broker.target_pipelines(make_spec("a", demand=5, k=4)) == 4  # explicit wins


def test_prefers_region_with_more_free_vram():
    broker = ResourceBroker()
    spec = make_spec("a")
    need = pipeline_vram_bytes(spec.model_info)
    plan = broker.plan(
        [PoolNode("small", "eu", int(need * 1.1)), PoolNode("big", "us", int(need * 3))],
        [spec],
    )
    assert list(plan.allocations["a"]) == ["big"]


def test_registry_crud_and_catalog(tmp_path):
    spec = make_spec("m1")
    registry = ModelRegistry([spec])
    assert registry.get("m1") is spec
    registry.remove("m1")
    assert registry.get("m1") is None

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        '{"models": [{"model_id": "x", "weights_uri": "u", "backend_type": "echo", '
        '"priority": 3, "model_info": %s}]}'
        % __import__("json").dumps(make_model_info_kwargs())
    )
    reg2 = ModelRegistry.from_catalog_file(catalog)
    assert reg2.ids() == ["x"]
    assert reg2.get("x").priority == 3
