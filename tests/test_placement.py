"""Hand-picked placement: which nodes run a model, and how many layers each.

Two things are being pinned down here. First, that a catalog entry no longer
means "running" — the orchestrator used to redeploy every model in the catalog
on startup, which makes a benchmark impossible to control and a restart a
surprise. Second, that a placement typed by a human is checked before anything
downloads a checkpoint: a gap, an overlap or a split that cannot fit is a
message on the screen, not a crash twenty minutes later.
"""

import asyncio
import json
from pathlib import Path

import pytest

from loom.orchestrator.config import OrchestratorConfig
from loom.orchestrator.controller import MultiModelController
from loom.orchestrator.placement import (
    Placement,
    PlacementError,
    even_split,
    max_layers_for,
    quota_for_layers,
    stages_from_request,
)
from loom.orchestrator.pool import NodeDescriptor
from loom.orchestrator.registry import ModelRegistry, ModelSpec
from loom.planning import NodeHardwareInfo

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GIB = 1024**3


# --------------------------------------------------------------- stage parsing
def test_layer_counts_become_a_contiguous_chain():
    stages = stages_from_request(
        [{"node_id": "a", "layers": 12}, {"node_id": "b", "layers": 8}],
        num_model_layers=20,
    )
    assert [(s.node_id, s.start_layer, s.end_layer) for s in stages] == [
        ("a", 0, 12),
        ("b", 12, 20),
    ]


def test_explicit_ranges_are_accepted_too():
    stages = stages_from_request(
        [
            {"node_id": "a", "start_layer": 0, "end_layer": 5},
            {"node_id": "b", "start_layer": 5, "end_layer": 20},
        ],
        num_model_layers=20,
    )
    assert stages[1].num_layers == 15


def test_the_order_given_is_the_pipeline_order():
    """Not cosmetic: stage 0 owns the embeddings and answers clients."""
    stages = stages_from_request(
        [{"node_id": "slow", "layers": 4}, {"node_id": "fast", "layers": 16}],
        num_model_layers=20,
    )
    assert stages[0].node_id == "slow" and stages[0].start_layer == 0


@pytest.mark.parametrize(
    "entries, expect",
    [
        ([{"node_id": "a", "layers": 19}], "cover 19 of 20"),
        ([{"node_id": "a", "layers": 25}], "5 too many"),
        (
            [
                {"node_id": "a", "start_layer": 0, "end_layer": 8},
                {"node_id": "b", "start_layer": 12, "end_layer": 20},
            ],
            "a gap of 4",
        ),
        (
            [
                {"node_id": "a", "start_layer": 0, "end_layer": 12},
                {"node_id": "b", "start_layer": 8, "end_layer": 20},
            ],
            "an overlap of 4",
        ),
        (
            [
                {"node_id": "a", "start_layer": 4, "end_layer": 12},
                {"node_id": "b", "start_layer": 12, "end_layer": 20},
            ],
            "must start at layer 0",
        ),
        (
            [{"node_id": "a", "layers": 10}, {"node_id": "a", "layers": 10}],
            "appears twice",
        ),
        ([{"node_id": "a", "layers": 0}], "at least one layer"),
        ([], "at least one node"),
        ([{"node_id": "", "layers": 20}], "node_id is required"),
        ([{"node_id": "a", "layers": "ten"}], "must be a number"),
    ],
)
def test_a_split_that_cannot_work_is_refused_with_a_reason(entries, expect):
    with pytest.raises(PlacementError, match=expect):
        stages_from_request(entries, num_model_layers=20)


def test_even_split_hands_the_remainder_to_the_front():
    stages = even_split(["a", "b", "c"], 20)
    assert [s.num_layers for s in stages] == [7, 7, 6]
    assert stages[-1].end_layer == 20


# ------------------------------------------------------------------ quota math
def test_a_stage_asks_for_memory_on_the_same_terms_as_the_broker():
    """quota x param_mem_ratio = the bytes the weights actually take."""
    quota = quota_for_layers(
        num_layers=10,
        is_first=False,
        is_last=False,
        per_layer_param_bytes=GIB,
        embedding_param_bytes=GIB,
        tie_embedding=False,
        param_mem_ratio=0.5,
    )
    assert quota == 20 * GIB  # 10 GB of weights at a 0.5 ratio

    with_head = quota_for_layers(
        num_layers=10,
        is_first=True,
        is_last=True,
        per_layer_param_bytes=GIB,
        embedding_param_bytes=GIB,
        tie_embedding=False,
        param_mem_ratio=0.5,
    )
    assert with_head == 24 * GIB  # plus embeddings and an untied LM head


def test_tied_embeddings_are_not_paid_for_twice():
    kwargs = dict(
        num_layers=4,
        is_first=True,
        is_last=True,
        per_layer_param_bytes=GIB,
        embedding_param_bytes=GIB,
        param_mem_ratio=1.0,
    )
    assert quota_for_layers(tie_embedding=True, **kwargs) == 5 * GIB
    assert quota_for_layers(tie_embedding=False, **kwargs) == 6 * GIB


def test_layer_ceiling_matches_the_capacity_formula():
    assert (
        max_layers_for(
            24 * GIB,
            per_layer_param_bytes=GIB,
            embedding_param_bytes=GIB,
            tie_embedding=False,
            param_mem_ratio=0.5,
        )
        == 12
    )


# ------------------------------------------------------- controller behaviour
def controller_with(*catalog_names, nodes=(("n1", 40), ("n2", 40), ("n3", 40))):
    specs = []
    for name in catalog_names:
        raw = json.loads((CONFIGS / name).read_text())
        entries = raw["models"] if isinstance(raw, dict) and "models" in raw else [raw]
        specs.extend(ModelSpec.from_dict(e) for e in entries)
    config = OrchestratorConfig(registry=ModelRegistry(specs), public_address="127.0.0.1:0")
    controller = MultiModelController(config)
    for node_id, gb in nodes:
        controller.nodes[node_id] = NodeDescriptor(
            node_id=node_id,
            region="default",
            hardware=NodeHardwareInfo(
                node_id=node_id,
                num_gpus=1,
                tflops_fp16=100.0,
                gpu_name="test-gpu",
                memory_gb=gb,
                memory_bandwidth_gbps=1000.0,
                device="cuda",
            ),
            vram_free_bytes=gb * GIB,
            vram_total_bytes=gb * GIB,
        )
    return controller


def test_a_catalog_model_does_not_run_until_someone_deploys_it():
    """The behaviour the demo asked for: restarting deploys nothing.

    The catalog used to be the deploy order, so every orchestrator restart
    re-launched whatever was listed in it — a model nobody asked for, taking
    GPUs a measurement needed.
    """
    controller = controller_with("catalog-demo.json")
    assert controller.registry.list(), "the catalog should still be loaded"
    assert controller.placements == {}

    asyncio.run(controller.rebalance(reason="startup"))
    assert controller.deployed == {}, "nothing may be placed on its own"


def test_deploying_by_hand_places_exactly_what_was_asked():
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers

    placement = asyncio.run(
        controller.deploy(
            model_id,
            stages=[
                {"node_id": "n1", "layers": layers - 3},
                {"node_id": "n2", "layers": 3},
            ],
        )
    )
    assert placement.is_manual
    assert [(s.node_id, s.num_layers) for s in placement.stages] == [
        ("n1", layers - 3),
        ("n2", 3),
    ]
    placed = {node_id: entry for (mid, node_id), entry in controller.deployed.items() if mid == model_id}
    assert set(placed) == {"n1", "n2"}
    # (quota, start, end, pipeline_id, stage_index, num_stages)
    assert placed["n1"][1:3] == (0, layers - 3)
    assert placed["n2"][1:3] == (layers - 3, layers)
    assert placed["n1"][4] == 0 and placed["n2"][4] == 1
    assert placed["n1"][5] == placed["n2"][5] == 2


def test_a_one_node_run_and_a_two_node_run_of_the_same_model():
    """The measurement this was built for: same model, different node counts."""
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers

    asyncio.run(controller.deploy(model_id, stages=[{"node_id": "n1", "layers": layers}]))
    assert [n for (m, n) in controller.deployed if m == model_id] == ["n1"]

    asyncio.run(
        controller.deploy(
            model_id,
            stages=[
                {"node_id": "n1", "layers": layers // 2},
                {"node_id": "n2", "layers": layers - layers // 2},
            ],
        )
    )
    assert sorted(n for (m, n) in controller.deployed if m == model_id) == ["n1", "n2"]


def test_undeploy_frees_the_nodes_and_keeps_the_catalog_entry():
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    asyncio.run(controller.deploy(model_id, stages=[{"node_id": "n1", "layers": layers}]))

    assert asyncio.run(controller.undeploy(model_id)) is True
    assert controller.deployed == {}
    assert controller.registry.get(model_id) is not None, "still on offer"
    assert asyncio.run(controller.undeploy(model_id)) is False


def test_a_node_that_is_not_connected_is_named_in_the_error():
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    with pytest.raises(PlacementError, match="not connected right now: ghost"):
        asyncio.run(
            controller.deploy(model_id, stages=[{"node_id": "ghost", "layers": layers}])
        )
    assert controller.placements == {}, "a rejected deploy changes nothing"


def test_a_split_that_will_not_fit_is_refused_before_anything_downloads():
    controller = controller_with("catalog-demo.json", nodes=(("tiny", 1), ("n2", 40)))
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    with pytest.raises(PlacementError, match="does not fit"):
        asyncio.run(
            controller.deploy(model_id, stages=[{"node_id": "tiny", "layers": layers}])
        )
    # ...but an operator who knows better can say so.
    placement = asyncio.run(
        controller.deploy(
            model_id, stages=[{"node_id": "tiny", "layers": layers}], force=True
        )
    )
    assert placement.is_manual


def test_deploying_a_model_that_is_not_in_the_catalog_is_refused():
    controller = controller_with("catalog-demo.json")
    with pytest.raises(PlacementError, match="not in the catalog"):
        asyncio.run(controller.deploy("nope", stages=[{"node_id": "n1", "layers": 1}]))


def test_hand_placed_nodes_are_kept_away_from_the_broker():
    """Otherwise a brokered model would be handed VRAM already spoken for."""
    controller = controller_with("catalog-demo.json")
    first, second = controller.registry.list()[0], controller.registry.list()[1]
    layers = first.model_info.num_layers

    asyncio.run(controller.deploy(first.model_id, stages=[{"node_id": "n1", "layers": layers}]))
    controller.placements[second.model_id] = Placement.auto(second.model_id)
    asyncio.run(controller.rebalance(reason="test"))

    brokered = {n for (m, n) in controller.deployed if m == second.model_id}
    assert "n1" not in brokered, "the broker took a node a human had pinned"


def test_removing_a_model_takes_its_placement_with_it():
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    asyncio.run(controller.deploy(model_id, stages=[{"node_id": "n1", "layers": layers}]))
    assert asyncio.run(controller.remove_model(model_id)) is True
    assert model_id not in controller.placements
    assert controller.deployed == {}


def test_placement_view_offers_what_the_form_needs():
    controller = controller_with("catalog-demo.json")
    view = controller.placement_view()
    assert {n["node_id"] for n in view["nodes"]} == {"n1", "n2", "n3"}
    model = view["models"][0]
    assert model["deployed"] is False and model["placement"] is None
    assert model["num_layers"] > 0
    # The ceiling the form validates against, per node, for THIS model.
    assert set(model["max_layers_per_node"]) == {"n1", "n2", "n3"}
    assert all(v > 0 for v in model["max_layers_per_node"].values())

    layers = model["num_layers"]
    asyncio.run(
        controller.deploy(model["model_id"], stages=[{"node_id": "n1", "layers": layers}])
    )
    after = controller.placement_view()["models"][0]
    assert after["deployed"] is True
    assert after["placement"]["stages"][0]["node_id"] == "n1"


# ------------------------------------------------- surviving an orchestrator restart
def running_shard(model_id, *, node_id, start, end, stage_index, num_stages, port=9100):
    from types import SimpleNamespace

    return SimpleNamespace(
        model_id=model_id,
        start_layer=start,
        end_layer=end,
        current_requests=0,
        healthy=True,
        status="serving",
        local_port=port,
        pipeline_id=f"{model_id}#0",
        stage_index=stage_index,
        num_stages=num_stages,
        avg_layer_latency_ms=0.0,
    )


def feed_telemetry(controller, node_id, shards):
    from types import SimpleNamespace

    session = SimpleNamespace(node_id=node_id)
    asyncio.run(controller.on_telemetry(session, SimpleNamespace(shards=shards)))


def test_a_restart_adopts_what_was_already_running_instead_of_killing_it():
    """A fresh orchestrator knows nothing; the workers still hold the model.

    Re-syncing the routing table from telemetry was already there, but without
    a placement behind it the next rebalance saw a model nobody had asked for
    and tore down a healthy pipeline. Adoption closes that: restarting the
    orchestrator changes nothing on the GPUs.
    """
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    half = layers // 2

    feed_telemetry(controller, "n1", [running_shard(model_id, node_id="n1", start=0, end=half, stage_index=0, num_stages=2)])
    # One stage is a fragment, not a pipeline: nothing is adopted yet.
    assert controller.placements == {}

    feed_telemetry(controller, "n2", [running_shard(model_id, node_id="n2", start=half, end=layers, stage_index=1, num_stages=2)])
    placement = controller.placements.get(model_id)
    assert placement is not None and placement.is_manual
    assert [(s.node_id, s.start_layer, s.end_layer) for s in placement.stages] == [
        ("n1", 0, half),
        ("n2", half, layers),
    ]

    # And the decisive part: a rebalance now leaves the running shards alone.
    before = dict(controller.deployed)
    asyncio.run(controller.rebalance(reason="after restart"))
    assert controller.deployed == before, "a healthy pipeline was torn down"


def test_adoption_never_overrides_a_placement_someone_chose():
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    asyncio.run(controller.deploy(model_id, stages=[{"node_id": "n3", "layers": layers}]))

    feed_telemetry(controller, "n1", [running_shard(model_id, node_id="n1", start=0, end=layers, stage_index=0, num_stages=1)])
    assert controller.placements[model_id].node_ids() == ["n3"]


def test_a_stale_model_can_be_taken_down_after_a_restart():
    """The escape hatch: whatever a restart adopted, the operator can drop."""
    controller = controller_with("catalog-demo.json")
    model_id = controller.registry.list()[0].model_id
    layers = controller.registry.get(model_id).model_info.num_layers
    feed_telemetry(controller, "n1", [running_shard(model_id, node_id="n1", start=0, end=layers, stage_index=0, num_stages=1)])
    assert controller.placements.get(model_id) is not None

    assert asyncio.run(controller.undeploy(model_id)) is True
    assert controller.deployed == {}


# --------------------------------------------------- the whole path, for real
def test_one_node_then_two_through_the_running_stack():
    """The measurement itself: same model, one node, then two, no worker touched.

    Everything below the admin call is real — gRPC control plane, LoadShard,
    StartServing, the tunnel, the endpoint registry. What is faked is only the
    model: an echo backend, because this is about placement, not arithmetic.
    """
    from stack_utils import OrchestratorHarness, WorkerHarness, wait_until

    registry = ModelRegistry(
        [ModelSpec.from_dict(json.loads((CONFIGS / "demo-echo.json").read_text()))]
    )
    orch = OrchestratorHarness(registry, auto_deploy=False).start()
    workers = [
        WorkerHarness(orch.grpc_port, join_key=orch.join_key, node_id=f"b{i}", memory_gb=8).start()
        for i in range(2)
    ]
    controller = orch.controller
    model_id = registry.list()[0].model_id
    layers = registry.get(model_id).model_info.num_layers
    try:
        assert wait_until(lambda: len(controller.nodes) == 2, 20)
        # Nothing runs until asked, however long we wait.
        assert not controller.endpoints.candidates(model_id)

        # --- one node
        orch.submit(controller.deploy(model_id, stages=[{"node_id": "b0", "layers": layers}]))
        assert wait_until(lambda: bool(controller.endpoints.candidates(model_id)), 30)
        assert {ep.node_id for ep in controller.endpoints.candidates(model_id)} == {"b0"}
        assert sorted(n for (m, n) in controller.deployed if m == model_id) == ["b0"]

        # --- two nodes, same model, one admin call
        orch.submit(
            controller.deploy(
                model_id,
                stages=[
                    {"node_id": "b0", "layers": layers // 2},
                    {"node_id": "b1", "layers": layers - layers // 2},
                ],
            )
        )
        assert wait_until(
            lambda: sorted(n for (m, n) in controller.deployed if m == model_id) == ["b0", "b1"],
            30,
        )
        # Only the head answers clients; the tail serves activations.
        assert wait_until(
            lambda: {ep.node_id for ep in controller.endpoints.candidates(model_id)} == {"b0"}, 30
        )
        stages = {n: e[1:3] for (m, n), e in controller.deployed.items() if m == model_id}
        assert stages == {"b0": (0, layers // 2), "b1": (layers // 2, layers)}

        # --- and off again: the workers stop serving it
        orch.submit(controller.undeploy(model_id))
        assert wait_until(lambda: not controller.endpoints.candidates(model_id), 30)

        def worker_serving():
            return {
                mid
                for w in workers
                for mid, shard in w.state.snapshot().items()
                if shard.status.value == "serving"
            }

        assert wait_until(lambda: model_id not in worker_serving(), 20)
        assert controller.registry.get(model_id) is not None, "still in the catalog"
    finally:
        for w in workers:
            w.stop()
        orch.stop()
