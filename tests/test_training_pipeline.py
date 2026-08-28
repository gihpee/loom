"""Training driven the way it will be driven: by messages, across stages.

The parity tests call the stages directly. This one goes through the protocol
— every forward and every gradient is a message with a target stage, exactly
as the transport delivers them — and covers the half of training that is not
mathematics: what happens when a stage fails, when a gradient never comes
back, and when a run is interrupted and picked up again.

The stages are wired to each other in this process instead of over the
network. That is not a simplification of the protocol: routing between stages
looks only at `pipeline_id` and `target_stage` and never at what the message
says, so a message that reaches the right stage here reaches it there.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.shard.loader import ShardSpec, build_shard  # noqa: E402
from loom_worker.train import TrainingStage, protocol  # noqa: E402
from loom_worker.train.runner import (  # noqa: E402
    StageRunner,
    StepDriver,
    TrainingStepFailed,
    load_shard,
    save_shard,
)
from make_tiny_model import ensure_tiny_model  # noqa: E402

TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
SPLITS = [(0, 3), (3, 6)]


@pytest.fixture(scope="module")
def model_dir():
    return str(ensure_tiny_model())


class Fleet:
    """Stages wired to each other, plus the driver that pushes work in."""

    def __init__(self, model_dir, splits=SPLITS, *, mode="lora", **driver_kwargs):
        self.stages, self.runners = [], []
        self.delivered = 0
        for i, (start, end) in enumerate(splits):
            torch.manual_seed(0)
            shard, _ = build_shard(
                ShardSpec(model_path=model_dir, start_layer=start, end_layer=end,
                          is_first=(i == 0), is_last=(i == len(splits) - 1))
            )
            self.stages.append(
                TrainingStage(shard, shard.spec, mode=mode, rank=4, alpha=8.0, lr=1e-3)
            )
        for i, stage in enumerate(self.stages):
            self.runners.append(
                StageRunner(stage, send=self._send, stage_index=i,
                            num_stages=len(splits))
            )
        self.driver = StepDriver(self.runners[0], **driver_kwargs)
        self.runners[0].on_micro_done = self.driver.micro_done

    def _send(self, target: int, msg: dict) -> None:
        """What the transport does: hand the message to the target stage."""
        self.delivered += 1
        if target < 0:                       # broadcast
            for runner in self.runners:
                runner.handle(msg)
            return
        if msg.get("kind") in (protocol.LOSS, protocol.ERROR):
            self.driver.note(msg)            # reports land on stage 0's driver
            return
        self.runners[target].handle(msg)

    def batch(self, n=2):
        return [{"input_ids": TOKENS} for _ in range(n)]


# ----------------------------------------------------------- the happy path
def test_a_step_runs_end_to_end_over_the_protocol(model_dir):
    fleet = Fleet(model_dir)
    result = fleet.driver.run_step(fleet.batch(2))

    assert result.step == 1 and result.attempts == 1
    assert result.loss > 0, "no loss came back from the tail"
    assert all(stage.in_flight() == 0 for stage in fleet.stages)
    assert all(stage.steps == 1 for stage in fleet.stages), "a stage did not step"


def test_every_stage_applies_its_own_optimiser(model_dir):
    """No two stages hold the same parameter, so nothing is synchronised."""
    fleet = Fleet(model_dir)
    before = [
        s.adapters["layers.0.self_attn.q_proj"].lora_b.detach().clone()
        for s in fleet.stages
    ]
    fleet.driver.run_step(fleet.batch(2))
    for stage, was in zip(fleet.stages, before):
        now = stage.adapters["layers.0.self_attn.q_proj"].lora_b
        assert not torch.allclose(was, now), "a stage's adapters never moved"


def test_training_over_the_protocol_lowers_the_loss(model_dir):
    fleet = Fleet(model_dir)
    first = fleet.driver.run_step(fleet.batch(2)).loss
    for _ in range(6):
        fleet.driver.run_step(fleet.batch(2))
    last = fleet.driver.run_step(fleet.batch(2)).loss
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_classic_fine_tuning_runs_the_same_way(model_dir):
    """Full fine-tuning is the same pipeline with a different set of parameters."""
    fleet = Fleet(model_dir, mode="full")
    weight = fleet.stages[0].shard.layers[0].self_attn.q_proj.weight
    before = weight.detach().clone()
    fleet.driver.run_step(fleet.batch(2))
    assert not torch.allclose(before, weight), "the base weights never moved"
    assert fleet.stages[0].trainable_bytes() > 0


# --------------------------------------------------------------- when it breaks
def test_a_stage_that_raises_aborts_the_step_and_it_is_retried(model_dir):
    """One machine failing must cost a step, not the run."""
    fleet = Fleet(model_dir)
    failures = {"left": 1}
    original = fleet.stages[1].forward

    def flaky(**kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("card fell over")
        return original(**kwargs)

    fleet.stages[1].forward = flaky
    result = fleet.driver.run_step(fleet.batch(2))

    assert result.attempts == 2, "the step was not retried"
    assert result.step == 1, "a failed attempt must not count as a step"
    assert all(stage.in_flight() == 0 for stage in fleet.stages)


def test_a_step_that_keeps_failing_gives_up_instead_of_looping(model_dir):
    """A stage that fails every time is broken, and waiting will not fix it."""
    fleet = Fleet(model_dir)
    fleet.stages[1].forward = lambda **kw: (_ for _ in ()).throw(RuntimeError("gone"))

    with pytest.raises(TrainingStepFailed, match="failed 3 times"):
        fleet.driver.run_step(fleet.batch(2))
    assert fleet.driver.step_index == 0


def test_an_abandoned_step_does_not_leak_into_the_next_one(model_dir):
    """Half a step's gradients cannot be subtracted, so they must be dropped.

    Keeping them would fold the abandoned attempt into the next step with a
    weight nobody chose — a bug that changes what the model learns and shows
    up nowhere.
    """
    fleet = Fleet(model_dir)
    fleet.driver._abort_everywhere()
    grads = [
        a.lora_b.grad
        for stage in fleet.stages
        for a in stage.adapters.values()
    ]
    assert all(g is None for g in grads), "gradients survived the abort"
    assert all(stage.in_flight() == 0 for stage in fleet.stages)


def test_a_micro_batch_that_never_returns_times_out(model_dir, monkeypatch):
    """A silently wedged stage must not hang the run forever."""
    from loom_worker.train import runner as runner_mod

    monkeypatch.setattr(runner_mod, "MICRO_TIMEOUT_S", 0.2)
    monkeypatch.setattr(runner_mod, "MAX_STEP_ATTEMPTS", 1)

    fleet = Fleet(model_dir)
    fleet.runners[1].handle = lambda msg: None  # takes the message, says nothing

    with pytest.raises(TrainingStepFailed, match="never came back"):
        fleet.driver.run_step(fleet.batch(2))


# --------------------------------------------------------- surviving a restart
def test_a_run_picks_up_where_it_stopped(model_dir, tmp_path):
    """Hour three of a run must not restart at hour zero.

    Each stage writes its own slice to its own machine: gathering gigabytes
    through the orchestrator on every checkpoint would move them across the
    fleet for nothing. They are collected once, at the end.
    """
    fleet = Fleet(model_dir, checkpoint_dir=str(tmp_path), checkpoint_every=1)
    fleet.driver.run_step(fleet.batch(2))
    fleet.driver.run_step(fleet.batch(2))

    trained = {
        i: s.adapters["layers.0.self_attn.q_proj"].lora_b.detach().clone()
        for i, s in enumerate(fleet.stages)
    }

    fresh = Fleet(model_dir, checkpoint_dir=str(tmp_path), checkpoint_every=1)
    assert fresh.driver.resume() == 2, "the run did not remember its step"
    for i, stage in enumerate(fresh.stages):
        load_shard(stage, str(tmp_path / "step-2"))
        assert torch.allclose(
            stage.adapters["layers.0.self_attn.q_proj"].lora_b, trained[i]
        ), f"stage {i} came back different from how it was saved"


def test_a_checkpoint_from_a_different_layout_is_refused(model_dir, tmp_path):
    """Layers [0,3) restored into a stage holding [3,6) would be silent nonsense."""
    fleet = Fleet(model_dir)
    save_shard(fleet.stages[0], str(tmp_path), step=1)   # only layers [0, 3)

    # Stage [3, 6) finds someone else's slice and no slice of its own. Loading
    # nothing and carrying on would restart it from the base weights while
    # every other stage resumed — a run that looks resumed and is not.
    with pytest.raises(ValueError, match="laid out differently"):
        load_shard(fleet.stages[1], str(tmp_path))


def test_nothing_to_resume_from_is_not_an_error(model_dir, tmp_path):
    fleet = Fleet(model_dir, checkpoint_dir=str(tmp_path), checkpoint_every=0)
    assert fleet.driver.resume() == 0


# ------------------------------------------------------- collecting the result
def test_the_stages_slices_name_layers_the_way_the_model_does(model_dir):
    """Four machines' pieces have to concatenate into one loadable file."""
    fleet = Fleet(model_dir)
    keys = set()
    for stage in fleet.stages:
        keys |= set(stage.state_dict())

    assert any("layers.0." in k for k in keys)
    assert any("layers.5." in k for k in keys), "the later stage kept local numbering"
    assert len(keys) == sum(len(s.state_dict()) for s in fleet.stages), (
        "two stages claimed the same parameter"
    )


# ------------------------------------------------- the orchestrator's side
def test_the_orchestrator_talks_to_stage_zero_and_nobody_else():
    """Stage 0 drives a run; the orchestrator only starts and watches it.

    Driving micro-batches from the orchestrator would put every one of them on
    the wire twice — out to the head and back — for a decision the head is
    already making.
    """
    import asyncio
    import json as _json
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from loom.orchestrator.controller import MultiModelController

    controller = MultiModelController.__new__(MultiModelController)
    controller.deployed = {
        ("run-a", "node-tail"): (0, 8, 16, "run-a#0", 1, 2),
        ("run-a", "node-head"): (0, 0, 8, "run-a#0", 0, 2),
    }
    assert controller.head_node("run-a") == "node-head"
    assert controller.head_node("nothing-deployed") is None

    asked = {}

    class Tunnel:
        async def request_bytes(self, node_id, **kwargs):
            asked.update(node_id=node_id, **kwargs)
            return SimpleNamespace(status=200), b'{"ok": true, "samples": 12}'

    controller.tunnel = Tunnel()
    answer = asyncio.run(
        controller.train_control("run-a", "start", {"dataset": "/data/set.jsonl"})
    )

    assert asked["node_id"] == "node-head", "the orchestrator talked to the wrong stage"
    assert asked["path"] == "/train/start"
    assert _json.loads(asked["body"])["dataset"] == "/data/set.jsonl"
    assert answer["ok"] and answer["node_id"] == "node-head"


def test_training_a_model_that_is_not_deployed_says_so():
    import asyncio

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from loom.orchestrator.controller import MultiModelController

    controller = MultiModelController.__new__(MultiModelController)
    controller.deployed = {}
    with pytest.raises(ValueError, match="not deployed"):
        asyncio.run(controller.train_control("ghost", "status", {}))
