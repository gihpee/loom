"""A model trained in pieces must learn what the whole model would learn.

Everything else about distributed training is an optimisation. This is the
correctness claim, and it is the same one the inference side rests on: if the
gradients differ from what one process would compute, the fleet is not
training the model — it is training something else that resembles it.

The chain here is real backpropagation cut into pieces. Each stage keeps the
graph between what came in and what went out; when the gradient of the loss
arrives from the stage after it, the stage differentiates its own piece and
hands the result to the stage before. Only numbers cross the boundary, never
the graph.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.shard.loader import ShardSpec, build_shard  # noqa: E402
from loom_worker.train import TrainingStage  # noqa: E402
from make_tiny_model import ensure_tiny_model  # noqa: E402

TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
SPLITS = [[(0, 6)], [(0, 3), (3, 6)], [(0, 2), (2, 4), (4, 6)]]


@pytest.fixture(scope="module")
def model_dir():
    return str(ensure_tiny_model())


def global_name(stage, local_name):
    """`layers.2.self_attn.q_proj` on stage 1 -> the whole model's numbering."""
    parts = local_name.split(".")
    parts[1] = str(int(parts[1]) + stage.spec.start_layer)
    return ".".join(parts)


def adapters_by_global_name(stages):
    return {
        global_name(stage, name): adapter
        for stage in stages
        for name, adapter in stage.adapters.items()
    }


def copy_adapters(source, target):
    """Give both layouts the same adapters, so only the split differs.

    Necessary because B starts at zero: with it, the forward pass ignores A
    entirely and any two layouts agree on the loss whatever A holds. The
    gradients do not — they are the first thing that sees A — so comparing
    them means anything only when A is the same on both sides.
    """
    src = adapters_by_global_name(source)
    for name, adapter in adapters_by_global_name(target).items():
        with torch.no_grad():
            adapter.lora_a.copy_(src[name].lora_a)
            adapter.lora_b.copy_(src[name].lora_b)


def build_stages(model_dir, splits, *, seed=0):
    """One TrainingStage per split."""
    stages = []
    for i, (start, end) in enumerate(splits):
        torch.manual_seed(seed)
        shard, _ = build_shard(
            ShardSpec(
                model_path=model_dir,
                start_layer=start,
                end_layer=end,
                is_first=(i == 0),
                is_last=(i == len(splits) - 1),
            )
        )
        stages.append(
            TrainingStage(shard, shard.spec, rank=4, alpha=8.0, lr=1e-3)
        )
    return stages


def train_step(stages, tokens, *, micro_id="m0"):
    """One micro-batch through the whole chain, forward then backward."""
    hidden, loss = None, None
    for i, stage in enumerate(stages):
        hidden, loss = stage.forward(
            micro_id=micro_id,
            input_ids=tokens if i == 0 else None,
            hidden=hidden,
            labels=tokens if i == len(stages) - 1 else None,
        )
    grad = None
    for stage in reversed(stages):
        grad = stage.backward(micro_id=micro_id, grad_output=grad)
    return float(loss.detach())


# --------------------------------------------------------------- the claim
@pytest.mark.parametrize("splits", SPLITS, ids=lambda s: f"{len(s)}-stage")
def test_the_loss_is_the_same_however_the_model_is_split(model_dir, splits):
    """Splitting changes who does the work, not what the work computes."""
    whole = train_step(build_stages(model_dir, SPLITS[0]), TOKENS)
    split = train_step(build_stages(model_dir, splits), TOKENS)
    assert split == pytest.approx(whole, rel=1e-5), (
        f"{len(splits)} stages disagreed with one about the loss"
    )


@pytest.mark.parametrize("splits", SPLITS[1:], ids=lambda s: f"{len(s)}-stage")
def test_the_gradients_are_the_same_however_the_model_is_split(model_dir, splits):
    """The loss matching is not enough — it is the gradients that train."""
    whole = build_stages(model_dir, SPLITS[0])
    stages = build_stages(model_dir, splits)
    copy_adapters(whole, stages)

    train_step(whole, TOKENS)
    train_step(stages, TOKENS)

    reference = {n: a.lora_b.grad for n, a in adapters_by_global_name(whole).items()}
    got = {n: a.lora_b.grad for n, a in adapters_by_global_name(stages).items()}

    assert set(got) == set(reference), "the adapters themselves came out different"
    for key, expected in reference.items():
        assert torch.allclose(got[key], expected, atol=1e-6), f"gradient differs at {key}"


# ------------------------------------------------------- what training does
def test_a_step_actually_moves_the_adapters(model_dir):
    """A pipeline that computes gradients and applies nothing trains nothing."""
    stages = build_stages(model_dir, SPLITS[1])
    before = stages[0].adapters["layers.0.self_attn.q_proj"].lora_b.detach().clone()

    train_step(stages, TOKENS)
    report = [stage.step() for stage in stages]

    after = stages[0].adapters["layers.0.self_attn.q_proj"].lora_b
    assert not torch.allclose(before, after), "the adapter never moved"
    assert all(r["step"] == 1 for r in report)


def test_training_lowers_the_loss_on_what_it_is_shown(model_dir):
    """The end-to-end claim, small but real: the pipeline learns."""
    stages = build_stages(model_dir, SPLITS[1])
    first = train_step(stages, TOKENS, micro_id="a")
    for stage in stages:
        stage.step()
    for i in range(6):
        train_step(stages, TOKENS, micro_id=f"m{i}")
        for stage in stages:
            stage.step()
    last = train_step(stages, TOKENS, micro_id="z")

    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_the_base_model_is_left_frozen(model_dir):
    """Only adapters train. A stage that moved base weights would drift apart
    from every other copy of the model in the fleet, including the ones
    serving inference from the same files."""
    stages = build_stages(model_dir, SPLITS[1])
    for stage in stages:
        for layer in stage.shard.layers:
            for name, param in layer.named_parameters(recurse=True):
                if "lora" in name:
                    continue
                assert not param.requires_grad, f"{name} is still trainable"


# ------------------------------------------------------- keeping the memory
def test_a_finished_micro_batch_stops_holding_its_activations(model_dir):
    """Activations are the biggest thing a step touches; leaking them ends it."""
    stages = build_stages(model_dir, SPLITS[1])
    train_step(stages, TOKENS, micro_id="one")
    assert all(stage.in_flight() == 0 for stage in stages)


def test_several_micro_batches_can_be_in_flight_at_once(model_dir):
    """What makes training worth spreading out at all.

    Decoding one sequence uses one stage at a time. Training overlaps: while a
    later stage goes backward over one micro-batch, an earlier one is already
    going forward over the next, so every machine has work and the hop between
    them is paid once per micro-batch instead of once per token.
    """
    stages = build_stages(model_dir, SPLITS[1])
    for micro in ("a", "b", "c"):
        hidden, _loss = None, None
        for i, stage in enumerate(stages):
            hidden, _loss = stage.forward(
                micro_id=micro,
                input_ids=TOKENS if i == 0 else None,
                hidden=hidden,
                labels=TOKENS if i == len(stages) - 1 else None,
            )
    assert [s.in_flight() for s in stages] == [3, 3]

    for micro in ("a", "b", "c"):  # gradients may come back in any order
        grad = None
        for stage in reversed(stages):
            grad = stage.backward(micro_id=micro, grad_output=grad)
    assert all(stage.in_flight() == 0 for stage in stages)


def test_a_gradient_for_an_unknown_micro_batch_is_refused(model_dir):
    stages = build_stages(model_dir, SPLITS[1])
    with pytest.raises(KeyError, match="no forward pass is waiting"):
        stages[0].backward(micro_id="never-happened")
