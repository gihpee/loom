"""A Mac serving one stage of a model on its GPU.

The load-bearing claim is parity: a model cut into stages must produce the same
tokens as the same model run whole. Everything else — that the split is cheap,
that a Mac stage interoperates with a CUDA one — matters only if that holds,
because a pipeline that computes something slightly different is not a faster
pipeline, it is a broken one.

Most of this needs a real Apple GPU and is skipped elsewhere. What is NOT
skipped is the part every image must get right: that asking for this backend on
a machine that cannot run it fails immediately and says why.
"""

import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.backends import BACKENDS  # noqa: E402
from loom_worker.mlx_stage import mlx_available  # noqa: E402

ON_APPLE_GPU = (
    platform.system() == "Darwin"
    and platform.machine().startswith("arm")
    and mlx_available()
)
needs_metal = pytest.mark.skipif(not ON_APPLE_GPU, reason="needs Apple Silicon + mlx")

# Small, quantised, and it downloads in seconds — the point is the split, not
# the model.
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Hello"
STEPS = 8


# ------------------------------------------------------- registration, anywhere
def test_the_backend_is_registered_and_can_be_a_stage():
    """A Mac must be placeable as one stage of a pipeline, not just a whole model."""
    assert "mlx_shard" in BACKENDS
    assert BACKENDS["mlx_shard"].serves_partial_shard is True
    # The older whole-model MLX backend stays what it was.
    assert BACKENDS["mlx"].serves_partial_shard is False


def test_the_orchestrator_will_split_a_model_across_macs():
    from loom.orchestrator.placement import (
        KNOWN_BACKENDS,
        SHARDABLE_BACKENDS,
        check_backend_can_split,
    )

    assert "mlx_shard" in KNOWN_BACKENDS and "mlx_shard" in SHARDABLE_BACKENDS
    check_backend_can_split("mlx_shard", 3)  # must not raise


@pytest.mark.skipif(ON_APPLE_GPU, reason="this is the refusal on other hosts")
def test_asking_for_metal_on_a_non_mac_fails_immediately():
    """A clear refusal at load time beats a mystery at inference time."""
    backend = BACKENDS["mlx_shard"](
        model_id="m",
        weights_uri="m",
        start_layer=0,
        end_layer=4,
        vram_quota_bytes=1,
    )
    with pytest.raises(NotImplementedError, match="Apple Silicon"):
        backend.prepare()


# ------------------------------------------------------------- the real thing
@pytest.fixture(scope="module")
def whole_model():
    from mlx_lm import load

    return load(MODEL)


@pytest.fixture(scope="module")
def model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL)


def greedy_reference(model, tokenizer, steps=STEPS):
    """What the undivided model answers, greedily."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    ids = mx.array([tokenizer.encode(PROMPT)])
    out = []
    for _ in range(steps):
        logits = model(ids, cache=cache)
        token = int(mx.argmax(logits[:, -1, :]).item())
        out.append(token)
        ids = mx.array([[token]])
    return out


def build_stages(model_path, num_layers, cuts):
    """Executors for a pipeline cut at `cuts`, e.g. (0, 12, 24)."""
    from loom_worker.mlx_stage import MlxStageConfig, MlxStageExecutor, build_stage_model

    stages = []
    for start, end in zip(cuts, cuts[1:]):
        config = MlxStageConfig(
            model_path=model_path,
            start_layer=start,
            end_layer=end,
            num_layers=num_layers,
        )
        model, _ = build_stage_model(config)
        spec = SimpleNamespace(
            start_layer=start,
            end_layer=end,
            is_first=config.is_first,
            is_last=config.is_last,
        )
        stages.append(MlxStageExecutor(model, config, spec))
    return stages


def run_pipeline(stages, tokenizer, steps=STEPS):
    """Drive the stages exactly as the stage server does, wire included."""
    ids = tokenizer.encode(PROMPT)
    out = []
    for _ in range(steps):
        payload = None
        for index, stage in enumerate(stages):
            hidden, logits = stage.forward(
                request_id="r",
                positions=list(range(len(ids))),
                input_ids=ids if index == 0 else None,
                hidden=payload,
            )
            if hidden is not None:
                # Through the wire, not around it: serialisation is part of
                # what has to be correct.
                payload = stages[index + 1].deserialize(*stage.serialize(hidden))
        out.append(stages[-1].sample(logits))
        ids = [out[-1]]
    return out


@needs_metal
def test_two_stages_answer_exactly_like_the_whole_model(whole_model, model_path):
    """The one claim everything else rests on."""
    model, tokenizer = whole_model
    num_layers = len(model.model.layers)
    reference = greedy_reference(model, tokenizer)

    stages = build_stages(model_path, num_layers, (0, num_layers // 2, num_layers))
    assert run_pipeline(stages, tokenizer) == reference


@needs_metal
def test_three_stages_answer_the_same_too(whole_model, model_path):
    """Splitting further must not change the arithmetic, only who does it."""
    model, tokenizer = whole_model
    num_layers = len(model.model.layers)
    reference = greedy_reference(model, tokenizer)

    third = num_layers // 3
    stages = build_stages(model_path, num_layers, (0, third, 2 * third, num_layers))
    assert run_pipeline(stages, tokenizer) == reference


@needs_metal
def test_a_stage_holds_only_its_own_layers(model_path, whole_model):
    """Economy, and the reason a Mac can host part of a model too big for it."""
    model, _ = whole_model
    num_layers = len(model.model.layers)
    head, tail = build_stages(model_path, num_layers, (0, 6, num_layers))

    assert len(head.layers) == 6
    assert len(tail.layers) == num_layers - 6
    assert head.spec.is_first and not head.spec.is_last
    assert tail.spec.is_last and not tail.spec.is_first


@needs_metal
def test_freeing_a_request_releases_its_cache(model_path, whole_model):
    """Unified memory: a forgotten KV cache is RAM the machine cannot reuse."""
    model, tokenizer = whole_model
    num_layers = len(model.model.layers)
    stages = build_stages(model_path, num_layers, (0, num_layers))
    stage = stages[0]

    stage.forward(request_id="a", positions=[0], input_ids=tokenizer.encode(PROMPT))
    assert stage.active_requests() == 1
    stage.free("a")
    assert stage.active_requests() == 0
    stage.free("a")  # freeing twice is not an error


@needs_metal
def test_a_mac_stage_and_a_cuda_stage_speak_the_same_wire(model_path, whole_model):
    """They stand in one pipeline, so their bytes must match exactly.

    The MLX executor writes with mlx_to_wire and a torch stage reads with
    from_wire; nothing but the shared dtype vocabulary keeps them compatible.
    """
    torch = pytest.importorskip("torch")
    from loom_worker.wire import from_wire

    model, tokenizer = whole_model
    num_layers = len(model.model.layers)
    head = build_stages(model_path, num_layers, (0, 4))[0]

    hidden, _ = head.forward(
        request_id="w", positions=[0], input_ids=tokenizer.encode(PROMPT)
    )
    raw, shape, dtype = head.serialize(hidden)

    as_torch = from_wire(torch, raw, shape, dtype)
    assert list(as_torch.shape) == shape
    # The batch axis is dropped on the wire: peers exchange (tokens, hidden).
    assert len(shape) == 2

    back = head.deserialize(raw, shape, dtype)
    assert [float(x) for x in back.flatten().tolist()[:8]] == pytest.approx(
        [float(x) for x in as_torch.flatten().tolist()[:8]], rel=1e-6
    )
