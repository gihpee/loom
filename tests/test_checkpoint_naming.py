"""Reading a checkpoint whose tensors are not named the way we assumed.

This is the failure it comes from. A 27B MoE model was deployed over four
nodes; every stage started, reported healthy and served. The first request
died several stages away with "list index out of range" and no indication of
where. The logs of the one machine the operator could reach said it plainly:

    checkpoint index uses unknown key naming; reading every shard file
    shard weights loaded: 0 tensors (1199 irrelevant keys skipped)
    shard built: layers [32, 48) of 64 on cuda

Zero tensors. The stage was holding uninitialised memory and answering with
it. The last stage matched exactly one key — `lm_head.weight`, the only name
in our table without a prefix — which is what named the cause: the model nests
its layers deeper than `model.layers.N`.

Two things had to be true and were not: the naming must be read from the file
rather than assumed, and a stage that loaded nothing must refuse to serve.
"""

import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.shard.loader import (  # noqa: E402
    detect_key_prefix,
    needed_weight_files,
    shard_target_key,
)


def layer_keys(prefix, layers, per_layer=("self_attn.q_proj.weight", "mlp.gate.weight")):
    return [f"{prefix}layers.{i}.{name}" for i in range(layers) for name in per_layer]


# ------------------------------------------------------------- the naming
def test_the_plain_layout_is_unchanged():
    keys = layer_keys("model.", 4) + ["model.embed_tokens.weight", "lm_head.weight"]
    assert detect_key_prefix(keys, 4) == "model."


def test_a_nested_language_model_is_found():
    """What the 27B actually ships: the layers live deeper."""
    keys = layer_keys("model.language_model.", 64) + ["lm_head.weight"]
    assert detect_key_prefix(keys, 64) == "model.language_model."


def test_a_vision_tower_does_not_win_the_election():
    """A multimodal checkpoint has other things called "layers" too.

    Picking the wrong one would load a vision encoder's weights into the text
    model — which materialises, passes every shape check it meets, and answers
    with nonsense.
    """
    keys = (
        layer_keys("visual.blocks.", 8)          # more keys per layer, fewer layers
        + layer_keys("model.language_model.", 64)
        + ["lm_head.weight"]
    )
    assert detect_key_prefix(keys, 64) == "model.language_model."


def test_a_checkpoint_with_no_layers_at_all_says_so():
    assert detect_key_prefix(["lm_head.weight", "something.else"], 32) is None


# ------------------------------------------------------- what it then maps
def test_a_nested_key_maps_onto_this_stage():
    target = shard_target_key(
        "model.language_model.layers.35.self_attn.q_proj.weight",
        start_layer=32, end_layer=48, is_first=False, is_last=False,
        prefix="model.language_model.",
    )
    assert target == "layers.3.self_attn.q_proj.weight"


def test_the_head_is_found_whether_nested_or_not():
    """It sits at the top level in this checkpoint even though nothing else does."""
    for key in ("lm_head.weight", "model.language_model.lm_head.weight"):
        assert shard_target_key(
            key, start_layer=48, end_layer=64, is_first=False, is_last=True,
            prefix="model.language_model.",
        ) == "lm_head.weight"


def test_another_stage_s_layers_are_still_refused():
    assert shard_target_key(
        "model.language_model.layers.10.mlp.gate.weight",
        start_layer=32, end_layer=48, is_first=False, is_last=False,
        prefix="model.language_model.",
    ) is None


def test_only_the_files_this_stage_needs_are_downloaded():
    """The saving that made the index worth reading in the first place."""
    weight_map = {}
    for i in range(64):
        weight_map[f"model.language_model.layers.{i}.mlp.gate.weight"] = f"s{i // 16}.safetensors"
    weight_map["lm_head.weight"] = "s3.safetensors"

    files = needed_weight_files(
        weight_map, start_layer=32, end_layer=48, is_first=False, is_last=False
    )
    assert files == ["s2.safetensors"]


# ------------------------------------------ a stage with nothing must not serve
def test_a_stage_that_loaded_nothing_refuses_to_start(tmp_path, monkeypatch):
    """The heart of it: silence here cost a whole deployment.

    Loading zero tensors used to be an INFO line. The stage then started,
    reported healthy, was given a quarter of a 27B model to hold, and answered
    with uninitialised memory — so the failure surfaced somewhere else
    entirely, as an unrelated-looking crash.
    """
    import pytest
    from types import SimpleNamespace

    from loom_worker.shard.loader import ShardModel, ShardSpec

    torch = pytest.importorskip("torch")

    spec = ShardSpec(model_path=str(tmp_path), start_layer=32, end_layer=48,
                     is_first=False, is_last=False)
    model = ShardModel(spec, SimpleNamespace(num_hidden_layers=64), torch.float32)
    model.layers = torch.nn.ModuleList()  # a stage holding nothing
    model.embed = model.norm = model.lm_head = None
    monkeypatch.setattr(model, "_weight_files", lambda: [])
    monkeypatch.setattr(model, "_resolve_key_prefix", lambda files: "model.")

    with pytest.raises(RuntimeError, match="no weights matched this stage"):
        model._load_weights()
