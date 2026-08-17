"""A stage fetches and loads ONLY the weights of its own layers.

Two properties, both of which a 14B model split over two nodes made visible:
  - correctness: the layers a stage runs are its assigned global layers, with
    their own weights — not layer 0..n of the checkpoint;
  - economy: a stage reads (and downloads) only the safetensors files those
    layers live in, per `model.safetensors.index.json`.

The economy half is proved the hard way: the files another stage owns are
DELETED from disk, and the stage must still load and produce identical output.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from loom_worker.shard.loader import (  # noqa: E402
    ShardSpec,
    _weight_patterns_for_shard,
    build_shard,
    needed_weight_files,
    shard_target_key,
)

PROMPT = [3, 17, 42, 8, 99]
LAYERS = 6


# ------------------------------------------------------- key filter (unit)
def qwen_like_weight_map(num_layers=40, per_file=8):
    """weight_map of a multi-file checkpoint: layers spread over N files."""
    weight_map = {"model.embed_tokens.weight": "model-00001-of-00005.safetensors"}
    for layer in range(num_layers):
        file_no = layer // per_file + 1
        name = f"model-{file_no:05d}-of-00005.safetensors"
        for suffix in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
            weight_map[f"model.layers.{layer}.{suffix}"] = name
    weight_map["model.norm.weight"] = "model-00005-of-00005.safetensors"
    weight_map["lm_head.weight"] = "model-00005-of-00005.safetensors"
    return weight_map


def test_key_filter_keeps_only_this_stages_layers():
    kw = dict(start_layer=20, end_layer=40, is_first=False, is_last=True)
    assert shard_target_key("model.layers.19.mlp.down_proj.weight", **kw) is None
    # Global layer 20 is this stage's local layer 0.
    assert shard_target_key("model.layers.20.mlp.down_proj.weight", **kw) == (
        "layers.0.mlp.down_proj.weight"
    )
    assert shard_target_key("model.layers.39.mlp.down_proj.weight", **kw) == (
        "layers.19.mlp.down_proj.weight"
    )
    assert shard_target_key("model.embed_tokens.weight", **kw) is None  # not first
    assert shard_target_key("lm_head.weight", **kw) == "lm_head.weight"  # is last


def test_stage_needs_only_the_files_its_layers_live_in():
    weight_map = qwen_like_weight_map()
    head = needed_weight_files(
        weight_map, start_layer=0, end_layer=20, is_first=True, is_last=False
    )
    tail = needed_weight_files(
        weight_map, start_layer=20, end_layer=40, is_first=False, is_last=True
    )
    all_files = sorted(set(weight_map.values()))
    assert head == [f"model-{i:05d}-of-00005.safetensors" for i in (1, 2, 3)]
    assert tail == [f"model-{i:05d}-of-00005.safetensors" for i in (3, 4, 5)]
    # Each stage skips files it has no tensor in — that is the whole point.
    assert set(head) < set(all_files) and set(tail) < set(all_files)
    assert set(all_files) - set(head) == {"model-00004-of-00005.safetensors",
                                          "model-00005-of-00005.safetensors"}


def test_tied_head_pulls_the_embedding_file_for_the_last_stage():
    weight_map = qwen_like_weight_map()
    del weight_map["lm_head.weight"]  # tie_word_embeddings=True checkpoints
    tail = needed_weight_files(
        weight_map,
        start_layer=20,
        end_layer=40,
        is_first=False,
        is_last=True,
        tie_word_embeddings=True,
    )
    assert "model-00001-of-00005.safetensors" in tail  # holds embed_tokens


def test_unknown_key_naming_falls_back_to_everything(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"transformer.h.0.attn.weight": "a.safetensors"}})
    )
    spec = ShardSpec(model_path=str(tmp_path), start_layer=0, end_layer=1,
                     is_first=True, is_last=True)
    assert _weight_patterns_for_shard(tmp_path, spec) == ["*.safetensors"]


def test_single_file_checkpoint_has_nothing_to_select(tmp_path):
    spec = ShardSpec(model_path=str(tmp_path), start_layer=0, end_layer=1,
                     is_first=True, is_last=True)
    assert _weight_patterns_for_shard(tmp_path, spec) == ["*.safetensors"]


# ------------------------------------------------- real multi-file checkpoint
@pytest.fixture(scope="module")
def sharded_model(tmp_path_factory):
    """A tiny model saved as SEVERAL safetensors files, with an index."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    path = tmp_path_factory.mktemp("sharded-llama")
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    # Force a multi-file layout, like any real 8B+ checkpoint.
    model.save_pretrained(path, safe_serialization=True, max_shard_size="60KB")
    index = path / "model.safetensors.index.json"
    assert index.exists(), "test needs a multi-file checkpoint"
    assert len(set(json.loads(index.read_text())["weight_map"].values())) > 2
    return path


def stage_spec(model_path, start, end, *, num_stages=2, index=0):
    return ShardSpec(
        model_path=str(model_path),
        start_layer=start,
        end_layer=end,
        is_first=index == 0,
        is_last=index == num_stages - 1,
    )


def test_stage_reads_a_strict_subset_of_the_files(sharded_model):
    shard, _ = build_shard(stage_spec(sharded_model, 0, 2))
    all_files = sorted(
        set(json.loads((sharded_model / "model.safetensors.index.json").read_text())["weight_map"].values())
    )
    used = {Path(f).name for f in shard._weight_files()}
    assert used < set(all_files), f"stage opened every file: {used}"


def test_stage_loads_with_the_other_stages_files_deleted(sharded_model, tmp_path):
    """The strongest form of 'only its own weights': the rest is not on disk."""
    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(sharded_model, dtype=torch.float32).eval()
    full_shard, _ = build_shard(stage_spec(sharded_model, 3, LAYERS, index=1))

    # Copy the checkpoint, then keep only what the tail stage needs.
    pruned = tmp_path / "tail-only"
    shutil.copytree(sharded_model, pruned)
    spec = stage_spec(pruned, 3, LAYERS, index=1)
    keep = {Path(f).name for f in full_shard._weight_files()}
    removed = []
    for file in pruned.glob("*.safetensors"):
        if file.name not in keep:
            file.unlink()
            removed.append(file.name)
    assert removed, "nothing was removable — the fixture is not multi-file"

    shard, _ = build_shard(spec)
    hidden = torch.randn(1, len(PROMPT), reference.config.hidden_size)
    with torch.no_grad():
        a = shard.layers[0](
            hidden,
            position_ids=torch.tensor([list(range(len(PROMPT)))]),
            position_embeddings=shard.rotary(hidden, torch.tensor([list(range(len(PROMPT)))])),
        )
        b = full_shard.layers[0](
            hidden,
            position_ids=torch.tensor([list(range(len(PROMPT)))]),
            position_embeddings=full_shard.rotary(
                hidden, torch.tensor([list(range(len(PROMPT)))])
            ),
        )
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert torch.equal(a, b)


def test_stage_layers_are_its_own_global_layers(sharded_model):
    """Stage [3,6) must hold the checkpoint's layers 3..5, not 0..2."""
    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(sharded_model, dtype=torch.float32).eval()
    shard, _ = build_shard(stage_spec(sharded_model, 3, LAYERS, index=1))

    assert len(shard.layers) == LAYERS - 3
    assert shard.embed is None, "a middle/tail stage must not carry embeddings"
    for local, global_idx in enumerate(range(3, LAYERS)):
        mine = shard.layers[local].self_attn.q_proj.weight
        theirs = reference.model.layers[global_idx].self_attn.q_proj.weight
        assert torch.equal(mine, theirs), f"local {local} != global {global_idx}"
        wrong = reference.model.layers[local].self_attn.q_proj.weight
        assert not torch.equal(mine, wrong), "stage loaded the checkpoint's first layers"


def test_head_stage_has_embeddings_but_no_head(sharded_model):
    shard, _ = build_shard(stage_spec(sharded_model, 0, 3, index=0))
    assert shard.embed is not None
    assert shard.lm_head is None and shard.norm is None
    assert len(shard.layers) == 3
