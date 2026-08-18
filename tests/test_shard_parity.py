"""The core capability: a model split across stages must compute EXACTLY the
same thing as the whole model.

These tests run the stage executors in-process (no networking) so a failure
points at the shard math itself, not at transport. Networking is covered by
tests/test_multistage_e2e.py.
"""

import sys
from pathlib import Path

import pytest
from make_tiny_model import ensure_tiny_model

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from loom_worker.shard.executor import ShardExecutor  # noqa: E402
from loom_worker.shard.loader import ShardSpec, build_shard  # noqa: E402

PROMPT = [3, 17, 42, 8, 99]
SPLITS = [
    [(0, 6)],                       # whole model on one node
    [(0, 3), (3, 6)],               # two stages
    [(0, 2), (2, 4), (4, 6)],       # three stages
    [(0, 1), (1, 2), (2, 4), (4, 6)],  # uneven stages
]


@pytest.fixture(scope="module")
def model_dir():
    return str(ensure_tiny_model())


@pytest.fixture(scope="module")
def reference(model_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        logits = model(torch.tensor([PROMPT])).logits[0]
        generated = model.generate(
            torch.tensor([PROMPT]), max_new_tokens=12, do_sample=False, pad_token_id=0
        )[0, len(PROMPT) :].tolist()
    return {"model": model, "prefill_logits": logits, "tokens": generated}


def build_stages(model_dir, splits):
    execs = []
    for i, (start, end) in enumerate(splits):
        shard, _ = build_shard(
            ShardSpec(
                model_path=model_dir,
                start_layer=start,
                end_layer=end,
                is_first=(i == 0),
                is_last=(i == len(splits) - 1),
            )
        )
        execs.append(ShardExecutor(shard))
    return execs


def run_chain(execs, *, request_id, positions, input_ids):
    """One pass through the whole pipeline, serialising between stages."""
    hidden, logits = None, None
    for i, ex in enumerate(execs):
        hidden, logits = ex.forward(
            request_id=request_id,
            positions=positions,
            input_ids=input_ids if i == 0 else None,
            hidden=hidden,
        )
        if hidden is not None:
            # Round-trip through the wire format, exactly as the relay does.
            data, shape, dtype = ex.serialize(hidden)
            hidden = ex.deserialize(data, shape, dtype)
    return logits


@pytest.mark.parametrize("splits", SPLITS, ids=lambda s: f"{len(s)}-stage")
def test_prefill_logits_match_whole_model(model_dir, reference, splits):
    execs = build_stages(model_dir, splits)
    logits = run_chain(
        execs, request_id="r", positions=list(range(len(PROMPT))), input_ids=PROMPT
    )
    ref_last = reference["prefill_logits"][-1]
    assert torch.allclose(logits, ref_last, atol=1e-4), (
        f"max diff {(logits - ref_last).abs().max().item()}"
    )
    assert int(logits.argmax()) == int(ref_last.argmax())


@pytest.mark.parametrize("splits", SPLITS, ids=lambda s: f"{len(s)}-stage")
def test_greedy_generation_matches_whole_model(model_dir, reference, splits):
    """Full autoregressive loop with per-stage KV caches."""
    execs = build_stages(model_dir, splits)
    ids = list(PROMPT)
    positions = list(range(len(PROMPT)))
    step_input = list(PROMPT)
    generated = []
    for _ in range(12):
        logits = run_chain(
            execs, request_id="gen", positions=positions, input_ids=step_input
        )
        token = execs[-1].sample(logits)  # greedy
        generated.append(token)
        ids.append(token)
        positions = [len(ids) - 1]
        step_input = [token]
    assert generated == reference["tokens"]


def test_kv_cache_is_actually_used(model_dir):
    """Guards the bug class where a renamed kwarg silently disabled the cache.

    Decode with a cache must equal a full forward over the same prefix; if the
    cache were ignored, the second token's logits would be wrong.
    """
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    (ex,) = build_stages(model_dir, [(0, 6)])

    logits = ex.forward(request_id="kv", positions=list(range(len(PROMPT))), input_ids=PROMPT)[1]
    first = int(logits.argmax())
    decoded = ex.forward(request_id="kv", positions=[len(PROMPT)], input_ids=[first])[1]

    with torch.no_grad():
        expected = ref(torch.tensor([PROMPT + [first]])).logits[0, -1]
    assert torch.allclose(decoded, expected, atol=1e-4)


def test_stage_state_is_per_request(model_dir):
    """Two concurrent requests must not share KV state."""
    (ex,) = build_stages(model_dir, [(0, 6)])
    ex.forward(request_id="a", positions=list(range(len(PROMPT))), input_ids=PROMPT)
    ex.forward(request_id="b", positions=list(range(3)), input_ids=PROMPT[:3])
    assert ex.active_requests() == 2
    ex.free("a")
    assert ex.active_requests() == 1


def test_rejects_impossible_layer_range(model_dir):
    with pytest.raises(ValueError):
        build_shard(
            ShardSpec(model_path=model_dir, start_layer=4, end_layer=99, is_first=True, is_last=True)
        )


def test_concurrent_requests_do_not_corrupt_each_other(model_dir):
    """Requests run side by side on a stage — that is what fills the bubble.

    Each holds only its own KV cache, so the answers must be identical to
    running them one after another. This is the safety net for dropping the
    stage-wide lock.
    """
    import threading

    (ex,) = build_stages(model_dir, [(0, 6)])
    prompts = {
        "a": [3, 17, 42, 8, 99],
        "b": [7, 1, 200, 5],
        "c": [11, 12, 13, 14, 15, 16],
    }
    # Ground truth: one at a time, each on a fresh request id.
    expected = {}
    for name, ids in prompts.items():
        logits = ex.forward(
            request_id=f"seq-{name}", positions=list(range(len(ids))), input_ids=ids
        )[1]
        token = ex.sample(logits)
        nxt = ex.forward(
            request_id=f"seq-{name}", positions=[len(ids)], input_ids=[token]
        )[1]
        expected[name] = (token, nxt)

    got = {}
    errors = []

    def work(name, ids):
        try:
            logits = ex.forward(
                request_id=f"par-{name}", positions=list(range(len(ids))), input_ids=ids
            )[1]
            token = ex.sample(logits)
            nxt = ex.forward(
                request_id=f"par-{name}", positions=[len(ids)], input_ids=[token]
            )[1]
            got[name] = (token, nxt)
        except Exception as exc:  # surfaced below, threads swallow otherwise
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n, ids)) for n, ids in prompts.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)

    assert not errors, errors
    assert set(got) == set(prompts)
    for name in prompts:
        assert got[name][0] == expected[name][0], f"{name}: different token under concurrency"
        assert torch.allclose(got[name][1], expected[name][1], atol=1e-5), name


def test_activations_survive_the_wire_in_every_dtype(model_dir, monkeypatch):
    """bf16 halves every hop's payload; the round trip must still be exact."""
    (ex,) = build_stages(model_dir, [(0, 6)])
    hidden = torch.randn(1, 4, 64)

    for name, atol in (("float32", 0.0), ("bfloat16", 5e-2), ("float16", 1e-2)):
        monkeypatch.setenv("LOOM_WIRE_DTYPE", name)
        ex._wire_dtype = ex._pick_wire_dtype()
        data, shape, dtype = ex.serialize(hidden)
        assert dtype == name
        back = ex.deserialize(data, shape, dtype).to(torch.float32)
        assert tuple(back.shape) == tuple(hidden.shape)
        assert torch.allclose(back, hidden, atol=max(atol, 1e-6)), name
        expected_bytes = hidden.numel() * (4 if name == "float32" else 2)
        assert len(data) == expected_bytes
