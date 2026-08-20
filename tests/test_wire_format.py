"""How activations are encoded between stages.

The old format upcast everything to float32 "for portability". On a bfloat16
model that doubled every byte and carried no extra information — the receiving
stage narrowed it straight back to compute. Portability actually comes from the
dtype travelling with the tensor, which it always did.

Two properties are load-bearing here. Bytes must survive the round trip exactly,
and a dtype this build does not understand must be REFUSED. The second is the
important one: misreading bfloat16 bytes as float32 does not crash, it produces
plausible numbers, and a pipeline running on plausible numbers answers
confidently and wrongly.
"""

import sys
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")

from loom_worker.wire import (  # noqa: E402
    WireFormatError,
    from_wire,
    to_wire,
    wire_dtype_for,
)

HIDDEN = 4096  # Qwen3-8B's width: one decode step between two stages


# ------------------------------------------------------------- round trips
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_a_tensor_survives_the_wire_unchanged(dtype):
    original = torch.randn(2, 8).to(dtype)
    raw, shape, name = to_wire(torch, original)
    restored = from_wire(torch, raw, shape, name)

    assert restored.dtype == original.dtype
    assert list(restored.shape) == list(original.shape)
    assert torch.equal(restored, original), "the bytes did not come back the same"


def test_a_bfloat16_stage_sends_half_the_bytes():
    """The whole point, in one number."""
    activation = torch.randn(1, HIDDEN)
    wide, _, _ = to_wire(torch, activation)
    narrow, _, name = to_wire(torch, activation.bfloat16())

    assert len(wide) == HIDDEN * 4
    assert len(narrow) == HIDDEN * 2
    assert name == "bfloat16"


def test_a_float32_stage_is_never_narrowed_behind_its_back():
    """A CPU stage really does compute in float32; squeezing it loses digits.

    This is why the default is "the tensor's own dtype" rather than "always
    bfloat16" — the saving is only free when the model was already narrow.
    """
    activation = torch.randn(1, 16, dtype=torch.float32)
    _, _, name = to_wire(torch, activation)
    assert name == "float32"
    assert wire_dtype_for(torch, activation) == torch.float32


def test_an_exotic_dtype_falls_back_to_something_both_ends_know():
    """The wire vocabulary is deliberately small; float64 is not in it."""
    _, _, name = to_wire(torch, torch.randn(4, dtype=torch.float64))
    assert name == "float32"


# --------------------------------------------------------- refusing to guess
def test_an_unknown_dtype_is_refused_rather_than_misread():
    """The failure this prevents is silent and awful.

    Reading bfloat16 bytes as float32 succeeds — it yields numbers, half as
    many of them, all wrong. The pipeline then produces fluent, confident
    nonsense, which is exactly the class of bug this project has already spent
    days on.
    """
    with pytest.raises(WireFormatError, match="does not"):
        from_wire(torch, b"\x00" * 8, [4], "float8_e5m2")


def test_an_empty_dtype_is_refused_too():
    with pytest.raises(WireFormatError):
        from_wire(torch, b"\x00" * 8, [2], "")


def test_the_error_names_the_likely_cause():
    """Mixed worker images in one pipeline is the only way to reach this."""
    with pytest.raises(WireFormatError, match="same worker image"):
        from_wire(torch, b"\x00" * 4, [1], "quantum")


# ---------------------------------------------------- the operator's override
def test_the_wire_dtype_can_be_pinned(monkeypatch):
    """For measuring what the format costs, and for a pipeline that disagrees."""
    import loom_worker.wire as wire

    monkeypatch.setattr(wire, "WIRE_DTYPE", "bfloat16")
    activation = torch.randn(1, HIDDEN, dtype=torch.float32)
    raw, _, name = wire.to_wire(torch, activation)
    assert name == "bfloat16" and len(raw) == HIDDEN * 2

    monkeypatch.setattr(wire, "WIRE_DTYPE", "float32")
    raw, _, name = wire.to_wire(torch, activation.bfloat16())
    assert name == "float32" and len(raw) == HIDDEN * 4


# --------------------------------------------------- through the real executors
def test_both_engines_agree_on_the_format():
    """A shard stage and a vLLM stage are neighbours in the same pipeline.

    They are different classes with their own serialize/deserialize, so the
    only thing keeping them compatible is that both defer to wire.py.
    """
    from loom_worker.shard.executor import ShardExecutor
    from loom_worker.vllm_stage.executor import VllmStageExecutor

    activation = torch.randn(1, 32, dtype=torch.bfloat16)
    shard = ShardExecutor.__new__(ShardExecutor)
    shard.torch = torch
    vllm = VllmStageExecutor.__new__(VllmStageExecutor)
    vllm.torch = torch

    sent = shard.serialize(activation)
    assert torch.equal(vllm.deserialize(*sent), activation)

    sent = vllm.serialize(activation)
    assert torch.equal(shard.deserialize(*sent), activation)
    assert sent[2] == "bfloat16", "the dtype name must survive the vLLM envelope"


def test_the_vllm_named_set_keeps_its_names_and_its_dtype(monkeypatch):
    """Llama-family stages carry hidden_states AND residual to the next stage."""
    from test_vllm_stage import install_fake_vllm

    install_fake_vllm(monkeypatch)  # only IntermediateTensors is needed here
    from loom_worker.vllm_stage.executor import VllmStageExecutor

    vllm = VllmStageExecutor.__new__(VllmStageExecutor)
    vllm.torch = torch

    class Bundle:
        def __init__(self, tensors):
            self.tensors = tensors

    tensors = {
        "hidden_states": torch.randn(1, 16, dtype=torch.bfloat16),
        "residual": torch.randn(1, 16, dtype=torch.bfloat16),
    }
    raw, shape, dtype = vllm.serialize(Bundle(tensors))
    assert dtype.startswith("bfloat16|")
    assert len(raw) == 2 * 16 * 2, "both tensors, two bytes an element"

    restored = vllm.deserialize(raw, shape, dtype)
    assert set(restored.tensors) == {"hidden_states", "residual"}
    for name, original in tensors.items():
        assert torch.equal(restored.tensors[name], original)
