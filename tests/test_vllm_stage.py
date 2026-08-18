"""The Loom side of the vLLM pipeline engine, tested without a GPU.

vLLM cannot even be imported without CUDA, so the engine itself is verified on
a GPU stand (docs/VLLM_PIPELINE.md has the procedure). What IS testable here is
everything Loom contributes: which layers a stage claims, how a request is
registered and freed, prefill vs decode, and that a middle stage returns hidden
states while the tail returns logits. Those are the parts that decide whether
the pipeline is correct — vLLM's job is only to be fast.

A fake `vllm` package stands in for the engine. It records what we ask of it,
which is exactly what a real integration bug would show up in.
"""

import sys
import types
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

torch = pytest.importorskip("torch")


# --------------------------------------------------------------- fake engine
class FakeBlocks:
    def __init__(self, ids):
        self._ids = ids

    def get_block_ids(self):
        return self._ids


class FakeKvManager:
    """Stands in for vLLM's paged KV cache manager."""

    def __init__(self, capacity_blocks: int = 64):
        self.capacity = capacity_blocks
        self.allocated = 0
        self.freed = []
        self.calls = []

    def allocate_slots(self, *, request, num_new_tokens, num_new_computed_tokens=0, **kw):
        self.calls.append(("allocate", request.request_id, num_new_tokens))
        blocks = max(1, num_new_tokens // 16)
        if self.allocated + blocks > self.capacity:
            return None
        self.allocated += blocks
        return FakeBlocks([list(range(blocks))])

    def free(self, request):
        self.freed.append(request.request_id)


class FakeRunner:
    def __init__(self, *, is_last: bool, hidden_dim: int = 8, vocab: int = 32):
        self.is_last = is_last
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.device = torch.device("cpu")
        self.requests = {}
        self.request_block_hasher = None
        self.seen = []

    def execute_model(self, *, scheduler_output, intermediate_tensors=None):
        self.seen.append((scheduler_output, intermediate_tensors))
        tokens = scheduler_output.total_num_scheduled_tokens
        out = types.SimpleNamespace()
        if self.is_last:
            out.logits = torch.randn(tokens, self.vocab)
            out.hidden_states = None
        else:
            out.hidden_states = torch.randn(tokens, self.hidden_dim)
            out.logits = None
        return out


def install_fake_vllm(monkeypatch):
    """Minimal `vllm` package: only the symbols our glue actually touches."""
    modules = {}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        modules[name] = mod
        return mod

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Request:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class IntermediateTensors(dict):
        pass

    class SchedulerOutput:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class CachedRequestData:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @staticmethod
        def make_empty():
            return CachedRequestData(req_ids=[])

    class NewRequestData:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    module("vllm")
    module("vllm.sampling_params", SamplingParams=SamplingParams)
    module("vllm.sequence", IntermediateTensors=IntermediateTensors)
    module("vllm.v1")
    module("vllm.v1.request", Request=Request)
    module("vllm.v1.core")
    module("vllm.v1.core.sched")
    module(
        "vllm.v1.core.sched.output",
        SchedulerOutput=SchedulerOutput,
        CachedRequestData=CachedRequestData,
        NewRequestData=NewRequestData,
    )
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return modules


@pytest.fixture
def executor_factory(monkeypatch):
    install_fake_vllm(monkeypatch)
    from loom_worker.vllm_stage.executor import VllmStageExecutor
    from loom_worker.vllm_stage.runtime import StageRuntimeConfig

    def build(*, start=0, end=20, num_layers=40, capacity=64):
        config = StageRuntimeConfig(
            model_path="/tmp/model",
            start_layer=start,
            end_layer=end,
            num_layers=num_layers,
        )
        runner = FakeRunner(is_last=config.is_last)
        kv = FakeKvManager(capacity_blocks=capacity)
        kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()], num_blocks=capacity)
        return VllmStageExecutor(runner, kv, kv_cache_config, config), runner, kv

    return build


# ------------------------------------------------------------------- tests
def test_head_stage_prefills_then_decodes(executor_factory):
    """First call registers the request; later calls only add one token."""
    ex, runner, kv = executor_factory(start=0, end=20)
    prompt = [5, 6, 7, 8]

    hidden, logits = ex.forward(request_id="r1", positions=[0, 1, 2, 3], input_ids=prompt)
    assert logits is None, "a head that is not the tail returns hidden states"
    assert hidden.shape[0] == len(prompt)
    assert kv.calls[0] == ("allocate", "r1", len(prompt))

    ex.forward(request_id="r1", positions=[4], input_ids=[9])
    assert kv.calls[1] == ("allocate", "r1", 1), "decode must ask for one token"
    prefill, decode = runner.seen[0][0], runner.seen[1][0]
    assert prefill.total_num_scheduled_tokens == len(prompt)
    assert decode.total_num_scheduled_tokens == 1
    assert decode.scheduled_new_reqs == [], "the request is already known to vLLM"


def test_middle_stage_consumes_hidden_states(executor_factory):
    """A stage that owns neither end is fed activations, not tokens."""
    ex, runner, _kv = executor_factory(start=20, end=30, num_layers=40)
    incoming = torch.randn(1, 4, 8)

    hidden, logits = ex.forward(request_id="r2", positions=[0, 1, 2, 3], hidden=incoming)
    assert logits is None
    assert hidden is not None
    _, intermediate = runner.seen[0]
    assert intermediate is not None, "vLLM must receive the activations"
    assert intermediate["hidden_states"].shape == (4, 8), "flattened to vLLM's token layout"


def test_tail_stage_returns_logits_for_sampling(executor_factory):
    ex, _runner, _kv = executor_factory(start=20, end=40, num_layers=40)
    hidden, logits = ex.forward(request_id="r3", positions=[0, 1], hidden=torch.randn(1, 2, 8))
    assert hidden is None
    assert logits.dim() == 1, "logits of the last position only"
    token = ex.sample(logits)
    assert isinstance(token, int) and 0 <= token < logits.shape[0]


def test_freeing_a_request_returns_its_blocks(executor_factory):
    """Paged cache only pays off if blocks come back to the pool."""
    ex, _runner, kv = executor_factory()
    ex.forward(request_id="r4", positions=[0, 1], input_ids=[1, 2])
    assert ex.active_requests() == 1
    ex.free("r4")
    assert ex.active_requests() == 0
    assert kv.freed == ["r4"]
    ex.free("r4")  # idempotent: the head frees on every finished request


def test_kv_exhaustion_is_reported_not_crashed(executor_factory):
    """Out of blocks must say so, and name the knob that fixes it."""
    from loom_worker.vllm_stage.executor import KvCacheExhausted

    ex, _runner, _kv = executor_factory(capacity=1)
    with pytest.raises(KvCacheExhausted, match="LOOM_MAX_REQUESTS|LOOM_MAX_MODEL_LEN"):
        ex.forward(request_id="big", positions=list(range(64)), input_ids=list(range(64)))
    assert ex.active_requests() == 0, "a rejected request must not linger"


def test_stage_roles_come_from_the_layer_range(executor_factory):
    """is_first / is_last are geometry, not configuration."""
    head, _, _ = executor_factory(start=0, end=20, num_layers=40)
    middle, _, _ = executor_factory(start=20, end=30, num_layers=40)
    tail, _, _ = executor_factory(start=20, end=40, num_layers=40)
    assert (head.spec.is_first, head.spec.is_last) == (True, False)
    assert (middle.spec.is_first, middle.spec.is_last) == (False, False)
    assert (tail.spec.is_first, tail.spec.is_last) == (False, True)


def test_activations_round_trip_through_the_wire_format(executor_factory):
    """The stage server serialises with the executor it happens to have."""
    ex, _runner, _kv = executor_factory()
    tensor = torch.randn(1, 3, 8)
    data, shape, dtype = ex.serialize(tensor)
    back = ex.deserialize(data, shape, dtype)
    assert torch.allclose(back, tensor, atol=1e-6)


# ------------------------------------------------------- wiring, no vLLM needed
def test_backend_launches_the_stage_server_with_the_vllm_engine():
    from loom_worker.backends import make_backend

    backend = make_backend(
        "vllm_shard",
        model_id="qwen3-14b",
        weights_uri="Qwen/Qwen3-14B",
        start_layer=0,
        end_layer=20,
        vram_quota_bytes=20 * 1024**3,
        device="cuda",
        relay_url="http://127.0.0.1:1/relay",
        topology={
            "pipeline_id": "p",
            "stage_index": 0,
            "num_stages": 2,
            "is_first": True,
            "is_last": False,
            "num_model_layers": 40,
        },
    )
    cmd = backend.command()
    assert cmd[cmd.index("--engine") + 1] == "vllm"
    assert cmd[cmd.index("--num-model-layers") + 1] == "40"
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16", "fp32 would halve the KV cache"
    assert cmd[cmd.index("--vram-quota-bytes") + 1] == str(20 * 1024**3)


def test_backend_refuses_to_run_on_a_non_cuda_node():
    from loom_worker.backends import make_backend

    backend = make_backend(
        "vllm_shard",
        model_id="m",
        weights_uri="w",
        start_layer=0,
        end_layer=2,
        vram_quota_bytes=1,
        device="cpu",
        relay_url="",
        topology={"stage_index": 0, "num_stages": 1, "is_first": True, "is_last": True},
    )
    with pytest.raises(NotImplementedError, match="CUDA"):
        backend.prepare()
