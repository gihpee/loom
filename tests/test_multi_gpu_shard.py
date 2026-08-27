"""A host with several cards is one node, not one card and some idle silicon.

The worker used to offer device 0 and nothing else: it reported that card's
free memory, placed the whole stage on it, and left the rest of the machine
unused. An owner plugging in four GPUs contributed one.

Splitting inside the host is also much cheaper than splitting across hosts.
A layer boundary between two cards costs a copy over PCIe — microseconds for
one token. The same boundary between two machines costs a network hop, which
measured ~20 ms per token on a live stand. Same number of layers, three orders
of magnitude apart.

The cross-card path is exercised here on a CPU-only machine, because torch
treats "cpu" and "cpu:0" as different devices while moving tensors between
them for real. Everything the layers touch therefore travels the same code
path it would on two GPUs.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.shard.executor import ShardExecutor  # noqa: E402
from loom_worker.shard.loader import (  # noqa: E402
    ShardSpec,
    build_shard,
    plan_layer_devices,
    resolve_devices,
)
from make_tiny_model import ensure_tiny_model  # noqa: E402

PROMPT = [1, 2, 3, 4, 5]
TWO_DEVICES = [torch.device("cpu"), torch.device("cpu:0")]


# ------------------------------------------------------------ which cards
def test_cuda_means_every_card_not_the_first(monkeypatch):
    """The whole point: "cuda" used to mean card 0 and nothing else."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_devices("cuda") == [torch.device(f"cuda:{i}") for i in range(4)]


def test_a_pinned_card_stays_pinned(monkeypatch):
    """How an operator splits one machine between two workers."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_devices("cuda:2") == [torch.device("cuda:2")]
    assert resolve_devices("cpu") == [torch.device("cpu")]


def test_a_single_card_host_is_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_devices("cuda") == [torch.device("cuda")]


# ------------------------------------------------------------- the split
def test_layers_are_split_by_free_memory_not_evenly(monkeypatch):
    """A 4090 beside a 3090 should carry more of the model, not half of it."""
    from loom_worker.shard import loader

    free = {"a": 24, "b": 8}
    monkeypatch.setattr(loader, "_free_bytes", lambda d: free[str(d)])
    plan = plan_layer_devices(32, ["a", "b"])

    assert len(plan) == 32
    assert plan.count("a") == 24 and plan.count("b") == 8


def test_each_card_gets_one_contiguous_run():
    """k cards must cost k-1 crossings; interleaving would cost one per layer."""
    from loom_worker.shard import loader

    plan = plan_layer_devices(9, ["a", "b", "c"])
    boundaries = sum(1 for x, y in zip(plan, plan[1:]) if x != y)
    assert boundaries == 2, plan


def test_a_full_card_is_skipped_rather_than_given_layers(monkeypatch):
    """A card with no room cannot hold a layer; pretending otherwise OOMs."""
    from loom_worker.shard import loader

    free = {"full": 0, "free": 24}
    monkeypatch.setattr(loader, "_free_bytes", lambda d: free[str(d)])
    plan = plan_layer_devices(8, ["full", "free"])
    assert plan == ["free"] * 8


def test_one_device_needs_no_plan():
    assert plan_layer_devices(5, ["only"]) == ["only"] * 5


# --------------------------------------------------- the arithmetic itself
@pytest.fixture(scope="module")
def model_dir():
    return str(ensure_tiny_model())


def build(model_dir, devices, monkeypatch=None):
    shard, _ = build_shard(
        ShardSpec(model_path=model_dir, start_layer=0, end_layer=6,
                  is_first=True, is_last=True)
    )
    return shard


def test_layers_spread_over_two_cards_answer_exactly_the_same(model_dir, monkeypatch):
    """The claim everything else rests on: splitting must not change the answer.

    Where the layers happen to sit is an implementation detail of the host. If
    it changed the arithmetic even slightly, a node's output would depend on
    how many cards its owner had.
    """
    from loom_worker.shard import loader

    one = build(model_dir, None)
    logits_one = ShardExecutor(one).forward(
        request_id="a", positions=list(range(len(PROMPT))), input_ids=PROMPT
    )[1]

    monkeypatch.setattr(loader, "resolve_devices", lambda device: list(TWO_DEVICES))
    two = build(model_dir, None)
    assert len(set(str(d) for d in two.layer_devices)) == 2, "the split did not happen"
    logits_two = ShardExecutor(two).forward(
        request_id="a", positions=list(range(len(PROMPT))), input_ids=PROMPT
    )[1]

    assert torch.allclose(logits_one, logits_two, atol=1e-5), (
        "the same layers on two cards produced different logits"
    )


def test_generation_across_cards_matches_a_single_card(model_dir, monkeypatch):
    """Several steps, so the KV cache is exercised on both cards too."""
    from loom_worker.shard import loader

    def greedy(shard):
        ex = ShardExecutor(shard)
        ids, out = list(PROMPT), []
        positions = list(range(len(ids)))
        for _ in range(8):
            _hidden, logits = ex.forward(
                request_id="g", positions=positions, input_ids=ids
            )
            token = ex.sample(logits)
            out.append(token)
            ids, positions = [token], [positions[-1] + 1]
        return out

    one = greedy(build(model_dir, None))
    monkeypatch.setattr(loader, "resolve_devices", lambda device: list(TWO_DEVICES))
    two = greedy(build(model_dir, None))
    assert one == two, "a KV cache spread over two cards diverged"


# ------------------------------------------------ the engines that use a card
def test_vllm_is_told_the_size_of_one_card_not_the_host():
    """Summing VRAM for placement must not mislead the engines.

    vLLM's --gpu-memory-utilization is a fraction of the single card it runs
    on. A node with two 24 GB cards now declares 48 GB, which is right for
    deciding what fits on it and wrong for that fraction: a 20 GB quota would
    become 20/48 = 0.42, and vLLM would take 10 GB of one card instead of 20.
    """
    from types import SimpleNamespace

    from loom_worker.main import _per_card

    two_cards = SimpleNamespace(vram_total_bytes=48 * 1024**3, num_gpus=2)
    one_card = SimpleNamespace(vram_total_bytes=24 * 1024**3, num_gpus=1)

    assert _per_card(two_cards) == 24 * 1024**3
    assert _per_card(one_card) == 24 * 1024**3

    from loom_worker.backends.vllm import VllmBackend

    backend = VllmBackend(
        model_id="m", weights_uri="m", start_layer=0, end_layer=32,
        vram_quota_bytes=20 * 1024**3,
        total_vram_bytes=_per_card(two_cards),
    )
    assert 0.8 < backend.gpu_memory_utilization() <= 0.95


# ------------------------------------------- two workers, one machine, one name
def test_a_worker_given_some_cards_names_itself_apart(monkeypatch):
    """Two containers on one host must not answer to the same node id.

    With --network host they share a hostname, so both called themselves
    "nv3". Each registration evicted the other's session: the node blinked in
    the table once a second and served nothing, which reads as a network fault
    and is not one.
    """
    from loom_worker import main as worker_main

    monkeypatch.setattr(worker_main.socket, "gethostname", lambda: "nv3")

    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")
    first = worker_main.default_node_id()
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    second = worker_main.default_node_id()

    assert first != second, "two workers on one host took the same name"
    assert first.startswith("nv3-") and second.startswith("nv3-")


def test_a_worker_with_the_whole_machine_keeps_the_plain_hostname(monkeypatch):
    """Upgrading must not turn every existing node into a new one."""
    from loom_worker import main as worker_main

    monkeypatch.setattr(worker_main.socket, "gethostname", lambda: "nv3")
    for value in ("", "all"):
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", value)
        assert worker_main.default_node_id() == "nv3"


def test_the_name_is_stable_across_restarts(monkeypatch):
    """A node that renames itself on restart looks like a stranger each time."""
    from loom_worker import main as worker_main

    monkeypatch.setattr(worker_main.socket, "gethostname", lambda: "nv3")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    assert worker_main.default_node_id() == worker_main.default_node_id()


def test_an_explicit_node_id_still_wins(monkeypatch):
    """The operator's own name is never second-guessed."""
    from loom_worker.main import parse_args

    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "1")
    assert parse_args(["--node-id", "chosen"]).node_id == "chosen"
