"""Батч, собранный снаружи.

vLLM тут подменён: проверяется не он, а наши решения — сколько токенов
считать, сколько блоков просить и что делать, когда их не дали. Ошибка здесь
не падает, а расходится позициями в KV-кэше, и стадия начинает отвечать
связной чушью.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def vllm(monkeypatch):
    """Внутренности vLLM, которых на этой машине нет."""
    made = {}

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.num_computed_tokens = 0
            self.outputs = []

        def append_output_token_ids(self, ids):
            self.outputs.extend(ids)

    class Blocks:
        def __init__(self, ids):
            self.ids = ids

        def get_block_ids(self, allow_none=False):
            return self.ids

        def __add__(self, other):
            return Blocks(self.ids + other.ids)

    class NewRequestData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class CachedRequestData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @staticmethod
        def make_empty():
            return CachedRequestData(req_ids=[])

    class SchedulerOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    for name, module in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.sampling_params", types.ModuleType("vllm.sampling_params")),
        ("vllm.v1", types.ModuleType("vllm.v1")),
        ("vllm.v1.request", types.ModuleType("vllm.v1.request")),
        ("vllm.v1.core", types.ModuleType("vllm.v1.core")),
        ("vllm.v1.core.sched", types.ModuleType("vllm.v1.core.sched")),
        ("vllm.v1.core.sched.output", types.ModuleType("vllm.v1.core.sched.output")),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.sampling_params"].SamplingParams = SamplingParams
    sys.modules["vllm.v1.request"].Request = Request
    out = sys.modules["vllm.v1.core.sched.output"]
    out.NewRequestData = NewRequestData
    out.CachedRequestData = CachedRequestData
    out.SchedulerOutput = SchedulerOutput
    made["Blocks"] = Blocks
    return made


class Manager:
    """Менеджер KV-кэша: выдаёт блоки, пока не откажет."""

    def __init__(self, *, refuse_after: int = 999, blocks_cls=None) -> None:
        self.refuse_after = refuse_after
        self.given = 0
        self.freed = []
        self.blocks_cls = blocks_cls

    def get_computed_blocks(self, request):
        return self.blocks_cls([]), 0

    def allocate_slots(self, *, request, num_new_tokens, num_new_computed_tokens,
                       new_computed_blocks=None):
        if self.given >= self.refuse_after:
            return None
        self.given += 1
        return self.blocks_cls([self.given])

    def free(self, request):
        self.freed.append(getattr(request, "request_id", "?"))


def runner_with(manager):
    return types.SimpleNamespace(kv_cache_manager=manager, requests={},
                                 kv_cache_config=types.SimpleNamespace(
                                     kv_cache_groups=[object()]))


def sequence(request_id="r1", prompt=None, outputs=None):
    from loom_stage.vllm_batch import Sequence

    return Sequence(request_id=request_id, prompt_ids=prompt or [1, 2, 3],
                    output_ids=list(outputs or []))


# --------------------------------------------------------- сколько посчитано
def test_на_первом_шаге_посчитан_весь_промпт():
    assert sequence(prompt=[1, 2, 3]).computed == 3


def test_дальше_промпт_плюс_выданное_кроме_последнего():
    """Последний выданный токен — вход этого шага, а не то, что уже посчитано.
    Ошибка здесь сдвигает позиции, и стадия отвечает чушью, нигде не падая."""
    assert sequence(prompt=[1, 2, 3], outputs=[9]).computed == 3
    assert sequence(prompt=[1, 2, 3], outputs=[9, 8]).computed == 4
    assert sequence(prompt=[1, 2, 3], outputs=[9, 8, 7]).computed == 5


# ----------------------------------------------------------------- prefill
def test_prefill_считает_весь_промпт(vllm):
    from loom_stage.vllm_batch import prefill

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    out = prefill([sequence(prompt=[1, 2, 3, 4])], runner)
    assert out.num_scheduled_tokens == {"r1": 4}
    assert out.total_num_scheduled_tokens == 4
    assert len(out.scheduled_new_reqs) == 1
    assert out.scheduled_new_reqs[0].num_computed_tokens == 0


def test_prefill_батчем_складывает_токены(vllm):
    from loom_stage.vllm_batch import prefill

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    out = prefill([sequence("a", [1, 2]), sequence("b", [1, 2, 3])], runner)
    assert out.total_num_scheduled_tokens == 5
    assert out.num_scheduled_tokens == {"a": 2, "b": 3}


def test_нехватка_кэша_отпускает_уже_взятое(vllm):
    """Половина выделенных блоков хуже, чем ни одного: они не вернутся сами, и
    следующий батч упрётся в память, которую никто не держит."""
    from loom_stage.vllm_batch import BatchRefused, prefill

    manager = Manager(refuse_after=1, blocks_cls=vllm["Blocks"])
    runner = runner_with(manager)
    with pytest.raises(BatchRefused, match="не хватило места"):
        prefill([sequence("a"), sequence("b")], runner)
    assert manager.freed == ["a"], "блоки первой последовательности остались висеть"


# ------------------------------------------------------------------ decode
def test_decode_считает_ровно_один_токен(vllm):
    from loom_stage.vllm_batch import decode

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    out = decode([sequence(prompt=[1, 2, 3], outputs=[9, 8])], runner)
    assert out.num_scheduled_tokens == {"r1": 1}
    assert out.scheduled_new_reqs == []


def test_decode_подаёт_только_последний_выданный(vllm):
    """Вход шага — один токен. Подать все выданные значит посчитать их заново."""
    from loom_stage.vllm_batch import decode

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    out = decode([sequence(prompt=[1, 2], outputs=[9, 8, 7])], runner)
    assert out.scheduled_cached_reqs.new_token_ids == [[7]]
    assert out.scheduled_cached_reqs.num_computed_tokens == [4]


def test_неголовная_стадия_берёт_длину_у_движка(vllm):
    """Ей токены не присылают: она их не видит и не сэмплирует. Но vLLM их
    помнит, и без этого её представление о длине разойдётся с головой."""
    from loom_stage.vllm_batch import decode

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    runner.requests["r1"] = types.SimpleNamespace(output_token_ids=[9, 8, 7])
    out = decode([sequence(prompt=[1, 2], outputs=[])], runner)
    assert out.scheduled_cached_reqs.new_token_ids == [[7]]


def test_пустой_батч_отвергается(vllm):
    from loom_stage.vllm_batch import BatchRefused, decode, prefill

    runner = runner_with(Manager(blocks_cls=vllm["Blocks"]))
    for form in (prefill, decode):
        with pytest.raises(BatchRefused, match="пустой батч"):
            form([], runner)


def test_без_менеджера_кэша_отказ_называет_причину(vllm):
    from loom_stage.vllm_batch import BatchRefused, prefill

    with pytest.raises(BatchRefused, match="менеджера KV-кэша"):
        prefill([sequence()], types.SimpleNamespace())


# --------------------------------------------------------------- уборка
def test_освобождение_незнакомого_запроса_безопасно(vllm):
    from loom_stage.vllm_batch import release

    manager = Manager(blocks_cls=vllm["Blocks"])
    release(runner_with(manager), "которого-нет")
    assert manager.freed == []


def test_освобождение_убирает_и_состояние(vllm):
    from loom_stage.vllm_batch import release

    manager = Manager(blocks_cls=vllm["Blocks"])
    runner = runner_with(manager)
    runner.requests["r1"] = types.SimpleNamespace(request_id="r1")
    release(runner, "r1")
    assert manager.freed == ["r1"]
    assert "r1" not in runner.requests
