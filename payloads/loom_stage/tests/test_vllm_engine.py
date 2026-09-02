"""Поднятие движка vLLM: порядок шагов и отказы.

Самого vLLM тут нет — он не ставится без карты. Проверяется то, что решает
исход: отказ до всякой работы, доля карты из квоты, и сверка «собралось ли
столько слоёв, сколько просили».
"""

from __future__ import annotations

import sys
import types

import pytest

from loom_stage import vllm_engine
from loom_stage.vllm_runner import RunnerRefused


# --------------------------------------------------------------- отказы
def test_без_карты_отказ_называет_замену(monkeypatch):
    """vLLM без CUDA не работает, и выясняется это глубоко внутри —
    сообщением, по которому не видно, что дело в железе."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RunnerRefused, match="движок torch"):
        vllm_engine.require_cuda()


def test_с_картой_молчит(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    vllm_engine.require_cuda()


# ------------------------------------------------------- сколько собралось
class Layer:
    pass


class PPMissingLayer:
    """Заглушка vLLM на месте чужого слоя. Считать её нашей нельзя."""


def runner_with(layers):
    inner = types.SimpleNamespace(layers=layers)
    return types.SimpleNamespace(model=types.SimpleNamespace(model=inner))


def test_считаются_только_настоящие_слои():
    """vLLM ставит заглушки на месте слоёв чужих стадий. Посчитать их —
    решить, что срез собрался верно, когда он собрался целиком."""
    layers = [Layer(), Layer(), PPMissingLayer(), PPMissingLayer()]
    assert vllm_engine._count_layers(runner_with(layers)) == 2


def test_модель_без_слоёв_не_ломает_счёт():
    assert vllm_engine._count_layers(types.SimpleNamespace(model=None)) == 0


def test_несовпадение_числа_слоёв_отвергается(monkeypatch):
    """Подмена get_pp_indices — единственное, что удерживает vLLM от сборки
    всей модели. Её молчаливый провал даёт стадию, которая считает всё и ест
    всю карту, не сказав ни слова."""
    monkeypatch.setattr(vllm_engine, "require_cuda", lambda: None)
    monkeypatch.setattr(vllm_engine, "_start_distributed", lambda: None)
    monkeypatch.setattr(vllm_engine, "replace_pipeline_group",
                        lambda *a, **k: None)
    monkeypatch.setattr(vllm_engine, "_build_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(vllm_engine, "layer_range", _nothing)
    monkeypatch.setattr(vllm_engine, "_count_layers", lambda _r: 36)
    monkeypatch.setattr(vllm_engine, "stage_runner_class",
                        lambda *_a: _FakeRunner)
    # Конфиг vLLM держится открытым на всю жизнь процесса — тут его нет.
    monkeypatch.setattr(vllm_engine, "_hold_config", lambda _c: None)

    # Именно атрибут модуля, а не запись в sys.modules: `from loom_stage
    # import vllm_patch` берёт атрибут пакета, и подмена через sys.modules
    # работала, только пока модуль не был импортирован кем-то ещё.
    monkeypatch.setattr("loom_stage.vllm_patch.allow_missing_ends",
                        lambda **_k: None)

    with pytest.raises(RunnerRefused, match="просили 18 слоёв"):
        vllm_engine.load_shard("модель", start_layer=0, end_layer=18,
                               num_model_layers=36)


def test_негодный_срез_отвергается_до_загрузки(monkeypatch):
    monkeypatch.setattr(vllm_engine, "require_cuda", lambda: None)
    with pytest.raises(RunnerRefused, match="не помещается"):
        vllm_engine.load_shard("модель", start_layer=30, end_layer=40,
                               num_model_layers=36)


class _FakeConfig:
    device_config = types.SimpleNamespace(device="cuda:0")


def _nothing(*_a, **_k):
    import contextlib

    return contextlib.nullcontext()


class _FakeRunner:
    """Исполнитель, который «загрузился», но собрал не тот срез."""

    def __init__(self, **_kwargs):
        self.model = None

    def load_model(self):
        pass

    def prepare_cache(self, **_kwargs):
        return None


# ---------------------------------------------------------------- уборка
def test_уборка_разбирает_что_подняла(monkeypatch):
    разобрано = []
    state = types.ModuleType("vllm.distributed.parallel_state")
    state.destroy_model_parallel = lambda: разобрано.append("модель")
    state.destroy_distributed_environment = lambda: разобрано.append("окружение")
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = state
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", state)

    torch = types.ModuleType("torch")
    torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: True,
        destroy_process_group=lambda: разобрано.append("torch"))
    monkeypatch.setitem(sys.modules, "torch", torch)

    vllm_engine.shutdown()
    assert разобрано == ["модель", "окружение", "torch"]


def test_уборка_не_падает_когда_разбирать_нечего(monkeypatch):
    """Падать на уборке — худшее, что можно сделать с процессом, который и
    так уходит: настоящая причина ухода потеряется."""
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    vllm_engine.shutdown()      # молча


# ------------------------------------------------------------------- шаг
class Intermediate:
    """Стенд-ин для vllm.sequence.IntermediateTensors."""


@pytest.fixture
def sequence_module(monkeypatch):
    module = types.ModuleType("vllm.sequence")
    module.IntermediateTensors = Intermediate
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sequence", module)
    return module


def test_тензоры_найдутся_как_бы_версия_их_ни_отдала(sequence_module):
    """Версии отличаются: одна возвращает их прямо, другая кладёт в состояние
    исполнителя. Обе должны пройти."""
    прямо = Intermediate()
    assert vllm_engine._hidden_from(types.SimpleNamespace(), прямо) is прямо

    в_состоянии = Intermediate()
    runner = types.SimpleNamespace(
        execute_model_state=types.SimpleNamespace(intermediate_tensors=в_состоянии))
    assert vllm_engine._hidden_from(runner, object()) is в_состоянии


def test_если_тензоров_нет_нигде_отказ_называет_что_пришло(sequence_module):
    """Гадать нельзя: молча вернуть None значит отправить дальше по конвейеру
    пустоту, и разбираться в этом будут на последней стадии."""
    with pytest.raises(RunnerRefused, match="вернул dict"):
        vllm_engine._hidden_from(types.SimpleNamespace(), {})


def test_логиты_ищутся_и_в_ответе_и_в_состоянии():
    ответ = types.SimpleNamespace(logits="прямо")
    assert vllm_engine._logits_from(types.SimpleNamespace(), ответ) == "прямо"

    runner = types.SimpleNamespace(
        execute_model_state=types.SimpleNamespace(logits="в состоянии"))
    assert vllm_engine._logits_from(runner, object()) == "в состоянии"


def test_если_логитов_нет_отказ_называет_что_пришло():
    with pytest.raises(RunnerRefused, match="вернул int"):
        vllm_engine._logits_from(types.SimpleNamespace(execute_model_state=None), 7)


def test_неголовной_стадии_без_тензоров_считать_нечего(monkeypatch, sequence_module):
    """Иначе она посчитает мусор из неинициализированного буфера и отдаст его
    дальше — молча."""
    shard = vllm_engine.LoadedShard(
        start_layer=18, end_layer=36, num_layers=36, is_first=False,
        is_last=True, dtype="bfloat16", runner=types.SimpleNamespace())
    monkeypatch.setattr("loom_stage.vllm_batch.prefill", lambda *a, **k: "батч")
    monkeypatch.setattr("loom_stage.vllm_batch.decode", lambda *a, **k: "батч")

    with pytest.raises(RunnerRefused, match="тензоры от предыдущей не пришли"):
        vllm_engine.step(shard, [object()], incoming=None, first_step=True)


# ------------------------------------------------- тензоры через файл
def test_тензоры_переживают_дорогу_через_файл(tmp_path, monkeypatch):
    """Складываются они нашим форматом провода — тем самым, которым поедут
    между машинами. Проверить его тут ничего не стоит, а разойдись он с
    ожиданием — стадия получит мусор и посчитает его молча."""
    import torch

    from loom_stage import vllm_engine as engine

    tensors = {
        "hidden_states": torch.randn(3, 8, dtype=torch.bfloat16),
        "residual": torch.randn(3, 8, dtype=torch.bfloat16),
    }
    path = str(tmp_path / "hidden.bin")
    engine._save_hidden(tensors, path)

    restored = {}
    import json

    from loom_stage import wire

    layout = json.loads((tmp_path / "hidden.bin.json").read_text())
    blob = engine.pathlib_read(path)
    for name, where in layout.items():
        piece = blob[where["at"]:where["at"] + where["size"]]
        restored[name] = wire.from_wire(torch, piece, where["shape"], where["dtype"])

    assert sorted(restored) == ["hidden_states", "residual"]
    for name, original in tensors.items():
        assert torch.equal(restored[name], original), name
        assert restored[name].dtype == original.dtype, "dtype не должен расширяться"


# ---------------------------------------------------- текущий конфиг vLLM
def test_конфиг_держится_открытым(monkeypatch):
    """Со стенда: загрузка прошла, а раскладка кэша упала на

        AssertionError: Current vLLM config is not set

    из бэкенда внимания — места, которое к конфигу отношения не имеет. Части
    движка спрашивают «текущий конфиг» сами, без аргументов, и вне контекста
    это падает где угодно.
    """
    import contextlib

    открыт = []

    @contextlib.contextmanager
    def set_current(config):
        открыт.append(config)
        try:
            yield
        finally:
            открыт.remove(config)

    module = types.ModuleType("vllm.config")
    module.set_current_vllm_config = set_current
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", module)
    monkeypatch.setattr(vllm_engine, "_CONFIG", None)

    vllm_engine._hold_config("конфиг")
    assert открыт == ["конфиг"], "контекст не установлен"

    # Второй вызов ничего не меняет: стадия в процессе одна.
    vllm_engine._hold_config("другой")
    assert открыт == ["конфиг"]

    vllm_engine.shutdown()
    assert открыт == [], "контекст не закрылся на уборке"


def test_уборка_без_конфига_молчит(monkeypatch):
    monkeypatch.setattr(vllm_engine, "_CONFIG", None)
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real = __import__

    def guarded(name, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    vllm_engine.shutdown()
