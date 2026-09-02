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

    # Именно атрибут модуля, а не запись в sys.modules: `from loom_stage
    # import vllm_patch` берёт атрибут пакета, и подмена через sys.modules
    # работала, только пока модуль не был импортирован кем-то ещё.
    monkeypatch.setattr("loom_stage.vllm_patch.allow_missing_ends",
                        lambda **_k: None)
    _fake_gpu_runner(monkeypatch)

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


def _fake_gpu_runner(monkeypatch):
    module = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def __init__(self, **_kwargs):
            self.model = None

        def load_model(self):
            pass

    module.GPUModelRunner = GPUModelRunner
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", module)
