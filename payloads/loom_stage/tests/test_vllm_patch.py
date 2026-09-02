"""Заплата на загрузчик vLLM.

Самого vLLM тут нет и не нужно: проверяется решение — что считать законным
отсутствием, а что настоящей бедой. Ошибка здесь молча превращает побитый
чекпоинт в «стадия работает», и обнаруживается это качеством ответов.
"""

from __future__ import annotations

import sys
import types

import pytest

from loom_stage import vllm_patch


def _original_load_weights(self, model, model_config):
    """Нетронутая копия. Держится отдельно, потому что заплата подменяет
    метод класса, и «восстановить как было» из самого класса уже нельзя —
    там лежит предыдущая заплата."""
    if type(self).raises is not None:
        raise type(self).raises


class FakeLoader:
    """Загрузчик vLLM, каким его видит заплата."""

    raises: Exception | None = None
    load_weights = _original_load_weights


@pytest.fixture
def vllm(monkeypatch):
    """Подставить внутренности vLLM, которых на этой машине нет."""
    module = types.ModuleType("vllm.model_executor.model_loader.default_loader")
    module.DefaultModelLoader = FakeLoader
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader.default_loader", module)
    monkeypatch.setattr(
        sys.modules["vllm.model_executor.model_loader"], "default_loader", module,
        raising=False)
    monkeypatch.setattr(vllm_patch, "_applied", False)
    FakeLoader.raises = None
    FakeLoader.load_weights = _original_load_weights
    yield FakeLoader
    FakeLoader.load_weights = _original_load_weights


def missing(what: str) -> ValueError:
    return ValueError(f"Some weights are not initialized from checkpoint: {{'{what}'}}")


def test_средней_стадии_не_нужны_ни_эмбеддинги_ни_голова(vllm):
    vllm_patch.allow_missing_ends(is_first=False, is_last=False)
    for what in ("model.embed_tokens.weight", "lm_head.weight"):
        vllm.raises = missing(what)
        vllm().load_weights(None, None)      # не бросает — так и задумано


def test_первой_стадии_эмбеддинги_обязательны(vllm):
    """Их отсутствие там — настоящая беда: либо чекпоинт неполон, либо срез
    слоёв посчитан неверно."""
    vllm_patch.allow_missing_ends(is_first=True, is_last=False)
    vllm.raises = missing("model.embed_tokens.weight")
    with pytest.raises(ValueError):
        vllm().load_weights(None, None)


def test_последней_стадии_голова_обязательна(vllm):
    vllm_patch.allow_missing_ends(is_first=False, is_last=True)
    vllm.raises = missing("lm_head.weight")
    with pytest.raises(ValueError):
        vllm().load_weights(None, None)


def test_чужая_беда_проходит_насквозь(vllm):
    """Заплата снимает ровно две проверки. Побитый чекпоинт обязан остаться
    ошибкой, иначе стадия «заработает» и начнёт отвечать чушью."""
    vllm_patch.allow_missing_ends(is_first=False, is_last=False)
    vllm.raises = ValueError("model.layers.7.mlp.down_proj.weight is corrupted")
    with pytest.raises(ValueError, match="corrupted"):
        vllm().load_weights(None, None)

    vllm.raises = missing("model.layers.3.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="q_proj"):
        vllm().load_weights(None, None)


def test_без_vllm_отказ_называет_причину(monkeypatch):
    """Стадия рассчитана на закреплённую версию, и лезет она внутрь неё."""
    monkeypatch.setattr(vllm_patch, "_applied", False)
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr("builtins.__import__", _no_vllm(__import__))
    with pytest.raises(vllm_patch.PatchRefused, match="версию"):
        vllm_patch.allow_missing_ends(is_first=True, is_last=True)


def _no_vllm(real):
    def guarded(name, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    return guarded
