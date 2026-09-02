"""Сколько карты просит стадия.

vLLM выделяет KV-кэш заранее и на всю отведённую долю. Доля наугад означает,
что стадия крошечной модели занимает столько же, сколько огромной, и соседу на
чужом узле места не остаётся. Здесь — арифметика, которая считает это от того,
что стадия обещает обслужить.
"""

from __future__ import annotations

import types

import pytest

from loom_stage import vllm_engine
from loom_stage.vllm_engine import (
    OVERHEAD_BYTES,
    RunnerRefused,
    dtype_bytes,
    kv_bytes_per_token,
    plan_memory,
)

ГБ = 1024 ** 3


def config(**kwargs):
    base = {"num_key_value_heads": 8, "num_attention_heads": 32,
            "hidden_size": 2560, "head_dim": 128}
    return types.SimpleNamespace(**{**base, **kwargs})


# ------------------------------------------------------------ байты dtype
@pytest.mark.parametrize("name, size", [
    ("bfloat16", 2), ("BF16", 2), ("float32", 4), ("float8", 1),
])
def test_известные_типы(name, size):
    assert dtype_bytes(name) == size


def test_неизвестный_тип_отказ_а_не_догадка():
    """Ошибка вдвое даёт вдвое неверный кэш: узел либо не поднимется, либо
    пообещает вдвое больше, чем сможет."""
    with pytest.raises(RunnerRefused, match="сколько байт занимает"):
        dtype_bytes("float4")


# ------------------------------------------------------- цена одного токена
def test_считается_по_kv_головам_а_не_по_головам_внимания():
    """У моделей с GQA их в несколько раз меньше. Считать по головам внимания
    значило бы завысить кэш вчетверо и не поднять стадию там, где памяти
    хватало."""
    по_kv = kv_bytes_per_token(config(), layers=1, dtype="bfloat16")
    по_вниманию = kv_bytes_per_token(config(num_key_value_heads=32),
                                     layers=1, dtype="bfloat16")
    assert по_kv * 4 == по_вниманию
    assert по_kv == 2 * 8 * 128 * 2       # ключи+значения × головы × размер × байты


def test_цена_растёт_со_слоями_среза():
    один = kv_bytes_per_token(config(), layers=1, dtype="bfloat16")
    восемнадцать = kv_bytes_per_token(config(), layers=18, dtype="bfloat16")
    assert восемнадцать == один * 18


def test_размер_головы_выводится_когда_не_назван():
    выведено = kv_bytes_per_token(config(head_dim=None, hidden_size=4096,
                                         num_attention_heads=32),
                                  layers=1, dtype="bfloat16")
    assert выведено == 2 * 8 * 128 * 2


def test_конфиг_без_голов_отвергается():
    with pytest.raises(RunnerRefused, match="посчитать размер кэша нечем"):
        kv_bytes_per_token(config(num_key_value_heads=None,
                                  num_attention_heads=None),
                           layers=1, dtype="bfloat16")


# -------------------------------------------------------------- план памяти
def plan(**kwargs):
    base = dict(per_token=72 * 1024, weights_bytes=2 * ГБ, total_bytes=24 * ГБ,
                budget_bytes=24 * ГБ, max_sequences=8, max_model_len=4096)
    return plan_memory(**{**base, **kwargs})


def test_просим_столько_сколько_обещаем():
    """Не долю карты, а место под обещанную ёмкость."""
    сделано = plan(max_sequences=8)
    кэш = 8 * 4096 * 72 * 1024
    assert сделано.bytes_needed == 2 * ГБ + OVERHEAD_BYTES + кэш
    assert сделано.max_sequences == 8


def test_маленькая_модель_не_забирает_всю_карту():
    """Ради этого всё и затевалось: рядом живёт вторая стадия того же
    кластера."""
    сделано = plan(per_token=8 * 1024, weights_bytes=ГБ, max_sequences=4)
    assert сделано.utilisation < 0.5


def test_не_влезло_уменьшаем_обещание_а_не_кэш():
    """Иначе узел принимал бы запросы, которые не может досчитать: голова
    видит свой потолок, а блоки кончаются у стадии."""
    сделано = plan(max_sequences=64, total_bytes=24 * ГБ)
    assert сделано.max_sequences < 64
    assert "узел обещает меньше" in сделано.why


def test_доля_не_переходит_потолок():
    """Выше него vLLM не оставляет места под собственные буферы."""
    сделано = plan(max_sequences=1000)
    assert сделано.utilisation <= vllm_engine.MAX_UTILISATION


def test_квота_ограничивает_сильнее_карты():
    просторно = plan(budget_bytes=24 * ГБ, max_sequences=64)
    тесно = plan(budget_bytes=8 * ГБ, max_sequences=64)
    assert тесно.max_sequences < просторно.max_sequences


def test_если_не_влезает_даже_одна_отказ_с_числами():
    """Молча взять меньше — значит пообещать то, чего нет."""
    with pytest.raises(RunnerRefused, match="даже под одну последовательность"):
        plan(total_bytes=4 * ГБ, budget_bytes=4 * ГБ, weights_bytes=2 * ГБ)


def test_объяснение_называет_из_чего_сложилось():
    """Число, взявшееся ниоткуда, невозможно оспорить."""
    why = plan().why
    for кусок in ("кэша", "веса", "запас", "итого"):
        assert кусок in why
