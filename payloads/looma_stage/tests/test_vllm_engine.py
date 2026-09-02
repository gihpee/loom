"""Поднятие движка vLLM: порядок шагов и отказы.

Самого vLLM тут нет — он не ставится без карты. Проверяется то, что решает
исход: отказ до всякой работы, доля карты из квоты, и сверка «собралось ли
столько слоёв, сколько просили».
"""

from __future__ import annotations

import sys
import types

import pytest

from looma_stage import vllm_engine
from looma_stage.vllm_runner import RunnerRefused


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
    monkeypatch.setattr(vllm_engine, "plan_for_shard",
                        lambda *a, **k: vllm_engine.Plan(
                            utilisation=0.5, max_sequences=8,
                            bytes_needed=0, why="в тесте"))
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
    # Иначе тест пойдёт качать модель с HuggingFace.
    monkeypatch.setattr(vllm_engine, "prepare_weights",
                        lambda weights, **_k: weights)

    # Именно атрибут модуля, а не запись в sys.modules: `from looma_stage
    # import vllm_patch` берёт атрибут пакета, и подмена через sys.modules
    # работала, только пока модуль не был импортирован кем-то ещё.
    monkeypatch.setattr("looma_stage.vllm_patch.allow_missing_ends",
                        lambda **_k: None)

    with pytest.raises(RunnerRefused, match="просили 18 слоёв"):
        vllm_engine.load_shard("модель", start_layer=0, end_layer=18,
                               num_model_layers=36)


def test_конфиг_ставится_раньше_распределённой_группы(monkeypatch):
    """Свежий vLLM спрашивает конфиг уже внутри `initialize_model_parallel`.

    Поставь его позже — падает на assert'е, в котором про конвейер нет ни
    слова: «Current vLLM config is not set... or a CustomOp was instantiated at
    module import time». Порядок этих двух шагов и есть весь смысл теста.
    """
    порядок = []
    monkeypatch.setattr(vllm_engine, "require_cuda", lambda: None)
    monkeypatch.setattr(vllm_engine, "plan_for_shard",
                        lambda *a, **k: vllm_engine.Plan(
                            utilisation=0.5, max_sequences=8,
                            bytes_needed=0, why="в тесте"))
    monkeypatch.setattr(vllm_engine, "_build_config", lambda *a, **k: _FakeConfig())
    monkeypatch.setattr(vllm_engine, "_hold_config",
                        lambda _c: порядок.append("конфиг"))
    monkeypatch.setattr(vllm_engine, "_start_distributed",
                        lambda: порядок.append("группа"))
    monkeypatch.setattr(vllm_engine, "replace_pipeline_group",
                        lambda *a, **k: порядок.append("подмена"))
    monkeypatch.setattr(vllm_engine, "layer_range", _nothing)
    monkeypatch.setattr(vllm_engine, "_count_layers", lambda _r: 18)
    monkeypatch.setattr(vllm_engine, "stage_runner_class", lambda *_a: _FakeRunner)
    monkeypatch.setattr(vllm_engine, "prepare_weights", lambda weights, **_k: weights)
    monkeypatch.setattr("looma_stage.vllm_patch.allow_missing_ends", lambda **_k: None)

    vllm_engine.load_shard("модель", start_layer=0, end_layer=18,
                           num_model_layers=36)
    assert порядок == ["конфиг", "группа", "подмена"]


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
    assert vllm_engine._logits_from(types.SimpleNamespace(), ответ, expected=1) == "прямо"

    runner = types.SimpleNamespace(
        execute_model_state=types.SimpleNamespace(logits="в состоянии"))
    assert vllm_engine._logits_from(runner, object(), expected=1) == "в состоянии"


def test_если_логитов_нет_отказ_называет_что_пришло():
    with pytest.raises(RunnerRefused, match="вернул int"):
        vllm_engine._logits_from(types.SimpleNamespace(execute_model_state=None), 7,
                                 expected=1)


def test_неголовной_стадии_без_тензоров_считать_нечего(monkeypatch, sequence_module):
    """Иначе она посчитает мусор из неинициализированного буфера и отдаст его
    дальше — молча."""
    shard = vllm_engine.LoadedShard(
        start_layer=18, end_layer=36, num_layers=36, is_first=False,
        is_last=True, dtype="bfloat16", runner=types.SimpleNamespace())
    monkeypatch.setattr("looma_stage.vllm_batch.prefill", lambda *a, **k: "батч")
    monkeypatch.setattr("looma_stage.vllm_batch.decode", lambda *a, **k: "батч")

    with pytest.raises(RunnerRefused, match="тензоры от предыдущей не пришли"):
        vllm_engine.step(shard, [object()], incoming=None, first_step=True)


# ------------------------------------------------- тензоры через файл
def test_тензоры_переживают_дорогу_через_файл(tmp_path, monkeypatch):
    """Складываются они нашим форматом провода — тем самым, которым поедут
    между машинами. Проверить его тут ничего не стоит, а разойдись он с
    ожиданием — стадия получит мусор и посчитает его молча."""
    import torch

    from looma_stage import vllm_engine as engine

    tensors = {
        "hidden_states": torch.randn(3, 8, dtype=torch.bfloat16),
        "residual": torch.randn(3, 8, dtype=torch.bfloat16),
    }
    path = str(tmp_path / "hidden.bin")
    engine._save_hidden(tensors, path)

    restored = {}
    import json

    from looma_stage import wire

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


# ------------------------------------------------- поля конфигов и версии
def test_неизвестное_поле_отбрасывается_и_называется(caplog):
    """Поля конфигов vLLM переезжают между версиями, и лишний аргумент роняет
    всё поднятие — сообщением про имя, а не про то, что версия другая."""
    import dataclasses
    import logging

    @dataclasses.dataclass
    class Старый:
        model: str = ""

    with caplog.at_level(logging.WARNING, logger="looma_stage.vllm_engine"):
        made = vllm_engine._config_with(Старый, model="м", enforce_eager=True)
    assert made.model == "м"
    assert any("enforce_eager" in r.getMessage() for r in caplog.records), (
        "молча потерянный enforce_eager вернёт захват графов и падение в нём")


def test_известные_поля_доходят():
    import dataclasses

    @dataclasses.dataclass
    class Новый:
        model: str = ""
        enforce_eager: bool = False

    made = vllm_engine._config_with(Новый, model="м", enforce_eager=True)
    assert made.enforce_eager is True


def test_не_датакласс_собирается_как_есть():
    class Обычный:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    assert vllm_engine._config_with(Обычный, a=1).kwargs == {"a": 1}


# ------------------------------------------------------ урезанный чекпоинт
def test_стадии_дают_урезанный_чекпоинт(monkeypatch):
    """vLLM перечисляет каждый файл из индекса и открывает его: недостающий —
    ошибка, а не экономия. Поэтому рядом собирается вид из симлинков с
    переписанным индексом."""
    просили = {}

    def resolve(weights, shard=None, **_kwargs):
        просили["скачано для"] = (shard.start_layer, shard.end_layer)
        просили["роли"] = (shard.is_first, shard.is_last)
        return "/локально"

    def view(path, shard):
        просили["вид из"] = path
        return "/локально/вид"

    monkeypatch.setattr("looma_stage.loader.resolve_model_path", resolve)
    monkeypatch.setattr("looma_stage.loader.build_stage_checkpoint_view", view)

    got = vllm_engine.prepare_weights("Qwen/Qwen3-4B", start_layer=0, end_layer=18,
                                      is_first=True, is_last=False, dtype="bfloat16")
    assert got == "/локально/вид"
    assert просили["скачано для"] == (0, 18), "скачали не свой срез"
    assert просили["роли"] == (True, False)
    assert просили["вид из"] == "/локально"


def test_если_урезать_нечем_читаем_целиком(monkeypatch):
    """Единственный файл или незнакомые имена ключей — не отказ: стадия просто
    прочитает больше, чем ей нужно."""
    monkeypatch.setattr("looma_stage.loader.resolve_model_path",
                        lambda *a, **k: "/локально")
    monkeypatch.setattr("looma_stage.loader.build_stage_checkpoint_view",
                        lambda path, shard: path)

    got = vllm_engine.prepare_weights("модель", start_layer=0, end_layer=18,
                                      is_first=True, is_last=False, dtype="bfloat16")
    assert got == "/локально"


# ------------------------------------------------------- батч из нескольких
class _Answer:
    def __init__(self, logits):
        self.logits = logits


class _Rows:
    """Тензор ровно настолько, насколько его щупает _logits_from."""

    def __init__(self, *shape):
        self.shape = shape


def test_число_строк_логитов_сверяется_с_батчем():
    """Строка логитов не подписана именем запроса: соответствие держится
    только на порядке. Разойдись оно молча — токен уехал бы чужому клиенту."""
    from looma_stage import vllm_engine

    with pytest.raises(vllm_engine.RunnerRefused, match="строк логитов"):
        vllm_engine._logits_from(object(), _Answer(_Rows(2, 151936)), expected=3)


def test_совпавший_батч_логитов_проходит():
    from looma_stage import vllm_engine

    logits = _Rows(3, 151936)
    assert vllm_engine._logits_from(object(), _Answer(logits), expected=3) is logits


def test_одномерные_логиты_считаются_одной_строкой():
    from looma_stage import vllm_engine

    logits = _Rows(151936)
    assert vllm_engine._logits_from(object(), _Answer(logits), expected=1) is logits


def test_шаг_без_последовательностей_отвергается():
    from looma_stage import vllm_engine

    with pytest.raises(vllm_engine.RunnerRefused, match="без единой"):
        vllm_engine.step(object(), [], first_step=True)


@pytest.mark.parametrize("text, expected", [
    ("1,2,3", [[1, 2, 3]]),
    ("1,2,3;4,5", [[1, 2, 3], [4, 5]]),
    ("1,2;;3", [[1, 2], [3]]),
    (" 1 , 2 ; 3 ", [[1, 2], [3]]),
])
def test_разбор_нескольких_промптов(text, expected):
    from looma_stage import vllm_engine

    assert vllm_engine._parse_prompts(text) == expected


def test_промпты_без_токенов_отвергаются():
    """Батч из пустой последовательности vLLM примет и посчитает ни за чем."""
    from looma_stage import vllm_engine

    with pytest.raises(vllm_engine.RunnerRefused, match="ни одного токена"):
        vllm_engine._parse_prompts(";;")


# --------------------------------------------------- двухфазный шаг vLLM
class _Logits:
    def __init__(self, rows=1, vocab=7):
        self.shape = (rows, vocab)
        self.copied = False

    def clone(self):
        made = _Logits(*self.shape)
        made.copied = True
        return made


class _TwoPhaseRunner:
    """Исполнитель vLLM 0.14: шаг разделён надвое."""

    def __init__(self, rows=1):
        self.execute_model_state = types.SimpleNamespace(logits=_Logits(rows))
        self.sampled = 0

    def sample_tokens(self, _grammar):
        if self.execute_model_state is None:
            raise AssertionError("позвали, когда закрывать было нечего")
        self.execute_model_state = None
        self.sampled += 1


def _shard(runner, *, is_last=True):
    return vllm_engine.LoadedShard(
        start_layer=18, end_layer=36, num_layers=36, is_first=False,
        is_last=is_last, dtype="bfloat16", runner=runner)


def test_шаг_закрывается_и_следующий_проходит(monkeypatch, sequence_module):
    """Не закрыть шаг — значит уронить СЛЕДУЮЩИЙ на «State error», уже после
    того, как первый токен уехал клиенту. Одиночной проверкой не ловится."""
    runner = _TwoPhaseRunner()
    monkeypatch.setattr("looma_stage.vllm_batch.prefill", lambda *a, **k: "батч")
    monkeypatch.setattr("looma_stage.vllm_batch.decode", lambda *a, **k: "батч")
    monkeypatch.setattr(runner, "execute_model", lambda *a, **k: None, raising=False)

    from looma_stage.scheduler import Sequence

    vllm_engine.step(_shard(runner), [Sequence("a", [1, 2, 3])],
                     incoming="тензоры", first_step=True)
    assert runner.execute_model_state is None and runner.sampled == 1


def test_логиты_забираются_копией(monkeypatch, sequence_module):
    """Сэмплер vLLM правит их на месте, а выбирать токен мы будем сами."""
    runner = _TwoPhaseRunner()
    monkeypatch.setattr("looma_stage.vllm_batch.prefill", lambda *a, **k: "батч")
    monkeypatch.setattr(runner, "execute_model", lambda *a, **k: None, raising=False)

    from looma_stage.scheduler import Sequence

    _hidden, logits = vllm_engine.step(_shard(runner), [Sequence("a", [1])],
                                       incoming="тензоры", first_step=True)
    assert logits.copied


def test_старая_версия_закрывать_нечего():
    """До 0.14 всё отдавалось одним вызовом."""
    vllm_engine._finish_step(types.SimpleNamespace())          # нет sample_tokens
    vllm_engine._finish_step(types.SimpleNamespace(sample_tokens=None,
                                                   execute_model_state=None))
