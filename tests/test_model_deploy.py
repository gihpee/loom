"""Разрезать модель между узлами: сколько слоёв кому и что уедет на узел.

Сеть здесь трогает только один тест — остальное про арифметику разреза и про
отказы, которые оператор увидит раньше, чем что-то запустится.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from looma.orchestrator.models import ModelError, describe, split_layers, stage_payload

from test_agent_gateway import stand  # noqa: F401 — фикстура стенда


# ------------------------------------------------------------------- разрез
def test_слои_делятся_поровну():
    assert split_layers(36, 3) == [(0, 12), (12, 24), (24, 36)]


def test_остаток_достаётся_первым_стадиям():
    """Хвост округления не должен потеряться: сумма всегда равна числу слоёв."""
    ranges = split_layers(37, 3)
    assert ranges[0] == (0, 13)
    assert ranges[-1][1] == 37
    assert sum(e - s for s, e in ranges) == 37


def test_диапазоны_идут_подряд_без_дыр():
    """Пропущенный слой — не ошибка загрузки, а тихо неверный ответ."""
    ranges = split_layers(48, 5)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 48
    for before, after in zip(ranges, ranges[1:]):
        assert before[1] == after[0]


def test_разрез_по_свободной_vram():
    """На стенде из 4090 и 3090 равный разрез упирается в меньшую карту."""
    ranges = split_layers(36, 2, [24.0, 48.0])
    first, second = (e - s for s, e in ranges)
    assert second > first
    assert first + second == 36


@pytest.mark.parametrize("weights", [None, [1, 1, 1], [10, 1, 1], [1, 1, 10]])
def test_сумма_всегда_сходится(weights):
    ranges = split_layers(29, 3, weights)
    assert sum(e - s for s, e in ranges) == 29
    assert all(e > s for s, e in ranges), "стадия без слоёв — переход, который не считает"


def test_стадий_больше_чем_слоёв_отвергается():
    with pytest.raises(ModelError) as exc:
        split_layers(4, 8)
    assert "не считает" in str(exc.value)


# ------------------------------------------------------------------- имена
@pytest.mark.parametrize("bad", ["", "нетслеша", "нет/такой", "a b/c", "/"])
def test_негодное_имя_модели_отвергается_до_сети(bad):
    """Иначе кириллица вылезает ошибкой кодировки из глубины urllib."""
    with pytest.raises(ModelError):
        describe(bad)


# ----------------------------------------------------------------- нагрузка
def test_код_стадии_уезжает_вместе_с_задачей():
    """Узел, который эту модель не обслуживал, получает код с задачей —
    без пакетного реестра посередине."""
    payload = stage_payload()
    assert "looma_stage/server.py" in payload
    assert "looma_stage/loader.py" in payload
    assert all(name.startswith("looma_stage/") for name in payload)
    assert 0 < sum(len(v) for v in payload.values()) < 5 * 1024 * 1024


def test_пусковой_слой_в_нагрузку_не_попадает():
    """Он живёт в образе: сломанный запускатель нельзя починить, прислав ещё один."""
    assert not any("launcher" in name for name in stage_payload())


# --------------------------------------------------------------------- сеть
@pytest.mark.slow
def test_настоящая_модель_читается_с_huggingface():
    info = describe("Qwen/Qwen3-8B")
    assert info.num_layers == 36
    assert info.hidden_size == 4096
    assert "Qwen3" in info.architecture


def test_код_стадии_находится_и_в_установленном_пакете(tmp_path, monkeypatch):
    """В образе оркестратор живёт в site-packages, где никакого payloads/ рядом нет.

    Симптом со стенда: деплой отказал с путём
    /usr/local/lib/python3.12/payloads/... — путь считался от исходников и
    установку не пережил.
    """
    where = tmp_path / "payloads" / "looma_stage" / "looma_stage"
    where.mkdir(parents=True)
    (where / "server.py").write_text("# стадия\n")
    (where / "loader.py").write_text("# загрузчик\n")
    monkeypatch.setenv("LOOMA_PAYLOADS_DIR", str(tmp_path / "payloads"))
    payload = stage_payload()
    assert set(payload) == {"looma_stage/server.py", "looma_stage/loader.py"}


def test_отсутствие_стадии_называет_где_искали(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMA_PAYLOADS_DIR", str(tmp_path / "нет-такого"))
    monkeypatch.setattr("looma.orchestrator.models._payload_dirs",
                        lambda: iter([tmp_path / "нет-такого"]))
    with pytest.raises(ModelError) as exc:
        stage_payload()
    assert "Искали в" in str(exc.value)
    assert "LOOMA_PAYLOADS_DIR" in str(exc.value)


# ------------------------------------------------------------------- движок
def deploy_body(**extra) -> dict:
    # Имя модели заведомо негодное: оба отказа ниже обязаны случиться ДО
    # обращения к HuggingFace, и если проверка движка когда-нибудь переедет
    # ниже, эти тесты упадут на другом сообщении, а не тихо уйдут в сеть.
    return {"repo": "не/модель", "stages": 1, **extra}


def api(orchestrator):
    from fastapi.testclient import TestClient

    from looma.api.app import create_app
    from test_agent_gateway import _Settings

    return TestClient(create_app(agents=orchestrator.hub, config=_Settings()))


def test_неизвестный_движок_отвергается_до_запуска(stand):
    """Оператор должен узнать об опечатке здесь, а не из логов узла, на который
    уже уехала задача."""
    orchestrator, _agent = stand
    answer = api(orchestrator).post("/admin/deploy",
                                    json=deploy_body(engine="vllm2"))
    assert answer.status_code == 400
    assert "не поддерживается" in answer.json()["error"]["message"]


def test_vllm_без_карты_отвергается_до_загрузки_весов(stand):
    """vLLM без карты не поднимается вовсе, а узнать об этом через десять минут
    качания весов — худший способ."""
    orchestrator, _agent = stand
    answer = api(orchestrator).post("/admin/deploy",
                                    json=deploy_body(engine="vllm", device="cpu"))
    assert answer.status_code == 400
    assert "только на cuda" in answer.json()["error"]["message"]


# ------------------------------------------------------------- vLLM и драйвер
@pytest.mark.parametrize("version, expected", [
    ("12.4", (12, 4)),
    ("12", (12, 0)),
    ("12.6.1", (12, 6)),
    ("", None),
    ("неизвестно", None),
    ("12.x", None),
])
def test_разбор_версии_cuda(version, expected):
    """Не разобрали — None, а не догадка: догадка тут стоит развёрнутой
    группы, которая упадёт через десять минут."""
    from looma.orchestrator.models import cuda_tuple

    assert cuda_tuple(version) == expected


def node(cuda="12.8", node_id="nv3") -> dict:
    return {"node_id": node_id, "cuda_version": cuda}


def test_свежий_драйвер_пропускает_vllm():
    from looma.orchestrator.models import vllm_refusal

    assert vllm_refusal(node("12.8")) == ""
    assert vllm_refusal(node("12.6")) == ""


def test_старый_драйвер_отвергается_с_названной_причиной():
    """Под cu124 нет сборки torch, которую требует vLLM; pip доставит её с
    PyPI поверх правильной, и упадёт это только при обращении к карте."""
    from looma.orchestrator.models import vllm_refusal

    reason = vllm_refusal(node("12.4"))
    assert "12.4" in reason and "12.6" in reason
    assert "nv3" in reason
    assert "transformers" in reason      # что делать вместо


def test_узел_без_карты_отвергается():
    from looma.orchestrator.models import vllm_refusal

    assert "не сообщил версию CUDA" in vllm_refusal(node(""))


def test_развёртывание_на_старом_драйвере_отвергается(stand, monkeypatch):
    """Оператор узнаёт об этом до запуска, а не из «CUDA initialization:
    driver is too old» через десять минут загрузки весов."""
    from looma.orchestrator.models import ModelInfo

    orchestrator, _agent = stand
    nodes = orchestrator.hub.node_list()
    monkeypatch.setattr(orchestrator.hub, "node_list",
                        lambda: [{**n, "cuda_version": "12.4"} for n in nodes])
    # Сеть здесь не при чём: config.json подменён, чтобы проверка драйвера
    # осталась единственным, из-за чего этот запрос может не пройти.
    monkeypatch.setattr("looma.api.app.describe",
                        lambda repo, **kw: ModelInfo(repo=repo, num_layers=36))
    answer = api(orchestrator).post("/admin/deploy",
                                    json=deploy_body(repo="Qwen/Qwen3-4B",
                                                     engine="vllm"))
    assert answer.status_code == 409
    assert "12.6" in answer.json()["error"]["message"]


# --------------------------------------------------------- версия движка
def test_vllm_ставится_с_версией():
    """Отпечаток окружения на узле считается от СТРОК требований. `vllm` без
    версии даёт один отпечаток для любой: узел, собравший окружение месяц
    назад, не обновится никогда, а соседний соберёт сегодняшний — и две стадии
    одного конвейера окажутся на разных версиях под одним именем каталога."""
    from looma.orchestrator.models import stage_requirements

    packages = stage_requirements("vllm")
    pinned = [name for name in packages if name.startswith("vllm")]
    assert pinned and "==" in pinned[0], packages


def test_переносимому_движку_vllm_не_ставится():
    """Это гигабайты и полчаса на узел — за то, чем он не будет считать."""
    from looma.orchestrator.models import stage_requirements

    assert not any(name.startswith("vllm") for name in stage_requirements("torch"))


def test_торч_под_vllm_пинится_той_версией_которую_он_требует():
    """Иначе первый проход агента поставит torch с индекса по драйверу, а vLLM
    вторым проходом стянет другую версию с PyPI поверх неё. Каталог окружения
    останется с именем cu128, а внутри окажется чужое колесо, и падать это
    будет не на установке, а при первом обращении к карте."""
    from looma.orchestrator.models import VLLM_TORCH, stage_requirements

    packages = stage_requirements("vllm")
    assert "torch" not in packages, "непинованный torch затянет подмену обратно"
    for pinned in VLLM_TORCH:
        assert pinned in packages
        assert "==" in pinned


def test_торч_остаётся_без_версии():
    """Агент выбирает сборку по драйверу узла, и наборы версий на индексах
    cu124/cu126/cu128 разные: пин, годный для одного, сделал бы неустановимым
    другой."""
    from looma.orchestrator.models import stage_requirements

    assert "torch" in stage_requirements("torch")


def test_смена_пина_меняет_окружение_узла():
    """Иначе узлы тихо останутся на старой версии: другого механизма
    обновления тут нет."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
    from looma_agent.tasks.spec import EnvSpec

    from looma.orchestrator.models import stage_requirements

    from looma.orchestrator.models import VLLM_PIN

    packages = stage_requirements("vllm")
    now = EnvSpec(kind="python", requirements=tuple(packages)).fingerprint()
    # Тот же список, но с другой версией vLLM — какой бы ни был пин сейчас.
    bumped = [VLLM_PIN + ".1" if name == VLLM_PIN else name for name in packages]
    later = EnvSpec(kind="python", requirements=tuple(bumped)).fingerprint()
    assert now != later
