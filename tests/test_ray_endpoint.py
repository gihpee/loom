"""Ручка, которая размещает Ray-кластер.

Ничего специфичного для Ray в оркестраторе нет и не должно быть: это обычная
группа с обычным окружением. Проверяется, что она собрана правильно — и что
она честно отказывает там, где ещё не работает.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from loom.orchestrator.payloads import PayloadMissing, collect, ray_payload
from test_agent_pipeline import two_nodes  # noqa: F401  (фикстура)


# ------------------------------------------------------------------ нагрузка
def test_код_ранга_уезжает_вместе_с_задачей():
    payload = ray_payload()
    assert "loom_ray/server.py" in payload
    assert "loom_ray/cluster.py" in payload
    assert "loom_ray/ports.py" in payload
    assert all(name.startswith("loom_ray/") for name in payload)


def test_обе_нагрузки_ищутся_одинаково():
    """Инференс здесь ничем не привилегирован — он просто первый жилец."""
    from loom.orchestrator.models import stage_payload

    assert set(ray_payload()) & set(stage_payload()) == set()


def test_пустой_каталог_с_подходящим_именем_не_считается_нагрузкой(tmp_path):
    """Такой встречается чаще, чем хотелось бы: остаток от переименования."""
    (tmp_path / "loom_ray" / "loom_ray").mkdir(parents=True)
    with pytest.raises(PayloadMissing) as exc:
        collect("loom_ray", human="кода ранга Ray",
                dirs=[tmp_path / "loom_ray" / "loom_ray"])
    assert "Искали в" in str(exc.value)


# -------------------------------------------------------------------- ручка
def client(hub):
    from fastapi.testclient import TestClient

    from loom.api.app import create_app
    from test_agent_gateway import _Settings

    return TestClient(create_app(agents=hub, config=_Settings()))


def test_кластер_ложится_на_несколько_узлов(two_nodes):
    """Раньше здесь стоял честный отказ: ранги на разных узлах друг друга не
    видели. Теперь между ними есть проброс портов, и группа размещается."""
    hub = two_nodes.hub
    answer = client(hub).post("/admin/ray", json={"size": 2, "label": "пара"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["size"] == 2
    assert sorted(body["nodes"]) == ["node-0", "node-1"]

    record = hub.groups[body["group_id"]]
    assert sorted(record.nodes.values()) == ["node-0", "node-1"]
    # Команда одна на всех: свой ранг задача узнаёт из окружения.
    for rank in record.tasks:
        assert hub.tasks[record.tasks[rank]].command[:3] == [
            "python", "-m", "loom_ray.server"]


def test_узлов_просят_больше_чем_есть(two_nodes):
    answer = client(two_nodes.hub).post("/admin/ray", json={"size": 5})
    assert answer.status_code == 409
    assert "5" in answer.json()["error"]["message"]


def test_на_одном_узле_группа_собирается_правильно(two_nodes):
    hub = two_nodes.hub
    answer = client(hub).post("/admin/ray", json={
        "node_ids": ["node-0"],
        "script": base64.b64encode(b"import ray\n").decode(),
        "label": "проба",
    })
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["size"] == 1
    assert body["entry"] == "job.py"
    assert body["nodes"] == ["node-0"]

    # Группа — обычная: то же размещение, тот же label, та же готовность.
    record = hub.groups[body["group_id"]]
    assert record.label == "проба"
    task = hub.tasks[record.tasks[0]]
    assert task.command[:3] == ["python", "-m", "loom_ray.server"]
    assert "--script" in task.command and "job.py" in task.command


def test_без_скрипта_кластер_просто_стоит(two_nodes):
    answer = client(two_nodes.hub).post("/admin/ray", json={"node_ids": ["node-1"]})
    assert answer.status_code == 200, answer.text
    assert answer.json()["entry"] == ""
    task = two_nodes.hub.tasks[
        two_nodes.hub.groups[answer.json()["group_id"]].tasks[0]]
    assert "--script" not in task.command


def test_испорченный_скрипт_называет_причину(two_nodes):
    answer = client(two_nodes.hub).post("/admin/ray", json={
        "node_ids": ["node-0"], "script": "это не base64 !!!"})
    assert answer.status_code == 400
    assert "base64" in answer.json()["error"]["message"]


def test_неподключённый_узел_назван_поимённо(two_nodes):
    answer = client(two_nodes.hub).post("/admin/ray",
                                        json={"node_ids": ["которого-нет"]})
    assert answer.status_code == 409
    assert "которого-нет" in answer.json()["error"]["message"]


# ---------------------------------------------------------------- связность
def test_ответ_называет_путь_которым_соберётся_кластер(two_nodes):
    """Медленный кластер должен быть объяснимым, а не загадочным."""
    answer = client(two_nodes.hub).post("/admin/ray", json={"size": 2})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["path"] in ("direct", "relay")
    assert "relayed_pairs" in body


def test_безнадёжная_пара_без_реле_отвергается_до_занятия_карт(two_nodes, monkeypatch):
    """Единственный случай, когда Ray действительно не поедет. Молча пустить
    его туда — значит показать зависание вместо причины."""
    monkeypatch.setattr("loom.api.app.relay_addrs", lambda: [])
    for session in two_nodes.hub.sessions.values():
        session.node.reachable = False
        session.node.symmetric_nat = True

    before = len(two_nodes.hub.groups)
    answer = client(two_nodes.hub).post("/admin/ray", json={"size": 2})
    assert answer.status_code == 409
    text = answer.json()["error"]["message"]
    assert "реле не развёрнуто" in text
    assert "node-0" in text and "node-1" in text
    assert len(two_nodes.hub.groups) == before, "группа не должна была появиться"


def test_с_реле_та_же_группа_поедет(two_nodes, monkeypatch):
    """Отказ — только про невозможность, а не про медленность."""
    monkeypatch.setattr("loom.api.app.relay_addrs", lambda: ["/ip4/1.2.3.4/tcp/47200"])
    for session in two_nodes.hub.sessions.values():
        session.node.reachable = False
        session.node.symmetric_nat = True

    answer = client(two_nodes.hub).post("/admin/ray", json={"size": 2})
    assert answer.status_code == 200, answer.text
    assert answer.json()["path"] == "relay"
    assert answer.json()["warning"]


def test_узлы_подбираются_по_связности(two_nodes, monkeypatch):
    """При выборе «самых свободных» связность важнее памяти: через реле у Ray
    идёт весь обмен, а не восемь килобайт на токен."""
    monkeypatch.setattr("loom.api.app.relay_addrs", lambda: ["/ip4/1.2.3.4/tcp/47200"])
    nodes = list(two_nodes.hub.sessions.values())
    nodes[0].node.symmetric_nat, nodes[0].node.reachable = True, False
    nodes[1].node.symmetric_nat, nodes[1].node.reachable = False, True

    answer = client(two_nodes.hub).post("/admin/ray", json={"size": 1})
    assert answer.status_code == 200, answer.text
    assert answer.json()["nodes"] == [nodes[1].node.node_id]


def test_узел_названный_дважды_получает_два_ранга(two_nodes):
    """На что опирается форма: собрать кластер там, где сеть между рангами не
    нужна вовсе. Соседи оказываются локальными, проброс не включается."""
    hub = two_nodes.hub
    answer = client(hub).post("/admin/ray", json={
        "node_ids": ["node-0", "node-0"], "label": "вдвоём-на-одном"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["size"] == 2
    assert body["nodes"] == ["node-0", "node-0"]
    # Связность тут ни при чём: узел с самим собой встречается всегда.
    assert body["path"] == "direct"

    record = hub.groups[body["group_id"]]
    assert set(record.nodes.values()) == {"node-0"}
    assert len(record.tasks) == 2
    # Оба ранга получают одну команду: свой номер задача берёт из окружения.
    assert (hub.tasks[record.tasks[0]].command[:3]
            == hub.tasks[record.tasks[1]].command[:3]
            == ["python", "-m", "loom_ray.server"])
