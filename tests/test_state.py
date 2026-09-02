"""Состояние, которое обязано пережить перезапуск оркестратора.

Узлы его переживают: их задачи никто не останавливал. Оркестратор, который
проснулся с пустой памятью, о них не знает — и не только не показывает, но и
не может снять, потому что снятие адресуется по task_id.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from loom.orchestrator.agents import (ADOPTION_GRACE_S, AgentHub, AgentNode,
                                      AgentSession, TaskRecord)
from loom.orchestrator.state import FORMAT, StateStore
from loom.proto_gen import agent_pb2


def hub_with(path: Path) -> AgentHub:
    return AgentHub(store=StateStore(path))


def connect(hub: AgentHub, node_id: str = "node-a") -> AgentSession:
    """Сессия без стрима: обработчикам докладов нужен только сам узел."""
    session = AgentSession(AgentNode(
        node_id=node_id, accepts_tasks=True, gpus_total=2, gpus_free=2,
        hardware=agent_pb2.Hardware(num_gpus=2, vram_free_bytes=24 * 1024**3,
                                    host_ram_gb=64.0)))
    hub.sessions[node_id] = session
    return session


def telemetry(node_id: str, tasks: dict) -> agent_pb2.Telemetry:
    report = agent_pb2.Telemetry(node_id=node_id, tasks_running=len(tasks))
    for task_id, state in tasks.items():
        report.tasks.add(task_id=task_id, state=state)
    return report


# ------------------------------------------------------------------ хранилище
def test_снимок_переживает_запись_и_чтение(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"tasks": [{"task_id": "t1"}], "groups": []})
    assert store.load() == {"format": FORMAT, "tasks": [{"task_id": "t1"}],
                            "groups": []}


def test_битый_файл_не_мешает_старту(tmp_path):
    """Оркестратор, отказавшийся стартовать из-за побитого состояния, не даёт
    сделать единственное, что тут помогает."""
    path = tmp_path / "state.json"
    path.write_text("{это не json")
    assert StateStore(path).load() == {}


def test_чужой_формат_не_читается_наполовину(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"format": 999, "tasks": [{"task_id": "t1"}]}')
    assert StateStore(path).load() == {}


def test_недописанный_файл_не_подменяет_прежний(tmp_path):
    """Запись идёт через временный файл: посреди неё на месте лежит либо
    прежний снимок, либо новый, но не половина нового."""
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.save({"tasks": [{"task_id": "t1"}], "groups": []})
    store.save({"tasks": [{"task_id": "t2"}], "groups": []})
    assert [t["task_id"] for t in store.load()["tasks"]] == ["t2"]
    assert list(path.parent.glob(".*part")) == [], "остался временный файл"


# --------------------------------------------------------------- перезапуск
def test_модель_и_задачи_переживают_перезапуск(tmp_path):
    """Тот самый симптом: пересобрали оркестратор — вкладки «Модели» и
    «Задачи» пусты, а узлы продолжают считать."""
    path = tmp_path / "state.json"
    before = hub_with(path)
    connect(before)
    group = before.submit_group(size=2, command=["python", "stage.py"],
                                node_ids=["node-a", "node-a"],
                                label="Qwen/Qwen3-4B")
    before.on_task_state(agent_pb2.TaskState(task_id=group.tasks[0], state="running"))
    before.on_task_state(agent_pb2.TaskState(task_id=group.tasks[1], state="running"))
    before.flush()

    after = hub_with(path)
    assert after.restore() == 2
    assert after.group_for("Qwen/Qwen3-4B") is not None, \
        "модель пропала из админки после перезапуска"
    assert after.tasks[group.tasks[0]].command == ["python", "stage.py"]
    assert after.tasks[group.tasks[0]].state == "running"
    assert after.groups[group.group_id].nodes[1] == "node-a"


def test_ресурсы_восстанавливаются_ровно(tmp_path):
    """Не через `as_dict`: тот округляет для показа, и восстановленная задача
    держала бы чуть-чуть не те карты, что держит на самом деле."""
    path = tmp_path / "state.json"
    before = hub_with(path)
    connect(before)
    task = before.submit(command=["sleep", "1"],
                         resources={"gpus": 1, "vram_gb": 17.17, "cpus": 2.5})
    before.flush()

    after = hub_with(path)
    after.restore()
    assert after.tasks[task.task_id].resources == task.resources


def test_снимок_пишется_только_на_переходах(tmp_path):
    """Телеметрия идёт с каждого узла каждые несколько секунд, и в ней меняется
    одно «сколько секунд работает». Писать диск на это — писать в пустоту."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    task = hub.submit(command=["sleep", "1"])
    hub.flush()

    hub.on_task_state(agent_pb2.TaskState(task_id=task.task_id, state="running"))
    assert hub._dirty, "переход в running обязан попасть на диск"
    hub.flush()

    hub.on_task_state(agent_pb2.TaskState(task_id=task.task_id, state="running",
                                          seconds=42.0))
    assert not hub._dirty, "запись из-за одного лишь тика секунд"


# ----------------------------------------------------------------- сведение
def test_незнакомая_задача_принимается_с_узла(tmp_path):
    """Состояние потеряли, а узел считает. Такую задачу надо хотя бы увидеть и
    суметь снять: снятие адресуется по task_id, которого иначе никто не знает."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    hub.on_telemetry(telemetry("node-a", {"task-осиротевшая": "running"}))

    record = hub.tasks.get("task-осиротевшая")
    assert record is not None, "узел доложил о задаче, а её не видно"
    assert record.state == "running"
    assert record.node_id == "node-a"
    assert record.adopted and record.command == []


def test_доклад_без_узла_не_плодит_задач(tmp_path):
    """Одиночный доклад не говорит, кто держит задачу, а задача без узла
    неостановима — и потому бесполезна."""
    hub = hub_with(tmp_path / "state.json")
    hub.on_task_state(agent_pb2.TaskState(task_id="ниоткуда", state="running"))
    assert hub.tasks == {}


def test_пропавшая_с_узла_задача_перестаёт_держать_карты(tmp_path):
    """Доклад узла — полная опись. Чего в нём нет, того на узле нет: узел
    перезапустили, машину выключили, сказать об этом было уже некому."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    task = hub.submit(command=["sleep", "1000"], resources={"gpus": 1})
    hub.on_task_state(agent_pb2.TaskState(task_id=task.task_id, state="running"))
    task.submitted_at = time.time() - ADOPTION_GRACE_S - 1

    hub.on_telemetry(telemetry("node-a", {}))
    assert hub.tasks[task.task_id].state == "gone"
    assert hub._promised_gpus().get("node-a", 0) == 0


def test_только_что_отправленная_задача_не_считается_пропавшей(tmp_path):
    """Между отправкой и первым докладом она не пропала, а не доехала."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    task = hub.submit(command=["sleep", "1000"])
    hub.on_telemetry(telemetry("node-a", {}))
    assert hub.tasks[task.task_id].state == "pending"


def test_задачи_чужого_узла_не_трогаются(tmp_path):
    """Опись присылает каждый узел про себя, и молчание одного ничего не
    говорит о задачах другого."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub, "node-a")
    connect(hub, "node-b")
    task = hub.submit(command=["sleep", "1000"], node_id="node-b")
    task.submitted_at = time.time() - ADOPTION_GRACE_S - 1
    hub.on_task_state(agent_pb2.TaskState(task_id=task.task_id, state="running"))

    hub.on_telemetry(telemetry("node-a", {}))
    assert hub.tasks[task.task_id].state == "running"


def test_без_хранилища_ничего_не_пишется(tmp_path, monkeypatch):
    """Тесты и старый способ запуска поднимают хаб без диска: файл не должен
    появляться от одного факта работы."""
    monkeypatch.chdir(tmp_path)
    hub = AgentHub()
    connect(hub)
    hub.submit(command=["sleep", "1"])
    hub.flush()
    assert hub.restore() == 0
    assert list(tmp_path.iterdir()) == []


def test_отпущенная_задача_не_воскресает(tmp_path):
    """Узел узнаёт об отпускании не мгновенно и успевает доложить о задаче
    ещё раз. Принять её обратно значило бы отменить отпускание."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    task = hub.submit(command=["true"])
    hub.on_task_state(agent_pb2.TaskState(task_id=task.task_id, state="done"))
    hub.release(task.task_id)

    hub.on_telemetry(telemetry("node-a", {task.task_id: "done"}))
    assert task.task_id not in hub.tasks


def test_собирающаяся_задача_не_считается_пропавшей(tmp_path):
    """Другая сторона той же поломки: узел докладывает задачу как
    provisioning, и сводить её как пропавшую нельзя — она просто ещё не
    запущена."""
    hub = hub_with(tmp_path / "state.json")
    connect(hub)
    task = hub.submit(command=["sleep", "1000"], resources={"gpus": 1})
    task.submitted_at = time.time() - ADOPTION_GRACE_S - 1

    hub.on_telemetry(telemetry("node-a", {task.task_id: "provisioning"}))
    assert hub.tasks[task.task_id].state == "provisioning"
    assert not hub.tasks[task.task_id].finished
