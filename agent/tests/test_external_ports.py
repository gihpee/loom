"""Разрешения на порты: соседям одно, оркестратору другое.

Без lattica: здесь проверяется не транспорт, а то, кому что открыто, — и
именно на этом стенд споткнулся последним.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class NoRegistry:
    """Реестр задач тут не участвует: проверяются разрешения, а не запуск."""

    channel_url = ""

    def get(self, _task_id):
        return None


@pytest.fixture
def commands():
    from loom_agent.control.tasks import TaskCommands
    from loom_agent.tasks.groups import Group, Member

    made = TaskCommands(registry=NoRegistry(), send=lambda _m: None, node_id="n")
    made.groups.join("t", Group(group_id="g", rank=0,
                                members={0: Member(rank=0, node_id="n")}))
    return made


def test_внешний_порт_разрешается_отдельно_от_соседских(commands):
    """Со стенда: клиентский вход Ray лежит в local_only, а не в crossing —
    соседним РАНГАМ он не нужен. Но до него должен дотянуться оркестратор,
    когда приходит loom-connect, и разрешения на это не было ни у кого:

        канал c-... не открылся: порт 21807 не открыт для соседей

    Две разные вещи, и путать их нельзя.
    """
    # Только внешний порт: соседей нет, значит и раскладки для них нет.
    answer = commands._task_forward("t", {"external": [21807]})
    assert answer["listening"] == 0
    assert 21807 in commands.allowed_ports
    assert commands.tunnels.allow(21807), "оркестратор не сможет открыть канал"
    assert not commands.tunnels.allow(21808), "разрешили лишнее"


def test_пустая_раскладка_по_прежнему_отвергается(commands):
    from loom_agent.tasks.spec import TaskRefused

    with pytest.raises(TaskRefused, match="ни одного порта"):
        commands._task_forward("t", {})
