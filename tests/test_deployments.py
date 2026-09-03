"""Память о том, чем подняли модель.

Единственная причина, по которой эта таблица существует: без неё каждая аренда
убивала бы инференс навсегда — группу сняли, а чем её поднимали, знал только тот
HTTP-запрос, который давно закончился.
"""

from __future__ import annotations

import os

import pytest

from looma.accounts.store import Accounts
from looma.store.database import Database
from looma.usage.deployments import EVICTED, GONE, RUNNING, Deployments
from looma.usage.ledger import COMPUTE, Ledger

URL = os.environ.get("LOOMA_TEST_DATABASE_URL", "").strip()


@pytest.fixture
async def parts():
    if not URL:
        pytest.skip("нет LOOMA_TEST_DATABASE_URL")
    database = Database(URL)
    await database.connect()
    async with database.acquire() as connection:
        await connection.execute(
            "DROP TABLE IF EXISTS deployments, token_usage, leases, rates,"
            " api_keys, sessions, accounts, schema_migrations CASCADE")
    await database.migrate()
    who = await Accounts(database).create(email="a@looma.ru",
                                          password="длинный пароль")
    yield Deployments(database), Ledger(database), who
    await database.close()


def запрос(repo="Qwen/Qwen3-4B", stages=2):
    return {"repo": repo, "stages": stages, "engine": "vllm", "dtype": "bfloat16"}


# ------------------------------------------------------------------ память
async def test_запрос_запоминается_целиком(parts):
    fleet, _ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    записи = await fleet.list()
    assert записи[0].request["repo"] == "Qwen/Qwen3-4B"
    assert записи[0].state == RUNNING


async def test_повторное_развёртывание_обновляет_запись(parts):
    fleet, _ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    await fleet.remember(group_id="g1", label="qwen3",
                         request=запрос(stages=4), account_id=who.id)
    assert (await fleet.list())[0].request["stages"] == 4


async def test_снятое_вручную_не_ждёт_возврата(parts):
    """Возвращать нечего: администратор снял намеренно."""
    fleet, _ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    await fleet.forget("g1")
    assert await fleet.list() == []


# -------------------------------------------------------------- вытеснение
async def test_снятое_арендой_помнит_чем_снято(parts):
    fleet, ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    аренда = await ledger.open_lease(account_id=who.id, resource=COMPUTE,
                                     group_id="ray-1")
    await fleet.mark_evicted(["g1"], аренда)

    ждут = await fleet.evicted_by(аренда)
    assert [d.group_id for d in ждут] == ["g1"]
    assert ждут[0].request["repo"] == "Qwen/Qwen3-4B"


async def test_возвращённое_больше_не_ждёт(parts):
    fleet, ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    аренда = await ledger.open_lease(account_id=who.id, resource=COMPUTE,
                                     group_id="ray-1")
    await fleet.mark_evicted(["g1"], аренда)
    await fleet.restored("g1")
    assert await fleet.evicted_by(аренда) == []


async def test_чужой_арендой_снятое_не_возвращается(parts):
    """Иначе закрытие одной аренды подняло бы модели, снятые другой, — и та
    осталась бы без узлов."""
    fleet, ledger, who = parts
    await fleet.remember(group_id="g1", label="a", request=запрос(), account_id=who.id)
    await fleet.remember(group_id="g2", label="b", request=запрос(), account_id=who.id)
    первая = await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="r1")
    вторая = await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="r2")
    await fleet.mark_evicted(["g1"], первая)
    await fleet.mark_evicted(["g2"], вторая)

    assert [d.group_id for d in await fleet.evicted_by(первая)] == ["g1"]


# ------------------------------------------------------------ после сбоя
async def test_ждущие_возврата_находятся_по_журналу(parts):
    """Оркестратор могли перезапустить между снятием и возвратом. Ищем по
    журналу, а не по памяти процесса, именно поэтому."""
    fleet, ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    аренда = await ledger.open_lease(account_id=who.id, resource=COMPUTE,
                                     group_id="ray-1")
    await fleet.mark_evicted(["g1"], аренда)

    assert await fleet.waiting() == []      # аренда ещё идёт — возвращать рано
    await ledger.close_lease("ray-1")
    assert [d.group_id for d in await fleet.waiting()] == ["g1"]


# --------------------------------------------------------------- защита
async def test_защита_ставится_и_видна(parts):
    fleet, _ledger, who = parts
    await fleet.remember(group_id="g1", label="qwen3", request=запрос(),
                         account_id=who.id)
    assert await fleet.set_protected("g1", True) is True
    assert await fleet.protected_ids() == {"g1", "qwen3"}


async def test_защита_несуществующего_отвергается(parts):
    fleet, _ledger, _who = parts
    assert await fleet.set_protected("нет-такой", True) is False


async def test_защита_ищется_и_по_имени_модели(parts):
    """Администратор думает именами, а вытеснение — группами."""
    fleet, _ledger, who = parts
    await fleet.remember(group_id="g1", label="витрина", request=запрос(),
                         account_id=who.id)
    await fleet.set_protected("g1", True)
    assert "витрина" in await fleet.protected_ids()
