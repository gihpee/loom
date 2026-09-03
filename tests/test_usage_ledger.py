"""Журнал потребления.

Счётчик показывает «сейчас», журнал отвечает на вопрос «за март» — и именно
этот вопрос задают, когда приходит время платить.

Часть проверок идёт против настоящей базы: почти всё, что тут может сломаться,
— поведение самой базы (частичный индекс по открытым, сравнение времени,
округление сумм).
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

from looma.accounts.store import Accounts
from looma.store.database import Database
from looma.usage.ledger import COMPUTE, INFERENCE, RELEASED, VANISHED, Ledger, cost

URL = os.environ.get("LOOMA_TEST_DATABASE_URL", "").strip()


# ------------------------------------------------------- арифметика счёта
def test_час_одной_карты_стоит_ставку():
    assert cost(per_hour=5000, gpus=1, seconds=3600) == 5000


def test_две_карты_стоят_вдвое():
    assert cost(per_hour=5000, gpus=2, seconds=3600) == 10000


def test_доли_копейки_округляются_вверх():
    """Вверх, а не к ближайшему: доли копейки за секунду складываются в
    заметную сумму на длинной аренде, и терять их систематически в одну
    сторону — значит незаметно раздавать мощность бесплатно."""
    assert cost(per_hour=1, gpus=1, seconds=1) == 1
    assert cost(per_hour=3600, gpus=1, seconds=1) == 1


def test_без_ставки_ничего_не_стоит():
    """Узел должен работать и до того, как оператор назначил цену."""
    assert cost(per_hour=0, gpus=4, seconds=99999) == 0


@pytest.mark.parametrize("gpus, seconds", [(0, 3600), (1, 0), (-1, 3600), (1, -5)])
def test_бессмысленные_величины_дают_ноль(gpus, seconds):
    assert cost(per_hour=5000, gpus=gpus, seconds=seconds) == 0


# ------------------------------------------------------------ против базы
pytestmark_db = pytest.mark.skipif(not URL, reason="нет LOOMA_TEST_DATABASE_URL")


@pytest.fixture
async def parts():
    if not URL:
        pytest.skip("нет LOOMA_TEST_DATABASE_URL")
    database = Database(URL)
    await database.connect()
    async with database.acquire() as connection:
        await connection.execute(
            "DROP TABLE IF EXISTS token_usage, leases, rates, api_keys,"
            " sessions, accounts, schema_migrations CASCADE")
    await database.migrate()
    accounts = Accounts(database)
    who = await accounts.create(email="client@looma.ru", password="длинный пароль")
    yield Ledger(database), who
    await database.close()


async def test_ставка_ставится_и_читается(parts):
    ledger, _who = parts
    await ledger.set_rate(COMPUTE, 12000)
    assert (await ledger.rate(COMPUTE)).per_hour == 12000


async def test_несуществующий_ресурс_отвергается(parts):
    ledger, _who = parts
    with pytest.raises(ValueError, match="не бывает"):
        await ledger.set_rate("looma-майнинг", 100)


async def test_ставка_запоминается_в_аренде(parts):
    """Иначе поднятая завтра цена перепишет вчерашние счета, и объяснить
    клиенту разницу будет нечем."""
    ledger, who = parts
    await ledger.set_rate(COMPUTE, 10000)
    await ledger.open_lease(account_id=who.id, resource=COMPUTE,
                            group_id="g1", gpus=1)
    await ledger.set_rate(COMPUTE, 99000)      # цена выросла после аренды

    # Проверяется сама записанная ставка, а не стоимость: на почти нулевом
    # времени обе стоимости равны нулю, и сравнение их ничего не значило бы.
    async with ledger.db.acquire() as connection:
        записано = await connection.fetchval(
            "SELECT per_hour FROM leases WHERE group_id = $1", "g1")
    assert записано == 10000


async def test_аренда_закрывается(parts):
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    assert await ledger.close_lease("g1") == 1
    assert await ledger.open_leases() == []


async def test_закрытая_аренда_не_закрывается_дважды(parts):
    """Иначе повторный вызов сдвинул бы конец времени вперёд."""
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    await ledger.close_lease("g1")
    assert await ledger.close_lease("g1") == 0


async def test_сверка_закрывает_исчезнувшие(parts):
    """Группа умеет исчезнуть сама: узел отвалился, оркестратор перезапустился.
    Без сверки такая аренда тикала бы вечно, и счёт рос бы молча."""
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="живая")
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="пропала")

    assert await ledger.reconcile(["живая"]) == 1
    осталось = await ledger.open_leases()
    assert [x["group_id"] for x in осталось] == ["живая"]


async def test_сверка_помечает_чем_закрыла(parts):
    """Во втором случае конец времени приблизителен, и при разборе это надо
    знать."""
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g2")
    await ledger.close_lease("g1")
    await ledger.reconcile([])

    async with ledger.db.acquire() as connection:
        rows = await connection.fetch(
            "SELECT group_id, closed_why FROM leases ORDER BY group_id")
    assert dict((r["group_id"], r["closed_why"]) for r in rows) == {
        "g1": RELEASED, "g2": VANISHED}


async def test_сверка_на_пустом_списке_закрывает_всё(parts):
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    assert await ledger.reconcile([]) == 1


async def test_идущая_аренда_видна_в_отчёте(parts):
    """Клиент, глядя на счёт посреди аренды, должен видеть натикавшее, а не
    ноль."""
    ledger, who = parts
    await ledger.set_rate(COMPUTE, 360000)     # рубль в секунду на карту
    await ledger.open_lease(account_id=who.id, resource=COMPUTE,
                            group_id="g1", gpus=1)
    отчёт = await ledger.report(account_id=who.id)
    assert отчёт["leases"][0]["running"] == 1
    assert отчёт["leases"][0]["seconds"] >= 0


async def test_ресурсы_считаются_раздельно(parts):
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    await ledger.open_lease(account_id=who.id, resource=INFERENCE, group_id="g2")
    отчёт = await ledger.report(account_id=who.id)
    # По алфавиту: looma-compute раньше looma-inference.
    assert [x["resource"] for x in отчёт["leases"]] == [COMPUTE, INFERENCE]


async def test_чужое_потребление_не_видно(parts):
    ledger, who = parts
    другой = await Accounts(ledger.db).create(email="petr@looma.ru",
                                              password="длинный пароль")
    await ledger.open_lease(account_id=другой.id, resource=COMPUTE, group_id="g1")
    assert (await ledger.report(account_id=who.id))["leases"] == []
    assert (await ledger.report())["leases"] != []      # админ видит всех


# ------------------------------------------------------------------ токены
async def test_токены_копятся_по_моделям(parts):
    ledger, who = parts
    await ledger.record_tokens(account_id=who.id, model="qwen3",
                               prompt_tokens=10, completion_tokens=90)
    await ledger.record_tokens(account_id=who.id, model="qwen3",
                               prompt_tokens=5, completion_tokens=5)
    отчёт = await ledger.report(account_id=who.id)
    assert отчёт["tokens"] == [{"model": "qwen3", "prompt": 15, "completion": 95}]


async def test_пустой_ответ_не_пишется(parts):
    """Иначе журнал распухает записями ни о чём."""
    ledger, who = parts
    await ledger.record_tokens(account_id=who.id, model="qwen3",
                               prompt_tokens=0, completion_tokens=0)
    assert (await ledger.report(account_id=who.id))["tokens"] == []


# --------------------------------------------------------------- владение
async def test_своя_аренда_числится_за_владельцем(parts):
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    assert await ledger.lease_belongs_to("g1", who.id) is True


async def test_чужая_аренда_не_числится(parts):
    """Иначе номер группы в адресе даёт власть над чужим кластером."""
    ledger, who = parts
    другой = await Accounts(ledger.db).create(email="petr@looma.ru",
                                              password="длинный пароль")
    await ledger.open_lease(account_id=другой.id, resource=COMPUTE, group_id="g1")
    assert await ledger.lease_belongs_to("g1", who.id) is False


async def test_закрытая_аренда_не_числится(parts):
    """Снять уже снятое нельзя — иначе повторный вызов трогал бы группу,
    которую тем временем занял кто-то другой."""
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="g1")
    await ledger.close_lease("g1")
    assert await ledger.lease_belongs_to("g1", who.id) is False


async def test_без_входа_ничего_не_числится(parts):
    ledger, _who = parts
    assert await ledger.lease_belongs_to("g1", None) is False


async def test_список_аренд_показывает_только_свои(parts):
    ledger, who = parts
    другой = await Accounts(ledger.db).create(email="petr@looma.ru",
                                              password="длинный пароль")
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="моя")
    await ledger.open_lease(account_id=другой.id, resource=COMPUTE, group_id="чужая")
    свои = await ledger.open_leases(account_id=who.id, resource=COMPUTE)
    assert [x["group_id"] for x in свои] == ["моя"]


async def test_список_фильтруется_по_ресурсу(parts):
    """Кабинет показывает кластеры, а не развёрнутые модели."""
    ledger, who = parts
    await ledger.open_lease(account_id=who.id, resource=COMPUTE, group_id="кластер")
    await ledger.open_lease(account_id=who.id, resource=INFERENCE, group_id="модель")
    только = await ledger.open_leases(account_id=who.id, resource=COMPUTE)
    assert [x["group_id"] for x in только] == ["кластер"]
