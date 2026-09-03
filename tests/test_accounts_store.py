"""Учётные записи в настоящей базе.

Пропускается, когда Postgres не поднят: `docker run -d --rm -p 55432:5432
-e POSTGRES_PASSWORD=looma -e POSTGRES_DB=looma postgres:16-alpine`, затем
LOOMA_TEST_DATABASE_URL=postgresql://postgres:looma@127.0.0.1:55432/looma

Проверять это заглушками бессмысленно: почти всё, что тут может сломаться, —
поведение самой базы (регистр адреса, каскад при удалении, срок сессии).
"""

from __future__ import annotations

import os

import pytest

from looma.accounts.store import ADMIN, Accounts, AccountError
from looma.store.database import Database

URL = os.environ.get("LOOMA_TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(not URL, reason="нет LOOMA_TEST_DATABASE_URL")


@pytest.fixture
async def accounts():
    database = Database(URL)
    await database.connect()
    async with database.acquire() as connection:
        await connection.execute(
            "DROP TABLE IF EXISTS api_keys, sessions, accounts,"
            " schema_migrations CASCADE")
    await database.migrate()
    yield Accounts(database)
    await database.close()


# ------------------------------------------------------------- создание
async def test_запись_создаётся_и_читается(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    assert (await accounts.by_id(made.id)).email == "ivan@looma.ru"
    assert made.is_admin is False


async def test_адрес_занят_независимо_от_регистра(accounts):
    """Иначе Ivan@ и ivan@ — два разных человека, и второй зарегистрируется
    поверх первого, ничего не заметив."""
    await accounts.create(email="Ivan@looma.ru", password="длинный пароль")
    with pytest.raises(AccountError, match="уже зарегистрирован"):
        await accounts.create(email="ivan@LOOMA.ru", password="другой пароль")


async def test_короткий_пароль_отвергается(accounts):
    with pytest.raises(AccountError, match="короче десяти"):
        await accounts.create(email="a@b.ru", password="1234")


def test_несуществующая_роль_отвергается_без_базы():
    """Проверка раньше запроса: незачем ходить в базу за тем, что и так не
    бывает."""
    import asyncio
    with pytest.raises(AccountError, match="не бывает"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            Accounts(None).create(email="a@b.ru", password="длинный пароль",
                                  role="superuser"))


# ------------------------------------------------------------------ вход
async def test_вход_по_паролю(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    assert (await accounts.sign_in(email="ivan@looma.ru",
                                   password="длинный пароль")).id == made.id
    assert await accounts.sign_in(email="ivan@looma.ru", password="не тот") is None


async def test_вход_не_чувствителен_к_регистру_адреса(accounts):
    await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    assert await accounts.sign_in(email="IVAN@Looma.RU", password="длинный пароль")


async def test_отключённая_запись_не_пускает(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    await accounts.set_disabled(made.id, True)
    assert await accounts.sign_in(email="ivan@looma.ru",
                                  password="длинный пароль") is None


# --------------------------------------------------------------- сессии
async def test_сессия_опознаёт_владельца(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    token = await accounts.start_session(made.id)
    assert (await accounts.by_session(token)).id == made.id


async def test_сессию_можно_погасить(accounts):
    """Ради этого сессии и лежат в базе, а не в подписанной cookie: подписанную
    нельзя отозвать до истечения срока."""
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    token = await accounts.start_session(made.id)
    await accounts.end_session(token)
    assert await accounts.by_session(token) is None


async def test_смена_пароля_выкидывает_все_сессии(accounts):
    """Смена пароля обычно и означает «меня взломали»."""
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    token = await accounts.start_session(made.id)
    await accounts.set_password(made.id, "совершенно другой пароль")
    assert await accounts.by_session(token) is None


async def test_отключение_записи_гасит_сессии(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    token = await accounts.start_session(made.id)
    await accounts.set_disabled(made.id, True)
    assert await accounts.by_session(token) is None


async def test_чужой_токен_не_подходит(accounts):
    assert await accounts.by_session("выдуманный") is None
    assert await accounts.by_session("") is None


# ------------------------------------------------------------ ключи API
async def test_ключ_опознаёт_владельца(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    key, record = await accounts.issue_key(made.id, name="ноутбук")
    assert (await accounts.by_api_key(key)).id == made.id
    assert record["hint"] in key


async def test_отозванный_ключ_перестаёт_работать(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    key, record = await accounts.issue_key(made.id)
    assert await accounts.revoke_key(made.id, record["id"]) is True
    assert await accounts.by_api_key(key) is None


async def test_чужой_ключ_не_отзывается(accounts):
    """Иначе номер ключа в адресе даёт власть над чужими ключами."""
    сам = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    чужой = await accounts.create(email="petr@looma.ru", password="длинный пароль")
    _key, record = await accounts.issue_key(сам.id)
    assert await accounts.revoke_key(чужой.id, record["id"]) is False


async def test_ключи_отключённой_записи_не_работают(accounts):
    made = await accounts.create(email="ivan@looma.ru", password="длинный пароль")
    key, _record = await accounts.issue_key(made.id)
    await accounts.set_disabled(made.id, True)
    assert await accounts.by_api_key(key) is None


async def test_роль_доезжает(accounts):
    made = await accounts.create(email="root@looma.ru", password="длинный пароль",
                                 role=ADMIN)
    assert made.is_admin
    assert (await accounts.by_id(made.id)).is_admin
