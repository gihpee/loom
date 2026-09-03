"""Кто прислал запрос и куда его пускать.

Главное, что здесь проверяется, — дыра, которую этот слой закрывает: раньше при
незаданном админском токене проверка пускала любого, и оркестратор, поднятый без
переменной окружения, не имел никакой защиты, ничем этого не выдавая.
"""

from __future__ import annotations

import pytest

from looma.api.auth import ANONYMOUS, Authenticator, Caller, bearer_token, refuse


class FakeAccount:
    def __init__(self, id=1, admin=False):
        self.id = id
        self.is_admin = admin
        self.email = "ivan@looma.ru"

    def as_dict(self):
        return {"id": self.id, "email": self.email,
                "role": "admin" if self.is_admin else "client"}


class FakeAccounts:
    def __init__(self, *, session=None, key=None):
        self._session = session
        self._key = key

    async def by_session(self, token):
        return self._session if token == "годная-сессия" else None

    async def by_api_key(self, key):
        return self._key if key == "looma_sk_годный" else None


# ------------------------------------------------------- разбор заголовка
@pytest.mark.parametrize("header, expected", [
    ("Bearer looma_sk_x", "looma_sk_x"),
    ("bearer looma_sk_x", "looma_sk_x"),
    ("BEARER  looma_sk_x ", "looma_sk_x"),
    ("looma_sk_x", ""),
    ("", ""),
    (None, ""),
])
def test_ключ_достаётся_из_заголовка(header, expected):
    """Регистр слова Bearer не важен: половина клиентов пишет его по-своему."""
    assert bearer_token(header) == expected


# ------------------------------------------------------------- опознание
async def test_незаданный_токен_означает_никого_а_не_всех():
    """Раньше здесь было `bool(token) and ...` — при пустом токене проверка
    возвращала False и пускала любого."""
    who = await Authenticator(emergency_token="").identify(admin_token="что угодно")
    assert who is ANONYMOUS
    assert refuse(who) == "нужно войти"


async def test_аварийный_токен_даёт_права_администратора():
    who = await Authenticator(emergency_token="секрет").identify(admin_token="секрет")
    assert who.how == "emergency" and who.is_admin


async def test_неверный_аварийный_токен_не_пускает():
    who = await Authenticator(emergency_token="секрет").identify(admin_token="не тот")
    assert who is ANONYMOUS


async def test_сессия_опознаётся():
    auth = Authenticator(accounts=FakeAccounts(session=FakeAccount()))
    who = await auth.identify(session="годная-сессия")
    assert who.how == "session" and who.account_id == 1


async def test_сессия_важнее_аварийного_токена():
    """Если человек вошёл, он и есть вызывающий, даже когда в окружении лежит
    общий токен."""
    auth = Authenticator(accounts=FakeAccounts(session=FakeAccount()),
                         emergency_token="секрет")
    who = await auth.identify(session="годная-сессия", admin_token="секрет")
    assert who.how == "session"


async def test_ключ_опознаётся():
    auth = Authenticator(accounts=FakeAccounts(key=FakeAccount(id=7)))
    who = await auth.identify(authorization="Bearer looma_sk_годный")
    assert who.how == "api_key" and who.account_id == 7


async def test_негодная_сессия_не_пускает():
    auth = Authenticator(accounts=FakeAccounts(session=FakeAccount()))
    assert await auth.identify(session="протухшая") is ANONYMOUS


# ----------------------------------------------------------------- права
def test_клиенту_нельзя_в_админское():
    who = Caller(account=FakeAccount(admin=False), how="session")
    assert refuse(who) is None
    assert refuse(who, admin=True) == "это может только администратор"


def test_администратору_можно():
    who = Caller(account=FakeAccount(admin=True), how="session")
    assert refuse(who, admin=True) is None


def test_причина_различает_вход_и_права():
    """«403» без слов заставляет гадать, дело в отсутствии входа или в нехватке
    прав, а это разные действия."""
    assert refuse(ANONYMOUS, admin=True) == "нужно войти"
    assert refuse(Caller(account=FakeAccount(), how="session"),
                  admin=True) == "это может только администратор"
