"""Кто прислал запрос и куда его пускать.

Три способа представиться, и они не равны по правам:

    сессия      cookie после входа через форму — человек в браузере;
    ключ API    заголовок `Authorization: Bearer looma_sk_…` — чужая программа;
    аварийный   общий админский токен из окружения.

Последний остаётся как запасной вход: если база недоступна или админ потерял
пароль, платформа не должна становиться неуправляемой. Но он именно аварийный —
о его использовании пишется в лог, и жить он должен ровно до того момента, когда
в базе появится настоящий администратор.

Отдельно про дыру, которую этот модуль закрывает. Раньше проверка выглядела как
`bool(admin_token) and token != admin_token`: при незаданном токене она
возвращала False, то есть **пускала любого**. Оркестратор, поднятый без
переменной окружения и выставленный в интернет, не имел никакой защиты и ничем
этого не выдавал. Отсутствие настройки теперь означает «никого», а не «всех».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("looma.api.auth")

SESSION_COOKIE = "looma_session"
BEARER = "bearer "


@dataclass(frozen=True)
class Caller:
    """Тот, от чьего имени пришёл запрос."""

    account: Optional[object] = None
    #: Как представился: session | api_key | emergency | anonymous
    how: str = "anonymous"

    @property
    def known(self) -> bool:
        return self.how != "anonymous"

    @property
    def is_admin(self) -> bool:
        if self.how == "emergency":
            return True
        return bool(self.account is not None and getattr(self.account, "is_admin", False))

    @property
    def account_id(self) -> Optional[int]:
        return getattr(self.account, "id", None)

    def as_dict(self) -> dict:
        if self.account is not None:
            return {**self.account.as_dict(), "how": self.how}
        return {"id": None, "email": "", "role": "admin" if self.is_admin else "",
                "how": self.how}


ANONYMOUS = Caller()


def bearer_token(header: Optional[str]) -> str:
    """Ключ из заголовка Authorization. Регистр слова Bearer не важен —
    половина клиентов пишет его по-своему, и отвергать их за это значит
    отлаживать чужой код вместо своего."""
    raw = (header or "").strip()
    if raw.lower().startswith(BEARER):
        return raw[len(BEARER):].strip()
    return ""


class Authenticator:
    """Опознаёт вызывающего. Ничего не решает про доступ — только «кто это»."""

    def __init__(self, *, accounts=None, emergency_token: str = "") -> None:
        self.accounts = accounts
        self.emergency_token = (emergency_token or "").strip()

    async def identify(self, *, session: Optional[str] = None,
                       authorization: Optional[str] = None,
                       admin_token: Optional[str] = None) -> Caller:
        """Порядок проверок — от точного к общему.

        Сессия первой: если человек вошёл, он и есть вызывающий, даже когда в
        окружении лежит аварийный токен.
        """
        if self.accounts is not None:
            if session:
                account = await self.accounts.by_session(session)
                if account is not None:
                    return Caller(account=account, how="session")
            key = bearer_token(authorization)
            if key:
                account = await self.accounts.by_api_key(key)
                if account is not None:
                    return Caller(account=account, how="api_key")

        supplied = (admin_token or "").strip() or bearer_token(authorization)
        if self.emergency_token and supplied == self.emergency_token:
            logger.warning("запрос принят по аварийному админскому токену — "
                           "заведите учётную запись администратора")
            return Caller(how="emergency")
        return ANONYMOUS


def refuse(caller: Caller, *, admin: bool = False) -> Optional[str]:
    """Почему этому вызывающему сюда нельзя. None — можно.

    Возвращает причину, а не булево: «403» без слов заставляет гадать, дело в
    отсутствии входа или в нехватке прав, а это разные действия.
    """
    if not caller.known:
        return "нужно войти"
    if admin and not caller.is_admin:
        return "это может только администратор"
    return None
