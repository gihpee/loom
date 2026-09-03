"""Учётные записи, сессии и ключи API — то, что лежит в базе.

Здесь только работа с хранилищем: кто такой, действует ли ещё, чем подписан
запрос. Решения «кого куда пускать» живут в api/auth.py — иначе правила доступа
расползутся по SQL, где их не прочитать целиком.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from looma.accounts.secrets_ import (
    hash_password,
    hash_token,
    key_hint,
    needs_rehash,
    new_api_key,
    new_session_token,
    verify_password,
)

logger = logging.getLogger("looma.accounts")

ADMIN = "admin"
CLIENT = "client"

# Сколько живёт сессия. Месяц — компромисс между «каждый день вводить пароль» и
# «украденная cookie работает вечно». Отзыв всё равно возможен: сессии лежат в
# базе, а не в подписанной cookie, и потому их можно погасить немедленно.
SESSION_TTL = timedelta(days=30)


class AccountError(RuntimeError):
    """То, что нужно показать человеку словами."""


@dataclass(frozen=True)
class Account:
    id: int
    email: str
    role: str
    display_name: str = ""
    disabled: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN

    def as_dict(self) -> dict:
        return {"id": self.id, "email": self.email, "role": self.role,
                "display_name": self.display_name, "disabled": self.disabled}


class Accounts:
    """Учётные записи поверх пула соединений."""

    def __init__(self, database) -> None:
        self.db = database

    # ------------------------------------------------------------ создание
    async def create(self, *, email: str, password: str, role: str = CLIENT,
                     display_name: str = "") -> Account:
        email = (email or "").strip()
        if "@" not in email:
            raise AccountError(f"{email!r} не похоже на адрес почты")
        if role not in (ADMIN, CLIENT):
            raise AccountError(f"роль {role!r} не бывает: только {ADMIN} и {CLIENT}")
        if len(password or "") < 10:
            # Десять, а не восемь: единственная защита утёкшей базы — цена
            # перебора, и короткий пароль сводит её к нулю независимо от того,
            # насколько дорого мы его считаем.
            raise AccountError("пароль короче десяти символов")
        async with self.db.acquire() as connection:
            try:
                row = await connection.fetchrow(
                    "INSERT INTO accounts (email, password, role, display_name)"
                    " VALUES ($1, $2, $3, $4)"
                    " RETURNING id, email, role, display_name, disabled_at",
                    email, hash_password(password), role, display_name)
            except Exception as exc:
                if "accounts_email_key" in str(exc):
                    raise AccountError(f"{email} уже зарегистрирован") from None
                raise
        return _account(row)

    async def set_password(self, account_id: int, password: str) -> None:
        if len(password or "") < 10:
            raise AccountError("пароль короче десяти символов")
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE accounts SET password = $2 WHERE id = $1",
                account_id, hash_password(password))
        # Все сессии — прочь: смена пароля обычно и означает «меня взломали».
        await self.end_all_sessions(account_id)

    async def set_disabled(self, account_id: int, disabled: bool) -> None:
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE accounts SET disabled_at = $2 WHERE id = $1",
                account_id, _now() if disabled else None)
        if disabled:
            await self.end_all_sessions(account_id)

    # -------------------------------------------------------------- чтение
    async def by_id(self, account_id: int) -> Optional[Account]:
        async with self.db.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT id, email, role, display_name, disabled_at"
                " FROM accounts WHERE id = $1", account_id)
        return _account(row) if row else None

    async def list(self) -> List[Account]:
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, email, role, display_name, disabled_at"
                " FROM accounts ORDER BY created_at")
        return [_account(row) for row in rows]

    async def count(self) -> int:
        async with self.db.acquire() as connection:
            return int(await connection.fetchval("SELECT count(*) FROM accounts"))

    # --------------------------------------------------------------- вход
    async def sign_in(self, *, email: str, password: str) -> Optional[Account]:
        """Проверить пару. None — не подошло, и почему именно, знать никому
        не нужно: «нет такого адреса» и «неверный пароль» вместе рассказывают,
        кто у нас зарегистрирован."""
        async with self.db.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT id, email, role, display_name, disabled_at, password"
                " FROM accounts WHERE lower(email) = lower($1)", (email or "").strip())
        if row is None or row["disabled_at"] is not None:
            return None
        try:
            if not verify_password(password or "", row["password"]):
                return None
        except Exception:
            logger.exception("испорченный хэш пароля у записи %s", row["id"])
            return None
        if needs_rehash(row["password"]):
            # Единственный момент, когда пароль есть в открытом виде. Другого
            # раза не будет — пересчитываем здесь или никогда.
            await self.set_password_quietly(row["id"], password)
        return _account(row)

    async def set_password_quietly(self, account_id: int, password: str) -> None:
        """Пересчитать хэш, не трогая сессии: пароль тот же, изменились только
        параметры, и выкидывать человека из-за нашей внутренней кухни незачем."""
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE accounts SET password = $2 WHERE id = $1",
                account_id, hash_password(password))

    # -------------------------------------------------------------- сессии
    async def start_session(self, account_id: int) -> str:
        token = new_session_token()
        async with self.db.acquire() as connection:
            await connection.execute(
                "INSERT INTO sessions (token_hash, account_id, expires_at)"
                " VALUES ($1, $2, $3)",
                hash_token(token), account_id, _now() + SESSION_TTL)
        return token

    async def by_session(self, token: str) -> Optional[Account]:
        if not token:
            return None
        async with self.db.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT a.id, a.email, a.role, a.display_name, a.disabled_at"
                " FROM sessions s JOIN accounts a ON a.id = s.account_id"
                " WHERE s.token_hash = $1 AND s.expires_at > now()",
                hash_token(token))
            if row is None or row["disabled_at"] is not None:
                return None
            await connection.execute(
                "UPDATE sessions SET last_seen_at = now() WHERE token_hash = $1",
                hash_token(token))
        return _account(row)

    async def end_session(self, token: str) -> None:
        async with self.db.acquire() as connection:
            await connection.execute("DELETE FROM sessions WHERE token_hash = $1",
                                     hash_token(token))

    async def end_all_sessions(self, account_id: int) -> None:
        async with self.db.acquire() as connection:
            await connection.execute("DELETE FROM sessions WHERE account_id = $1",
                                     account_id)

    async def forget_expired_sessions(self) -> int:
        async with self.db.acquire() as connection:
            done = await connection.execute(
                "DELETE FROM sessions WHERE expires_at <= now()")
        return int(done.rsplit(" ", 1)[-1] or 0)

    # ----------------------------------------------------------- ключи API
    async def issue_key(self, account_id: int, *, name: str = "") -> tuple:
        """(что показать один раз, запись для списка). Ключ не хранится."""
        key, stored = new_api_key()
        async with self.db.acquire() as connection:
            row = await connection.fetchrow(
                "INSERT INTO api_keys (account_id, token_hash, hint, name)"
                " VALUES ($1, $2, $3, $4)"
                " RETURNING id, hint, name, created_at",
                account_id, stored, key_hint(key), name)
        return key, dict(row)

    async def by_api_key(self, key: str) -> Optional[Account]:
        if not key:
            return None
        async with self.db.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT a.id, a.email, a.role, a.display_name, a.disabled_at"
                " FROM api_keys k JOIN accounts a ON a.id = k.account_id"
                " WHERE k.token_hash = $1 AND k.revoked_at IS NULL",
                hash_token(key))
            if row is None or row["disabled_at"] is not None:
                return None
            await connection.execute(
                "UPDATE api_keys SET last_used_at = now() WHERE token_hash = $1",
                hash_token(key))
        return _account(row)

    async def keys_of(self, account_id: int) -> List[dict]:
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT id, hint, name, created_at, last_used_at, revoked_at"
                " FROM api_keys WHERE account_id = $1 ORDER BY created_at",
                account_id)
        return [dict(row) for row in rows]

    async def revoke_key(self, account_id: int, key_id: int) -> bool:
        """Отзыв, а не удаление: запись о том, что ключ существовал, нужна при
        разборе того, кто и что потребил."""
        async with self.db.acquire() as connection:
            done = await connection.execute(
                "UPDATE api_keys SET revoked_at = now()"
                " WHERE id = $1 AND account_id = $2 AND revoked_at IS NULL",
                key_id, account_id)
        return done.endswith("1")


def _account(row) -> Account:
    return Account(id=row["id"], email=row["email"], role=row["role"],
                   display_name=row["display_name"],
                   disabled=row["disabled_at"] is not None)


def _now() -> datetime:
    return datetime.now(timezone.utc)
