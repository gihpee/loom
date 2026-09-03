#!/usr/bin/env python3
"""Завести учётную запись. Первая — администратора, иначе войти будет некому.

    docker compose exec orchestrator python scripts/create_admin.py ivan@looma.ru

Пароль спрашивается без эха и не берётся из аргумента намеренно: аргументы
видны в `ps` любому на машине и оседают в истории командной строки.

Ставит роль администратора, только если записей ещё нет вовсе. Второй
администратор заводится существующим — через панель: команда, раздающая права
без входа, обесценивает вход.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from looma.accounts.store import ADMIN, CLIENT, Accounts, AccountError  # noqa: E402
from looma.store.database import Database, DatabaseUnavailable, database_url  # noqa: E402


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Завести учётную запись Looma")
    parser.add_argument("email")
    parser.add_argument("--name", default="", help="как показывать в панели")
    parser.add_argument("--client", action="store_true",
                        help="завести клиента, а не администратора")
    args = parser.parse_args(argv)

    url = database_url()
    if not url:
        print("LOOMA_DATABASE_URL не задан — заводить запись негде")
        return 2

    database = Database(url)
    try:
        await database.connect(maximum=2)
    except DatabaseUnavailable as exc:
        print(exc)
        return 2

    try:
        await database.migrate()
        accounts = Accounts(database)
        первая = await accounts.count() == 0
        role = CLIENT if args.client else (ADMIN if первая else CLIENT)
        if not args.client and not первая:
            print("Записи уже есть, поэтому эта заводится клиентом.")
            print("Права администратора выдаются из панели существующим "
                  "администратором.")

        password = getpass.getpass("пароль: ")
        if password != getpass.getpass("ещё раз: "):
            print("пароли не совпали")
            return 1
        try:
            made = await accounts.create(email=args.email, password=password,
                                         role=role, display_name=args.name)
        except AccountError as exc:
            print(exc)
            return 1
        print(f"заведена запись {made.email}, роль {made.role}")
        return 0
    finally:
        await database.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
