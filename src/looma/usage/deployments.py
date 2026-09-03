"""Чем подняли модель — чтобы вернуть её на место.

Единственная причина, по которой эта таблица существует: без неё каждая аренда
убивала бы инференс **навсегда**. Группу сняли, а чем её поднимали, знал только
тот HTTP-запрос, который давно закончился.

Возврат — не украшение, а вторая половина вытеснения. Механику «клиент вытесняет
базовую загрузку» без неё нельзя показывать никому: она выглядит работающей ровно
до конца первой аренды, а потом платформа тихо остаётся без инференса, и заметит
это не тот, кто арендовал.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("looma.usage.deployments")

RUNNING = "running"
EVICTED = "evicted"
GONE = "gone"


@dataclass(frozen=True)
class Deployment:
    group_id: str
    label: str
    request: dict
    account_id: Optional[int]
    state: str
    protected: bool

    def as_dict(self) -> dict:
        return {"group_id": self.group_id, "label": self.label,
                "state": self.state, "protected": self.protected,
                "account_id": self.account_id}


class Deployments:
    """Что развёрнуто, чем это поднимали и что с ним стало."""

    def __init__(self, database) -> None:
        self.db = database

    async def remember(self, *, group_id: str, label: str, request: dict,
                       account_id: Optional[int]) -> None:
        async with self.db.acquire() as connection:
            await connection.execute(
                "INSERT INTO deployments (group_id, label, request, account_id)"
                " VALUES ($1, $2, $3::jsonb, $4)"
                " ON CONFLICT (group_id) DO UPDATE"
                " SET label = $2, request = $3::jsonb, state = 'running',"
                "     evicted_by = NULL, evicted_at = NULL",
                group_id, label, json.dumps(request), account_id)

    async def forget(self, group_id: str) -> None:
        """Снято намеренно — возвращать нечего.

        Пометка, а не удаление: запись о том, что модель стояла, нужна при
        разборе счетов, а строка тут стоит байты.
        """
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE deployments SET state = $2, evicted_by = NULL"
                " WHERE group_id = $1", group_id, GONE)

    async def mark_evicted(self, group_ids: List[str], lease_id: int) -> None:
        """Снято ради аренды. Вернём, когда она закроется."""
        if not group_ids:
            return
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE deployments SET state = $3, evicted_by = $2,"
                " evicted_at = now() WHERE group_id = ANY($1::text[])",
                group_ids, lease_id, EVICTED)
        logger.info("сняты ради аренды %s: %s", lease_id, ", ".join(group_ids))

    async def evicted_by(self, lease_id: int) -> List[Deployment]:
        """Что снято этой арендой. По ним и восстанавливаем."""
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT group_id, label, request, account_id, state, protected"
                " FROM deployments WHERE evicted_by = $1 AND state = $2"
                " ORDER BY evicted_at", lease_id, EVICTED)
        return [_made(row) for row in rows]

    async def restored(self, old_group_id: str) -> None:
        """Вернули: старая запись больше не ждёт возврата.

        Новая группа получит свой идентификатор и свою запись — старую держим
        как след того, что модель однажды снимали.
        """
        async with self.db.acquire() as connection:
            await connection.execute(
                "UPDATE deployments SET state = $2, evicted_by = NULL"
                " WHERE group_id = $1", old_group_id, GONE)

    async def set_protected(self, group_id: str, protected: bool) -> bool:
        async with self.db.acquire() as connection:
            done = await connection.execute(
                "UPDATE deployments SET protected = $2 WHERE group_id = $1",
                group_id, protected)
        return done.endswith("1")

    async def protected_ids(self) -> set:
        """Кого не трогать. И по группе, и по имени модели: администратор
        думает именами, а вытеснение — группами."""
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT group_id, label FROM deployments WHERE protected")
        found = set()
        for row in rows:
            found.add(row["group_id"])
            if row["label"]:
                found.add(row["label"])
        return found

    async def list(self) -> List[Deployment]:
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT group_id, label, request, account_id, state, protected"
                " FROM deployments WHERE state <> $1 ORDER BY created_at", GONE)
        return [_made(row) for row in rows]

    async def waiting(self) -> List[Deployment]:
        """Снятое, чья аренда уже закрыта, а возврат не случился.

        Бывает, когда оркестратор перезапустили между снятием и возвратом.
        Ищется по журналу, а не по памяти процесса, именно поэтому.
        """
        async with self.db.acquire() as connection:
            rows = await connection.fetch(
                "SELECT d.group_id, d.label, d.request, d.account_id, d.state,"
                " d.protected FROM deployments d"
                " JOIN leases l ON l.id = d.evicted_by"
                " WHERE d.state = $1 AND l.closed_at IS NOT NULL"
                " ORDER BY d.evicted_at", EVICTED)
        return [_made(row) for row in rows]


def _made(row) -> Deployment:
    request = row["request"]
    return Deployment(
        group_id=row["group_id"], label=row["label"],
        request=json.loads(request) if isinstance(request, str) else dict(request),
        account_id=row["account_id"], state=row["state"],
        protected=row["protected"])
