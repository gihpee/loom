"""Локальный слушатель, который возит байты до оркестратора.

Одно соединение WebSocket на одно TCP-соединение. Мультиплексировать незачем:
Ray Client открывает их единицы, а свой формат кадров поверх чужого — лишний
слой, в котором заводятся собственные ошибки.

Байты идут двоичными кадрами как есть. Никакой своей рамки: WebSocket уже
даёт границы сообщений, и добавлять к ним ещё одни значит склеивать то, что
кто-то потом будет расклеивать.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("loom_connect")

# Кусок, которым читается локальный сокет. Крупнее — меньше кадров на мегабайт;
# слишком крупно — растёт задержка на мелком обмене, а у Ray Client его
# большинство.
CHUNK = 64 * 1024


class Upstream:
    """То, что на другом конце: умеет отправлять и принимать байты.

    Отдельным протоколом, а не конкретным websocket'ом, чтобы весь клиентский
    путь проверялся против заглушки — до того, как появится настоящий тоннель
    до агента.
    """

    async def send(self, data: bytes) -> None:      # pragma: no cover - интерфейс
        raise NotImplementedError

    async def recv(self) -> bytes:                  # pragma: no cover - интерфейс
        raise NotImplementedError

    async def close(self) -> None:                  # pragma: no cover - интерфейс
        raise NotImplementedError


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
               upstream: Upstream) -> None:
    """Возить байты в обе стороны, пока одна из сторон не кончится.

    Две задачи, а не одна: чтение из сокета и чтение из канала блокируются
    независимо, и совместить их в одном цикле можно только поставив одно в
    зависимость от другого.
    """
    async def outbound() -> None:
        try:
            while True:
                piece = await reader.read(CHUNK)
                if not piece:
                    return
                await upstream.send(piece)
        except (OSError, asyncio.IncompleteReadError):
            return

    async def inbound() -> None:
        try:
            while True:
                piece = await upstream.recv()
                if not piece:
                    return
                writer.write(piece)
                await writer.drain()
        except (OSError, ConnectionError):
            return

    tasks = [asyncio.create_task(outbound()), asyncio.create_task(inbound())]
    try:
        # Как только кончилась любая сторона, вторую держать незачем: TCP тут
        # закрывается целиком, полузакрытых состояний мы не поддерживаем.
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await upstream.close()
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionError):
            pass


class Listener:
    """Порт на 127.0.0.1, за которым стоит удалённый кластер."""

    def __init__(self, *, port: int,
                 connect: Callable[[], Awaitable[Upstream]]) -> None:
        self.port = port
        self.connect = connect
        # К чему привязались на самом деле. Проверяется в тестах: адрес
        # назначения 0.0.0.0 ядро трактует как локалхост, так что «снаружи не
        # достучаться» доказывается привязкой, а не неудачным подключением.
        self.host = ""
        self.opened = 0
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        # Только 127.0.0.1, и никогда 0.0.0.0: за этим портом — исполнение
        # произвольного кода на чужой машине, и открывать его в сеть означало
        # бы раздать её всем, кто дотянется.
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", self.port)
        self.host, self.port = self._server.sockets[0].getsockname()[:2]
        return self.port

    async def serve_forever(self) -> None:
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _accept(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        try:
            upstream = await self.connect()
        except Exception as exc:
            # Отказ наверху — не наша поломка, но молчать нельзя: клиент увидит
            # оборванное соединение и пойдёт искать причину в своём коде.
            logger.error("не удалось открыть канал до кластера: %s", exc)
            writer.close()
            return
        self.opened += 1
        await pump(reader, writer, upstream)
