"""Весь клиентский путь — до того, как появится тоннель до агента.

Заглушка на том конце позволяет проверить программу целиком, не поднимая ни
узла, ни оркестратора. Ради этого канал и описан протоколом, а не конкретным
websocket'ом.
"""

from __future__ import annotations

import asyncio

import pytest

from loom_connect.tunnel import Listener, Upstream


class Echo(Upstream):
    """Тот конец, который возвращает всё, что ему прислали."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.sent = bytearray()

    async def send(self, data: bytes) -> None:
        self.sent.extend(data)
        await self.queue.put(data)

    async def recv(self) -> bytes:
        return await self.queue.get()

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(b"")


@pytest.mark.asyncio
async def test_байты_доходят_туда_и_обратно():
    upstream = Echo()
    listener = Listener(port=0, connect=lambda: _ready(upstream))
    port = await listener.start()
    serving = asyncio.create_task(listener.serve_forever())
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"loom")
        await writer.drain()
        assert await reader.readexactly(4) == b"loom"
        writer.close()
    finally:
        serving.cancel()
        await listener.close()


@pytest.mark.asyncio
async def test_двоичные_данные_не_портятся():
    """Через канал идёт чужой протокол: нули и то, что не UTF-8, обязаны
    пережить дорогу."""
    upstream = Echo()
    listener = Listener(port=0, connect=lambda: _ready(upstream))
    port = await listener.start()
    serving = asyncio.create_task(listener.serve_forever())
    payload = bytes(range(256)) * 64
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(payload)
        await writer.drain()
        assert await reader.readexactly(len(payload)) == payload
        writer.close()
    finally:
        serving.cancel()
        await listener.close()


@pytest.mark.asyncio
async def test_отказ_наверху_закрывает_соединение_а_не_вешает():
    """Иначе клиент увидит зависший ray.init и пойдёт искать причину у себя."""
    async def refuse() -> Upstream:
        raise RuntimeError("токен не подошёл")

    listener = Listener(port=0, connect=refuse)
    port = await listener.start()
    serving = asyncio.create_task(listener.serve_forever())
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        assert await reader.read(1) == b"", "соединение должно было закрыться"
        writer.close()
    finally:
        serving.cancel()
        await listener.close()


@pytest.mark.asyncio
async def test_конец_канала_закрывает_локальный_сокет():
    upstream = Echo()
    listener = Listener(port=0, connect=lambda: _ready(upstream))
    port = await listener.start()
    serving = asyncio.create_task(listener.serve_forever())
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await upstream.queue.put(b"")     # «поток кончился»
        assert await reader.read(1) == b""
        writer.close()
    finally:
        serving.cancel()
        await listener.close()


@pytest.mark.asyncio
async def test_слушает_только_локалхост():
    """За этим портом — исполнение произвольного кода на чужой машине."""
    listener = Listener(port=0, connect=lambda: _ready(Echo()))
    await listener.start()
    try:
        # Именно привязка: 0.0.0.0 как адрес НАЗНАЧЕНИЯ ядро трактует как
        # локалхост, поэтому неудачным подключением это не доказать.
        assert listener.host == "127.0.0.1"
    finally:
        await listener.close()


async def _ready(upstream: Upstream) -> Upstream:
    return upstream
