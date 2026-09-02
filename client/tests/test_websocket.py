"""Тот же путь, но через настоящий WebSocket.

Заглушка проверяет нашу логику, а здесь проверяется транспорт: двоичные кадры,
заголовок с токеном, поведение при отказе. Сервер поднимается прямо тут —
оркестратор для этого не нужен, в этом и смысл отдельной программы.
"""

from __future__ import annotations

import asyncio

import pytest
from websockets.asyncio.server import serve

from looma_connect.tunnel import Listener
from looma_connect.websocket import TOKEN_HEADER, opener


@pytest.fixture
async def echo_server():
    """Оркестратор-заглушка: возвращает всё, что прислали, и требует токен."""
    seen = {"tokens": []}

    async def handle(socket):
        seen["tokens"].append(socket.request.headers.get(TOKEN_HEADER))
        if socket.request.headers.get(TOKEN_HEADER) != "fde901ffd176":
            await socket.close(code=1008, reason="токен не подошёл")
            return
        async for message in socket:
            await socket.send(message)

    server = await serve(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}/connect/group-a", seen
    server.close()
    await server.wait_closed()


async def through(url: str, token: str, payload: bytes, *, want: int) -> bytes:
    listener = Listener(port=0, connect=opener(url, token))
    port = await listener.start()
    serving = asyncio.create_task(listener.serve_forever())
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(payload)
        await writer.drain()
        try:
            return await asyncio.wait_for(reader.readexactly(want), timeout=10)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return b""
        finally:
            writer.close()
    finally:
        serving.cancel()
        await listener.close()


@pytest.mark.asyncio
async def test_байты_проходят_через_настоящий_websocket(echo_server):
    url, _seen = echo_server
    assert await through(url, "fde901ffd176", b"looma",
                         want=len(b"looma")) == b"looma"


@pytest.mark.asyncio
async def test_двоичное_переживает_дорогу(echo_server):
    """Через канал идёт чужой протокол, и он не текст."""
    url, _seen = echo_server
    payload = bytes(range(256)) * 32
    assert await through(url, "fde901ffd176", payload, want=len(payload)) == payload


@pytest.mark.asyncio
async def test_токен_едет_заголовком(echo_server):
    """Не параметром адреса: параметры оседают в логах прокси и в истории
    команд, а токен даёт право исполнять код на чужой машине."""
    url, seen = echo_server
    await through(url, "fde901ffd176", b"x", want=1)
    assert seen["tokens"] == ["fde901ffd176"]


@pytest.mark.asyncio
async def test_неверный_токен_закрывает_соединение(echo_server):
    """И закрывает, а не вешает: зависший ray.init отправит клиента искать
    причину в своём коде."""
    url, _seen = echo_server
    assert await through(url, "0000deadbeef", b"x", want=1) == b""


@pytest.mark.asyncio
async def test_причина_отказа_доходит_до_человека(caplog):
    """Со стенда: канал открывался и сразу закрывался, клиент видел таймаут
    ray.init, а объяснение оставалось в логе агента — там, куда за ним никто
    не пойдёт."""
    import logging

    from websockets.asyncio.server import serve

    async def refuse(socket):
        await socket.close(code=1011, reason="порт 29607 не открыт для соседей")

    server = await serve(refuse, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with caplog.at_level(logging.ERROR, logger="looma_connect"):
            await through(f"ws://127.0.0.1:{port}/connect/g", "token", b"x", want=1)
    finally:
        server.close()
        await server.wait_closed()

    сказано = " ".join(r.getMessage() for r in caplog.records)
    assert "не открыт для соседей" in сказано, "причина не дошла до клиента"
