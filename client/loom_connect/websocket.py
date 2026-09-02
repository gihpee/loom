"""Канал до оркестратора поверх WebSocket.

Почему не gRPC, хотя у оркестратора он уже есть: `grpcio` — это сорок мегабайт
ради байтопровода, а утилита ставится клиенту, который про Loom больше ничего
знать не должен. WebSocket даёт двунаправленный поток, проходит через
корпоративные прокси и TLS, и на клиенте это одна маленькая библиотека.
"""

from __future__ import annotations

from urllib.parse import quote

from loom_connect.tunnel import Upstream

# Заголовок с токеном, а не параметр в адресе: параметры оседают в логах
# прокси и в истории команд, а токен даёт право исполнять код на чужой машине.
TOKEN_HEADER = "X-Loom-Admin-Token"


def endpoint(orchestrator: str, cluster: str, *, insecure: bool = False) -> str:
    """Адрес канала для этого кластера.

    Принимает и голый хост, и адрес со схемой: клиент почти наверняка
    скопирует его из панели, а там он с http://.
    """
    address = orchestrator.strip().rstrip("/")
    for prefix, secure in (("https://", True), ("http://", False)):
        if address.startswith(prefix):
            address = address[len(prefix):]
            insecure = not secure
            break
    scheme = "ws" if insecure else "wss"
    return f"{scheme}://{address}/connect/{quote(cluster, safe='')}"


class WebSocketUpstream(Upstream):
    """Одно соединение WebSocket, притворяющееся байтовым каналом."""

    def __init__(self, socket) -> None:
        self.socket = socket

    async def send(self, data: bytes) -> None:
        await self.socket.send(data)

    async def recv(self) -> bytes:
        message = await self.socket.recv()
        # Текстовый кадр здесь означает, что на том конце не тоннель, а,
        # например, страница ошибки прокси. Отдаём пустое: для вызывающего это
        # «поток кончился», и соединение закроется, а не зависнет.
        return message if isinstance(message, bytes) else b""

    async def close(self) -> None:
        try:
            await self.socket.close()
        except Exception:
            pass


def opener(url: str, token: str):
    """Сделать функцию, открывающую новый канал. По одному на TCP-соединение."""
    async def connect() -> Upstream:
        import ssl

        from websockets.asyncio.client import connect as ws_connect

        headers = {TOKEN_HEADER: token} if token else {}
        try:
            socket = await ws_connect(url, additional_headers=headers,
                                      max_size=None, open_timeout=30)
        except ssl.SSLError as exc:
            # «wrong version number» означает ровно одно: на том конце обычный
            # HTTP, а мы пришли с TLS. Сообщение OpenSSL про это не говорит, и
            # искать причину идут в сертификаты.
            raise ConnectionError(
                f"на {url} отвечает не TLS — похоже, оркестратор без https. "
                f"Добавьте --insecure ({exc})") from None
        return WebSocketUpstream(socket)

    return connect
