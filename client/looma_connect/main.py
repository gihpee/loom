"""looma-connect — локальный порт до Ray-кластера в Looma.

    looma-connect 201.34.135.177:8000 group-cf6421224f --token <токен>
    # слушаю 127.0.0.1:10001 → group-cf6421224f

Дальше клиент пишет обычный код у себя:

    import ray
    ray.init("ray://127.0.0.1:10001")

Кластер при этом живёт на машине, до которой нельзя дозвониться: у неё нет
входящих портов и не будет. Наружу смотрит только оркестратор, и он же
доносит байты до узла по соединению, которое узел открыл сам.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from looma_connect import __version__
from looma_connect.tunnel import Listener
from looma_connect.websocket import endpoint, opener

# Порт Ray Client по умолчанию: клиент пишет ray://127.0.0.1:10001 и не думает.
DEFAULT_PORT = 10001


def parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="looma-connect",
        description="Локальный порт до Ray-кластера, живущего за NAT")
    parser.add_argument("orchestrator", help="адрес оркестратора, например 1.2.3.4:8000")
    parser.add_argument("cluster", help="идентификатор группы из панели")
    parser.add_argument("--token", default=os.environ.get("LOOMA_TOKEN", ""),
                        help="токен доступа; по умолчанию из LOOMA_TOKEN")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"локальный порт (по умолчанию {DEFAULT_PORT}, 0 — любой)")
    parser.add_argument("--insecure", action="store_true",
                        help="ws:// вместо wss:// — только для своего стенда")
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    if not args.token.isascii():
        # Иначе это всплывёт из глубины библиотеки как «invalid
        # X-Looma-Admin-Token header» — сообщением, которое называет заголовок,
        # а не то, что человек скопировал токен вместе с чем-то лишним.
        print("в токене есть символы, которых не бывает в заголовке HTTP — "
              "похоже, он скопирован с лишним", file=sys.stderr)
        return 2
    url = endpoint(args.orchestrator, args.cluster, insecure=args.insecure)
    listener = Listener(port=args.port, connect=opener(url, args.token))
    port = await listener.start()
    print(f"слушаю 127.0.0.1:{port} → {args.cluster}")
    print(f'    ray.init("ray://127.0.0.1:{port}")')
    if not args.token:
        # Не отказ: на своём стенде оркестратор может не спрашивать токена.
        # Но промолчать значит дать клиенту гадать над «connection refused».
        print("токен не задан — если оркестратор его требует, соединения будут "
              "отвергнуты", file=sys.stderr)
    try:
        await listener.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        await listener.close()
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
