"""Ранг Ray-кластера, запущенный как задача Loom.

    python -m loom_ray.server --size 3 [--script job.py]

Что делает ранг 0: поднимает голову, ждёт остальных, запускает скрипт клиента.
Что делают остальные: подключаются к голове и держатся, пока их не снимут.

Здесь нет ничего про модели, слои и веса — и не должно быть. Это второй жилец
того же слота, что и стадия инференса: агент не отличает эту задачу от любой
другой и знать про Ray не обязан.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from loom_ray import cluster
from loom_ray.ports import crossing_for_group, ports_for

logging.basicConfig(level=os.environ.get("LOOM_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("loom_ray.server")

AGENT_URL = os.environ.get("LOOM_AGENT_URL", "")
TASK_ID = os.environ.get("LOOM_TASK_ID", "")

STATE: dict = {"ready": False, "phase": "starting", "nodes": 0, "size": 0,
               "error": "", "client_port": 0}
_STOP = threading.Event()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/health":
            self._json(404, {"error": "нет такого адреса"})
            return
        # Тот же контракт, что у стадии: «ok» означает, что работать можно, а
        # не что процесс запустился. Запущен он за минуты до готовности, и
        # запрос, пришедший в этом промежутке, падает непонятно.
        ready = bool(STATE["ready"])
        self._json(200 if ready else 503, {
            "status": "ok" if ready else STATE["phase"],
            "rank": STATE.get("rank", 0),
            "nodes": STATE["nodes"],
            "size": STATE["size"],
            "error": STATE["error"],
            # Порт клиентского входа. Спрашивает его оркестратор, когда до
            # кластера приходит loom-connect: раскладку портов определяет
            # версия Ray, и знать её оркестратору значит обновлять его вместе
            # с ней. Ноль означает «внешнего входа нет».
            "client_port": STATE["client_port"],
        })

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def announce(port: int) -> None:
    """Сказать агенту, на каком порту мы в итоге встали.

    Агент может только предложить номер; между предложением и bind его мог
    занять другой процесс на этой же машине.
    """
    if not AGENT_URL:
        return
    request = urllib.request.Request(
        f"{AGENT_URL}/ready", data=json.dumps({"port": port}).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Loom-Task": TASK_ID})
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            answer.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.error("не удалось сообщить агенту про порт %d: %s", port, exc)


def ask_forwarding(size: int, rank: int) -> dict:
    """Сказать агенту, что кому открыть.

    Две разные вещи, и путать их нельзя:

    `ports`    — что должны видеть СОСЕДНИЕ РАНГИ. Только для них агент поднимает
                 слушателей, и только когда ранги на разных машинах.
    `external` — что может открыть ОРКЕСТРАТОР, когда до кластера приходит
                 loom-connect. Это клиентский вход, и соседям он не нужен —
                 из-за чего и не попадал в разрешения вовсе.

    Раскладку присылаем МЫ: её определяет версия Ray, а не версия агента.
    Знай её агент — обновление Ray стало бы обновлением всего парка.
    """
    if not AGENT_URL:
        return {"listening": 0}
    body: dict = {}
    if size >= 2:
        body["ports"] = {str(r): p for r, p in crossing_for_group(size).items()}
    if rank == 0 and STATE["client_port"]:
        body["external"] = [STATE["client_port"]]
    if not body:
        return {"listening": 0}
    request = urllib.request.Request(
        f"{AGENT_URL}/forward", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Loom-Task": TASK_ID})
    with urllib.request.urlopen(request, timeout=60) as answer:
        return json.loads(answer.read() or b"{}")


def serve_health(port: int) -> ThreadingHTTPServer:
    """Поднять /health ДО того, как начнём собирать кластер: иначе минуты
    сборки снаружи неотличимы от зависшей задачи.

    Предложенный номер — предложение, а не приказ: между ним и нашим bind его
    мог занять кто угодно, а привилегированный мы и вовсе не займём. Свой
    выбор сообщаем агенту, иначе он будет стучаться не туда.
    """
    for candidate in (port, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError as exc:
            logger.warning("порт %s занять не вышло (%s)", candidate, exc)
    else:
        raise SystemExit("нет порта, на котором можно отвечать")
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    announce(server.server_address[1])
    return server


def run_script(path: str, address: str) -> int:
    """Запустить код клиента на ранге 0, когда кластер уже собран.

    `RAY_ADDRESS` вместо правки его кода: клиентский `ray.init()` без
    аргументов подключится к нашему кластеру, а не поднимет свой рядом — а
    именно это и произошло бы, и заметно бы не было.
    """
    env = dict(os.environ, RAY_ADDRESS=address)
    logger.info("кластер собран, запускаю %s", path)
    return subprocess.run([sys.executable, path], env=env).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int,
                        default=int(os.environ.get("LOOM_GROUP_SIZE", "1")))
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("LOOM_RANK", "0")))
    parser.add_argument("--serve-port", type=int,
                        default=int(os.environ.get("LOOM_SERVE_PORT", "0")))
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--script", default="",
                        help="что запустить на ранге 0, когда кластер соберётся; "
                             "без него кластер просто стоит и ждёт")
    args = parser.parse_args(argv)

    # ДО сборки, а не после: снятая на середине задача обязана успеть убрать
    # за собой Ray, иначе на чужой машине остаётся работающий кластер.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (cluster.STOP.set(), _STOP.set()))

    STATE.update(rank=args.rank, size=args.size, phase="starting")
    health = serve_health(args.serve_port) if args.serve_port else None

    # Плазма-сокет ложится внутрь временного каталога Ray, а путь unix-сокета
    # не может быть длиннее 103 байт. Каталог задачи в лимит не влезает — см.
    # docs/RAY.md; LOOM_TASK_TMP агент даёт как раз для этого.
    temp_dir = os.environ.get("LOOM_TASK_TMP", "")
    if temp_dir:
        os.environ.setdefault("RAY_TMPDIR", temp_dir)

    code = 0
    try:
        STATE["phase"] = "прошу проброс портов"
        if args.rank == 0:
            STATE["client_port"] = cluster.client_port(args.size)
        bridged = ask_forwarding(args.size, args.rank)
        if bridged.get("listening"):
            logger.info("агент слушает %d чужих портов для рангов %s",
                        bridged["listening"], bridged.get("ranks"))
        STATE["phase"] = "поднимаю ray"
        address = cluster.start_node(args.rank, args.size, gpus=args.gpus,
                                     temp_dir=temp_dir)
        STATE["phase"] = "жду остальные ранги"
        STATE["nodes"] = cluster.wait_for_group(args.size)
        STATE["ready"] = True
        STATE["phase"] = "ok"
        logger.info("ранг %d/%d в кластере, узлов %d, голова %s",
                    args.rank, args.size, STATE["nodes"], address)

        if args.rank == 0 and args.script:
            code = run_script(args.script, address)
            logger.info("скрипт клиента завершился с кодом %d", code)
        else:
            _hold()
    except cluster.Stopped as exc:
        # Не ошибка: так выглядит снятие задачи.
        STATE.update(ready=False, phase="stopped", error=str(exc))
        logger.info("%s", exc)
    except cluster.ClusterRefused as exc:
        STATE.update(ready=False, phase="failed", error=str(exc))
        logger.error("%s", exc)
        code = 1
    finally:
        cluster.stop_node()
        if health is not None:
            health.shutdown()
    return code


def _hold() -> None:
    """Стоять, пока не снимут. Ранги, кроме нулевого, только держат кластер —
    работу им раздаёт Ray, а не мы."""
    while not _STOP.is_set():
        STATE["nodes"] = cluster.alive_nodes()
        _STOP.wait(5.0)


if __name__ == "__main__":
    raise SystemExit(main())
