"""Поднять узел Ray и дождаться, пока кластер соберётся.

`ray start` подпроцессом, а не `ray.init()` из этого процесса: узел кластера
должен пережить наш скрипт, а сам процесс — остаться свободным, чтобы отвечать
на /health, пока Ray поднимается. Это те же минуты, что у стадии уходят на
веса, и всё это время снаружи надо видеть разницу между «поднимается» и
«отвечает».
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional

from loom_ray.ports import RankPorts, head_address, ports_for

logger = logging.getLogger("loom_ray.cluster")

# Сколько ранг ждёт голову. Голова поднимается быстро, но её задача может
# стоять в очереди за окружением, а окружение — это pip.
HEAD_WAIT_S = float(os.environ.get("LOOM_RAY_HEAD_WAIT_S", "900"))
# Сколько голова ждёт остальных, прежде чем считать это провалом.
JOIN_WAIT_S = float(os.environ.get("LOOM_RAY_JOIN_WAIT_S", "900"))


# Взводится, когда задачу снимают. Ожидания смотрят на него, иначе SIGTERM во
# время сборки не прерывал бы её, а просто убивал процесс — и `ray stop` не
# успевал бы отработать, оставляя чужой машине работающий кластер.
STOP = threading.Event()


class ClusterRefused(RuntimeError):
    """Кластер не собрался, и вот почему."""


class Stopped(ClusterRefused):
    """Сборку прервали снаружи. Не отказ — решение."""


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_head(base: int = 0, stride: int = 0, *, timeout_s: float = 0.0) -> str:
    """Дождаться, пока голова начнёт принимать соединения.

    Проверяется соединением, а не файлом и не паузой: между «ranks стартовали
    одновременно» и «голова готова» лежит установка окружения на её узле, и
    сколько она займёт, отсюда знать нельзя.
    """
    address = head_address(base, stride)
    host, port = address.split(":")
    deadline = time.time() + (timeout_s or HEAD_WAIT_S)
    while time.time() < deadline:
        if STOP.is_set():
            raise Stopped("сборку кластера прервали")
        if _reachable(host, int(port)):
            return address
        STOP.wait(1.0)
    raise ClusterRefused(
        f"голова кластера не отозвалась на {address} за "
        f"{timeout_s or HEAD_WAIT_S:.0f}с. Если ранги на разных узлах, это "
        "означает, что между ними нет проброса портов (docs/RAY.md)")


def _common_flags(ports: RankPorts, gpus: Optional[int]) -> List[str]:
    flags = [
        "--node-ip-address", "127.0.0.1",
        "--node-manager-port", str(ports.node_manager),
        "--object-manager-port", str(ports.object_manager),
        "--runtime-env-agent-port", str(ports.runtime_env_agent),
        "--dashboard-agent-listen-port", str(ports.dashboard_listen),
        "--dashboard-agent-grpc-port", str(ports.dashboard_grpc),
        "--metrics-export-port", str(ports.metrics),
        "--min-worker-port", str(ports.worker_first),
        "--max-worker-port", str(ports.worker_last),
        # Узел чужой: молча слать телеметрию с него наружу мы не будем.
        "--disable-usage-stats",
    ]
    if gpus is not None:
        flags += ["--num-gpus", str(gpus)]
    return flags


def start_node(rank: int, size: int, *, gpus: Optional[int] = None,
               base: int = 0, stride: int = 0, temp_dir: str = "") -> str:
    """Поднять узел Ray для этого ранга. Возвращает адрес головы."""
    ports = ports_for(rank, base=base, stride=stride)
    argv = [sys.executable, "-m", "ray.scripts.scripts", "start"]
    if rank == 0:
        # Без --ray-client-server-port: он требует ray[client], которого в
        # минимальной установке нет, и ray start отказывается стартовать
        # вовсе. Порт под него в раскладке зарезервирован — понадобится, когда
        # появится клиентский вход (docs/RAY.md).
        # --include-dashboard только здесь: Ray отвергает его у неголовных
        # рангов целиком, а не игнорирует. Смотреть на дашборд всё равно
        # неоткуда — у узла нет входящих портов.
        argv += ["--head", "--port", str(ports.gcs),
                 "--include-dashboard", "false"]
        address = head_address(base, stride)
    else:
        address = wait_for_head(base, stride)
        argv += ["--address", address]
    argv += _common_flags(ports, gpus)
    if temp_dir:
        argv += ["--temp-dir", temp_dir]

    logger.info("ранг %d/%d: %s", rank, size,
                "поднимаю голову" if rank == 0 else f"подключаюсь к {address}")
    result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise ClusterRefused(
            f"ray start для ранга {rank} не отработал: "
            + " / ".join(detail[-4:] or ["вывода не было"]))
    return address


def wait_for_group(size: int, *, timeout_s: float = 0.0) -> int:
    """Дождаться, пока в кластере окажутся ВСЕ ранги.

    Не «голова поднялась»: клиентский код, запущенный на половине кластера,
    не падает — он считает вдвое дольше и молча. Разницу видно только отсюда.
    """
    import ray

    deadline = time.time() + (timeout_s or JOIN_WAIT_S)
    seen = 0
    while time.time() < deadline:
        if STOP.is_set():
            raise Stopped("ожидание рангов прервали")
        seen = alive_nodes()
        if seen >= size:
            return seen
        STOP.wait(1.0)
    raise ClusterRefused(
        f"в кластере {seen} узлов из {size} — остальные не подключились за "
        f"{timeout_s or JOIN_WAIT_S:.0f}с")


def alive_nodes() -> int:
    """Сколько узлов сейчас живо. Ноль означает и «нет узлов», и «Ray ещё не
    поднялся» — для /health разница неважна, оба значат «не готов»."""
    try:
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", log_to_driver=False,
                     ignore_reinit_error=True, configure_logging=False)
        return sum(1 for node in ray.nodes() if node.get("Alive"))
    except Exception:
        return 0


def stop_node() -> None:
    """Убрать за собой. Группу процессов агент всё равно снесёт, но снести её
    посреди записи — не то же самое, что дать Ray остановиться самому."""
    try:
        subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"],
                       capture_output=True, timeout=120)
    except Exception:
        logger.debug("ray stop не отработал", exc_info=True)
