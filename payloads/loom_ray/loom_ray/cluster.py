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

from loom_ray.ports import RankPorts, group_base, head_address, ports_for

logger = logging.getLogger("loom_ray.cluster")

# Сколько ранг ждёт голову. Голова поднимается быстро, но её задача может
# стоять в очереди за окружением, а окружение — это pip.
HEAD_WAIT_S = float(os.environ.get("LOOM_RAY_HEAD_WAIT_S", "900"))
# Сколько голова ждёт остальных, прежде чем считать это провалом.
JOIN_WAIT_S = float(os.environ.get("LOOM_RAY_JOIN_WAIT_S", "900"))
# Сколько раз ранг пробует присоединиться и сколько ждёт между попытками.
# Голова занимает свой порт задолго до того, как становится готова принимать.
JOIN_ATTEMPTS = int(os.environ.get("LOOM_RAY_JOIN_ATTEMPTS", "5"))
JOIN_RETRY_S = float(os.environ.get("LOOM_RAY_JOIN_RETRY_S", "15"))


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
    # Сколько воркеров поднимать. Без этого Ray считает своими ВСЕ ядра машины
    # и пред-запускает воркер на каждое — а ядра на узле делятся, и два ранга
    # на одной машине заводят вдвое больше процессов, чем она стоит. Каждый со
    # своими потоками, и упирается это в лимит задолго до пользы.
    cpus = _own_cpus()
    if cpus:
        flags += ["--num-cpus", str(cpus)]
    return flags


def _own_cpus() -> int:
    """Своя доля процессора, как её назвал агент. Ноль — «решай сам»."""
    try:
        share = float(os.environ.get("LOOM_TASK_CPUS", "") or 0)
    except ValueError:
        return 0
    return max(1, int(share)) if share > 0 else 0


def start_node(rank: int, size: int, *, gpus: Optional[int] = None,
               base: int = 0, stride: int = 0, temp_dir: str = "") -> str:
    """Поднять узел Ray для этого ранга. Возвращает адрес головы."""
    # Окно группы, а не общее для всех: брошенный кластер занимает СВОИ порты,
    # и новый его больше не встречает.
    base = base or group_base(size, stride=stride)
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

    if rank == 0 and _occupied(ports.gcs):
        # Иначе Ray подключится к чужому кластеру и упадёт на несовпадении
        # имени сессии — сообщении, из которого причина не следует вовсе, и
        # искать её пойдут в своём коде, а не в списке процессов.
        raise ClusterRefused(
            f"порт {ports.gcs} уже занят: похоже, там живёт кластер прошлой "
            "попытки. Снимите старую группу — или, если это чужой процесс, "
            "сдвиньте окно через LOOM_RAY_PORT_BASE")

    logger.info("ранг %d/%d: %s", rank, size,
                "поднимаю голову" if rank == 0 else f"подключаюсь к {address}")
    _run_start(argv, rank=rank, retries=0 if rank == 0 else JOIN_ATTEMPTS)
    return address


def _occupied(port: int) -> bool:
    """Слушает ли кто-то этот порт прямо сейчас."""
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _run_start(argv: List[str], *, rank: int, retries: int) -> None:
    """Запустить `ray start`, повторяя, пока голова не примет.

    Повтор нужен именно неголовным рангам. Открытый порт головы не значит, что
    голова готова: GCS занимает его в первую секунду, а собирается ещё
    десятки — и присоединение в этом промежутке падает по таймауту raylet'а.
    Снаружи «занимает порт» и «готова» неотличимы, поэтому вместо угадывания
    здесь просто пробуют ещё раз.
    """
    last = ""
    for attempt in range(retries + 1):
        if STOP.is_set():
            raise Stopped("сборку кластера прервали")
        result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        last = " / ".join(detail[-4:] or ["вывода не было"])
        if attempt < retries:
            logger.warning("ранг %d: голова ещё не принимает (попытка %d из %d): %s",
                           rank, attempt + 1, retries + 1, last[:200])
            STOP.wait(JOIN_RETRY_S)
    raise ClusterRefused(f"ray start для ранга {rank} не отработал: {last}")


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


# Подключение к Ray делается один раз и переиспользуется. Раньше на каждый
# опрос заводился поток, а опрос идёт раз в секунду до пятнадцати минут — и
# это ровно то, что добивало узел, уже занятый чужими процессами.
_CONNECTED = threading.Event()
_BROKEN = threading.Event()


def _connect(timeout_s: float) -> bool:
    """Подключиться к своему же узлу Ray. Один раз за жизнь процесса.

    С потолком по времени: `ray.init` своего таймаута не имеет и против
    умершей головы висит молча. Один такой вызов подвешивал весь ранг — он
    оставался «running» и не отвечал ничего, что хуже любой ошибки.
    """
    if _CONNECTED.is_set():
        return True
    if _BROKEN.is_set():
        return False

    def dial() -> None:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(address="auto", log_to_driver=False,
                         ignore_reinit_error=True, configure_logging=False)
            _CONNECTED.set()
        except Exception as exc:
            logger.warning("к Ray не подключиться: %s", exc)
            _BROKEN.set()

    worker = threading.Thread(target=dial, name="ray-connect", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if not _CONNECTED.is_set() and not _BROKEN.is_set():
        logger.warning("Ray не ответил за %.0fс; больше не ждём", timeout_s)
        _BROKEN.set()
    return _CONNECTED.is_set()


def alive_nodes(timeout_s: float = 30.0) -> int:
    """Сколько узлов сейчас живо. Ноль означает и «нет узлов», и «Ray ещё не
    поднялся» — для /health разница неважна, оба значат «не готов»."""
    if not _connect(timeout_s):
        return 0
    try:
        import ray

        return sum(1 for node in ray.nodes() if node.get("Alive"))
    except Exception:
        return 0


def stop_node() -> None:
    """Ничего не делать — и это осознанно.

    Напрашивается `ray stop --force`, и он ЛОМАЕТ соседей: команда снимает все
    процессы Ray этого пользователя на машине, а не наши. Два ранга на одном
    узле работают под одним uid, так что упавший ранг своей уборкой убивал
    голову живого — тот потом стоял и ждал кластер, которого уже нет.

    Убирает за нами агент: задача работает в своей группе процессов, и снятие
    задачи сносит её целиком вместе со всем, что Ray наплодил.
    """
    return
