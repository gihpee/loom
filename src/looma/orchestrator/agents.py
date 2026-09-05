"""The orchestrator's half of the agent protocol.

One session per connected node, one record per task, and the rule that keeps
both honest: **the orchestrator never blocks on a node**. A node can be slow,
asleep or gone, and an API call that waited on one would take the whole
orchestrator down with it.

The old ControlGateway is untouched and still serves the workers that speak it.
This is the service the new agent uses (docs/AGENT_PLAN.md).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from looma.logging_config import get_logger
from looma.orchestrator.resources import Resources, choose_node
from looma.orchestrator.state import StateStore
from looma.proto_gen import agent_pb2, agent_pb2_grpc

logger = get_logger(__name__)

# Big enough that a gigabyte moves in reasonable time, small enough to stay
# well inside the gRPC message limit with room for the envelope.
CHUNK_BYTES = 1024 * 1024
# How long an API call waits for a node to answer before giving up on it. A
# node that has gone quiet must not hold a request open forever.
NODE_REPLY_TIMEOUT_S = 120.0
# Как часто изменившееся состояние уходит на диск. Не при каждом изменении:
# телеметрия идёт каждые несколько секунд с каждого узла, и запись на каждый
# доклад — это запись в пустоту, потому что меняется в ней одно лишь «сколько
# секунд работает задача».
FLUSH_INTERVAL_S = 2.0
# Сколько ждать, прежде чем счесть задачу пропавшей с узла. Узел докладывает
# только то, что держит; между отправкой задачи и первым докладом о ней есть
# промежуток, и в нём задача ещё не пропала, а просто не доехала.
ADOPTION_GRACE_S = 90.0
# Сколько законченная группа лежит в списке, прежде чем её уберут сами.
# Дольше, чем узел держит каталог задачи (час), чтобы результат успели забрать.
KEEP_FINISHED_S = float(os.environ.get("LOOMA_KEEP_FINISHED_S", str(24 * 3600)))


class AgentError(RuntimeError):
    """Something the caller asked for cannot be done, with the reason."""


# ------------------------------------------------------------------- records
@dataclass
class ResultFile:
    name: str
    size_bytes: int
    digest: str

    def as_dict(self) -> dict:
        return {"name": self.name, "size_bytes": self.size_bytes, "digest": self.digest}

    @classmethod
    def from_stored(cls, raw: dict) -> "ResultFile":
        return cls(name=raw.get("name", ""), size_bytes=int(raw.get("size_bytes", 0)),
                   digest=raw.get("digest", ""))


@dataclass
class TaskRecord:
    task_id: str
    node_id: str
    command: List[str]
    state: str = "pending"
    error: str = ""
    exit_code: int = 0
    devices: List[int] = field(default_factory=list)
    seconds: float = 0.0
    results: List[ResultFile] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)
    resources: Optional[Resources] = None
    group_id: str = ""
    rank: int = 0
    # Задача, о которой узел доложил, а мы её не заводили: пережила наш
    # перезапуск, а снимок состояния до неё не дошёл. Команду её никто не
    # знает — узел докладывает не то, что запускал, а как оно себя чувствует.
    adopted: bool = False

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "cancelled", "gone")

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "command": self.command,
            "state": self.state,
            "error": self.error,
            "exit_code": self.exit_code,
            "devices": self.devices,
            "seconds": round(self.seconds, 1),
            "submitted_at": self.submitted_at,
            "results": [r.as_dict() for r in self.results],
            "group_id": self.group_id,
            "rank": self.rank,
            "adopted": self.adopted,
        }

    def stored(self) -> dict:
        """Снимок для диска.

        Не `as_dict`: тот округляет и досчитывает для показа, а прочитать
        обратно надо ровно то, что было, — иначе после перезапуска задача
        держит чуть-чуть не те ресурсы, чем держала до него.
        """
        return {
            "task_id": self.task_id, "node_id": self.node_id,
            "command": list(self.command), "state": self.state, "error": self.error,
            "exit_code": self.exit_code, "devices": list(self.devices),
            "seconds": self.seconds, "submitted_at": self.submitted_at,
            "results": [r.as_dict() for r in self.results],
            "group_id": self.group_id, "rank": self.rank, "adopted": self.adopted,
            "resources": None if self.resources is None else {
                "vram_bytes": self.resources.vram_bytes,
                "ram_bytes": self.resources.ram_bytes,
                "cpus": self.resources.cpus,
                "gpus": self.resources.gpus,
                "disk_bytes": self.resources.disk_bytes,
            },
        }

    @classmethod
    def from_stored(cls, raw: dict) -> "TaskRecord":
        held = raw.get("resources")
        return cls(
            task_id=raw["task_id"], node_id=raw.get("node_id", ""),
            command=list(raw.get("command") or []), state=raw.get("state", "pending"),
            error=raw.get("error", ""), exit_code=int(raw.get("exit_code", 0)),
            devices=list(raw.get("devices") or []),
            seconds=float(raw.get("seconds", 0.0)),
            submitted_at=float(raw.get("submitted_at", time.time())),
            results=[ResultFile.from_stored(r) for r in (raw.get("results") or [])],
            resources=Resources(**held) if isinstance(held, dict) else None,
            group_id=raw.get("group_id", ""), rank=int(raw.get("rank", 0)),
            adopted=bool(raw.get("adopted", False)),
        )


@dataclass
class GroupRecord:
    """A job spread over several nodes: a pipeline, a training run.

    All-or-nothing by construction. A pipeline missing a stage does not run
    slower, it does not run — so a group that cannot be placed whole is not
    placed at all, and nothing is left holding resources for it.
    """

    group_id: str
    # What this group serves, when it serves something a client asks for by
    # name — a model id, usually. Empty for a group that is just work.
    label: str = ""
    tasks: Dict[int, str] = field(default_factory=dict)   # rank -> task_id
    nodes: Dict[int, str] = field(default_factory=dict)   # rank -> node_id
    submitted_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "size": len(self.tasks),
            "ranks": [{"rank": r, "task_id": self.tasks[r], "node_id": self.nodes.get(r, "")}
                      for r in sorted(self.tasks)],
            "submitted_at": self.submitted_at,
        }

    def stored(self) -> dict:
        return {
            "group_id": self.group_id, "label": self.label,
            "tasks": {str(r): t for r, t in self.tasks.items()},
            "nodes": {str(r): n for r, n in self.nodes.items()},
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_stored(cls, raw: dict) -> "GroupRecord":
        return cls(
            group_id=raw["group_id"], label=raw.get("label", ""),
            tasks={int(r): t for r, t in (raw.get("tasks") or {}).items()},
            nodes={int(r): n for r, n in (raw.get("nodes") or {}).items()},
            submitted_at=float(raw.get("submitted_at", time.time())),
        )


@dataclass
class AgentNode:
    node_id: str
    region: str = ""
    agent_version: str = ""
    hardware: Optional[agent_pb2.Hardware] = None
    accepts_tasks: bool = False
    refusal: str = ""
    environment_kinds: List[str] = field(default_factory=list)
    gpus_total: int = 0
    gpus_free: int = 0
    tasks_running: int = 0
    env_cache_bytes: int = 0
    #: Сколько занимают скачанные веса. Растёт быстрее окружений, и владелец
    #: узла должен видеть, за что отдал диск.
    model_cache_bytes: int = 0
    #: Место на томе узла. Ноль означает «этот агент ещё не умеет о нём
    #: рассказывать», а не «диск кончился» — различать это должен показывающий.
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    peer_id: str = ""
    symmetric_nat: bool = False
    reachable: bool = False
    in_network: bool = False
    #: Адреса, по которым узел себя объявляет. Хранятся целиком, а не сводятся
    #: к одному «доступен»: когда сосед не может дозвониться, первый вопрос —
    #: куда именно он звонил, и без этого списка ответа нет нигде.
    visible_addrs: List[str] = field(default_factory=list)
    direct_share: float = 0.0
    direct: int = 0
    relayed: int = 0
    link_rtt_ms: float = 0.0
    update_state: str = ""
    update_version: str = ""
    update_error: str = ""
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        hardware = self.hardware
        return {
            "node_id": self.node_id,
            "region": self.region,
            "agent_version": self.agent_version,
            "device": hardware.device if hardware else "",
            "gpu_name": hardware.gpu_name if hardware else "",
            "cuda_version": hardware.cuda_version if hardware else "",
            "gpus_total": self.gpus_total,
            "gpus_free": self.gpus_free,
            "vram_free_bytes": hardware.vram_free_bytes if hardware else 0,
            "host_ram_gb": hardware.host_ram_gb if hardware else 0.0,
            "accepts_tasks": self.accepts_tasks,
            "refusal": self.refusal,
            "environment_kinds": list(self.environment_kinds),
            "tasks_running": self.tasks_running,
            "env_cache_bytes": self.env_cache_bytes,
            "model_cache_bytes": self.model_cache_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "peer_id": self.peer_id,
            "symmetric_nat": self.symmetric_nat,
            "reachable": self.reachable,
            "in_network": self.in_network,
            "visible_addrs": list(self.visible_addrs),
            "direct": self.direct,
            "relayed": self.relayed,
            "direct_share": round(self.direct_share, 3),
            "link_rtt_ms": round(self.link_rtt_ms, 1),
            "update_state": self.update_state,
            "update_version": self.update_version,
            "update_error": self.update_error,
            "connected_at": self.connected_at,
            "seconds_since_seen": round(time.time() - self.last_seen, 1),
        }

    def placement_view(self) -> dict:
        hardware = self.hardware
        return {
            "vram_free_bytes": hardware.vram_free_bytes if hardware else 0,
            "ram_bytes": int((hardware.host_ram_gb if hardware else 0.0) * 1024**3),
            "cpus": 8.0,
            "num_gpus": self.gpus_free,
            "disk_bytes": 1 << 62,  # not tracked yet; see docs/AGENT_PLAN.md §7
        }


class Tunnel:
    """Одно соединение снаружи внутрь. Байты в обе стороны и ничего больше.

    Держит очередь входящих, а не колбэк: читающая сторона — веб-сокет, и ей
    удобнее забирать по мере готовности, чем принимать в чужом потоке.
    """

    def __init__(self, *, hub, conn_id: str, session, inbox: asyncio.Queue,
                 node_id: str = "") -> None:
        self.hub = hub
        self.conn_id = conn_id
        self.session = session
        self.inbox = inbox
        self.node_id = node_id
        self.closed = False

    def send(self, data: bytes) -> None:
        if self.closed:
            return
        self.session.send(agent_pb2.ServerMessage(
            tunnel_chunk=agent_pb2.TunnelChunk(conn_id=self.conn_id, data=data)))

    async def recv(self) -> bytes:
        """Следующий кусок, или пусто, когда канал кончился."""
        chunk = await self.inbox.get()
        if chunk.error:
            logger.info("канал %s закрыт узлом: %s", self.conn_id, chunk.error)
            self.error = chunk.error
            return b""
        if chunk.last:
            return b""
        return chunk.data

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.hub._tunnels.pop(self.conn_id, None)
        try:
            self.session.send(agent_pb2.ServerMessage(
                tunnel_chunk=agent_pb2.TunnelChunk(conn_id=self.conn_id, last=True)))
        except Exception:
            logger.debug("закрытие канала %s не ушло", self.conn_id, exc_info=True)

    error = ""


class AgentSession:
    """One connected node, and the only way to say anything to it."""

    def __init__(self, node: AgentNode) -> None:
        self.node = node
        self.outbox: "asyncio.Queue[agent_pb2.ServerMessage]" = asyncio.Queue()
        self.closed = asyncio.Event()

    def send(self, message: agent_pb2.ServerMessage) -> None:
        self.outbox.put_nowait(message)


# ----------------------------------------------------------------------- hub
class AgentHub:
    """Everything the orchestrator knows about agents and their tasks."""

    def __init__(self, *, keystore=None, rendezvous=None, releases=None,
                 release_base_url: str = "", store: Optional[StateStore] = None) -> None:
        self.keystore = keystore
        # Only for the one address handed out at registration. The hub knows
        # nothing else about p2p and should not.
        self.rendezvous = rendezvous
        self.releases = releases
        # Where a node fetches a payload from. Left empty when the orchestrator
        # has no reachable HTTP address, in which case no release is offered:
        # naming one a node cannot fetch would make every registration start a
        # download that fails.
        self.release_base_url = release_base_url.rstrip("/")
        # Куда сохраняется то, что должно пережить перезапуск. None — значит
        # никуда: так работают тесты и так работал весь хаб раньше.
        self.store = store
        self._dirty = False
        self._sweeps = 0
        self.sessions: Dict[str, AgentSession] = {}
        self.tasks: Dict[str, TaskRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}
        # Группы, чьи стадии уже доложили о готовности. Загруженная стадия не
        # разгружается, так что проверять повторно незачем.
        self._ready_groups: set = set()
        # Задачи, которые мы только что отпустили. Узел узнаёт об этом не
        # мгновенно и успевает доложить о них ещё раз — а незнакомая задача в
        # докладе принимается обратно, и отпущенная воскресала бы.
        self._released: Dict[str, float] = {}
        # Replies a caller is waiting for, by command id. A node that never
        # answers leaves a future nobody resolves, which is why every wait has
        # a timeout.
        self._pending_logs: Dict[str, asyncio.Future] = {}
        self._collecting: Dict[str, Tuple[bytearray, asyncio.Future]] = {}
        # Очередь, а не future: ответ может прийти частями, и собрать его
        # целиком — частный случай, а не наоборот.
        self._serving: Dict[str, asyncio.Queue] = {}
        # Открытые байтовые каналы наружу: conn_id -> очередь входящих кусков.
        self._tunnels: Dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------ хранение
    def _touch(self) -> None:
        """Пометить, что снимок на диске устарел. Пишет `flush`."""
        self._dirty = True

    def restore(self) -> int:
        """Поднять задачи и группы прошлого запуска. Возвращает число задач.

        Сессий здесь ещё нет: узлы дозвонятся сами и своей телеметрией скажут,
        что из этого живо (`on_telemetry`). До тех пор задачи числятся в том
        состоянии, в котором их застал прошлый процесс, — что и есть правда:
        оркестратор ничего не останавливал, а узел ничего не заметил.
        """
        if self.store is None:
            return 0
        raw = self.store.load()
        for entry in raw.get("groups") or []:
            try:
                record = GroupRecord.from_stored(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self.groups[record.group_id] = record
        for entry in raw.get("tasks") or []:
            try:
                record = TaskRecord.from_stored(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self.tasks[record.task_id] = record
        if self.tasks or self.groups:
            logger.info("восстановлено задач: %d, групп: %d",
                        len(self.tasks), len(self.groups))
        return len(self.tasks)

    def flush(self) -> None:
        """Записать снимок, если с прошлого раза что-то изменилось."""
        if self.store is None or not self._dirty:
            return
        self._dirty = False
        self.store.save({
            "tasks": [t.stored() for t in self.tasks.values()],
            "groups": [g.stored() for g in self.groups.values()],
        })

    async def flush_loop(self) -> None:
        """Писать снимок в фоне, пока живёт процесс.

        Отдельной задачей, а не из обработчиков: те выполняются в потоке gRPC
        и в горячем пути запроса, где обращение к диску — это задержка ровно в
        том месте, где её меньше всего ждут.
        """
        try:
            while True:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                self._sweeps += 1
                # Уборка редко: она ходит по всем группам, а меняется в ней
                # что-то раз в сутки.
                if self._sweeps % int(300 / FLUSH_INTERVAL_S) == 0:
                    self.prune()
                self.flush()
        except asyncio.CancelledError:
            self.flush()
            raise

    # ---------------------------------------------------------------- nodes
    def node_list(self) -> List[dict]:
        return [s.node.as_dict() for s in self.sessions.values()]

    def _placement_nodes_with(self, promised: Dict[str, int]) -> Dict[str, dict]:
        view: Dict[str, dict] = {}
        for node_id, session in self.sessions.items():
            if not session.node.accepts_tasks:
                continue
            free = session.node.placement_view()
            free["num_gpus"] = max(0, free["num_gpus"] - promised.get(node_id, 0))
            view[node_id] = free
        return view

    def _placement_nodes(self) -> Dict[str, dict]:
        """What each node has free, with cards already promised taken out.

        GPUs are counted here rather than left to `jobs._subtract`, which
        deliberately does not decrement them: on the shard path a card is a
        requirement, not a consumable, and two shards share one perfectly well
        if the VRAM fits. An agent does the opposite — it hands a task its own
        devices and refuses when too few are free — so a placement that
        double-booked a card would be accepted here and rejected there, which
        looks like tasks failing at random on a busy node.
        """
        return self._placement_nodes_with(self._promised_gpus())

    def _promised_gpus(self) -> Dict[str, int]:
        """Cards held by tasks already placed but not yet reported by telemetry.

        Telemetry says the same thing a few seconds later. Two tasks submitted
        in one breath would both see the card free without this.
        """
        held: Dict[str, int] = {}
        for task in self.tasks.values():
            if task.finished or task.resources is None:
                continue
            held[task.node_id] = held.get(task.node_id, 0) + task.resources.gpus
        return held

    def _reserved(self) -> Dict[str, Resources]:
        """What tasks already placed are holding, so a card is not given twice.

        Telemetry says the same thing eventually, but it arrives seconds later
        and two tasks submitted in one breath would both see a free card.
        """
        held: Dict[str, Resources] = {}
        for task in self.tasks.values():
            if task.finished or task.resources is None:
                continue
            held[task.node_id] = (held.get(task.node_id) or Resources(cpus=0.0)).plus(
                task.resources)
        return held

    # ---------------------------------------------------------------- tasks
    def submit(
        self,
        *,
        command: List[str],
        environment: Optional[dict] = None,
        resources: Optional[dict] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_s: int = 3600,
        inputs: Optional[Dict[str, bytes]] = None,
        node_id: str = "",
    ) -> TaskRecord:
        """Place a task and send it. Returns as soon as it is on its way.

        Does not wait for the node to start it: provisioning an environment can
        take half an hour, and an HTTP request must not be open for it. The
        caller follows the task by its state.
        """
        if not command:
            raise AgentError("a task needs a command to run")
        wanted = Resources.from_request(resources)
        if node_id:
            session = self.sessions.get(node_id)
            if session is None:
                raise AgentError(f"node {node_id!r} is not connected")
            if not session.node.accepts_tasks:
                raise AgentError(session.node.refusal or f"node {node_id!r} takes no tasks")
        else:
            chosen, why = choose_node(
                nodes=self._placement_nodes(), resources=wanted, reserved=self._reserved()
            )
            if not chosen:
                raise AgentError(why or "no connected node can take this task")
            node_id = chosen
            session = self.sessions[node_id]

        task_id = f"task-{uuid.uuid4().hex[:12]}"
        inputs = inputs or {}
        record = TaskRecord(task_id=task_id, node_id=node_id, command=list(command),
                            state="pending", resources=wanted)
        self.tasks[task_id] = record
        self._touch()

        session.send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
            command_id=f"run-{task_id}",
            task_id=task_id,
            command=list(command),
            env=dict(env or {}),
            timeout_s=max(1, int(timeout_s)),
            resources=agent_pb2.Resources(
                gpus=wanted.gpus, ram_bytes=wanted.ram_bytes,
                cpus=wanted.cpus, disk_bytes=wanted.disk_bytes,
            ),
            environment=agent_pb2.Environment(**(environment or {"kind": "none"})),
            inputs=[
                agent_pb2.InputFile(name=name, size_bytes=len(data),
                                    digest=hashlib.sha256(data).hexdigest())
                for name, data in inputs.items()
            ],
        )))
        for name, data in inputs.items():
            for offset in range(0, len(data), CHUNK_BYTES):
                session.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
                    task_id=task_id, name=name, data=data[offset:offset + CHUNK_BYTES])))
            session.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
                task_id=task_id, name=name, last=True)))
        logger.info("task %s placed on %s (%s)", task_id, node_id, " ".join(command[:4]))
        return record

    def stop(self, task_id: str, *, reason: str = "cancelled") -> TaskRecord:
        record, session = self._locate(task_id)
        session.send(agent_pb2.ServerMessage(stop_task=agent_pb2.StopTask(
            command_id=f"stop-{task_id}", task_id=task_id, reason=reason)))
        return record

    def release(self, task_id: str) -> None:
        record, session = self._locate(task_id)
        session.send(agent_pb2.ServerMessage(release_task=agent_pb2.ReleaseTask(
            command_id=f"rel-{task_id}", task_id=task_id)))
        self.tasks.pop(task_id, None)
        self._released[task_id] = time.time()
        self._touch()

    async def logs(self, task_id: str, *, tail_lines: int = 200) -> str:
        record, session = self._locate(task_id)
        command_id = f"log-{task_id}-{uuid.uuid4().hex[:6]}"
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_logs[command_id] = waiter
        session.send(agent_pb2.ServerMessage(fetch_logs=agent_pb2.FetchLogs(
            command_id=command_id, task_id=task_id, tail_lines=tail_lines)))
        try:
            return await asyncio.wait_for(waiter, NODE_REPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AgentError(f"node {record.node_id} did not answer in time") from None
        finally:
            self._pending_logs.pop(command_id, None)

    async def collect(self, task_id: str, name: str) -> bytes:
        record, session = self._locate(task_id)
        key = f"{task_id}/{name}"
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._collecting[key] = (bytearray(), waiter)
        session.send(agent_pb2.ServerMessage(fetch_result=agent_pb2.FetchResult(
            command_id=f"get-{task_id}", task_id=task_id, name=name)))
        try:
            return await asyncio.wait_for(waiter, NODE_REPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AgentError(f"node {record.node_id} did not send {name!r} in time") from None
        finally:
            self._collecting.pop(key, None)

    # ---------------------------------------------------------------- groups
    def submit_group(
        self,
        *,
        size: int,
        command: List[str],
        environment: Optional[dict] = None,
        resources: Optional[dict] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_s: int = 3600,
        serve_port: int = 0,
        node_ids: Optional[List[str]] = None,
        per_rank: Optional[List[dict]] = None,
        label: str = "",
        inputs: Optional[Dict[str, bytes]] = None,
    ) -> GroupRecord:
        """Place a whole job, or none of it.

        `per_rank` says what differs between members: a pipeline stage runs the
        same program on a different slice of the model, and which slice is the
        orchestrator's decision because only it knows what each node can hold.
        Each entry may carry `command` and `env`, overriding the shared ones.

        Members are chosen first and started second. Starting them as they are
        chosen would leave a half-placed pipeline running on three nodes with
        nothing to send to, holding cards nobody can rent.
        """
        if size < 1:
            raise AgentError("a group needs at least one member")
        if per_rank is not None and len(per_rank) != size:
            raise AgentError(
                f"this group has {size} members and {len(per_rank)} were described")
        wanted = Resources.from_request(resources)
        chosen = self._choose_for_group(size, wanted, node_ids)

        group_id = f"group-{uuid.uuid4().hex[:10]}"
        record = GroupRecord(group_id=group_id, label=label)
        members = []
        for rank, node_id in enumerate(chosen):
            task_id = f"{group_id}-r{rank}"
            record.tasks[rank] = task_id
            record.nodes[rank] = node_id
            session = self.sessions[node_id]
            members.append(agent_pb2.GroupMember(
                rank=rank, node_id=node_id,
                peer_id=session.node.peer_id,
            ))
        self.groups[group_id] = record

        # The same files to every member: a pipeline's stages run one program,
        # and shipping it with the task is how a node that has never served
        # this model gets it without a package registry in the middle.
        declared = [
            agent_pb2.InputFile(name=name, size_bytes=len(data),
                                digest=hashlib.sha256(data).hexdigest())
            for name, data in (inputs or {}).items()
        ]
        for rank, node_id in enumerate(chosen):
            task_id = record.tasks[rank]
            self.tasks[task_id] = TaskRecord(
                task_id=task_id, node_id=node_id,
                command=list(((per_rank[rank] if per_rank else {}) or {}).get("command")
                             or command),
                state="pending", resources=wanted, group_id=group_id, rank=rank,
            )
            mine = (per_rank[rank] if per_rank else {}) or {}
            self.sessions[node_id].send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
                command_id=f"run-{task_id}",
                task_id=task_id,
                command=list(mine.get("command") or command),
                env={**(env or {}), **(mine.get("env") or {})},
                timeout_s=max(1, int(timeout_s)),
                serve_port=serve_port,
                resources=agent_pb2.Resources(
                    gpus=wanted.gpus, ram_bytes=wanted.ram_bytes,
                    cpus=wanted.cpus, disk_bytes=wanted.disk_bytes,
                ),
                environment=agent_pb2.Environment(**(environment or {"kind": "none"})),
                group=agent_pb2.TaskGroup(group_id=group_id, rank=rank, members=members),
                inputs=declared,
            )))
            for name, data in (inputs or {}).items():
                session = self.sessions[node_id]
                for offset in range(0, len(data), CHUNK_BYTES):
                    session.send(agent_pb2.ServerMessage(
                        input_chunk=agent_pb2.InputChunk(
                            task_id=task_id, name=name,
                            data=data[offset:offset + CHUNK_BYTES])))
                session.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
                    task_id=task_id, name=name, last=True)))
        self._touch()
        logger.info("group %s placed across %s", group_id, ", ".join(chosen))
        return record

    def _choose_for_group(self, size: int, wanted: Resources,
                          node_ids: Optional[List[str]]) -> List[str]:
        if node_ids:
            for node_id in node_ids:
                if node_id not in self.sessions:
                    raise AgentError(f"node {node_id!r} is not connected")
            if len(node_ids) != size:
                raise AgentError(
                    f"this group has {size} members and {len(node_ids)} nodes were named")
            return list(node_ids)
        chosen: List[str] = []
        reserved = dict(self._reserved())
        promised = dict(self._promised_gpus())
        for rank in range(size):
            # A node may hold two ranks — a two-stage model on one machine is
            # the fastest arrangement there is — but only if it still fits both,
            # which is what carrying `promised` and `reserved` forward checks.
            nodes = self._placement_nodes_with(promised)
            node_id, why = choose_node(nodes=nodes, resources=wanted, reserved=reserved)
            if not node_id:
                raise AgentError(
                    f"could not place member {rank + 1} of {size}: {why}")
            chosen.append(node_id)
            promised[node_id] = promised.get(node_id, 0) + wanted.gpus
            reserved[node_id] = (reserved.get(node_id) or Resources(cpus=0.0)).plus(wanted)
        return chosen

    def group_for(self, label: str) -> Optional[GroupRecord]:
        """Группа, обслуживающая это имя — та, что ЗАПУЩЕНА.

        Запущена не значит готова: процесс слушает за минуты до того, как
        веса окажутся в памяти. Готовность спрашивают отдельно — see
        `serving()`; здесь только выбор среди кандидатов.

        Свежайшая первой: передеплой оставляет старую группу дослуживать, и
        запрос должен идти в то, что развернули последним.
        """
        candidates = [g for g in self.groups.values() if g.label == label]
        for record in sorted(candidates, key=lambda g: g.submitted_at, reverse=True):
            head = self.tasks.get(record.tasks.get(0, ""))
            if head is not None and head.state == "running":
                return record
        return None

    async def serving(self, label: str) -> Optional[GroupRecord]:
        """Группа, которая реально ОТВЕЧАЕТ.

        Состояние задачи говорит только о процессе. Стадия сообщает о своей
        готовности сама, и спрашивать надо её: иначе запрос уходит в стадию,
        которая ещё качает веса, и падает через минуту после того, как панель
        показала «отвечает».

        Спрашивается один раз: загруженная стадия не разгружается обратно, так
        что в установившемся режиме это ничего не стоит.
        """
        record = self.group_for(label)
        if record is None:
            return None
        if record.group_id in self._ready_groups:
            return record
        try:
            status, _headers, body = await self.request(
                record.tasks[0], path="/health", timeout_s=15)
        except AgentError:
            return None
        if status != 200:
            return None
        try:
            if json.loads(body).get("status") != "ok":
                return None
        except ValueError:
            return None
        self._ready_groups.add(record.group_id)
        return record

    def stop_group(self, group_id: str, *, reason: str = "cancelled") -> GroupRecord:
        record = self.groups.get(group_id)
        if record is None:
            raise AgentError(f"no group {group_id!r}")
        self._ready_groups.discard(group_id)
        for task_id in record.tasks.values():
            try:
                self.stop(task_id, reason=reason)
            except AgentError:
                continue
        return record

    def group_finished(self, record: GroupRecord) -> bool:
        """Кончилась ли группа целиком. Пустая — да: держать нечего."""
        return all(
            (self.tasks.get(task_id) or TaskRecord(task_id=task_id, node_id="",
                                                   state="gone")).finished
            for task_id in record.tasks.values()
        )

    def forget_group(self, group_id: str) -> GroupRecord:
        """Убрать группу совсем: отпустить задачи и забыть запись.

        Остановка этого не делает намеренно — у остановленной задачи ещё
        лежит результат, за которым придут. А вот забытая группа не нужна
        никому, и без этого записи копились бы вечно, теперь ещё и в снимке
        состояния.
        """
        record = self.groups.get(group_id)
        if record is None:
            raise AgentError(f"нет группы {group_id!r}")
        for task_id in record.tasks.values():
            try:
                self.release(task_id)
            except AgentError:
                # Узел отключился — отпускать некому, но запись убрать всё
                # равно надо: иначе она остаётся навсегда именно в том случае,
                # когда мешает больше всего.
                self.tasks.pop(task_id, None)
        self.groups.pop(group_id, None)
        self._ready_groups.discard(group_id)
        self._touch()
        return record

    def prune(self, older_than_s: float = KEEP_FINISHED_S) -> int:
        """Убрать законченные группы, за которыми давно не приходили.

        Иначе список растёт монотонно: остановленная группа остаётся в нём
        навсегда, и через месяц работы панель показывает историю вместо
        состояния.
        """
        deadline = time.time() - max(60.0, older_than_s)
        stale = [
            group_id for group_id, record in self.groups.items()
            if record.submitted_at < deadline and self.group_finished(record)
        ]
        for group_id in stale:
            try:
                self.forget_group(group_id)
            except AgentError:
                continue
        if stale:
            logger.info("убрано законченных групп: %d", len(stale))
        return len(stale)

    def announce_release(self) -> int:
        """Сказать подключённым узлам, что версия сменилась.

        Объявление в ответе на регистрацию доходит только до тех, кто
        переподключился — то есть до упавших. Исправный узел держит поток
        месяцами, и без этой рассылки выкатка на нём не начиналась бы никогда.

        Само по себе это не push: узел получает то же самое объявление и решает
        сам, как и при регистрации.
        """
        if self.releases is None or not self.release_base_url:
            return 0
        told = 0
        for node_id, session in list(self.sessions.items()):
            offer = self._release_message(node_id)
            if offer is None:
                continue
            session.send(agent_pb2.ServerMessage(release=offer))
            told += 1
        if told:
            logger.info("told %d node(s) about release %s", told,
                        self.releases.current.version if self.releases.current else "?")
        return told

    def _release_message(self, node_id: str):
        """Каким релизом этот узел должен быть, если он в текущей волне."""
        store = self.releases
        if store is None or not self.release_base_url:
            return None
        release = store.offer_to(node_id)
        if release is None:
            return None
        return agent_pb2.AgentRelease(
            version=release.version,
            url=f"{self.release_base_url}/agent/release/{release.version}.tar.gz",
            sha256=release.sha256,
            signature=release.signature,
        )

    # ---------------------------------------------------- байтовый канал
    def open_tunnel(self, task_id: str, port: int) -> "Tunnel":
        """Соединение с портом задачи, поверх стрима, который узел открыл сам.

        Ничего не ждёт: узел ответит либо байтами, либо закрытием с причиной, и
        оба ответа придут в очередь этого канала.
        """
        record, session = self._locate(task_id)
        conn_id = f"c-{uuid.uuid4().hex[:12]}"
        inbox: asyncio.Queue = asyncio.Queue()
        self._tunnels[conn_id] = inbox
        session.send(agent_pb2.ServerMessage(tunnel_open=agent_pb2.TunnelOpen(
            conn_id=conn_id, task_id=task_id, port=int(port))))
        return Tunnel(hub=self, conn_id=conn_id, session=session, inbox=inbox,
                      node_id=record.node_id)

    def on_tunnel_chunk(self, chunk: agent_pb2.TunnelChunk) -> None:
        inbox = self._tunnels.get(chunk.conn_id)
        if inbox is None:
            return
        inbox.put_nowait(chunk)

    def route(self, message: agent_pb2.TaskMessage) -> None:
        """Carry a message between two members that have no direct link.

        The long way round — node, orchestrator, node — and the reason p2p
        exists. It is the correct path when hole punching cannot work, and
        wrong to skip when it can, which is why the node decides and this only
        does what it is asked.
        """
        record = self.groups.get(message.group_id)
        if record is None:
            logger.warning("a message for unknown group %s", message.group_id)
            return
        node_id = record.nodes.get(message.to_rank)
        session = self.sessions.get(node_id or "")
        if session is None:
            logger.warning("rank %d of %s is on %s, which is not connected",
                           message.to_rank, message.group_id, node_id)
            return
        session.send(agent_pb2.ServerMessage(task_message=message))

    # ------------------------------------------------------------- serving
    async def request(self, task_id: str, *, method: str = "GET", path: str = "/",
                      body: bytes = b"", headers: Optional[Dict[str, str]] = None,
                      timeout_s: float = 600.0) -> Tuple[int, Dict[str, str], bytes]:
        """Спросить у задачи её HTTP и дождаться ответа целиком."""
        status, head, whole = 502, {}, bytearray()
        async for piece in self.request_stream(
                task_id, method=method, path=path, body=body,
                headers=headers, timeout_s=timeout_s):
            if isinstance(piece, tuple):
                status, head = piece
            else:
                whole.extend(piece)
        return status, head, bytes(whole)

    async def request_stream(self, task_id: str, *, method: str = "GET", path: str = "/",
                             body: bytes = b"", headers: Optional[Dict[str, str]] = None,
                             timeout_s: float = 600.0):
        """То же, но по частям: сначала (status, headers), потом куски тела.

        Так модель на чьей-то домашней машине отвечает интернету, и первое
        слово видно сразу — а не через минуту, неотличимую от зависания.
        """
        record, session = self._locate(task_id)
        command_id = f"req-{uuid.uuid4().hex[:10]}"
        pieces: asyncio.Queue = asyncio.Queue()
        self._serving[command_id] = pieces
        session.send(agent_pb2.ServerMessage(task_request=agent_pb2.TaskRequest(
            command_id=command_id, task_id=task_id, method=method, path=path,
            body=body or b"", headers=dict(headers or {}),
        )))
        started = True
        try:
            while True:
                try:
                    answer = await asyncio.wait_for(pieces.get(), timeout_s)
                except asyncio.TimeoutError:
                    raise AgentError(
                        f"{record.node_id} не ответил за {timeout_s:.0f}s"
                    ) from None
                if answer.error:
                    raise AgentError(answer.error)
                if started:
                    yield answer.status, dict(answer.headers)
                    started = False
                if answer.body:
                    yield answer.body
                if answer.last:
                    return
        finally:
            self._serving.pop(command_id, None)

    def on_task_response(self, answer: agent_pb2.TaskResponse) -> None:
        pieces = self._serving.get(answer.command_id)
        if pieces is not None:
            pieces.put_nowait(answer)

    def _locate(self, task_id: str) -> Tuple[TaskRecord, AgentSession]:
        record = self.tasks.get(task_id)
        if record is None:
            raise AgentError(f"no task {task_id!r}")
        session = self.sessions.get(record.node_id)
        if session is None:
            raise AgentError(
                f"the node that holds {task_id} ({record.node_id}) is not connected"
            )
        return record, session

    # ------------------------------------------------- what agents tell us
    def on_task_state(self, state: agent_pb2.TaskState, *, node_id: str = "") -> None:
        record = self.tasks.get(state.task_id)
        if record is None:
            if not node_id:
                # Одиночный доклад без узла: сказать о нём нечего, кроме
                # task_id, а задача без узла неостановима и потому бесполезна.
                return
            if state.task_id in self._released:
                return
            record = self._adopt(state.task_id, node_id)
        was = (record.state, len(record.results))
        record.state = state.state
        record.error = state.error or record.error
        record.exit_code = state.exit_code
        record.devices = list(state.devices)
        record.seconds = state.seconds
        if state.results:
            record.results = [
                ResultFile(name=f.name, size_bytes=f.size_bytes, digest=f.digest)
                for f in state.results
            ]
        if was != (record.state, len(record.results)):
            self._touch()

    def _adopt(self, task_id: str, node_id: str) -> TaskRecord:
        """Завести запись для задачи, о которой доложил узел, а мы её не знаем.

        Так выглядит потерянное состояние с той стороны: узел считает, карты
        заняты, а оркестратор об этой задаче не слышал — и не мог бы её снять,
        потому что снятие адресуется по task_id. Заводим запись: команду мы уже
        не узнаем, но остановить и увидеть — сможем.
        """
        logger.info("узел %s держит незнакомую задачу %s; принимаем её",
                    node_id, task_id)
        record = TaskRecord(task_id=task_id, node_id=node_id, command=[],
                            state="pending", adopted=True)
        self.tasks[task_id] = record
        self._touch()
        return record

    def on_result_chunk(self, chunk: agent_pb2.ResultChunk) -> None:
        key = f"{chunk.task_id}/{chunk.name}"
        entry = self._collecting.get(key)
        if entry is None:
            return
        buffer, waiter = entry
        if chunk.data:
            buffer.extend(chunk.data)
        if chunk.last and not waiter.done():
            if chunk.error:
                waiter.set_exception(AgentError(chunk.error))
            else:
                waiter.set_result(bytes(buffer))

    def on_logs(self, logs: agent_pb2.TaskLogs) -> None:
        waiter = self._pending_logs.get(logs.command_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(logs.text)

    def on_telemetry(self, report: agent_pb2.Telemetry) -> None:
        session = self.sessions.get(report.node_id)
        if session is None:
            return
        node = session.node
        node.last_seen = time.time()
        node.gpus_total = report.gpus_total
        node.gpus_free = report.gpus_free
        node.tasks_running = report.tasks_running
        node.env_cache_bytes = report.env_cache_bytes
        node.model_cache_bytes = report.model_cache_bytes
        node.disk_free_bytes = report.disk_free_bytes
        node.disk_total_bytes = report.disk_total_bytes
        peer = report.peer
        node.peer_id = peer.peer_id or node.peer_id
        node.symmetric_nat = peer.symmetric_nat
        # Адрес без /p2p-circuit значит, что до узла можно дозвониться прямо;
        # circuit — это тот же путь через реле под другим именем.
        node.reachable = any("/p2p-circuit" not in a for a in peer.visible_addrs)
        # Разные вопросы: reachable — «дозвонятся ли до него», in_network —
        # «дозвонится ли он». Узел без DHT принимает входящие как ни в чём не
        # бывало и при этом не находит никого.
        node.in_network = bool(peer.in_network)
        node.visible_addrs = list(peer.visible_addrs)
        node.direct = peer.direct
        node.relayed = peer.relayed
        node.direct_share = peer.direct_share
        node.link_rtt_ms = peer.link_rtt_ms
        node.update_state = report.update.state
        node.update_version = report.update.version
        node.update_error = report.update.error
        if node.hardware is not None and report.vram_free_bytes:
            node.hardware.vram_free_bytes = report.vram_free_bytes
        for state in report.tasks:
            self.on_task_state(state, node_id=node.node_id)
        self._reconcile(node.node_id, {s.task_id for s in report.tasks})

    def _reconcile(self, node_id: str, held: set) -> None:
        """Свести наш список задач узла с тем, что узел действительно держит.

        Доклад узла — это полная опись, а не список изменений: чего в нём нет,
        того на узле нет. Задача, пропавшая оттуда, кончилась так, что сказать
        об этом было уже некому — узел перезапустили, машину выключили.
        Оставить её числиться работающей значит держать под неё карты, которых
        она не занимает, и не пускать на узел новую работу.

        Со скидкой на дорогу: между отправкой задачи и первым докладом о ней
        проходит время, и в нём задача не пропала, а ещё не доехала.
        """
        deadline = time.time() - ADOPTION_GRACE_S
        self._released = {t: at for t, at in self._released.items() if at > deadline}
        for record in self.tasks.values():
            if record.node_id != node_id or record.finished:
                continue
            if record.task_id in held or record.submitted_at > deadline:
                continue
            logger.info("узел %s больше не держит %s; считаем её пропавшей",
                        node_id, record.task_id)
            record.state = "gone"
            record.error = record.error or "узел больше не держит эту задачу"
            self._touch()


# ------------------------------------------------------------------ servicer
class AgentGatewayServicer(agent_pb2_grpc.AgentGatewayServicer):
    def __init__(self, *, hub: AgentHub) -> None:
        self.hub = hub

    async def Attach(self, request_iterator, context):
        session: Optional[AgentSession] = None
        reader = None
        try:
            first = await request_iterator.__anext__()
        except StopAsyncIteration:
            return
        if first.WhichOneof("msg") != "register":
            await context.abort(16, "the first message on this stream must be a registration")
            return

        registration = first.register
        node = self._node_from(registration)
        if self.hub.keystore is not None:
            ok, why = self._check_key(registration.join_key, node.node_id)
            if not ok:
                yield agent_pb2.ServerMessage(
                    register_ack=agent_pb2.RegisterAck(ok=False, error=why))
                return

        session = AgentSession(node)
        # A reconnect replaces the old session rather than living beside it:
        # two sessions for one node means commands go to whichever is found
        # first, which is a coin toss nobody can debug.
        previous = self.hub.sessions.get(node.node_id)
        if previous is not None:
            previous.closed.set()
        self.hub.sessions[node.node_id] = session
        logger.info("agent %s attached (%s, %d GPU)", node.node_id, node.agent_version,
                    node.hardware.num_gpus if node.hardware else 0)

        reader = asyncio.create_task(self._read(request_iterator, session))
        yield agent_pb2.ServerMessage(register_ack=agent_pb2.RegisterAck(
            ok=True,
            node_id=node.node_id,
            rendezvous=self._rendezvous_addrs(),
            relays=self._relay_addrs(),
            release=self._release_for(node.node_id),
        ))
        try:
            while not session.closed.is_set():
                try:
                    message = await asyncio.wait_for(session.outbox.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                yield message
        finally:
            if reader is not None:
                reader.cancel()
            if self.hub.sessions.get(node.node_id) is session:
                del self.hub.sessions[node.node_id]
                logger.info("agent %s detached", node.node_id)

    async def _read(self, request_iterator, session: AgentSession) -> None:
        try:
            async for message in request_iterator:
                kind = message.WhichOneof("msg")
                if kind == "task_state":
                    self.hub.on_task_state(message.task_state)
                elif kind == "telemetry":
                    self.hub.on_telemetry(message.telemetry)
                elif kind == "result_chunk":
                    self.hub.on_result_chunk(message.result_chunk)
                elif kind == "logs":
                    self.hub.on_logs(message.logs)
                elif kind == "task_message":
                    self.hub.route(message.task_message)
                elif kind == "task_response":
                    self.hub.on_task_response(message.task_response)
                elif kind == "tunnel_chunk":
                    self.hub.on_tunnel_chunk(message.tunnel_chunk)
                elif kind == "ack" and not message.ack.ok:
                    logger.warning("node %s refused %s: %s", session.node.node_id,
                                   message.ack.command_id, message.ack.error)
        except Exception:
            logger.debug("agent stream ended", exc_info=True)
        finally:
            session.closed.set()

    # ------------------------------------------------------------------ small
    def _node_from(self, registration: agent_pb2.Register) -> AgentNode:
        readiness = registration.readiness
        return AgentNode(
            node_id=registration.node_id,
            region=registration.region,
            agent_version=registration.agent_version,
            hardware=registration.hardware,
            accepts_tasks=readiness.accepts_tasks,
            refusal=readiness.refusal,
            environment_kinds=list(readiness.environment_kinds),
            gpus_total=registration.hardware.num_gpus,
            gpus_free=registration.hardware.num_gpus,
            peer_id=registration.peer.peer_id,
        )

    def _check_key(self, join_key: str, node_id: str) -> Tuple[bool, str]:
        try:
            secret = self.hub.keystore.validate(join_key, node_id=node_id)
        except Exception as exc:
            return False, f"this join key was refused: {exc}"
        if not secret:
            return False, "this join key is not valid; get a new one from the admin page"
        return True, ""

    def _release_for(self, node_id: str) -> Optional[agent_pb2.AgentRelease]:
        """What this node should move to, if it is in the current wave.

        Absent means "stay where you are", which is what most nodes hear for
        most of a rollout and is the correct default at every other moment.
        """
        return self.hub._release_message(node_id)

    def _rendezvous_addrs(self) -> List[str]:
        node = self.hub.rendezvous
        return list(node.multiaddrs()) if node is not None else []

    def _relay_addrs(self) -> List[str]:
        from looma.orchestrator.rendezvous import relay_addrs

        return relay_addrs()


def add_agent_gateway_to_server(server, hub: AgentHub) -> AgentGatewayServicer:
    servicer = AgentGatewayServicer(hub=hub)
    agent_pb2_grpc.add_AgentGatewayServicer_to_server(servicer, server)
    return servicer
