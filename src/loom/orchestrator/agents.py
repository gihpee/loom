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
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom.logging_config import get_logger
from loom.orchestrator.resources import Resources, choose_node
from loom.proto_gen import agent_pb2, agent_pb2_grpc

logger = get_logger(__name__)

# Big enough that a gigabyte moves in reasonable time, small enough to stay
# well inside the gRPC message limit with room for the envelope.
CHUNK_BYTES = 1024 * 1024
# How long an API call waits for a node to answer before giving up on it. A
# node that has gone quiet must not hold a request open forever.
NODE_REPLY_TIMEOUT_S = 120.0


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
        }


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
    peer_id: str = ""
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
            "gpus_total": self.gpus_total,
            "gpus_free": self.gpus_free,
            "vram_free_bytes": hardware.vram_free_bytes if hardware else 0,
            "host_ram_gb": hardware.host_ram_gb if hardware else 0.0,
            "accepts_tasks": self.accepts_tasks,
            "refusal": self.refusal,
            "environment_kinds": list(self.environment_kinds),
            "tasks_running": self.tasks_running,
            "env_cache_bytes": self.env_cache_bytes,
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
                 release_base_url: str = "") -> None:
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
        self.sessions: Dict[str, AgentSession] = {}
        self.tasks: Dict[str, TaskRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}
        # Replies a caller is waiting for, by command id. A node that never
        # answers leaves a future nobody resolves, which is why every wait has
        # a timeout.
        self._pending_logs: Dict[str, asyncio.Future] = {}
        self._collecting: Dict[str, Tuple[bytearray, asyncio.Future]] = {}
        self._serving: Dict[str, asyncio.Future] = {}

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
            )))
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
        """A group serving this name, if one is up.

        Newest first: redeploying a model leaves the old group draining for a
        while, and a request should go to what was deployed last.
        """
        candidates = [g for g in self.groups.values() if g.label == label]
        for record in sorted(candidates, key=lambda g: g.submitted_at, reverse=True):
            head = self.tasks.get(record.tasks.get(0, ""))
            if head is not None and head.state == "running":
                return record
        return None

    def stop_group(self, group_id: str, *, reason: str = "cancelled") -> GroupRecord:
        record = self.groups.get(group_id)
        if record is None:
            raise AgentError(f"no group {group_id!r}")
        for task_id in record.tasks.values():
            try:
                self.stop(task_id, reason=reason)
            except AgentError:
                continue
        return record

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
        """Ask a task's own HTTP server something, through the node's stream.

        This is how a model on somebody's home machine answers a request from
        the internet without that machine opening a single port.
        """
        record, session = self._locate(task_id)
        command_id = f"req-{uuid.uuid4().hex[:10]}"
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._serving[command_id] = waiter
        session.send(agent_pb2.ServerMessage(task_request=agent_pb2.TaskRequest(
            command_id=command_id, task_id=task_id, method=method, path=path,
            body=body or b"", headers=dict(headers or {}),
        )))
        try:
            return await asyncio.wait_for(waiter, timeout_s)
        except asyncio.TimeoutError:
            raise AgentError(
                f"{record.node_id} did not answer for {task_id} in {timeout_s:.0f}s"
            ) from None
        finally:
            self._serving.pop(command_id, None)

    def on_task_response(self, answer: agent_pb2.TaskResponse) -> None:
        waiter = self._serving.get(answer.command_id)
        if waiter is None or waiter.done():
            return
        if answer.error:
            waiter.set_exception(AgentError(answer.error))
            return
        waiter.set_result((answer.status, dict(answer.headers), answer.body))

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
    def on_task_state(self, state: agent_pb2.TaskState) -> None:
        record = self.tasks.get(state.task_id)
        if record is None:
            return
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
        if node.hardware is not None and report.vram_free_bytes:
            node.hardware.vram_free_bytes = report.vram_free_bytes
        for state in report.tasks:
            self.on_task_state(state)


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
        store = self.hub.releases
        if store is None or not self.hub.release_base_url:
            return None
        release = store.offer_to(node_id)
        if release is None:
            return None
        return agent_pb2.AgentRelease(
            version=release.version,
            url=f"{self.hub.release_base_url}/agent/release/{release.version}.tar.gz",
            sha256=release.sha256,
            signature=release.signature,
        )

    def _rendezvous_addrs(self) -> List[str]:
        node = self.hub.rendezvous
        return list(node.multiaddrs()) if node is not None else []

    def _relay_addrs(self) -> List[str]:
        from loom.orchestrator.rendezvous import relay_addrs

        return relay_addrs()


def add_agent_gateway_to_server(server, hub: AgentHub) -> AgentGatewayServicer:
    servicer = AgentGatewayServicer(hub=hub)
    agent_pb2_grpc.add_AgentGatewayServicer_to_server(servicer, server)
    return servicer
