"""Turning task commands from the stream into work on this machine.

The shape of this file is set by one constraint: **nothing slow may happen on
the stream thread**. Provisioning an environment can take half an hour and
waiting for a client's dataset can take longer, and a node that stops reading
its control stream while either happens looks dead to the orchestrator and
stops answering everything else.

So the stream thread only ever files things away, and a thread per task does
the waiting.

The other constraint is that a task must not start before its input is all
here. A task running against half a dataset does not fail, it produces a wrong
answer, which is worse.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
from typing import Callable, Dict, List, Optional

from loom_agent.proto import agent_pb2
from loom_agent.tasks import channel as channel_mod
from loom_agent.tasks.groups import GroupTable, group_from_proto
from loom_agent.tasks.registry import TaskRegistry
from loom_agent.tasks.runner import Task
from loom_agent.tasks.spec import EnvSpec, Resources, TaskRefused, TaskSpec
from loom_agent.transport.files import Inbox, IncomingFile, TransferRefused, safe_target

logger = logging.getLogger("loom_agent.control.tasks")

# How long a task waits for input that stopped arriving before giving the node
# its resources back. Generous: a large dataset over a home connection is slow,
# and killing it would waste what already arrived.
INPUT_IDLE_TIMEOUT_S = 600.0

_CLOSED = object()


class TaskCommands:
    def __init__(self, *, registry: TaskRegistry, send: Callable[[agent_pb2.AgentMessage], None],
                 node_id: str = "", links=None) -> None:
        self.registry = registry
        self.send = send
        self.node_id = node_id
        # The direct path to peers, when this node has one. None means every
        # message between nodes goes through the orchestrator — correct, and
        # two wide-area crossings instead of one.
        self.links = links
        self.groups = GroupTable()
        self.channel = channel_mod.TaskChannel(on_send=self._from_task,
                                              on_ready=self._task_ready)
        self.registry.channel_url = self.channel.url if self.channel.port else ""
        self._inputs: Dict[str, "queue.Queue"] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        """Open the loopback endpoint tasks send through."""
        self.channel.start()
        self.registry.channel_url = self.channel.url

    def shutdown(self) -> None:
        """Close the loopback endpoint. Not `stop` — that is the StopTask handler."""
        self.channel.stop()

    # ------------------------------------------------------- from the stream
    def run(self, command: agent_pb2.RunTask) -> None:
        try:
            spec = _spec_of(command)
        except TaskRefused as exc:
            # Both, not just the ack: whoever is following this task by its
            # state would otherwise watch it sit in "pending" forever, because
            # a task refused before it existed never reports anything else.
            self._ack(command.command_id, False, str(exc))
            self._report(command.task_id, "failed", error=str(exc))
            return
        group = group_from_proto(command.group)
        if group is not None:
            self.groups.join(spec.task_id, group)
        expected = list(command.inputs)
        if expected:
            with self._lock:
                self._inputs[spec.task_id] = queue.Queue()
        self._ack(command.command_id, True, "")
        self._report(spec.task_id, "provisioning")
        threading.Thread(
            target=self._carry_out, args=(spec, expected, group),
            name=f"submit-{spec.task_id}", daemon=True,
        ).start()

    def input_chunk(self, chunk: agent_pb2.InputChunk) -> None:
        """Hand a piece of input to the thread waiting for it.

        Never blocks and never validates: the waiting thread owns the file and
        is the only place that can refuse it coherently.
        """
        with self._lock:
            pending = self._inputs.get(chunk.task_id)
        if pending is None:
            logger.warning("input for %s arrived with no task waiting for it",
                           chunk.task_id)
            return
        pending.put(chunk)

    def stop(self, command: agent_pb2.StopTask) -> None:
        try:
            self.registry.stop(command.task_id, reason=command.reason or "cancelled")
            self._ack(command.command_id, True, "")
        except TaskRefused as exc:
            self._ack(command.command_id, False, str(exc))

    def release(self, command: agent_pb2.ReleaseTask) -> None:
        self.groups.leave(command.task_id)
        self._close_input(command.task_id)
        self.registry.release(command.task_id)
        self._ack(command.command_id, True, "")
        self.send(agent_pb2.AgentMessage(
            task_state=agent_pb2.TaskState(task_id=command.task_id, state="gone")))

    def fetch_result(self, command: agent_pb2.FetchResult) -> None:
        """Stream one result file back, on its own thread.

        A multi-gigabyte checkpoint must not be read on the stream thread, and
        it must not be assembled in memory either.
        """
        threading.Thread(target=self._send_result, args=(command,),
                         name=f"result-{command.task_id}", daemon=True).start()

    def fetch_logs(self, command: agent_pb2.FetchLogs) -> None:
        task = self.registry.get(command.task_id)
        text = task.logs(tail=command.tail_lines) if task else ""
        self.send(agent_pb2.AgentMessage(logs=agent_pb2.TaskLogs(
            command_id=command.command_id, task_id=command.task_id, text=text)))

    # -------------------------------------------------------- on own threads
    def _carry_out(self, spec: TaskSpec, expected: List[agent_pb2.InputFile],
                   group=None) -> None:
        deliver = self._deliver(spec.task_id, expected) if expected else None
        try:
            task = self.registry.submit(spec, deliver_input=deliver, group=group)
        except TaskRefused as exc:
            logger.warning("refusing task %s: %s", spec.task_id, exc)
            self._report(spec.task_id, "failed", error=str(exc))
            return
        except Exception as exc:
            logger.exception("task %s could not be started", spec.task_id)
            self._report(spec.task_id, "failed", error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._close_input(spec.task_id)
        self.send(agent_pb2.AgentMessage(task_state=_state_of(task)))
        task.wait()
        # The finished state carries the manifest, so whoever asked knows what
        # there is to collect without having to ask again.
        self.send(agent_pb2.AgentMessage(task_state=_state_of(task, with_results=True)))
        self.groups.leave(spec.task_id)

    def _deliver(self, task_id: str, expected: List[agent_pb2.InputFile]):
        def deliver(inbox: Inbox) -> None:
            _receive_all(inbox, self._queue_for(task_id), expected)

        return deliver

    def _send_result(self, command: agent_pb2.FetchResult) -> None:
        task = self.registry.get(command.task_id)
        if task is None:
            self.send(agent_pb2.AgentMessage(result_chunk=agent_pb2.ResultChunk(
                task_id=command.task_id, name=command.name, last=True,
                error=f"no task {command.task_id!r} on this node")))
            return
        try:
            for piece in task.read_result(command.name):
                self.send(agent_pb2.AgentMessage(result_chunk=agent_pb2.ResultChunk(
                    task_id=command.task_id, name=command.name, data=piece)))
            self.send(agent_pb2.AgentMessage(result_chunk=agent_pb2.ResultChunk(
                task_id=command.task_id, name=command.name, last=True)))
        except (TransferRefused, OSError) as exc:
            self.send(agent_pb2.AgentMessage(result_chunk=agent_pb2.ResultChunk(
                task_id=command.task_id, name=command.name, last=True, error=str(exc))))


    # ------------------------------------------- messages between tasks
    def _from_task(self, task_id: str, to_rank: int, payload: bytes,
                   content_type: str) -> None:
        """A task wants to reach another member of its job.

        Three destinations and the task knows about none of them: the same machine,
        a peer we have a direct link to, or the orchestrator. Choosing is the whole
        reason the task sends to a RANK rather than to an address.
        """
        group = self.groups.of(task_id)
        if group is None:
            raise TaskRefused(f"task {task_id} is not part of a group and has nobody to write to")
        if to_rank < 0:
            # Every other member. A finished request tells the whole pipeline
            # to drop its cache for it, and doing that with one call per rank
            # from the task would put the loop in the wrong place.
            for rank in group.members:
                if rank != group.rank:
                    self._send_to(group, rank, payload, content_type)
            return
        member = group.member(to_rank)
        if member is None:
            raise TaskRefused(
                f"there is no rank {to_rank} in {group.group_id} (it has {group.size} members)"
            )
        self._send_to(group, to_rank, payload, content_type)

    def _send_to(self, group, to_rank: int, payload: bytes, content_type: str) -> None:
        """One hop, by whichever path is shorter."""
        local = self.groups.local_task(group.group_id, to_rank)
        if local is not None:
            # Same machine. No network at all, which is the fastest path there
            # is and the one a two-stage model on one node should always take.
            self._deliver_local(local, payload, content_type, group.rank)
            return
        member = group.member(to_rank)
        if member is None:
            return
        message = agent_pb2.TaskMessage(
            group_id=group.group_id, from_rank=group.rank, to_rank=to_rank,
            payload=payload, content_type=content_type,
        )
        if self.links is not None and member.peer_id:
            self.links.send(
                group.group_id, to_rank,
                {"group_id": group.group_id, "from_rank": group.rank, "to_rank": to_rank,
                 "payload": base64.b64encode(payload).decode(), "content_type": content_type},
                lambda _m: self.send(agent_pb2.AgentMessage(task_message=message)),
            )
            return
        self.send(agent_pb2.AgentMessage(task_message=message))

    def _task_ready(self, task_id: str, port: int) -> None:
        task = self.registry.get(task_id)
        if task is None:
            return
        if port and port != task.serve_port:
            logger.info("task %s serves on %d, not the %d it was offered",
                        task_id, port, task.serve_port)
        task.serve_port = port

    def _deliver_local(self, task_id: str, payload: bytes, content_type: str,
                       from_rank: int) -> None:
        task = self.registry.get(task_id)
        if task is None or not task.serve_port:
            logger.warning("nothing here is listening for %s", task_id)
            return
        channel_mod.deliver(task.serve_port, payload,
                            content_type=content_type, from_rank=from_rank)

    def on_task_message(self, message: agent_pb2.TaskMessage) -> None:
        """A message that came the long way round, through the orchestrator."""
        local = self.groups.local_task(message.group_id, message.to_rank)
        if local is None:
            logger.warning("a message for rank %d of %s arrived here, and it is not here",
                           message.to_rank, message.group_id)
            return
        self._deliver_local(local, message.payload, message.content_type, message.from_rank)

    def on_peer_message(self, raw: dict) -> None:
        """A message that came straight from a peer."""
        try:
            payload = base64.b64decode(raw.get("payload") or "")
            self.on_task_message(agent_pb2.TaskMessage(
                group_id=raw.get("group_id", ""),
                from_rank=int(raw.get("from_rank") or 0),
                to_rank=int(raw.get("to_rank") or 0),
                payload=payload,
                content_type=raw.get("content_type", "application/octet-stream"),
            ))
        except Exception:
            logger.exception("a direct message could not be delivered")

    def task_request(self, command: agent_pb2.TaskRequest) -> None:
        """An HTTP request for whatever a task is serving.

        On its own thread: a generation can take a minute, and the stream must keep
        being read while it does or this node looks dead.
        """
        threading.Thread(target=self._answer_request, args=(command,),
                         name=f"serve-{command.task_id}", daemon=True).start()

    def _answer_request(self, command: agent_pb2.TaskRequest) -> None:
        task = self.registry.get(command.task_id)
        if task is None or not task.serve_port:
            self._respond(command.command_id, status=503, last=True,
                          error=f"task {command.task_id!r} is not serving on this node")
            return
        pieces = channel_mod.request_stream(
            task.serve_port, method=command.method, path=command.path,
            body=command.body, headers=dict(command.headers),
        )
        try:
            status, headers = next(pieces)
        except StopIteration:
            self._respond(command.command_id, status=502, last=True,
                          error="the task closed the connection without answering")
            return
        except Exception as exc:
            self._respond(command.command_id, status=502, last=True, error=str(exc))
            return
        clean = {k: v for k, v in headers.items() if k.lower() != "transfer-encoding"}
        try:
            previous = None
            for chunk in pieces:
                # Одна часть придерживается: последнюю надо пометить, а узнать,
                # что она последняя, можно только не увидев следующей.
                if previous is not None:
                    self._respond(command.command_id, status=status, headers=clean,
                                  body=previous)
                    clean = {}
                previous = chunk
            self._respond(command.command_id, status=status, headers=clean,
                          body=previous or b"", last=True)
        except Exception as exc:
            self._respond(command.command_id, status=status, last=True, error=str(exc))

    def _respond(self, command_id: str, *, status: int, body: bytes = b"",
                 headers: Optional[Dict[str, str]] = None, error: str = "",
                 last: bool = False) -> None:
        self.send(agent_pb2.AgentMessage(task_response=agent_pb2.TaskResponse(
            command_id=command_id, status=status, body=body,
            headers=headers or {}, error=error, last=last)))


    # ----------------------------------------------------------------- small
    def _queue_for(self, task_id: str) -> "queue.Queue":
        with self._lock:
            return self._inputs.setdefault(task_id, queue.Queue())

    def _close_input(self, task_id: str) -> None:
        with self._lock:
            pending = self._inputs.pop(task_id, None)
        if pending is not None:
            pending.put(_CLOSED)

    def _ack(self, command_id: str, ok: bool, error: str) -> None:
        if command_id:
            self.send(agent_pb2.AgentMessage(
                ack=agent_pb2.Ack(command_id=command_id, ok=ok, error=error)))

    def _report(self, task_id: str, state: str, *, error: str = "") -> None:
        self.send(agent_pb2.AgentMessage(task_state=agent_pb2.TaskState(
            task_id=task_id, state=state, error=error)))


# ------------------------------------------------------------------ receiving
def _receive_all(inbox: Inbox, pending: "queue.Queue",
                 expected: List[agent_pb2.InputFile]) -> None:
    """Collect every declared input, or refuse the lot.

    Files may interleave: a sender is free to alternate between them, so an
    open handle is kept per name rather than assuming one finishes before the
    next begins.
    """
    wanted = {f.name: f for f in expected}
    for name in wanted:
        safe_target(inbox.work, name)  # refuse a hostile name before writing anything
    open_files: Dict[str, IncomingFile] = {}
    done: set = set()
    try:
        while len(done) < len(wanted):
            try:
                item = pending.get(timeout=INPUT_IDLE_TIMEOUT_S)
            except queue.Empty:
                raise TransferRefused(
                    f"input stopped arriving after {INPUT_IDLE_TIMEOUT_S:.0f}s; "
                    f"{len(done)} of {len(wanted)} files were complete"
                ) from None
            if item is _CLOSED:
                raise TransferRefused("the connection closed while input was arriving")
            declared = wanted.get(item.name)
            if declared is None:
                raise TransferRefused(
                    f"{item.name!r} was sent but never declared; the agent only "
                    "writes what the task said it would need"
                )
            handle = open_files.get(item.name)
            if handle is None:
                handle = IncomingFile(
                    safe_target(inbox.work, item.name),
                    expected_bytes=declared.size_bytes,
                    digest=declared.digest,
                )
                handle.__enter__()
                open_files[item.name] = handle
            if item.data:
                handle.write(item.data)
            if item.last:
                written = handle.finish()
                inbox._hand_over(written)
                open_files.pop(item.name, None)
                done.add(item.name)
    except Exception:
        for handle in open_files.values():
            handle.abort()
        raise


# -------------------------------------------------------------- translation
def _spec_of(command: agent_pb2.RunTask) -> TaskSpec:
    return TaskSpec.from_dict({
        "task_id": command.task_id,
        "command": list(command.command),
        "env": dict(command.env),
        "timeout_s": command.timeout_s,
        "serve_port": command.serve_port,
        "resources": {
            "gpus": command.resources.gpus,
            "ram_bytes": command.resources.ram_bytes,
            "cpus": command.resources.cpus,
            "disk_bytes": command.resources.disk_bytes,
        },
        "environment": {
            "kind": command.environment.kind or "none",
            "requirements": list(command.environment.requirements),
            "source": command.environment.source,
        },
    })


def _state_of(task: Task, *, with_results: bool = False) -> agent_pb2.TaskState:
    status = task.status()
    state = agent_pb2.TaskState(
        task_id=status["task_id"],
        state=status["state"],
        exit_code=status["exit_code"] or 0,
        error=status["error"],
        devices=list(status["devices"]),
        seconds=status["seconds"],
    )
    if with_results:
        try:
            for result in task.results():
                state.results.add(name=result.name, size_bytes=result.size_bytes,
                                  digest=result.digest)
        except TransferRefused as exc:
            state.error = state.error or str(exc)
    return state
