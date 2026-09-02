"""A real gRPC orchestrator, small enough to assert against.

Real, not a mock. The properties this phase has to prove — that a task starts
only once its input is whole, that a result comes back over the same stream,
that a broken connection does not strand resources — only happen over an actual
connection.

It is also the reference for what the real orchestrator has to do: the order of
messages here is the order it will send them in.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent import futures
from typing import Dict, List, Optional

import grpc

from looma_agent.proto import agent_pb2, agent_pb2_grpc

CHUNK = 64 * 1024


class FakeOrchestrator(agent_pb2_grpc.AgentGatewayServicer):
    def __init__(self) -> None:
        self.registrations: List[agent_pb2.Register] = []
        self.acks: List[agent_pb2.Ack] = []
        self.telemetry: List[agent_pb2.Telemetry] = []
        self.logs: List[agent_pb2.TaskLogs] = []
        self.states: Dict[str, List[agent_pb2.TaskState]] = {}
        self.results: Dict[str, bytearray] = {}
        self.result_errors: Dict[str, str] = {}
        self._finished: Dict[str, threading.Event] = {}
        self._collected: Dict[str, threading.Event] = {}
        self._got_registration = threading.Event()
        self._lock = threading.Lock()
        self._to_send: List[agent_pb2.ServerMessage] = []
        self._drop_next_stream = False
        self._server = None
        self.port = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> str:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        agent_pb2_grpc.add_AgentGatewayServicer_to_server(self, self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        self._server.start()
        return f"127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop(0)

    # -------------------------------------------------------- what it can do
    def send(self, message: agent_pb2.ServerMessage) -> None:
        with self._lock:
            self._to_send.append(message)

    def run_task(self, task_id: str, command, *, inputs: Dict[str, bytes] = None,
                 environment=None, resources=None, timeout_s: int = 60,
                 command_id: str = "") -> None:
        """Send a task the way the orchestrator will: declare, then deliver."""
        inputs = inputs or {}
        declared = [
            agent_pb2.InputFile(name=name, size_bytes=len(data),
                                digest=hashlib.sha256(data).hexdigest())
            for name, data in inputs.items()
        ]
        self.send(agent_pb2.ServerMessage(run_task=agent_pb2.RunTask(
            command_id=command_id or f"cmd-{task_id}",
            task_id=task_id,
            command=list(command),
            timeout_s=timeout_s,
            inputs=declared,
            environment=agent_pb2.Environment(**(environment or {"kind": "none"})),
            resources=agent_pb2.Resources(**(resources or {})),
        )))
        for name, data in inputs.items():
            for offset in range(0, max(len(data), 1), CHUNK):
                piece = data[offset:offset + CHUNK]
                self.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
                    task_id=task_id, name=name, data=piece)))
            self.send(agent_pb2.ServerMessage(input_chunk=agent_pb2.InputChunk(
                task_id=task_id, name=name, last=True)))

    def collect(self, task_id: str, name: str) -> None:
        self.send(agent_pb2.ServerMessage(fetch_result=agent_pb2.FetchResult(
            command_id=f"get-{task_id}", task_id=task_id, name=name)))

    def drop_next_stream(self) -> None:
        self._drop_next_stream = True

    # ---------------------------------------------------------------- waiting
    def wait_registered(self, timeout: float = 10.0) -> bool:
        return self._got_registration.wait(timeout)

    def reset_registration_flag(self) -> None:
        self._got_registration.clear()

    def wait_finished(self, task_id: str, timeout: float = 120.0) -> Optional[agent_pb2.TaskState]:
        """Wait for a terminal state and return it, or None if it never came."""
        if not self._event(self._finished, task_id).wait(timeout):
            return None
        with self._lock:
            return self.states[task_id][-1]

    def wait_collected(self, task_id: str, name: str, timeout: float = 60.0) -> bool:
        return self._event(self._collected, f"{task_id}/{name}").wait(timeout)

    def state_names(self, task_id: str) -> List[str]:
        with self._lock:
            return [s.state for s in self.states.get(task_id, [])]

    def _event(self, where: Dict[str, threading.Event], key: str) -> threading.Event:
        with self._lock:
            return where.setdefault(key, threading.Event())

    # ------------------------------------------------------------------- rpc
    def Attach(self, request_iterator, context):
        threading.Thread(target=self._consume, args=(request_iterator,),
                         daemon=True).start()
        if not self._got_registration.wait(10.0):
            return
        if self._drop_next_stream:
            self._drop_next_stream = False
            context.abort(grpc.StatusCode.UNAVAILABLE, "orchestrator went away")
            return
        yield agent_pb2.ServerMessage(register_ack=agent_pb2.RegisterAck(
            ok=True, node_id=self.registrations[-1].node_id))
        while context.is_active():
            with self._lock:
                pending, self._to_send = self._to_send, []
            for message in pending:
                yield message
            if not pending:
                time.sleep(0.01)

    def _consume(self, request_iterator) -> None:
        try:
            for message in request_iterator:
                self._record(message)
        except grpc.RpcError:
            pass

    def _record(self, message: agent_pb2.AgentMessage) -> None:
        kind = message.WhichOneof("msg")
        if kind == "register":
            with self._lock:
                self.registrations.append(message.register)
            self._got_registration.set()
        elif kind == "ack":
            with self._lock:
                self.acks.append(message.ack)
        elif kind == "telemetry":
            with self._lock:
                self.telemetry.append(message.telemetry)
        elif kind == "logs":
            with self._lock:
                self.logs.append(message.logs)
        elif kind == "task_state":
            state = message.task_state
            with self._lock:
                self.states.setdefault(state.task_id, []).append(state)
            if state.state in ("done", "failed", "cancelled"):
                self._event(self._finished, state.task_id).set()
        elif kind == "result_chunk":
            chunk = message.result_chunk
            key = f"{chunk.task_id}/{chunk.name}"
            with self._lock:
                if chunk.error:
                    self.result_errors[key] = chunk.error
                if chunk.data:
                    self.results.setdefault(key, bytearray()).extend(chunk.data)
            if chunk.last:
                self._event(self._collected, key).set()
