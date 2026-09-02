"""Where a command from the orchestrator goes.

A dispatcher and nothing else: it decides which part of the agent a message
belongs to and does none of the work itself. Anything that could take time
happens on a thread the task machinery owns, because a stream that stops being
read makes this node look dead.
"""

from __future__ import annotations

import logging

from looma_agent.control.tasks import TaskCommands
from looma_agent.proto import agent_pb2

logger = logging.getLogger("looma_agent.handlers")


class CommandHandlers:
    def __init__(self, *, tasks: TaskCommands, telemetry, on_release=None) -> None:
        self.tasks = tasks
        self._telemetry = telemetry
        # Подсказка «посмотри на версию сейчас». Приходит, когда оператор
        # перевёл ступень, а этот узел никуда не переподключался.
        self.on_release = on_release or (lambda _release: None)

    def handle(self, message: agent_pb2.ServerMessage) -> None:
        kind = message.WhichOneof("msg")
        if kind == "input_chunk":
            self.tasks.input_chunk(message.input_chunk)
        elif kind == "run_task":
            self.tasks.run(message.run_task)
        elif kind == "stop_task":
            self.tasks.stop(message.stop_task)
        elif kind == "release_task":
            self.tasks.release(message.release_task)
        elif kind == "fetch_result":
            self.tasks.fetch_result(message.fetch_result)
        elif kind == "fetch_logs":
            self.tasks.fetch_logs(message.fetch_logs)
        elif kind == "task_message":
            self.tasks.on_task_message(message.task_message)
        elif kind == "task_request":
            self.tasks.task_request(message.task_request)
        elif kind == "tunnel_open":
            self.tasks.tunnel_open(message.tunnel_open)
        elif kind == "tunnel_chunk":
            self.tasks.tunnel_chunk(message.tunnel_chunk)
        elif kind == "release":
            self.on_release(message.release)
        elif kind == "probe":
            self.tasks.send(self._telemetry())
        else:
            # Loud, not silent: a node that quietly drops commands looks
            # healthy while doing nothing, which is the worst way to fail.
            logger.error("no handler for %r; this agent and the orchestrator "
                         "disagree about the protocol", kind)
