"""Assembly. Read this first — every connection in the agent is visible here.

    join key  ──→ identity        where to call, and the secret to verify with
    machine   ──→ hwinfo          what this node has to offer
                     │
              control/client      one outbound stream, held open
                     │
              control/handlers    what to do with a command
                     │
              tasks/              (phase 1: directories, environments, running)

Phase 0 stops after the stream: the node registers, reports itself, and refuses
work it cannot do yet. That is a complete, useful thing — it proves onboarding,
reconnection and the image size before anything harder is built on top.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

from loom_agent import __version__
from loom_agent.config import Config, parse_args
from loom_agent.control.client import ControlClient
from loom_agent.control.handlers import CommandHandlers
from loom_agent.hwinfo import detect_hardware
from loom_agent.identity import BadJoinKey, default_node_id, parse_join_key
from loom_agent.p2p.layer import PeerLayer
from loom_agent.tasks.env import EnvironmentCache
from loom_agent.tasks.env.cache import BUILDERS as ENVIRONMENT_KINDS
from loom_agent.tasks.limits import resolve_isolation
from loom_agent.tasks.registry import TaskRegistry
from loom_agent.update import Updater, mark_healthy
from loom_agent.control.tasks import TaskCommands
from loom_agent.proto import agent_pb2

logger = logging.getLogger("loom_agent")


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOOM_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def hardware_message() -> agent_pb2.Hardware:
    """What this machine has, detected — never declared by its owner.

    Logged before it is sent: when a node turns out not to fit a model, this
    line is the first thing worth looking at, and `detection_source` says which
    of NVML / torch / nvidia-smi / sysctl actually answered.
    """
    hw = detect_hardware()
    logger.info(
        "hardware: device=%s gpu=%s x%d vram_free=%.1fGB tflops=%.1f (%s)",
        hw.device, hw.gpu_name, hw.num_gpus,
        hw.vram_free_bytes / 1024**3, hw.tflops_fp16, hw.detection_source,
    )
    return agent_pb2.Hardware(
        num_gpus=hw.num_gpus,
        tflops_fp16=hw.tflops_fp16,
        gpu_name=hw.gpu_name,
        memory_gb=hw.memory_gb,
        memory_bandwidth_gbps=hw.memory_bandwidth_gbps,
        device=hw.device,
        vram_free_bytes=hw.vram_free_bytes,
        vram_total_bytes=hw.vram_total_bytes,
        host_ram_gb=hw.host_ram_gb,
        detection_source=hw.detection_source,
    )


class Agent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.key = parse_join_key(config.join_key)
        self.node_id = config.node_id or default_node_id()
        self.hardware = hardware_message()
        self._stop = threading.Event()
        self.isolation = resolve_isolation()
        self.tasks = TaskRegistry(
            root=config.tasks_dir,
            isolation=self.isolation,
            environments=EnvironmentCache(config.envs_dir),
            total_gpus=self.hardware.num_gpus,
        )
        # Inbound direct messages go to the task machinery, which is built
        # just below — hence the late binding rather than a direct reference.
        self.peers = PeerLayer(on_message=lambda raw: self.commands.on_peer_message(raw))
        self.client = ControlClient(
            address=self.key.address,
            register_message=self._register_message,
            on_message=lambda msg: self.handlers.handle(msg),
            on_registered=self._on_registered,
            reconnect_delay_s=config.reconnect_delay_s,
        )
        self.commands = TaskCommands(
            registry=self.tasks, send=self.client.send,
            node_id=self.node_id, links=self.peers.links,
        )
        self.updater = Updater(
            current_version=__version__,
            drain=self.tasks.drain,
            stop=self.stop,
        )
        self.handlers = CommandHandlers(tasks=self.commands, telemetry=self._telemetry)

    def _on_registered(self, ack: agent_pb2.RegisterAck) -> None:
        """The ack carries the one address this node needs to reach every other.

        Bootstrap peers are fixed when a libp2p node is built, so the p2p node
        cannot come up before now. It is started on the ack rather than on the
        first piece of work: joining the network takes seconds, and a
        deployment should not be the thing that waits for it.
        """
        self.peers.on_rendezvous(list(ack.rendezvous), list(ack.relays))
        # Only now: reaching the orchestrator is what "this payload works"
        # means, and the launcher's rollback reads exactly this.
        mark_healthy()
        self.updater.on_release(ack.release)

    def _register_message(self) -> agent_pb2.Register:
        message = agent_pb2.Register(
            node_id=self.node_id,
            join_key=self.key.raw,
            hardware=self.hardware,
            region=self.config.region,
            agent_version=__version__,
            readiness=self._readiness(),
        )
        # Absent on a first registration and present after a reconnect: by then
        # the node knows its own peer id and addresses, and the orchestrator can
        # hand them to its neighbours.
        identity = self.peers.identity_message()
        if identity is not None:
            message.peer.peer_id = identity.peer_id
            message.peer.listen_addrs.extend(identity.listen_addrs)
            message.peer.symmetric_nat = bool(identity.symmetric_nat)
        return message

    def _readiness(self) -> agent_pb2.Readiness:
        """What this node can actually do, said at registration.

        A node that cannot isolate a task declares it rather than accepting
        work and failing every time — the orchestrator can then place around it
        instead of discovering the problem one task at a time.
        """
        refusal = self.tasks.unusable
        if not refusal and not self.isolation.drops_privileges and \
                not self.isolation.unprivileged_fallback:
            refusal = "this node cannot run tasks as a separate user"
        return agent_pb2.Readiness(
            accepts_tasks=not refusal,
            refusal=refusal,
            environment_kinds=sorted(["none", *ENVIRONMENT_KINDS]),
        )

    def _telemetry(self) -> agent_pb2.AgentMessage:
        snapshot = self.tasks.snapshot()
        report = agent_pb2.Telemetry(
            node_id=self.node_id,
            vram_free_bytes=self.hardware.vram_free_bytes,
            reported_at_unix_ms=int(time.time() * 1000),
            gpus_total=snapshot["gpus_total"],
            gpus_free=snapshot["gpus_free"],
            tasks_running=snapshot["running"],
            env_cache_bytes=snapshot["environments"]["bytes"],
        )
        for task in self.tasks.list():
            status = task.status()
            report.tasks.add(
                task_id=status["task_id"], state=status["state"],
                exit_code=status["exit_code"] or 0, error=status["error"],
                devices=list(status["devices"]), seconds=status["seconds"],
            )
        return agent_pb2.AgentMessage(telemetry=report)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.config.heartbeat_interval_s):
            if self.client.registered:
                self.client.send(self._telemetry())

    def _report_readiness(self) -> None:
        """Say once, at startup, whether this node can actually take work.

        A node that refuses every task looks identical to a healthy idle one
        from the outside. Saying so here means the owner finds out from their
        own logs rather than from a support thread a week later.
        """
        if self.tasks.unusable:
            logger.warning("this node will refuse every task: %s", self.tasks.unusable)
            return
        if self.isolation.drops_privileges:
            logger.info("tasks will run as %s on %d GPU(s)",
                        self.isolation.user, self.hardware.num_gpus)
        elif self.isolation.unprivileged_fallback:
            logger.warning("tasks will run as this agent's own user: weaker isolation "
                           "was accepted explicitly on this node")
        else:
            logger.warning("this node will REFUSE every task until it can run them as "
                           "a separate user; it will still report itself and idle")

    def run(self) -> int:
        logger.info("agent %s: node %s -> %s", __version__, self.node_id, self.key.address)
        self._report_readiness()
        self.commands.start()
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        self.client.run_forever()
        return 0

    def stop(self) -> None:
        self._stop.set()
        # Tasks first: they are somebody's work, and stopping them politely
        # while the stream is still up means the orchestrator hears why.
        self.tasks.stop_all()
        self.commands.shutdown()
        self.client.stop()


def main(argv=None) -> int:
    _setup_logging()
    config = parse_args(argv)
    if not config.join_key:
        logger.error("no join key: pass --key loom_... (get one from the admin page)")
        return 2
    try:
        agent = Agent(config)
    except BadJoinKey as exc:
        logger.error("%s", exc)
        return 2

    # SIGTERM is how `docker stop` and the launcher ask us to finish. Answering
    # it means the stream closes cleanly instead of the orchestrator waiting
    # for a keepalive to time out.
    def _shutdown(signum, _frame):
        logger.info("shutting down on signal %s", signum)
        agent.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _shutdown)
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
