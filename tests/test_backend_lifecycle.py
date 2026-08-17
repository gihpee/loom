"""Regression tests for the engine-launch storm seen on a real GPU node.

A cold vLLM start takes minutes (checkpoint download + warm-up). During that
window the orchestrator kept re-issuing StartServing, and every retry spawned
ANOTHER `vllm serve` on the same port: the previous process was orphaned but
kept the whole VRAM quota, so each new attempt died with
"Free memory on device cuda:0 (7.77/23.6 GiB) ... less than desired".

The invariants pinned here:
  1. one backend process per shard, whatever the orchestrator repeats;
  2. a backend that never became healthy is stopped, not left holding VRAM;
  3. the orchestrator does not re-place a shard while its start is in flight,
     and backs off after a failure instead of retrying every pass.
"""

import sys
import time
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
sys.path.insert(0, str(WORKER_DIR))

from loom_worker.backends.echo import EchoBackend  # noqa: E402
from loom_worker.handlers import CommandHandlers  # noqa: E402
from loom_worker.proto import gateway_pb2 as w_gateway_pb2  # noqa: E402
from loom_worker.proto import worker_control_pb2 as w_control_pb2  # noqa: E402
from loom_worker.state import ShardStatus, WorkerState  # noqa: E402
from loom_worker.watchdog import QuotaWatchdog  # noqa: E402

GIB = 1024**3


def echo(**overrides):
    kwargs = dict(
        model_id="lifecycle",
        weights_uri="",
        start_layer=0,
        end_layer=1,
        vram_quota_bytes=4 * GIB,
    )
    kwargs.update(overrides)
    return EchoBackend(**kwargs)


# ------------------------------------------------------------------ backend
def test_second_start_does_not_spawn_a_second_process():
    backend = echo()
    backend.prepare()
    backend.start()
    try:
        assert backend.wait_healthy(timeout_s=15)
        first_pid = backend.pid()
        backend.start()  # orchestrator retry / reconnect
        assert backend.pid() == first_pid, "a second engine was launched on the same port"
    finally:
        backend.stop()
    assert backend.pid() is None


def test_start_after_stop_is_allowed():
    """The guard must not turn into a one-shot: a stopped shard can restart."""
    backend = echo()
    backend.prepare()
    backend.start()
    assert backend.wait_healthy(timeout_s=15)
    first_pid = backend.pid()
    backend.stop()
    backend.start()
    try:
        assert backend.wait_healthy(timeout_s=15)
        assert backend.pid() not in (None, first_pid)
    finally:
        backend.stop()


def test_wait_healthy_returns_when_the_process_dies():
    backend = echo(startup_delay_s=30)
    backend.prepare()
    backend.start()
    pid = backend.pid()
    assert pid is not None
    backend.stop()
    # No hanging for the full readiness timeout: a dead process is a verdict.
    started = time.time()
    assert backend.wait_healthy(timeout_s=30) is False
    assert time.time() - started < 5


# ------------------------------------------------------------------ handlers
class _Collector:
    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(msg)

    def acks(self):
        return [m.ack for m in self.messages if m.WhichOneof("msg") == "ack"]


def make_handlers(**kwargs):
    state = WorkerState(node_id="lifecycle-node", advertise_host="127.0.0.1")
    sent = _Collector()
    handlers = CommandHandlers(
        state,
        send=sent,
        backend_kwargs={"echo": kwargs.pop("echo_kwargs", {})},
        watchdog_poll_s=0.2,
        rss_overhead_bytes=8 * GIB,
        **kwargs,
    )
    return state, handlers, sent


def load(handlers, *, command_id="c1", model_id="lifecycle"):
    return handlers.load_shard(
        w_control_pb2.LoadShardRequest(
            model_id=model_id,
            start_layer=0,
            end_layer=1,
            backend_type="echo",
            weights_uri="",
            vram_quota_bytes=4 * GIB,
            meta=w_control_pb2.CommandMeta(command_id=command_id),
        )
    )


def start(handlers, *, command_id, model_id="lifecycle"):
    return handlers.start_serving(
        w_control_pb2.ModelRequest(
            model_id=model_id, meta=w_control_pb2.CommandMeta(command_id=command_id)
        )
    )


def test_repeated_start_serving_launches_one_backend():
    state, handlers, sent = make_handlers(echo_kwargs={"startup_delay_s": 3})
    assert load(handlers).ack.ok
    start(handlers, command_id="s1")
    # The shard is claimed immediately, before the engine is up.
    assert state.get("lifecycle").status == ShardStatus.STARTING
    backend = state.get("lifecycle").backend
    try:
        # Retries arriving mid-start: acked, but no second engine.
        for i in range(5):
            reply = start(handlers, command_id=f"s-retry-{i}")
            assert reply.ack.ok
        # Wait for the shard itself, not for the process: the serve thread
        # spawns it, so polling backend.pid() from here would race.
        deadline = time.time() + 30
        while time.time() < deadline and state.get("lifecycle").status != ShardStatus.SERVING:
            time.sleep(0.1)
        assert state.get("lifecycle").status == ShardStatus.SERVING
        pid = backend.pid()
        assert pid is not None
        # Still the one process the first StartServing launched.
        assert start(handlers, command_id="s-late").ack.ok
        assert backend.pid() == pid
    finally:
        backend.stop()


def test_load_shard_never_replaces_a_starting_backend():
    """LoadShard is idempotent while a start is in flight — no orphaned engine."""
    state, handlers, _ = make_handlers(echo_kwargs={"startup_delay_s": 3})
    assert load(handlers).ack.ok
    start(handlers, command_id="s1")
    backend = state.get("lifecycle").backend
    try:
        assert load(handlers, command_id="c2").ack.ok
        assert state.get("lifecycle").backend is backend
    finally:
        backend.stop()


def test_failed_start_releases_the_process():
    """A backend that never answers /health must not keep holding the GPU."""
    state, handlers, sent = make_handlers(echo_kwargs={"startup_delay_s": 60})
    assert load(handlers).ack.ok
    backend = state.get("lifecycle").backend
    # Health check gives up long before the (fake) engine is ready.
    original = backend.wait_healthy
    backend.wait_healthy = lambda *a, **k: original(timeout_s=2)
    start(handlers, command_id="s1")
    deadline = time.time() + 30
    while time.time() < deadline and state.get("lifecycle").status != ShardStatus.FAILED:
        time.sleep(0.1)
    shard = state.get("lifecycle")
    assert shard.status == ShardStatus.FAILED
    assert backend.pid() is None, "failed backend left a process holding VRAM"
    nacks = [a for a in sent.acks() if not a.ok]
    assert nacks and "health" in nacks[0].error


def test_telemetry_reports_starting_shards():
    state, handlers, _ = make_handlers(echo_kwargs={"startup_delay_s": 5})
    assert load(handlers).ack.ok
    start(handlers, command_id="s1")
    backend = state.get("lifecycle").backend
    try:
        report = handlers.telemetry_report().telemetry
        shard = report.shards[0]
        assert shard.status == "starting"
        assert shard.local_port == 0  # not routable yet
        assert shard.healthy is False
    finally:
        backend.stop()


# ------------------------------------------------------------------ watchdog
def test_vram_overhead_is_not_charged_to_the_quota():
    """The CUDA context sits on top of weights+KV; killing for it is a false positive."""
    kills = []
    watchdog = QuotaWatchdog(
        get_pid=lambda: 1,
        quota_bytes=10 * GIB,
        on_kill=kills.append,
        device="cuda",
        poll_interval_s=0.05,
        vram_overhead_bytes=GIB,
    )
    watchdog._measure = lambda pid: (10 * GIB + GIB // 2, "vram")  # quota + context
    watchdog._kill_tree = lambda pid: None  # nothing real to kill in this test
    watchdog.start()
    try:
        time.sleep(0.4)
        assert not kills
        watchdog._measure = lambda pid: (12 * GIB, "vram")  # genuinely over
        deadline = time.time() + 3
        while time.time() < deadline and not kills:
            time.sleep(0.05)
        assert kills and "vram" in kills[0]
    finally:
        watchdog.stop()


def test_a_serving_shard_whose_process_died_is_restarted_not_re_announced():
    """Idempotency is checked against the process, not the bookkeeping.

    A backend killed behind the agent's back (watchdog, OOM killer, crash)
    leaves the status saying SERVING. Answering "already serving" then hands
    the orchestrator an endpoint with nothing behind it.
    """
    state, handlers, sent = make_handlers()
    assert load(handlers).ack.ok
    start(handlers, command_id="s1")
    shard = state.get("lifecycle")
    backend = shard.backend
    try:
        deadline = time.time() + 30
        while time.time() < deadline and shard.status != ShardStatus.SERVING:
            time.sleep(0.1)
        assert shard.status == ShardStatus.SERVING
        first_pid = backend.pid()

        backend.stop()  # simulate the watchdog killing it
        assert backend.pid() is None

        # StartServing must relaunch instead of re-announcing the dead port.
        assert start(handlers, command_id="s2") is None, "no restart was attempted"
        deadline = time.time() + 30
        while time.time() < deadline and shard.status != ShardStatus.SERVING:
            time.sleep(0.1)
        assert shard.status == ShardStatus.SERVING
        assert backend.pid() not in (None, first_pid), "a fresh process should be serving"
    finally:
        backend.stop()
