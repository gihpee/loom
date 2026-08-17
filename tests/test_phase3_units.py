"""Phase 3 units: new adapters, command signing/replay, NVML watchdog, SLO boosts."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from loom_worker.backends.mlx import MlxBackend  # noqa: E402
from loom_worker.backends.sglang import SglangBackend  # noqa: E402
from loom_worker.proto import gateway_pb2 as w_gateway_pb2  # noqa: E402
from loom_worker.security import CommandVerifier  # noqa: E402
from loom_worker.watchdog import QuotaWatchdog  # noqa: E402

from conftest import make_model_info_kwargs  # noqa: E402

from loom.orchestrator.broker import PoolNode, ResourceBroker, pipeline_vram_bytes  # noqa: E402
from loom.orchestrator.gateway import new_meta  # noqa: E402
from loom.orchestrator.registry import ModelSpec  # noqa: E402
from loom.orchestrator.signing import sign_control_message  # noqa: E402
from loom.planning import ModelInfo  # noqa: E402
from loom.proto_gen import gateway_pb2, worker_control_pb2  # noqa: E402

GIB = 1024**3


# --------------------------------------------------------------- new adapters
def make_kwargs(**over):
    kwargs = dict(
        model_id="m",
        weights_uri="Qwen/Qwen3-0.6B",
        start_layer=0,
        end_layer=28,
        vram_quota_bytes=12 * GIB,
        port=8200,
    )
    kwargs.update(over)
    return kwargs


def test_sglang_command_and_fraction():
    b = SglangBackend(total_vram_bytes=24 * GIB, **make_kwargs())
    cmd = b.command()
    assert "sglang.launch_server" in cmd
    assert cmd[cmd.index("--model-path") + 1] == "Qwen/Qwen3-0.6B"
    assert float(cmd[cmd.index("--mem-fraction-static") + 1]) == pytest.approx(0.5, abs=0.001)
    with pytest.raises(NotImplementedError):
        SglangBackend(total_vram_bytes=0, **make_kwargs(start_layer=3)).prepare()


def test_mlx_command_uses_launcher_with_memory_limit():
    b = MlxBackend(**make_kwargs(vram_quota_bytes=5 * GIB))
    cmd = b.command()
    assert "loom_worker.backends.mlx_launcher" in cmd
    assert cmd[cmd.index("--memory-limit-bytes") + 1] == str(5 * GIB)
    assert b.health_path() == "/v1/models"
    with pytest.raises(NotImplementedError):
        MlxBackend(**make_kwargs(start_layer=1)).prepare()


# ------------------------------------------------------- signing / replay
def signed_worker_message(key="tok"):
    """Orchestrator-side signed message, reparsed as the worker-side class."""
    meta = new_meta()
    msg = gateway_pb2.ControlMessage(
        load_shard=worker_control_pb2.LoadShardRequest(
            model_id="m", start_layer=0, end_layer=28, backend_type="echo", meta=meta
        )
    )
    sign_control_message(msg, key)
    wire = msg.SerializeToString()
    worker_msg = w_gateway_pb2.ControlMessage()
    worker_msg.ParseFromString(wire)
    return worker_msg


def test_sign_verify_roundtrip_across_packages():
    verifier = CommandVerifier("tok")
    ok, err = verifier.verify(signed_worker_message())
    assert ok, err


def test_tampered_command_rejected():
    verifier = CommandVerifier("tok")
    msg = signed_worker_message()
    msg.load_shard.vram_quota_bytes = 999  # tamper after signing
    ok, err = verifier.verify(msg)
    assert not ok and err == "signature mismatch"


def test_wrong_key_and_unsigned_rejected():
    msg = signed_worker_message(key="tok")
    ok, err = CommandVerifier("other").verify(msg)
    assert not ok and err == "signature mismatch"
    unsigned = signed_worker_message(key="tok")
    unsigned.load_shard.meta.signature = b""
    ok, err = CommandVerifier("tok").verify(unsigned)
    assert not ok and err == "signature mismatch"


def test_replay_rejected():
    verifier = CommandVerifier("tok")
    msg = signed_worker_message()
    assert verifier.verify(msg)[0]
    ok, err = verifier.verify(msg)  # same command_id again
    assert not ok and err == "replay rejected"


def test_stale_command_rejected():
    verifier = CommandVerifier("tok", max_skew_ms=1000)
    meta = new_meta()
    meta.issued_at_unix_ms = int(time.time() * 1000) - 10_000
    msg = gateway_pb2.ControlMessage(
        start_serving=worker_control_pb2.ModelRequest(model_id="m", meta=meta)
    )
    sign_control_message(msg, "tok")
    worker_msg = w_gateway_pb2.ControlMessage()
    worker_msg.ParseFromString(msg.SerializeToString())
    ok, err = verifier.verify(worker_msg)
    assert not ok and err == "stale command"


# ------------------------------------------------------------- NVML watchdog
class FakeNvml:
    """Minimal NVML surface reporting fixed GPU usage for given pids."""

    def __init__(self, usage_by_pid):
        self.usage_by_pid = usage_by_pid

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return object()

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        class Info:
            def __init__(self, pid, used):
                self.pid = pid
                self.usedGpuMemory = used

        return [Info(pid, used) for pid, used in self.usage_by_pid.items()]


def test_watchdog_kills_on_gpu_quota_via_nvml():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    kills = []
    watchdog = QuotaWatchdog(
        get_pid=lambda: proc.pid if proc.poll() is None else None,
        quota_bytes=1 * GIB,
        on_kill=kills.append,
        device="cuda",
        poll_interval_s=0.1,
    )
    # Inject fake NVML: the process "uses" 2 GiB of VRAM > 1 GiB quota.
    watchdog._pynvml = FakeNvml({proc.pid: 2 * GIB})
    watchdog.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
        assert proc.poll() is not None, "process should have been killed via VRAM quota"
        assert kills and "vram quota" in kills[0]
    finally:
        watchdog.stop()
        if proc.poll() is None:
            proc.kill()


# ------------------------------------------------------------- SLO boosts
def test_broker_boost_flips_ordering():
    broker = ResourceBroker()
    mi = ModelInfo(**make_model_info_kwargs())
    need = pipeline_vram_bytes(mi)
    pool = [PoolNode("w0", "default", int(need * 1.2))]  # room for exactly one

    def spec(mid):
        return ModelSpec(
            model_id=mid, weights_uri="u", backend_type="echo", model_info=mi, priority=1
        )

    models = [spec("first"), spec("second")]
    # Equal scores: tie broken by model_id -> "first" wins the single slot.
    assert set(broker.plan(pool, models).allocations) == {"first"}
    # SLO boost on "second" flips the ordering.
    boosted = broker.plan(pool, models, score_boosts={"second": 2.0})
    assert set(boosted.allocations) == {"second"}
    assert boosted.unscheduled == ["first"]
