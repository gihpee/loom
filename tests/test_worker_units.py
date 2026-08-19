"""Worker unit tests: vLLM adapter command construction, quota watchdog."""

import sys
import time
from pathlib import Path

import pytest

WORKER_DIR = Path(__file__).resolve().parent.parent / "worker"
sys.path.insert(0, str(WORKER_DIR))

from loom_worker.backends.echo import EchoBackend  # noqa: E402
from loom_worker.backends.vllm import MAX_GPU_UTILISATION as MAX_UTIL  # noqa: E402
from loom_worker.backends.vllm import VllmBackend  # noqa: E402
from loom_worker.watchdog import QuotaWatchdog  # noqa: E402

GIB = 1024**3


def make_vllm(**overrides):
    kwargs = dict(
        model_id="qwen3-0.6b",
        weights_uri="Qwen/Qwen3-0.6B",
        start_layer=0,
        end_layer=28,
        vram_quota_bytes=12 * GIB,
        total_vram_bytes=24 * GIB,
        port=8100,
    )
    kwargs.update(overrides)
    return VllmBackend(**kwargs)


def test_vllm_command_and_quota_fraction():
    backend = make_vllm()
    cmd = backend.command()
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-0.6B"]
    assert "--served-model-name" in cmd and "qwen3-0.6b" in cmd
    util = float(cmd[cmd.index("--gpu-memory-utilization") + 1])
    assert util == pytest.approx(0.5, abs=0.001)  # 12/24 GiB


def test_vllm_quota_fraction_clamped():
    assert make_vllm(vram_quota_bytes=1).gpu_memory_utilization() == 0.05
    # Never the whole card: vLLM refuses to start unless that share is free at
    # startup, and the driver/another tenant always holds a few hundred MB.
    assert make_vllm(vram_quota_bytes=100 * GIB).gpu_memory_utilization() == MAX_UTIL
    assert MAX_UTIL < 0.95
    # Unknown total VRAM -> vLLM default share, still capped.
    assert make_vllm(total_vram_bytes=0).gpu_memory_utilization() == min(0.9, MAX_UTIL)


def test_vllm_rejects_partial_shard():
    backend = make_vllm(start_layer=4, end_layer=28)
    with pytest.raises(NotImplementedError):
        backend.prepare()


def test_watchdog_kills_over_quota_backend():
    backend = EchoBackend(
        model_id="wd-test",
        weights_uri="",
        start_layer=0,
        end_layer=1,
        vram_quota_bytes=1,  # 1 byte: any real process exceeds it
    )
    backend.prepare()
    backend.start()
    assert backend.wait_healthy(timeout_s=15)
    kills = []
    watchdog = QuotaWatchdog(
        get_pid=backend.pid, quota_bytes=1, on_kill=kills.append, poll_interval_s=0.2
    )
    watchdog.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and backend.pid() is not None:
            time.sleep(0.1)
        assert backend.pid() is None, "process should have been killed"
        assert kills and "quota" in kills[0]
    finally:
        watchdog.stop()
        backend.stop()


def test_watchdog_leaves_within_quota_backend_alone():
    backend = EchoBackend(
        model_id="wd-ok",
        weights_uri="",
        start_layer=0,
        end_layer=1,
        vram_quota_bytes=4 * GIB,
    )
    backend.prepare()
    backend.start()
    assert backend.wait_healthy(timeout_s=15)
    kills = []
    watchdog = QuotaWatchdog(
        get_pid=backend.pid, quota_bytes=4 * GIB, on_kill=kills.append, poll_interval_s=0.2
    )
    watchdog.start()
    try:
        time.sleep(1.0)
        assert backend.pid() is not None
        assert not kills
    finally:
        watchdog.stop()
        backend.stop()


# ------------------------------------------------- what a node knows about itself
def test_stage_reports_its_measured_speed_only_once_it_means_something():
    """Per-layer ms is what the scheduler splits layers by.

    A warm-up sample is worse than no sample: it would tell the planner this
    node is slow and move layers away permanently. So the stage stays silent
    until it has seen enough steps.
    """
    from loom_worker.shard.server import StageSpeed

    speed = StageSpeed()
    assert speed.snapshot() is None
    for _ in range(4):
        speed.record(compute_ms=40.0, num_layers=20)
    assert speed.snapshot() is None, "four samples is still warm-up"

    for _ in range(20):
        speed.record(compute_ms=40.0, num_layers=20)
    assert speed.snapshot() == pytest.approx(2.0, rel=0.05)


def test_a_single_slow_step_does_not_redefine_the_node():
    """One 300 ms hiccup must not cost this node its layers."""
    from loom_worker.shard.server import StageSpeed

    speed = StageSpeed()
    for _ in range(40):
        speed.record(compute_ms=40.0, num_layers=20)
    steady = speed.snapshot()
    speed.record(compute_ms=6000.0, num_layers=20)
    assert speed.snapshot() < steady * 2


def test_unknown_gpus_do_not_all_look_alike_forever():
    """The spec table is a starting guess, and it says so in the logs."""
    from loom_worker.hwinfo import match_gpu_specs

    # The card the stand actually runs, previously unknown to the table.
    assert match_gpu_specs("NVIDIA A30", 24.0)["tflops_fp16"] == 165.0
    # Longest match wins: "a40" is a substring of "RTX A4000".
    assert match_gpu_specs("NVIDIA RTX A4000", 16.0)["tflops_fp16"] == 76.0
    assert match_gpu_specs("NVIDIA A40", 48.0)["tflops_fp16"] == 149.0
