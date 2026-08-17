"""Quota watchdog: kills a backend process tree that exceeds its memory quota.

Enforcement source depends on the device:
- cuda: per-process GPU memory via NVML (nvidia-ml-py), summed over the
  backend's process tree. Falls back to RSS if NVML is unavailable.
- cpu / mlx: host RSS (MLX unified memory shows up as RSS, so this is the
  hard-kill safety net on top of the in-process mx.set_memory_limit).

The kill terminates ONLY the backend subprocess tree — the worker agent stays
up, keeps heartbeating, and reports the shard as failed so the orchestrator
can re-place it.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import psutil


def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


class QuotaWatchdog:
    def __init__(
        self,
        *,
        get_pid: Callable[[], Optional[int]],
        quota_bytes: int,
        on_kill: Callable[[str], None],
        device: str = "cpu",
        poll_interval_s: float = 2.0,
        rss_overhead_bytes: int = 0,
    ) -> None:
        self._get_pid = get_pid
        self._quota = quota_bytes
        self._on_kill = on_kill
        self._device = device
        # A VRAM quota says nothing about host RSS: the interpreter, torch and
        # CUDA runtime add hundreds of MB that are not the model's weights. In
        # RSS mode (cpu/mlx, or cuda without NVML) allow that fixed overhead on
        # top of the quota, otherwise every backend is killed on startup.
        self._rss_overhead = max(0, int(rss_overhead_bytes))
        self._poll = poll_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pynvml = _nvml() if device.startswith("cuda") else None

    def set_quota(self, quota_bytes: int) -> None:
        self._quota = quota_bytes

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="quota-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll * 2)

    def _tree_pids(self, pid: int) -> List[int]:
        try:
            proc = psutil.Process(pid)
            return [pid] + [c.pid for c in proc.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []

    def _tree_rss(self, pids: List[int]) -> int:
        total = 0
        for p in pids:
            try:
                total += psutil.Process(p).memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total

    def _tree_gpu_bytes(self, pids: List[int]) -> Optional[int]:
        """Sum GPU memory of the process tree across all NVML devices."""
        if self._pynvml is None:
            return None
        nv = self._pynvml
        pid_set = set(pids)
        total = 0
        try:
            for i in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(i)
                for info in nv.nvmlDeviceGetComputeRunningProcesses(handle):
                    if info.pid in pid_set and info.usedGpuMemory is not None:
                        total += int(info.usedGpuMemory)
            return total
        except Exception:
            return None

    def _measure(self, pid: int) -> tuple[int, str]:
        pids = self._tree_pids(pid)
        if not pids:
            return 0, "none"
        gpu = self._tree_gpu_bytes(pids)
        if gpu is not None:
            return gpu, "vram"
        return self._tree_rss(pids), "rss"

    def _kill_tree(self, pid: int) -> None:
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.kill()
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            pid = self._get_pid()
            if pid is None or self._quota <= 0:
                continue
            used, kind = self._measure(pid)
            limit = self._quota if kind == "vram" else self._quota + self._rss_overhead
            if used > limit:
                self._kill_tree(pid)
                self._on_kill(
                    f"backend pid={pid} exceeded {kind} quota: used={used} > {limit}"
                )
