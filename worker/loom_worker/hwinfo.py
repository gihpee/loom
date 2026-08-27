# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/parallax/server/server_info.py — HardwareInfo.detect(),
# NvidiaHardwareInfo._GPU_DB/_match_gpu_specs(), AppleSiliconHardwareInfo._APPLE_PEAK_FP16,
# detect_node_hardware().
# Изменения:
#   1. Убрана обязательная зависимость от torch/mlx: основной источник данных для
#      NVIDIA — NVML (nvidia-ml-py), затем torch, затем парсинг nvidia-smi.
#      Причина: воркер должен определять железо и в slim-образе без torch.
#   2. Неизвестная карта не выбрасывает исключение (как для Apple в оригинале),
#      а получает консервативную оценку — недоверенный узел не должен падать
#      из-за незнакомого GPU.
#   3. Добавлены vram_total/vram_free (нужны Resource Broker'у для квот) и
#      detection_source (наблюдаемость: чем именно определили).
#   4. Таблица _GPU_DB расширена актуальными датацентр/консьюмер картами.
"""Automatic hardware detection on the worker.

The node owner declares nothing: everything below is measured locally and sent
to the orchestrator at registration. Environment variables are honored only as
explicit overrides for testing (LOOM_DEVICE, LOOM_MEMORY_GB, ...).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger("loom_worker.hwinfo")

GIB = 1024**3

# Best-effort device database; key = lowercase substring of the device name.
_GPU_DB: Dict[str, Dict[str, float]] = {
    "h200": {"tflops_fp16": 989.0, "bandwidth_gbps": 4800.0},
    "h100": {"tflops_fp16": 989.0, "bandwidth_gbps": 3350.0},
    "h800": {"tflops_fp16": 989.0, "bandwidth_gbps": 3350.0},
    "a100-80g": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
    "a100 80": {"tflops_fp16": 312.0, "bandwidth_gbps": 2039.0},
    "a100-40g": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
    "a100 40": {"tflops_fp16": 312.0, "bandwidth_gbps": 1935.0},
    "l40s": {"tflops_fp16": 362.0, "bandwidth_gbps": 864.0},
    "l40": {"tflops_fp16": 181.0, "bandwidth_gbps": 864.0},
    "l4": {"tflops_fp16": 121.0, "bandwidth_gbps": 300.0},
    "a10g": {"tflops_fp16": 125.0, "bandwidth_gbps": 600.0},
    "a10": {"tflops_fp16": 125.0, "bandwidth_gbps": 600.0},
    "a30": {"tflops_fp16": 165.0, "bandwidth_gbps": 933.0},
    "a16": {"tflops_fp16": 35.9, "bandwidth_gbps": 200.0},
    "a2": {"tflops_fp16": 18.0, "bandwidth_gbps": 200.0},
    "rtx a6000": {"tflops_fp16": 155.0, "bandwidth_gbps": 768.0},
    "rtx a5000": {"tflops_fp16": 111.0, "bandwidth_gbps": 768.0},
    "rtx a4500": {"tflops_fp16": 94.0, "bandwidth_gbps": 640.0},
    "rtx a4000": {"tflops_fp16": 76.0, "bandwidth_gbps": 448.0},
    "a6000": {"tflops_fp16": 155.0, "bandwidth_gbps": 768.0},
    "a40": {"tflops_fp16": 149.0, "bandwidth_gbps": 696.0},
    "v100": {"tflops_fp16": 112.0, "bandwidth_gbps": 900.0},
    "t4": {"tflops_fp16": 65.0, "bandwidth_gbps": 320.0},
    "rtx 6000 ada": {"tflops_fp16": 182.0, "bandwidth_gbps": 960.0},
    "rtx 5090": {"tflops_fp16": 104.8, "bandwidth_gbps": 1792.0},
    "rtx 4090": {"tflops_fp16": 82.6, "bandwidth_gbps": 1008.0},
    "rtx 4080": {"tflops_fp16": 48.7, "bandwidth_gbps": 717.0},
    "rtx 3090": {"tflops_fp16": 35.6, "bandwidth_gbps": 936.0},
    "rtx 3080": {"tflops_fp16": 29.8, "bandwidth_gbps": 760.0},
}

_APPLE_PEAK_FP16: Dict[str, float] = {
    "M1": 4.58, "M1 Pro": 10.6, "M1 Max": 21.2,
    "M2": 7.1, "M2 Pro": 11.36, "M2 Max": 26.98, "M2 Ultra": 53.96,
    "M3": 7.1, "M3 Pro": 9.94, "M3 Max": 28.4, "M3 Ultra": 57.34,
    "M4": 8.52, "M4 Pro": 17.04, "M4 Max": 34.08,
    "M5": 9.37, "M5 Pro": 18.74, "M5 Max": 37.49, "M5 Ultra": 74.98,
}

_FALLBACK_GPU = {"tflops_fp16": 50.0, "bandwidth_gbps": 600.0}


@dataclass
class DetectedHardware:
    device: str  # "cuda" | "mlx" | "cpu"
    num_gpus: int
    gpu_name: str
    tflops_fp16: float
    memory_gb: float  # schedulable device memory (VRAM for cuda)
    memory_bandwidth_gbps: float
    vram_total_bytes: int
    vram_free_bytes: int
    host_ram_gb: float
    detection_source: str


def match_gpu_specs(name: str, vram_gb: float) -> Dict[str, float]:
    """Map a device name to peak FP16 TFLOPs and memory bandwidth.

    Longest key first, because the keys are substrings and short ones swallow
    long ones: "a40" sits inside "RTX A4000", which made a 76 TFLOPS card
    report itself as a 149 TFLOPS one — and the scheduler then handed it twice
    the layers it could keep up with.

    An unmatched card gets `_FALLBACK_GPU`, which is a guess and looks the same
    for every unknown device. Two different unknown cards therefore appear
    identical to the planner; the measured ms-per-layer that stages report back
    is what eventually corrects that (see loom/planning/layer_allocation.py).
    """
    key = name.lower()
    if "a100" in key and ("80" in key or vram_gb >= 60):
        return _GPU_DB["a100-80g"]
    if "a100" in key:
        return _GPU_DB["a100-40g"]
    for sub in sorted(_GPU_DB, key=len, reverse=True):
        if sub in key:
            return _GPU_DB[sub]
    logger.warning(
        "unknown GPU %r: scheduling with generic %.0f TFLOPS / %.0f GB/s until "
        "this node reports its measured speed",
        name,
        _FALLBACK_GPU["tflops_fp16"],
        _FALLBACK_GPU["bandwidth_gbps"],
    )
    return dict(_FALLBACK_GPU)


def _host_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / GIB, 1)
    except Exception:
        return 0.0


def gpu_fingerprint() -> str:
    """A short, stable id for the SET OF CARDS this process can see.

    Two workers on one machine, each given different cards, are two nodes —
    but with `--network host` they share a hostname, so the orchestrator saw
    one node registering twice a second, each registration evicting the other.
    The node blinked and served nothing.

    Card UUIDs rather than indices: Docker renumbers what it passes through,
    so the card handed over as `device=1` is index 0 inside the container.
    UUIDs survive that, and survive restarts, so the id stays the same node.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        uuids = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid.decode() if isinstance(uuid, bytes) else str(uuid))
    except Exception:
        uuids = []
    if not uuids:
        # No NVML: fall back to whatever the operator asked for. Worse than a
        # UUID (it moves if the request changes) but still distinguishes two
        # containers on one host, which is the whole point.
        uuids = [_visible_devices()]
    import hashlib

    return hashlib.sha256("|".join(sorted(uuids)).encode()).hexdigest()[:6]


def _visible_devices() -> str:
    """What this container was told it may use, if anything."""
    for name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def sees_only_some_gpus() -> bool:
    """Was this process handed a subset of the machine's cards?

    Only then does the node id need a suffix. A worker with the whole machine
    keeps the plain hostname it has always had, so existing nodes do not turn
    into new ones on upgrade.
    """
    value = _visible_devices()
    return bool(value) and value.lower() not in ("all", "void", "none")


# --- NVIDIA detection paths (in priority order) -----------------------------
def _nvidia_via_nvml() -> Optional[Tuple[int, str, int, int]]:
    """(num_gpus, name, total_bytes, free_bytes) over EVERY visible GPU.

    Summed, not device 0. A host with four cards used to offer one of them:
    the memory of device 0 was reported, a stage was placed to fit it, and the
    other three sat idle while their owner believed all four were earning.

    Free memory rather than total, per card: a card someone else is already
    using contributes only what is left on it, and one that is full
    contributes nothing without making the node unusable.

    The name is device 0's. Mixed-card hosts exist and the label is then
    approximate — the memory figure, which is what placement actually uses,
    stays exact.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        name, total, free = "", 0, 0
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            if not name:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total += int(mem.total)
            free += int(mem.free)
        return count, name, total, free
    except Exception:
        return None


def _nvidia_via_torch() -> Optional[Tuple[int, str, int, int]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
        # Every card, same as the NVML path: a multi-GPU host offers all of it.
        name, total, free = "", 0, 0
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            name = name or str(props.name)
            total += int(props.total_memory)
            try:
                on_card, _ = torch.cuda.mem_get_info(index)
                free += int(on_card)
            except Exception:
                free += int(props.total_memory)
        return count, name, total, free
    except Exception:
        return None


def _nvidia_via_smi() -> Optional[Tuple[int, str, int, int]]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        lines = [l for l in out.splitlines() if l.strip()]
        if not lines:
            return None
        name, total, free = "", 0, 0
        for line in lines:  # one line per card; all of them count
            card, total_mib, free_mib = (p.strip() for p in line.split(","))
            name = name or card
            total += int(float(total_mib)) * 1024 * 1024
            free += int(float(free_mib)) * 1024 * 1024
        return len(lines), name, total, free
    except Exception:
        return None


def _detect_nvidia() -> Optional[DetectedHardware]:
    for source, probe in (
        ("nvml", _nvidia_via_nvml),
        ("torch", _nvidia_via_torch),
        ("nvidia-smi", _nvidia_via_smi),
    ):
        result = probe()
        if result is None:
            continue
        count, name, total, free = result
        total_gb = total / GIB
        spec = match_gpu_specs(name, total_gb)
        return DetectedHardware(
            device="cuda",
            num_gpus=count,
            gpu_name=name,
            tflops_fp16=float(spec["tflops_fp16"]),
            # Schedulable memory: what is actually free on the device now.
            memory_gb=round(free / GIB, 2),
            memory_bandwidth_gbps=float(spec["bandwidth_gbps"]),
            vram_total_bytes=total,
            vram_free_bytes=free,
            host_ram_gb=_host_ram_gb(),
            detection_source=source,
        )
    return None


def _detect_apple() -> Optional[DetectedHardware]:
    if platform.system() != "Darwin" or not platform.machine().startswith("arm"):
        return None
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=10
        ).strip()
    except Exception:
        chip = "Apple Silicon"
    short = chip.rsplit("Apple ", maxsplit=1)[-1].replace(" (Virtual)", "")
    tflops = _APPLE_PEAK_FP16.get(short, 10.0)
    ram_gb = _host_ram_gb()
    # Unified memory: reserve headroom for the OS rather than offering it all.
    usable_gb = round(max(1.0, ram_gb * 0.7), 2)
    return DetectedHardware(
        device="mlx",
        num_gpus=1,
        gpu_name=chip,
        tflops_fp16=tflops,
        memory_gb=usable_gb,
        memory_bandwidth_gbps=200.0,
        vram_total_bytes=int(ram_gb * GIB),
        vram_free_bytes=int(usable_gb * GIB),
        host_ram_gb=ram_gb,
        detection_source="sysctl",
    )


def _detect_cpu() -> DetectedHardware:
    ram_gb = _host_ram_gb() or 8.0
    usable_gb = round(max(1.0, ram_gb * 0.5), 2)
    return DetectedHardware(
        device="cpu",
        num_gpus=1,
        gpu_name=platform.processor() or platform.machine() or "cpu",
        tflops_fp16=2.0,
        memory_gb=usable_gb,
        memory_bandwidth_gbps=50.0,
        vram_total_bytes=int(ram_gb * GIB),
        vram_free_bytes=int(usable_gb * GIB),
        host_ram_gb=ram_gb,
        detection_source="fallback",
    )


def detect_hardware() -> DetectedHardware:
    """Detect local hardware: NVIDIA -> Apple Silicon -> CPU fallback.

    Env overrides (testing only): LOOM_DEVICE, LOOM_MEMORY_GB,
    LOOM_TFLOPS_FP16, LOOM_MEM_BW_GBPS, LOOM_NUM_GPUS, LOOM_GPU_NAME.
    """
    forced = os.environ.get("LOOM_DEVICE")
    hw: Optional[DetectedHardware] = None
    if forced == "cuda" or forced is None:
        hw = _detect_nvidia()
    if hw is None and forced in (None, "mlx"):
        hw = _detect_apple()
    if hw is None:
        hw = _detect_cpu()

    if forced and forced != hw.device:
        hw.device = forced
    if "LOOM_MEMORY_GB" in os.environ:
        hw.memory_gb = float(os.environ["LOOM_MEMORY_GB"])
        hw.vram_free_bytes = int(hw.memory_gb * GIB)
        hw.vram_total_bytes = max(hw.vram_total_bytes, hw.vram_free_bytes)
        hw.detection_source = "env"
    if "LOOM_TFLOPS_FP16" in os.environ:
        hw.tflops_fp16 = float(os.environ["LOOM_TFLOPS_FP16"])
        hw.detection_source = "env"
    if "LOOM_MEM_BW_GBPS" in os.environ:
        hw.memory_bandwidth_gbps = float(os.environ["LOOM_MEM_BW_GBPS"])
    if "LOOM_NUM_GPUS" in os.environ:
        hw.num_gpus = int(os.environ["LOOM_NUM_GPUS"])
    if "LOOM_GPU_NAME" in os.environ:
        hw.gpu_name = os.environ["LOOM_GPU_NAME"]
    return hw
