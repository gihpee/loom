"""Building a Python environment for a task.

`--system-site-packages` is the whole point: the image already carries whatever
it carries, and on a node that has served one inference task it also carries
torch. Reinstalling several gigabytes of it per task, into a directory that
will be deleted, would make the cache pointless.

So the venv inherits what is already there and installs only what is missing.
A task that needs a different version of something inherited gets it — pip puts
it in the venv, which shadows the system copy.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from loom_agent.tasks.spec import TaskRefused

logger = logging.getLogger("loom_agent.tasks.env.python")

# Installing a large dependency set over a home connection is slow, and a node
# whose network died mid-install must not hold a build slot forever.
INSTALL_TIMEOUT_S = int(os.environ.get("LOOM_ENV_INSTALL_TIMEOUT_S", "1800"))

# Сборки torch под конкретные версии CUDA. По убыванию: берём самую новую,
# которую держит драйвер узла.
TORCH_INDEX = "https://download.pytorch.org/whl"
TORCH_WHEELS = (
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
)


def build(target: Path, requirements: Sequence[str]) -> None:
    """Create the venv and install into it. Raises TaskRefused with the output.

    Never partially succeeds from the caller's point of view: the caller builds
    into a temporary directory and only moves it into the cache when this
    returns.
    """
    _run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(target)],
        what="create the environment",
    )
    _inherit_our_packages(target)
    if not requirements:
        return
    python = _interpreter(target)
    _run(
        [str(python), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
         *_wheel_source(requirements), *requirements],
        what=f"install {', '.join(requirements)}",
    )


def wheel_variant(requirements: Sequence[str]) -> str:
    """Короткое имя того, что на этом узле реально поставится: cu124, cpu, "".

    Входит в имя кэшированного окружения, поэтому смена драйвера даёт новое
    окружение, а не тихое переиспользование несовместимого.
    """
    source = _wheel_source(requirements)
    return source[-1].rsplit("/", 1)[-1] if source else ""


def _wheel_source(requirements: Sequence[str]) -> list:
    """Откуда брать torch, чтобы он завёлся именно на этой машине.

    По умолчанию pip ставит сборку под самую новую CUDA. На узле с драйвером
    постарее она ставится молча и падает при первом обращении к карте:
    "The NVIDIA driver on your system is too old (found version 12040)" —
    сообщение, из которого не следует ни что делать, ни кто виноват.

    Выбирает узел, а не оркестратор: тот говорит, ЧТО нужно, а какое колесо
    подходит этому железу, знает только само железо.
    """
    if not any(_is_torch(name) for name in requirements):
        return []
    from loom_agent.hwinfo import cuda_driver_version

    driver = cuda_driver_version()
    if driver is None:
        # Карты нет или её не видно — CPU-колёса вместо гигабайтов CUDA,
        # которые всё равно не пригодятся.
        logger.info("environment: no CUDA driver here, taking CPU wheels")
        return ["--extra-index-url", f"{TORCH_INDEX}/cpu"]
    for version, tag in TORCH_WHEELS:
        if driver >= version:
            logger.info("environment: driver supports CUDA %d.%d, taking %s wheels",
                        driver[0], driver[1], tag)
            return ["--extra-index-url", f"{TORCH_INDEX}/{tag}"]
    logger.warning(
        "environment: driver supports only CUDA %d.%d, which is older than any "
        "torch build we know; letting pip choose and hoping", driver[0], driver[1])
    return []


def _is_torch(requirement: str) -> bool:
    name = requirement.strip().lower()
    for separator in ("==", ">=", "<=", "~=", ">", "<", "[", " "):
        name = name.split(separator)[0]
    return name in ("torch", "torchvision", "torchaudio")


def _inherit_our_packages(target: Path) -> None:
    """Make the environment see what THIS interpreter sees, not just the base one.

    `--system-site-packages` inherits from `sys.base_prefix`, which is the
    wrong place whenever the agent is not running from a plain system install:
    a virtualenv (what a downloaded update payload looks like) or an injected
    overlay both put the packages that matter somewhere the flag never looks.
    The task then starts fine and fails on `import torch` — correct-looking,
    and unrelated to its cause.

    `sys.path` is the honest source, because it is what this interpreter
    actually resolves imports against however it was started. Entries already
    covered by the flag simply appear twice, which costs nothing.
    """
    inherited = [
        entry for entry in sys.path
        if entry and Path(entry).is_dir() and not entry.startswith(str(target))
    ]
    if not inherited:
        return
    for site_packages in target.glob("lib/python*/site-packages"):
        site_packages.joinpath("loom-inherit.pth").write_text("\n".join(inherited) + "\n")
        return


def _interpreter(target: Path) -> Path:
    for candidate in (target / "bin" / "python", target / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    raise TaskRefused("the environment was created without an interpreter in it")


def _run(argv, *, what: str) -> None:
    logger.info("environment: %s", what)
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise TaskRefused(
            f"could not {what}: it was still going after {INSTALL_TIMEOUT_S}s "
            "and was stopped"
        ) from None
    if result.returncode != 0:
        # The tail of pip's own output, because the reason is almost always in
        # it and paraphrasing it would only lose the detail that matters.
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise TaskRefused(
            f"could not {what}: " + " / ".join(detail[-4:] or ["no output"])
        )
