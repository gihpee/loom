"""Environments, kept so they are built once.

Without this the whole design is worse than what it replaces: provisioning
moves gigabytes out of `docker pull` and into `pip install`, which is slower.
The cache is what turns that into a one-time cost per node.

Three properties it must have, in order of how badly their absence hurts:

1. A half-built environment is never handed to a task. Builds happen in a
   temporary directory and are moved into place with a rename, which is
   atomic: a reader sees a finished environment or no environment at all.
2. Two tasks wanting the same environment build it once. The second waits for
   the first rather than racing it into the same directory.
3. Eviction never removes an environment something is using. Environments are
   leased while a task holds one, and a leased environment is not a candidate
   however old it is.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from looma_agent.tasks.env import binary as binary_env
from looma_agent.tasks.env import oci as oci_env
from looma_agent.tasks.env import python as python_env
from looma_agent.tasks.env.base import (
    NO_ENVIRONMENT,
    Environment,
    directory_size,
    read_marker,
    write_marker,
)
from looma_agent.tasks.spec import EnvSpec, TaskRefused

logger = logging.getLogger("looma_agent.tasks.env.cache")

BUILDING_PREFIX = ".building-"
# Сколько каталог сборки должен пролежать нетронутым, чтобы считаться брошенным.
# Дольше самой долгой честной установки и короче человеческого терпения.
STALE_BUILD_S = float(os.environ.get("LOOMA_ENV_STALE_S", str(6 * 3600)))


def _variant(spec: EnvSpec) -> str:
    """Чем сборка на ЭТОМ узле отличается от такой же на другом."""
    if spec.kind != "python":
        return ""
    from looma_agent.tasks.env.python import wheel_variant

    return wheel_variant(spec.requirements)


def _build_python(target, spec) -> dict:
    python_env.build(target, spec.requirements)
    return {}


def _build_binary(target, spec) -> dict:
    binary_env.build(target, spec.source)
    return {}


def _build_oci(target, spec) -> dict:
    """An image also carries how it expects to be run.

    Kept in the marker rather than re-read from config.json on every cache hit,
    so a hit costs one small read and not a second parse of the image.
    """
    image = oci_env.pull(spec.source, target)
    return {
        "image_env": image.environment(),
        "image_workdir": image.working_dir(),
        "entrypoint": image.entrypoint,
        "default_command": image.default_command,
    }


# What "provision this environment" means, per kind. Adding a kind is adding a
# row here and nothing else: the cache, the leases and the eviction do not care
# what is inside a directory.
BUILDERS = {
    "python": _build_python,
    "binary": _build_binary,
    "oci": _build_oci,
}
# What the node owner is willing to give up to caching. Generous by default
# because the alternative is re-downloading it, but bounded because it is
# their disk.
DEFAULT_QUOTA_BYTES = int(os.environ.get("LOOMA_ENV_QUOTA_BYTES", str(64 * 1024**3)))


@contextlib.contextmanager
def _across_processes(root: Path, fingerprint: str):
    """Не дать двум агентам на одной машине собирать одно и то же дважды.

    Замок внутри процесса покрывает только свои задачи, а том с этим кэшем
    может быть общим: многокарточный хост держат двумя агентами. Файловая
    блокировка — единственное, что они оба видят.

    Не получилось взять — работаем дальше без неё. Хуже всего тут лишняя
    параллельная сборка, и она уже безопасна: у каждой свой каталог, а
    проигравший гонку переименования просто берёт чужой результат.
    """
    handle = None
    try:
        import fcntl

        root.mkdir(parents=True, exist_ok=True)
        handle = open(root / f".lock-{fingerprint}", "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
    except Exception:
        if handle is not None:
            handle.close()
            handle = None
    try:
        yield
    finally:
        if handle is not None:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()


class EnvironmentCache:
    def __init__(self, root: Path, *, quota_bytes: int = DEFAULT_QUOTA_BYTES) -> None:
        self.root = root
        self.quota_bytes = max(0, int(quota_bytes))
        self.unusable = ""
        self._leases: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._building: Dict[str, threading.Lock] = {}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.unusable = f"cannot use {self.root} for environments ({exc.strerror})"
            logger.error("%s", self.unusable)
            return
        self._sweep_unfinished()

    # ------------------------------------------------------------- provisioning
    def acquire(self, spec: EnvSpec) -> Environment:
        """Get the environment this spec asks for, building it if need be.

        The caller holds a lease until it calls release(), and a leased
        environment is never evicted.
        """
        if spec.kind == "none":
            return NO_ENVIRONMENT
        if self.unusable:
            raise TaskRefused(self.unusable)
        if spec.kind not in BUILDERS:
            raise TaskRefused(
                f"this agent cannot build a {spec.kind!r} environment; it knows "
                f"{', '.join(sorted(BUILDERS))}"
            )
        fingerprint = self._key(spec)
        existing = self._ready(fingerprint)
        if existing is not None:
            self._lease(fingerprint)
            self._touch(existing.path)
            logger.info("environment %s was already here", fingerprint)
            return existing
        with self._build_lock(fingerprint), _across_processes(self.root, fingerprint):
            # Кто-то мог закончить, пока мы ждали замок. Ради этого и ждём, а
            # не гоняемся: две одинаковые сборки — это два полных torch,
            # скачанных на канал владельца машины.
            existing = self._ready(fingerprint)
            if existing is None:
                existing = self._build(spec, fingerprint)
            self._lease(fingerprint)
        self._evict_to_quota()
        return existing

    def release(self, fingerprint: str) -> None:
        if fingerprint in (None, "", "none"):
            return
        with self._lock:
            remaining = self._leases.get(fingerprint, 0) - 1
            if remaining > 0:
                self._leases[fingerprint] = remaining
            else:
                self._leases.pop(fingerprint, None)

    # ------------------------------------------------------------------ private
    def _key(self, spec: EnvSpec) -> str:
        """Имя окружения на диске: что просили ПЛЮС что здесь из этого выйдет.

        Отпечаток спецификации считает оркестратор, и одинаковых требований ему
        достаточно. Но собранное окружение зависит ещё и от узла: колесо torch
        выбирается под его драйвер. Без этого нода, у которой обновили драйвер —
        или у которой мы починили выбор колеса, — молча переиспользовала бы
        окружение, собранное под старые условия, и падала бы ровно так же.
        """
        base = spec.fingerprint()
        variant = _variant(spec)
        return f"{base}-{variant}" if variant else base

    def _build(self, spec: EnvSpec, fingerprint: str) -> Environment:
        # uuid, НЕ pid: два агента на одной машине делят том с этим кэшем, а в
        # своих контейнерах оба — процесс номер 7. Совпавшее имя означало, что
        # один сносил недостроенное окружение другого: у первого падал
        # ensurepip, у второго mkdir с "File exists", и ни одно из сообщений не
        # называло настоящую причину.
        staging = self.root / f"{BUILDING_PREFIX}{fingerprint}-{uuid.uuid4().hex[:12]}"
        started = time.time()
        final = self.root / fingerprint
        try:
            extra = BUILDERS[spec.kind](staging, spec) or {}
            size = directory_size(staging)
            write_marker(staging, fingerprint=fingerprint, kind=spec.kind,
                         size_bytes=size, extra=extra)
            try:
                # Атомарно: задача видит либо готовое окружение, либо никакого.
                # Недостроенное состояние, на диагностику которого уходит
                # вечер, снаружи этой функции наблюдать нельзя.
                os.rename(staging, final)
            except OSError:
                # Кто-то собрал то же самое, пока собирали мы. Его результат
                # ничем не хуже — выкидываем свой, а не ломаем чужой.
                theirs = self._ready(fingerprint)
                if theirs is None:
                    raise
                logger.info("environment %s was built elsewhere while we built it too",
                            fingerprint)
                shutil.rmtree(staging, ignore_errors=True)
                return theirs
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        logger.info(
            "environment %s built in %.0fs, %.1f GB",
            fingerprint, time.time() - started, size / 1024**3,
        )
        return Environment(
            fingerprint=fingerprint, kind=spec.kind, path=final, size_bytes=size,
            image_env=dict(extra.get("image_env") or {}),
            image_workdir=extra.get("image_workdir") or "",
        )

    def _ready(self, fingerprint: str) -> Optional[Environment]:
        """An environment only counts as present when its marker says it finished."""
        path = self.root / fingerprint
        if not path.is_dir():
            return None
        marker = read_marker(path)
        if marker is None:
            logger.warning("environment %s has no marker; treating it as absent",
                           fingerprint)
            return None
        return Environment(
            fingerprint=fingerprint,
            kind=marker.get("kind", "python"),
            path=path,
            size_bytes=int(marker.get("size_bytes") or 0),
            image_env=dict(marker.get("image_env") or {}),
            image_workdir=marker.get("image_workdir") or "",
        )

    def _build_lock(self, fingerprint: str) -> threading.Lock:
        with self._lock:
            return self._building.setdefault(fingerprint, threading.Lock())

    def _lease(self, fingerprint: str) -> None:
        with self._lock:
            self._leases[fingerprint] = self._leases.get(fingerprint, 0) + 1

    def _touch(self, path: Optional[Path]) -> None:
        """Mark an environment as recently used, for eviction order."""
        if path is None:
            return
        try:
            os.utime(path / ".looma-env.json", None)
        except OSError:
            pass

    def _sweep_unfinished(self) -> None:
        """Убрать то, что осталось от упавшего агента.

        По возрасту, а не просто по префиксу: том может быть общим с другим
        агентом на этой же машине, и он прямо сейчас может что-то собирать.
        Настоящая сборка не длится часами, а живая никогда не бывает такой
        старой — так что старое чужое и старое своё одинаково безопасно
        удалять, а свежее чужое остаётся нетронутым.
        """
        deadline = time.time() - STALE_BUILD_S
        for entry in self.root.glob(f"{BUILDING_PREFIX}*"):
            try:
                if entry.stat().st_mtime > deadline:
                    continue
            except OSError:
                continue
            logger.info("removing a half-built environment left behind: %s", entry.name)
            shutil.rmtree(entry, ignore_errors=True)

    # ----------------------------------------------------------------- eviction
    def _candidates(self) -> List[Tuple[float, Path, int]]:
        found: List[Tuple[float, Path, int]] = []
        with self._lock:
            leased = set(self._leases)
        for entry in self.root.iterdir():
            if not entry.is_dir() or entry.name.startswith(BUILDING_PREFIX):
                continue
            if entry.name in leased:
                continue
            marker = read_marker(entry)
            if marker is None:
                continue
            try:
                used_at = (entry / ".looma-env.json").stat().st_mtime
            except OSError:
                used_at = 0.0
            found.append((used_at, entry, int(marker.get("size_bytes") or 0)))
        return sorted(found)

    def total_bytes(self) -> int:
        total = 0
        for entry in self.root.iterdir():
            if entry.is_dir() and not entry.name.startswith(BUILDING_PREFIX):
                marker = read_marker(entry)
                if marker is not None:
                    total += int(marker.get("size_bytes") or 0)
        return total

    def _evict_to_quota(self) -> None:
        if not self.quota_bytes:
            return
        total = self.total_bytes()
        if total <= self.quota_bytes:
            return
        for used_at, path, size in self._candidates():
            if total <= self.quota_bytes:
                return
            logger.info(
                "evicting environment %s (%.1f GB, last used %.0f min ago) to stay "
                "under the %.0f GB cache quota",
                path.name, size / 1024**3, (time.time() - used_at) / 60,
                self.quota_bytes / 1024**3,
            )
            shutil.rmtree(path, ignore_errors=True)
            total -= size
        if total > self.quota_bytes:
            # Everything left is in use. Not an error: the quota is a target,
            # and breaking a running task to meet it would be the wrong trade.
            logger.warning(
                "the environment cache is %.1f GB over its quota and everything "
                "left is in use by a running task",
                (total - self.quota_bytes) / 1024**3,
            )

    def snapshot(self) -> dict:
        with self._lock:
            leased = dict(self._leases)
        return {
            "unusable": self.unusable,
            "bytes": self.total_bytes(),
            "quota_bytes": self.quota_bytes,
            "leased": leased,
        }
