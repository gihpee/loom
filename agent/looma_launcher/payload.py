"""Which agent to run.

Two places an agent can come from:

  bundled   installed in the image. Always present, never changes, and the
            floor we fall back to when everything else is broken.
  payload   unpacked under LOOMA_ROOT/agent/<version>/, selected by the
            `current` symlink. This is what a network update replaces.

Switching versions is switching a symlink, which is atomic on POSIX: a reader
sees either the old target or the new one, never a half-written state.

The agent fetches; the launcher installs. That split is the point: the agent is
the part an update replaces, so it is not the part that decides what may be
installed. It drops an archive and a manifest into `incoming/` and stops; the
launcher verifies the signature, unpacks, and switches.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from looma_launcher.signature import Manifest, Untrusted, check_archive, verify

logger = logging.getLogger("looma_launcher.payload")

BUILDING_PREFIX = ".building-"


def root() -> Path:
    return Path(os.environ.get("LOOMA_ROOT", "/var/lib/looma"))


def agents_dir() -> Path:
    return root() / "agent"


def current_link() -> Path:
    return agents_dir() / "current"


def previous_link() -> Path:
    """The last payload known to have registered. The floor for a rollback."""
    return agents_dir() / "previous"


@dataclass(frozen=True)
class Payload:
    """How to start one agent."""

    version: str
    # Directory to put on PYTHONPATH, or None for the agent installed in the
    # image (already importable, nothing to prepend).
    path: Optional[Path]

    @property
    def bundled(self) -> bool:
        return self.path is None

    def describe(self) -> str:
        return f"{self.version} ({'bundled' if self.bundled else self.path})"


def bundled() -> Payload:
    """The agent shipped inside the image."""
    return Payload(version=_bundled_version(), path=None)


def _bundled_version() -> str:
    try:
        from importlib.metadata import version

        return version("looma-agent")
    except Exception:
        return "unknown"


def resolve(link: Path = None) -> Payload:
    """The payload to run now: whatever `current` points at, else the bundled one.

    A dangling or unreadable symlink is not an error worth stopping for — it
    means an update went wrong, and the right answer is to run the agent we
    know is intact rather than to leave the node dead.
    """
    link = link or current_link()
    try:
        target = link.resolve(strict=True)
    except (OSError, RuntimeError):
        return bundled()
    if not (target / "looma_agent" / "main.py").is_file():
        return bundled()
    return Payload(version=target.name, path=target)


def incoming_dir() -> Path:
    """Where the agent leaves what it downloaded."""
    return agents_dir() / "incoming"


def refused_dir() -> Path:
    """Версии, которые этот узел уже отказался ставить.

    Отказ обязан пережить перезапуск, иначе он ничего не значит. Оркестратор
    предлагает релиз при каждом подключении, а состояние агента живёт ровно
    столько же, сколько его процесс: без записи на диск узел качает отвергнутое
    заново, отказ повторяется, агент перезапускается — и так по кругу, раз в
    секунду, пока кто-нибудь не заметит.

    Рядом с `incoming`, потому что читать это будет агент, а он про раскладку
    лаунчера знает только один путь — тот, что ему передали.
    """
    return incoming_dir().parent / "refused"


def remember_refusal(version: str, reason: str) -> None:
    """Записать, что этот релиз ставить не будем и почему.

    Не падает ни при какой ошибке записи: отказ и так уже случился, и уронить
    на нём лаунчер значит превратить негодный релиз в мёртвый узел.
    """
    try:
        directory = refused_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{version}.txt").write_text(reason + "\n")
    except OSError as exc:
        logger.warning("не удалось запомнить отказ от %s (%s); узел может "
                       "начать качать его по кругу", version, exc)


def health_marker(version: str) -> Path:
    """Written by the agent once it has actually registered.

    "The process is alive" is not the same as "the agent works": a payload can
    start, fail to reach the orchestrator and sit there. This file is the
    difference, and it is what a rollback decision reads.
    """
    return agents_dir() / version / ".healthy"


def pending() -> List[Path]:
    """Manifests waiting to be installed, oldest first."""
    try:
        return sorted(incoming_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return []


def install(manifest_path: Path, *, installed_version: str = "") -> Optional[Payload]:
    """Verify and install one downloaded release. Returns it, or None.

    Never raises: a bad payload is a reason to keep running the agent we have,
    not a reason for the node to go down. The archive and manifest are removed
    either way, so a rejected release is not retried forever.
    """
    archive = manifest_path.with_suffix(".tar.gz")
    try:
        raw = json.loads(manifest_path.read_text())
        manifest = Manifest(version=raw["version"], sha256=raw["sha256"])
        signature = bytes.fromhex(raw.get("signature", ""))
    except (OSError, ValueError, KeyError) as exc:
        logger.error("ignoring an unreadable release manifest: %s", exc)
        _discard(manifest_path, archive)
        return None
    try:
        verify(manifest, signature, installed_version=installed_version)
        check_archive(archive, manifest)
        payload = _unpack(archive, manifest.version)
    except Untrusted as exc:
        logger.error("REFUSING release %s: %s", manifest.version, exc)
        # На диск, а не только в лог: без этого агент скачает тот же релиз при
        # следующем же подключении, и отказ повторится вместе с перезапуском.
        remember_refusal(manifest.version, str(exc))
        _discard(manifest_path, archive)
        return None
    except (OSError, tarfile.TarError) as exc:
        logger.error("release %s could not be unpacked: %s", manifest.version, exc)
        _discard(manifest_path, archive)
        return None
    _discard(manifest_path, archive)
    logger.info("installed agent %s", manifest.version)
    return payload


def _discard(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


# Сколько каталог распаковки считается живым. Распаковка нескольких мегабайт
# укладывается в секунды; всё, что старше, осталось от убитого процесса.
STALE_STAGING_S = 3600


def _sweep_stale(where: Path) -> None:
    """Убрать брошенные каталоги распаковки — и только их.

    С уникальным именем такой каталог больше не будет переиспользован, поэтому
    убитый на полпути процесс оставлял бы мусор навсегда. Возраст здесь не
    предосторожность, а условие: рядом может распаковываться СОСЕДНИЙ агент, и
    снести его каталог значило бы вернуть ту самую ошибку, от которой уходим.
    """
    import time

    try:
        entries = list(where.iterdir())
    except OSError:
        return
    for path in entries:
        if not path.name.startswith(BUILDING_PREFIX) or not path.is_dir():
            continue
        try:
            if time.time() - path.stat().st_mtime < STALE_STAGING_S:
                continue
        except OSError:
            continue
        logger.info("removing a leftover staging directory %s", path.name)
        shutil.rmtree(path, ignore_errors=True)


def _unpack(archive: Path, version: str) -> Payload:
    """Into a staging directory, then into place with a rename.

    Same reason as everywhere else in this codebase: a half-unpacked payload
    that something tries to run is a failure that looks nothing like its cause.
    """
    # uuid, НЕ pid — ровно по той же причине, по которой он уже стоит в имени
    # файла при скачивании (looma_agent/update.py): том бывает общим со вторым
    # агентом на этой же машине, а в своих контейнерах оба — процесс номер 7.
    # Совпавшее имя означало, что оба распаковываются в один каталог и чужая
    # уборка стирает наполовину распакованное. Наружу это выходило как «the
    # release archive holds no agent» — то есть обвинением исправного архива.
    staging = agents_dir() / f"{BUILDING_PREFIX}{version}-{uuid.uuid4().hex[:12]}"
    _sweep_stale(agents_dir())
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk() or member.name.startswith(("/", "..")) \
                        or ".." in Path(member.name).parts:
                    raise Untrusted(
                        f"the release archive contains {member.name!r}, which would "
                        "write outside where it is unpacked"
                    )
            try:
                # Belt and braces: the members were already checked above, and
                # the stdlib filter catches anything that check did not think of.
                tar.extractall(staging, filter="data")
            except TypeError:
                tar.extractall(staging)  # older Python without the filter
        if not (staging / "looma_agent" / "main.py").is_file():
            raise Untrusted("the release archive holds no agent")
        final = agents_dir() / version
        if (final / "looma_agent" / "main.py").is_file():
            # Второй агент на этой же машине уже распаковал ту же версию.
            # Его результат ничем не хуже: подпись проверена и там, и здесь.
            shutil.rmtree(staging, ignore_errors=True)
            return Payload(version=version, path=final)
        shutil.rmtree(final, ignore_errors=True)
        os.rename(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Payload(version=version, path=agents_dir() / version)


def switch_to(payload: Payload) -> None:
    """Point `current` at this payload, remembering what it replaced.

    The symlink is replaced with a rename, which is atomic: nothing ever sees
    a moment with no `current` at all.
    """
    agents_dir().mkdir(parents=True, exist_ok=True)
    was = current_link()
    try:
        previous_target = was.resolve(strict=True)
    except (OSError, RuntimeError):
        previous_target = None
    if previous_target is not None and previous_target != payload.path:
        _point(previous_link(), previous_target)
    _point(current_link(), payload.path)


def roll_back() -> Optional[Payload]:
    """Go back to the last payload that worked, if there is one.

    Decided on the node, without asking anyone: the connection to the
    orchestrator may be exactly what the new version broke.
    """
    try:
        target = previous_link().resolve(strict=True)
    except (OSError, RuntimeError):
        logger.error("nothing to roll back to; falling back to the bundled agent")
        _discard(current_link())
        return bundled()
    _point(current_link(), target)
    logger.warning("rolled back to agent %s", target.name)
    return Payload(version=target.name, path=target)


def _point(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + ".swap")
    try:
        temporary.unlink()
    except OSError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)
