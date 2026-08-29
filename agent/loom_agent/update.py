"""Converging on the version this node is supposed to be running.

Not a push. The orchestrator states what it wants in every registration ack and
the agent moves towards it, which is why a node that was switched off for a
week catches up by itself and a node joining today arrives correct.

The agent only FETCHES. It writes the archive and its manifest where the
launcher looks and then stops; the launcher — the part an update cannot replace
— is what checks the signature and installs. An agent that could install its
own successor would be an agent that could be told to install anything.

Before stopping it drains: a task in flight is somebody's work, and throwing it
away to save a few minutes on a rollout is not a trade we get to make.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from loom_agent.proto import agent_pb2

logger = logging.getLogger("loom_agent.update")

# A payload is a few megabytes. Longer than this means the connection is not
# going to finish, and the node is better off staying on the version it has.
DOWNLOAD_TIMEOUT_S = 300
# How long running tasks get to finish before the agent restarts anyway.
DRAIN_TIMEOUT_S = float(os.environ.get("LOOM_DRAIN_TIMEOUT_S", "600"))


def health_file() -> Optional[Path]:
    raw = os.environ.get("LOOM_AGENT_HEALTH_FILE", "").strip()
    return Path(raw) if raw else None


def incoming_dir() -> Optional[Path]:
    raw = os.environ.get("LOOM_AGENT_INCOMING", "").strip()
    return Path(raw) if raw else None


def mark_healthy() -> None:
    """Say that this payload actually works.

    Called after a successful registration, not at startup: an agent that comes
    up and cannot reach anyone is alive and useless, and telling the launcher
    otherwise would disarm the rollback that exists for exactly that case.
    """
    marker = health_file()
    if marker is None:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()))
    except OSError:
        logger.debug("could not write the health marker", exc_info=True)


class Updater:
    def __init__(
        self,
        *,
        current_version: str,
        drain: Callable[[float], bool],
        stop: Callable[[], None],
    ) -> None:
        self.current_version = current_version
        self._drain = drain
        self._stop = stop
        self._working = threading.Lock()
        self.last_refusal = ""

    def on_release(self, release: agent_pb2.AgentRelease) -> None:
        """What the orchestrator says this node should run."""
        if not release.version or release.version == self.current_version:
            return
        if not release.url:
            logger.warning("release %s was named with no way to fetch it",
                           release.version)
            return
        if not self._working.acquire(blocking=False):
            return  # already fetching one
        threading.Thread(target=self._carry_out, args=(release,),
                         name="update", daemon=True).start()

    def _carry_out(self, release: agent_pb2.AgentRelease) -> None:
        try:
            target = incoming_dir()
            if target is None:
                # Started outside the launcher — in a test, or by hand. There
                # is nothing that could install an update, so saying so is more
                # useful than downloading one nobody will read.
                self.last_refusal = "this agent was not started by the launcher"
                logger.info("ignoring release %s: %s", release.version,
                            self.last_refusal)
                return
            logger.info("fetching agent %s from %s", release.version, release.url)
            if not self._download(release, target):
                return
            logger.info("agent %s is downloaded; draining before restart",
                        release.version)
            if not self._drain(DRAIN_TIMEOUT_S):
                logger.warning("tasks were still running after %.0fs; restarting anyway",
                               DRAIN_TIMEOUT_S)
            self._stop()
        finally:
            self._working.release()

    def _download(self, release: agent_pb2.AgentRelease, target: Path) -> bool:
        """Fetch the archive and leave a manifest beside it. Never fatal."""
        try:
            target.mkdir(parents=True, exist_ok=True)
            archive = target / f"{release.version}.tar.gz"
            partial = archive.with_suffix(".part")
            with urllib.request.urlopen(release.url, timeout=DOWNLOAD_TIMEOUT_S) as source, \
                    open(partial, "wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            # Only now does it get its real name: the launcher must never find
            # half a download and try to install it.
            os.replace(partial, archive)
            (target / f"{release.version}.json").write_text(json.dumps({
                "version": release.version,
                "sha256": release.sha256,
                "signature": release.signature.hex(),
            }))
            return True
        except Exception as exc:
            # A node that cannot fetch an update is a node running an old
            # version, which is a much smaller problem than a node that stops.
            self.last_refusal = f"could not fetch release {release.version}: {exc}"
            logger.warning("%s", self.last_refusal)
            return False
