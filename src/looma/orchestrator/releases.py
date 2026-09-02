"""What version the fleet should be running, and who is on it yet.

The orchestrator does not push updates. It states what it wants and agents
converge on it, which is why a node that was off for a week catches up by
itself and one joining today arrives correct.

Two things this deliberately does NOT hold: the signing key, and the power to
advance a wave on its own.

The key is absent because an orchestrator that has been taken over should be
able to name any release we ever signed — bad — and not to run new code on
every machine in the fleet — catastrophic. Signing happens elsewhere; this only
carries the signature.

Waves advance by hand because "the step succeeded" has no honest automatic
definition yet: registering is not the same as working, and a rollout that
advanced on registrations alone would spread a broken version confidently.
The operator advances them looking at the version map below.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from looma.logging_config import get_logger

logger = get_logger(__name__)


class ReleaseError(ValueError):
    """This release cannot be published, and why."""


@dataclass
class Release:
    version: str
    sha256: str
    signature: bytes
    path: Path
    published_at: float = field(default_factory=time.time)
    # Share of the fleet that should be on it, 0-100. Starts small on purpose:
    # a bad build reaching every node at once takes the network down and, at
    # worst, takes with it the ability to ship the fix.
    wave_percent: int = 0

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "wave_percent": self.wave_percent,
            "published_at": self.published_at,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


class ReleaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.current: Optional[Release] = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("cannot store agent releases in %s: %s", root, exc)

    # ------------------------------------------------------------ publishing
    def publish(self, *, version: str, signature: bytes, archive: bytes) -> Release:
        """Take a signed build. Does not roll it out — that is a separate act.

        The signature is not checked here. It cannot be: the public key lives
        in the agent image, and an orchestrator that could validate a release
        would be an orchestrator that could be tricked into vouching for one.
        The launcher on each node is what decides, and it is the only thing
        that should.
        """
        version = (version or "").strip()
        if not version:
            raise ReleaseError("a release needs a version")
        if not signature:
            raise ReleaseError(
                "a release needs its signature; agents refuse anything unsigned "
                "and would simply never take this one"
            )
        if not archive:
            raise ReleaseError("the release archive is empty")
        path = self.root / f"{version}.tar.gz"
        path.write_bytes(archive)
        release = Release(
            version=version,
            sha256=hashlib.sha256(archive).hexdigest(),
            signature=signature,
            path=path,
        )
        self.current = release
        logger.info("agent release %s published (%d bytes), rollout at 0%%",
                    version, len(archive))
        return release

    def set_wave(self, percent: int) -> Release:
        if self.current is None:
            raise ReleaseError("no release has been published")
        self.current.wave_percent = max(0, min(100, int(percent)))
        logger.info("agent release %s rolling out to %d%% of the fleet",
                    self.current.version, self.current.wave_percent)
        return self.current

    def withdraw(self) -> None:
        """Stop offering the current release. Nodes already on it stay there.

        Not a rollback: taking a version back from a node means publishing an
        older one, and agents refuse to move backwards for good reason. This
        only stops the spread.
        """
        if self.current is not None:
            logger.warning("agent release %s withdrawn; no node will be offered it",
                           self.current.version)
            self.current.wave_percent = 0

    # -------------------------------------------------------------- rollout
    def offer_to(self, node_id: str) -> Optional[Release]:
        """The release this node should move to, or None to leave it alone."""
        release = self.current
        if release is None or release.wave_percent <= 0:
            return None
        return release if in_wave(node_id, release.wave_percent) else None

    def archive_bytes(self, version: str) -> Optional[bytes]:
        path = self.root / f"{version}.tar.gz"
        try:
            return path.read_bytes()
        except OSError:
            return None

    # ------------------------------------------------------------ reporting
    def version_map(self, nodes: List[dict]) -> dict:
        """Who is on what — the thing an operator reads before advancing a wave."""
        counts: Dict[str, int] = {}
        for node in nodes:
            version = node.get("agent_version") or "unknown"
            counts[version] = counts.get(version, 0) + 1
        target = self.current.version if self.current else ""
        return {
            "release": self.current.as_dict() if self.current else None,
            "versions": counts,
            "nodes_total": len(nodes),
            "nodes_on_target": counts.get(target, 0) if target else 0,
            "nodes_in_wave": sum(
                1 for n in nodes
                if self.current and in_wave(n.get("node_id", ""),
                                            self.current.wave_percent)
            ),
        }


def in_wave(node_id: str, percent: int) -> bool:
    """Whether this node is in the first `percent` of the fleet.

    By a hash of the name, not by arrival order or chance: a node must land in
    the same wave every time it reconnects. Rolling a die per registration
    would let a node flip in and out of a rollout, updating and being told to
    update again a minute later.
    """
    if percent >= 100:
        return True
    if percent <= 0 or not node_id:
        return False
    digest = hashlib.sha256(node_id.encode()).digest()
    return (int.from_bytes(digest[:4], "big") % 100) < percent
