"""Deciding whether a payload is ours.

This is the file that makes an update channel safe to have at all. Everything
else in the launcher is plumbing; this is the part that stands between a
stranger's machine and arbitrary code.

Three rules, and each one exists because leaving it out has a name:

  the signature is asymmetric      The orchestrator holds no key that can sign
                                   a release. Taking it over lets an attacker
                                   name any release we ever signed — bad, but
                                   not the same as running new code on every
                                   machine in the fleet.

  the signature covers the version The manifest is signed as a whole, not just
                                   the archive digest. Signing bytes alone
                                   would let a genuine old payload be served
                                   under a new version number.

  versions never go backwards      A release we signed a year ago is still
                                   validly signed, and installing it is how a
                                   fixed vulnerability comes back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("looma_launcher.signature")

# Where the public half lives in the image. A file rather than a constant so
# anyone running their own Looma can ship their own key, and so it is obvious
# in a diff when it changes.
KEY_FILE = Path(__file__).with_name("release_key.pub")


class Untrusted(Exception):
    """This payload will not be installed, and why."""


@dataclass(frozen=True)
class Manifest:
    version: str
    sha256: str

    def canonical(self) -> bytes:
        """What actually gets signed.

        Sorted keys and no whitespace, so the same manifest always produces the
        same bytes on both sides — a signature over "whatever json.dumps did
        this time" verifies by luck.
        """
        return json.dumps(
            {"version": self.version, "sha256": self.sha256},
            sort_keys=True, separators=(",", ":"),
        ).encode()


def public_key_bytes() -> Optional[bytes]:
    """The key this image trusts, or None when it was built without one."""
    override = os.environ.get("LOOMA_RELEASE_PUBKEY", "").strip()
    if override:
        # Legitimate for a self-hosted fleet and for tests. Said out loud
        # because it changes who may write code to this machine.
        logger.warning("trusting a release key from the environment, not the image")
        return bytes.fromhex(override)
    try:
        raw = KEY_FILE.read_text().strip()
    except OSError:
        return None
    return bytes.fromhex(raw) if raw else None


def verify(manifest: Manifest, signature: bytes, *, installed_version: str = "") -> None:
    """Raise Untrusted unless this payload may be installed."""
    if not manifest.version or not manifest.sha256:
        raise Untrusted("the release manifest is incomplete")
    if installed_version and not _is_newer(manifest.version, installed_version):
        raise Untrusted(
            f"release {manifest.version} is not newer than the running "
            f"{installed_version}; refusing to move backwards"
        )
    key = public_key_bytes()
    if key is None:
        raise Untrusted(
            "this image carries no release key, so it can verify nothing and "
            "will install nothing. Updates are off on this node"
        )
    if not signature:
        raise Untrusted("the release is unsigned")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - the dependency is required
        raise Untrusted(f"cannot check signatures here: {exc}") from None
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(signature, manifest.canonical())
    except InvalidSignature:
        raise Untrusted(
            f"the signature on release {manifest.version} is not ours"
        ) from None
    except Exception as exc:
        raise Untrusted(f"the release signature could not be checked: {exc}") from None


def digest_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_archive(path: Path, manifest: Manifest) -> None:
    """The bytes on disk are the bytes that were signed."""
    actual = digest_of(path)
    if actual != manifest.sha256.lower():
        raise Untrusted(
            f"release {manifest.version} arrived with the wrong contents "
            f"({actual[:12]}… instead of {manifest.sha256[:12]}…)"
        )


def _is_newer(candidate: str, installed: str) -> bool:
    return _order(candidate) > _order(installed)


def _order(version: str) -> Tuple:
    """Compare versions numerically where possible, textually where not.

    A dotted numeric version is the normal case; anything else falls back to a
    string compare rather than being rejected, because refusing to parse a
    version is not a reason to leave a node unable to update.
    """
    parts = []
    for piece in str(version).split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append((0, int(digits), piece) if digits else (1, 0, piece))
    return tuple(parts)
