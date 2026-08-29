"""Pulling a container image without a container daemon.

The point of the whole design: images give universality — any language, any
system library, a registry full of ready-made environments — and a daemon gives
none of that. It gives privileges we do not want, nesting we cannot have, and a
cache that dies with the container. So we take the format and leave the daemon.

What this does is the registry protocol and nothing more: fetch the manifest,
fetch the layers, apply them in order into a directory. That directory is then
an ordinary filesystem tree, and running a process in it is a separate problem
solved in tasks/limits.py.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loom_agent.tasks.env.archive import BadArchive, apply_layer

logger = logging.getLogger("loom_agent.tasks.env.oci")

DEFAULT_REGISTRY = "registry-1.docker.io"
DEFAULT_TAG = "latest"
PULL_TIMEOUT_S = int(os.environ.get("LOOM_OCI_TIMEOUT_S", "600"))

MANIFEST_TYPES = ",".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


class PullFailed(RuntimeError):
    """The image could not be fetched, with the reason a human needs."""


@dataclass(frozen=True)
class Reference:
    registry: str
    repository: str
    tag: str

    @property
    def url_base(self) -> str:
        scheme = "http" if self.registry.startswith(("localhost", "127.0.0.1")) else "https"
        return f"{scheme}://{self.registry}/v2/{self.repository}"

    def describe(self) -> str:
        return f"{self.registry}/{self.repository}:{self.tag}"


def parse_reference(source: str) -> Reference:
    """`alpine`, `library/alpine:3.19`, `ghcr.io/owner/name:tag` — all of them.

    The rule for "is the first part a registry" is the one everyone uses: it
    counts as a host if it has a dot or a colon, or is exactly localhost.
    Without it, `owner/name` would be read as a registry called `owner`.
    """
    source = (source or "").strip()
    if not source:
        raise PullFailed("an oci environment needs an image, e.g. 'python:3.12-slim'")
    remainder = source
    registry = DEFAULT_REGISTRY
    head, _, rest = source.partition("/")
    if rest and ("." in head or ":" in head or head == "localhost"):
        registry, remainder = head, rest
    repository, _, tag = remainder.partition(":")
    if "/" in tag:  # a colon inside a path, not a tag
        repository, tag = remainder, ""
    if registry == DEFAULT_REGISTRY and "/" not in repository:
        repository = f"library/{repository}"
    return Reference(registry=registry, repository=repository, tag=tag or DEFAULT_TAG)


def platform_pair() -> Tuple[str, str]:
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "amd64", "amd64": "amd64",
        "aarch64": "arm64", "arm64": "arm64",
    }.get(machine, machine)
    return "linux", architecture


@dataclass
class Image:
    reference: Reference
    root: Path
    config: dict = field(default_factory=dict)

    @property
    def entrypoint(self) -> List[str]:
        cfg = self.config.get("config") or {}
        return list(cfg.get("Entrypoint") or [])

    @property
    def default_command(self) -> List[str]:
        cfg = self.config.get("config") or {}
        return list(cfg.get("Cmd") or [])

    def environment(self) -> Dict[str, str]:
        """PATH and friends as the image defines them.

        Without these a task in a python image cannot find `python`, because
        nothing outside the image knows where the image decided to put it.
        """
        cfg = self.config.get("config") or {}
        values: Dict[str, str] = {}
        for entry in cfg.get("Env") or []:
            name, _, value = str(entry).partition("=")
            if name:
                values[name] = value
        return values

    def working_dir(self) -> str:
        cfg = self.config.get("config") or {}
        return cfg.get("WorkingDir") or "/"


def pull(source: str, target: Path) -> Image:
    """Fetch an image and lay it out under `target`. Blocking and slow."""
    reference = parse_reference(source)
    logger.info("pulling %s", reference.describe())
    token = _anonymous_token(reference)
    manifest = _manifest(reference, token)
    config = _blob_json(reference, token, manifest["config"]["digest"])
    root = target / "rootfs"
    root.mkdir(parents=True, exist_ok=True)
    layers = manifest.get("layers") or []
    for number, layer in enumerate(layers, start=1):
        logger.info("  layer %d/%d (%s)", number, len(layers),
                    _human(layer.get("size", 0)))
        _apply_blob(reference, token, layer["digest"], root)
    (target / "config.json").write_text(json.dumps(config))
    return Image(reference=reference, root=root, config=config)


# ------------------------------------------------------------------ registry
def _get(url: str, token: str = "", accept: str = "", *, raw: bool = False):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    try:
        return urllib.request.urlopen(request, timeout=PULL_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode(errors="replace") if not raw else ""
        raise PullFailed(f"{url} answered {exc.code}: {detail or exc.reason}") from None
    except urllib.error.URLError as exc:
        raise PullFailed(f"could not reach {url}: {exc.reason}") from None


def _anonymous_token(reference: Reference) -> str:
    """A pull token for a public image.

    Private images need credentials we do not have anywhere to put yet; the
    registry says so plainly with a 401, which is a better error than anything
    this function could invent.
    """
    if reference.registry != DEFAULT_REGISTRY:
        return ""
    query = urllib.parse.urlencode({
        "service": "registry.docker.io",
        "scope": f"repository:{reference.repository}:pull",
    })
    with _get(f"https://auth.docker.io/token?{query}") as answer:
        return json.load(answer).get("token", "")


def _manifest(reference: Reference, token: str) -> dict:
    url = f"{reference.url_base}/manifests/{reference.tag}"
    with _get(url, token, MANIFEST_TYPES) as answer:
        document = json.load(answer)
    if "manifests" in document:
        document = _manifest_for_this_machine(reference, token, document)
    if "layers" not in document or "config" not in document:
        raise PullFailed(
            f"{reference.describe()} is not an image we can unpack "
            "(no layers in its manifest)"
        )
    return document


def _manifest_for_this_machine(reference: Reference, token: str, index: dict) -> dict:
    wanted_os, wanted_arch = platform_pair()
    available = []
    for entry in index.get("manifests", []):
        where = entry.get("platform") or {}
        available.append(f"{where.get('os')}/{where.get('architecture')}")
        if where.get("os") == wanted_os and where.get("architecture") == wanted_arch:
            url = f"{reference.url_base}/manifests/{entry['digest']}"
            with _get(url, token, MANIFEST_TYPES) as answer:
                return json.load(answer)
    raise PullFailed(
        f"{reference.describe()} has no build for {wanted_os}/{wanted_arch}; "
        f"it offers {', '.join(sorted(set(available))) or 'nothing'}"
    )


def _blob_json(reference: Reference, token: str, digest: str) -> dict:
    with _get(f"{reference.url_base}/blobs/{digest}", token) as answer:
        return json.load(answer)


def _apply_blob(reference: Reference, token: str, digest: str, root: Path) -> None:
    """Stream one layer to disk, then apply it. Never held in memory.

    A layer is routinely hundreds of megabytes, and a node lending us its
    machine did not agree to have that in RAM.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with _get(f"{reference.url_base}/blobs/{digest}", token, raw=True) as source:
            with open(staging, "wb") as sink:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    sink.write(chunk)
        try:
            apply_layer(staging, root)
        except (tarfile.TarError, BadArchive) as exc:
            raise PullFailed(f"layer {digest[:19]} could not be applied: {exc}") from None
    finally:
        staging.unlink(missing_ok=True)


def _human(size: int) -> str:
    return f"{size / 1024**2:.0f} MB" if size else "unknown size"
