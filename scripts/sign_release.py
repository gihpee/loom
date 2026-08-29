#!/usr/bin/env python3
"""Make and sign an agent release.

The private key made here is what lets code run on every machine in the fleet.
It does not belong on the orchestrator, in the repository, or in CI that anyone
can trigger — an orchestrator that has been taken over should be able to name
any release we ever signed, and nothing more.

    # once, and keep agent-release.key somewhere the orchestrator cannot reach
    python scripts/sign_release.py keygen --out agent-release.key

    # per release
    python scripts/sign_release.py sign --key agent-release.key --version 0.2.0

`sign` prints the version, the signature and a ready-made curl for publishing.
Publishing does not roll anything out; the wave is advanced separately, from
the Release tab.

Run with: uv run --with cryptography python scripts/sign_release.py ...
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
PUBLIC_KEY_FILE = AGENT / "loom_launcher" / "release_key.pub"


def _keys():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey, serialization


def keygen(args) -> int:
    Ed25519PrivateKey, serialization = _keys()
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists. Overwriting it means every node that trusts "
              f"the old key stops accepting updates; pass --force if that is "
              f"really what you want.", file=sys.stderr)
        return 2
    key = Ed25519PrivateKey.generate()
    out.write_bytes(key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    ))
    out.chmod(0o600)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    PUBLIC_KEY_FILE.write_text(public.hex() + "\n")
    print(f"private key -> {out}  (keep it off the orchestrator)")
    print(f"public key  -> {PUBLIC_KEY_FILE}  (goes into the agent image)")
    print("\nRebuild and redistribute the agent image before publishing a release: "
          "nodes running an image without this key will refuse every update, "
          "which is the safe behaviour and also means they will never take one.")
    return 0


def build_archive(version: str, out: Path) -> Path:
    """Pack the agent payload — the part an update replaces, and nothing else.

    The launcher is deliberately absent: it lives in the image and is never
    updated over the network, because a broken launcher shipped that way could
    not be fixed by shipping another one.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(AGENT / "loom_agent", arcname="loom_agent",
                filter=lambda info: None if "__pycache__" in info.name else info)
    return out


def sign(args) -> int:
    Ed25519PrivateKey, serialization = _keys()
    key = Ed25519PrivateKey.from_private_bytes(Path(args.key).read_bytes())
    archive = build_archive(args.version, Path(args.out or f"dist/loom-agent-{args.version}.tar.gz"))
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # The manifest is signed as a whole, not just the digest: signing bytes
    # alone would let a genuine old payload be served under a new version.
    manifest = json.dumps({"version": args.version, "sha256": digest},
                          sort_keys=True, separators=(",", ":")).encode()
    signature = key.sign(manifest).hex()

    print(f"archive   {archive}  ({len(payload)} bytes)")
    print(f"version   {args.version}")
    print(f"sha256    {digest}")
    print(f"signature {signature}")
    if args.emit_curl:
        body = json.dumps({"version": args.version, "signature": signature,
                           "archive": base64.b64encode(payload).decode()})
        Path("dist/publish.json").write_text(body)
        print("\npublish with:\n"
              f"  curl -X POST http://<orchestrator>:8000/admin/release \\\n"
              f"    -H 'X-Loom-Admin-Token: <token>' -H 'Content-Type: application/json' \\\n"
              f"    --data @dist/publish.json")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("keygen", help="make a release key pair")
    generate.add_argument("--out", default="agent-release.key")
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=keygen)

    signer = sub.add_parser("sign", help="pack and sign the agent payload")
    signer.add_argument("--key", required=True)
    signer.add_argument("--version", required=True)
    signer.add_argument("--out", default="")
    signer.add_argument("--emit-curl", action="store_true", default=True)
    signer.set_defaults(func=sign)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
