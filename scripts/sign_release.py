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
import gzip
import io
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
PUBLIC_KEY_FILE = AGENT / "looma_launcher" / "release_key.pub"


def _keys():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey, serialization


def keygen(args) -> int:
    Ed25519PrivateKey, serialization = _keys()
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} уже существует. Перезапись означает, что каждый узел, "
              f"доверяющий старому ключу, перестанет принимать обновления; "
              f"--force, если это правда то, что нужно.", file=sys.stderr)
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
    beside = out.with_suffix(out.suffix + ".pub")
    beside.write_text(public.hex() + "\n")
    print(f"приватный ключ -> {out}  (держать подальше от оркестратора)")
    print(f"публичный      -> {beside}")

    # Ключ в дереве — это якорь доверия всего парка. Подменить его значит
    # сделать так, что каждый уже розданный образ перестанет принимать
    # обновления, и вернуть это можно только походом на все машины. Поэтому
    # молча — никогда.
    if PUBLIC_KEY_FILE.exists() and not args.install:
        print(f"\n{PUBLIC_KEY_FILE} уже есть и НЕ тронут.", file=sys.stderr)
        print("Это ключ, которому доверяют уже розданные образы. Подменить его "
              "значит лишить обновлений каждый узел с таким образом — вернуть "
              "можно только руками на каждой машине.", file=sys.stderr)
        print("Если это действительно новый парк: --install", file=sys.stderr)
        return 3
    PUBLIC_KEY_FILE.write_text(public.hex() + "\n")
    print(f"в образ       -> {PUBLIC_KEY_FILE}")
    print("\nПересоберите и раздайте образ агента до первой публикации: узел с "
          "образом без этого ключа отвергает любое обновление — состояние "
          "безопасное и означающее, что он никогда его и не примет.")
    return 0


def pubkey(args) -> int:
    """Восстановить публичную половину из приватной.

    Нужно, когда публичный ключ потерян или перезаписан: пересоздавать пару
    нельзя — это отрежет от обновлений все розданные образы.
    """
    Ed25519PrivateKey, serialization = _keys()
    key = Ed25519PrivateKey.from_private_bytes(Path(args.key).read_bytes())
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    print(public.hex())
    if args.install:
        PUBLIC_KEY_FILE.write_text(public.hex() + "\n")
        print(f"записан в {PUBLIC_KEY_FILE}", file=sys.stderr)
    return 0


def build_archive(version: str, out: Path) -> Path:
    """Pack the agent payload — the part an update replaces, and nothing else.

    The launcher is deliberately absent: it lives in the image and is never
    updated over the network, because a broken launcher shipped that way could
    not be fixed by shipping another one.

    Reproducible: the same sources give byte-identical output, and therefore
    the same signature. Otherwise re-running this to recover a signature you
    lost hands you one that does not match the archive you already have —
    which is exactly the trap it looks like it is not.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in (AGENT / "looma_agent").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for path in files:
            info = tar.gettarinfo(path, arcname=str(
                Path("looma_agent") / path.relative_to(AGENT / "looma_agent")))
            # Всё, что меняется от прогона к прогону, но не меняет содержимое.
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(path, "rb") as handle:
                tar.addfile(info, handle)
    with open(out, "wb") as sink:
        # mtime=0 и пустое имя: gzip пишет в свой заголовок и время, и имя
        # исходного файла, так что архив зависел бы ещё и от того, куда его
        # положили.
        with gzip.GzipFile(filename="", fileobj=sink, mode="wb", mtime=0) as zipped:
            zipped.write(raw.getvalue())
    return out


def sign(args) -> int:
    Ed25519PrivateKey, serialization = _keys()
    key = Ed25519PrivateKey.from_private_bytes(Path(args.key).read_bytes())
    archive = build_archive(args.version, Path(args.out or f"dist/looma-agent-{args.version}.tar.gz"))
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # The manifest is signed as a whole, not just the digest: signing bytes
    # alone would let a genuine old payload be served under a new version.
    manifest = json.dumps({"version": args.version, "sha256": digest},
                          sort_keys=True, separators=(",", ":")).encode()
    signature = key.sign(manifest).hex()

    # Рядом с архивом, а не только в терминале: подпись, отделённая от своего
    # файла, бесполезна, а вывод команды теряется при первом же git pull.
    manifest_path = archive.with_name(archive.name.replace(".tar.gz", "") + ".json")
    manifest_path.write_text(json.dumps(
        {"version": args.version, "sha256": digest, "signature": signature},
        indent=2) + "\n")

    print(f"archive   {archive}  ({len(payload)} bytes)")
    print(f"manifest  {manifest_path}  — его и загружайте в админке")
    print(f"version   {args.version}")
    print(f"sha256    {digest}")
    print(f"signature {signature}")
    print("\nВ админке, вкладка Release: сначала манифест, потом архив.\n"
          "Или запросом:\n"
          f"  curl -X POST http://<оркестратор>:8000/admin/release \\\n"
          f"    -H 'X-Looma-Admin-Token: <token>' -H 'Content-Type: application/json' \\\n"
          f"    -d \"$(python - <<'EOF'\n"
          f"import base64, json, pathlib\n"
          f"m = json.loads(pathlib.Path({str(manifest_path)!r}).read_text())\n"
          f"m['archive'] = base64.b64encode(pathlib.Path({str(archive)!r}).read_bytes()).decode()\n"
          f"print(json.dumps(m))\nEOF\n)\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("keygen", help="make a release key pair")
    generate.add_argument("--out", default="agent-release.key")
    generate.add_argument("--force", action="store_true",
                          help="перезаписать приватный ключ, если он уже есть")
    generate.add_argument("--install", action="store_true",
                          help="заменить публичный ключ в дереве — отрежет от "
                               "обновлений все уже розданные образы")
    generate.set_defaults(func=keygen)

    public = sub.add_parser("pubkey", help="публичная половина из приватной")
    public.add_argument("--key", required=True)
    public.add_argument("--install", action="store_true",
                        help="записать в agent/looma_launcher/release_key.pub")
    public.set_defaults(func=pubkey)

    signer = sub.add_parser("sign", help="pack and sign the agent payload")
    signer.add_argument("--key", required=True)
    signer.add_argument("--version", required=True)
    signer.add_argument("--out", default="")
    signer.set_defaults(func=sign)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
