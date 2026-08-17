#!/usr/bin/env python3
"""Generate python gRPC stubs from src/loom/proto into both consumers:

- src/loom/proto_gen/          (orchestrator side, package loom.proto_gen)
- worker/loom_worker/proto/    (worker side, self-contained copy)

Generated files are committed so neither runtime image needs protoc.
Run: uv run --with grpcio-tools python scripts/gen_proto.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "src" / "loom" / "proto"
TARGETS = [ROOT / "src" / "loom" / "proto_gen", ROOT / "worker" / "loom_worker" / "proto"]
PROTOS = ["worker_control.proto", "gateway.proto", "dataplane.proto"]


def relativize_imports(path: Path) -> None:
    """Rewrite `import x_pb2 ...` to package-relative imports."""
    text = path.read_text()
    text = re.sub(
        r"^import (\w+_pb2)( as [\w.]+)?$",
        r"from . import \1\2",
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def main() -> int:
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={target}",
            f"--grpc_python_out={target}",
            *[str(PROTO_DIR / p) for p in PROTOS],
        ]
        subprocess.run(cmd, check=True)
        for f in target.glob("*_pb2*.py"):
            relativize_imports(f)
        init = target / "__init__.py"
        init.write_text(
            '"""Generated gRPC stubs (scripts/gen_proto.py). Do not edit."""\n'
        )
        print(f"generated stubs in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
