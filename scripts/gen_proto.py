#!/usr/bin/env python3
"""Generate the gRPC stubs both sides speak, from the one .proto that defines them.

    uv run --with grpcio-tools python scripts/gen_proto.py

Generated files are committed so neither runtime image needs protoc, and so a
change to the contract between the orchestrator and its nodes shows up in a
diff — which it should, because it is a change to what two machines have agreed.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "src" / "looma" / "proto"
PROTOS = ["agent.proto"]
TARGETS = [
    ROOT / "src" / "looma" / "proto_gen",        # orchestrator side
    ROOT / "agent" / "looma_agent" / "proto",    # node side, self-contained
]


def relativize_imports(path: Path) -> None:
    """`import x_pb2` -> `from . import x_pb2`.

    protoc writes a flat import that only resolves when the files sit directly
    on sys.path. Rewriting it makes the directory an ordinary package.
    """
    text = path.read_text()
    text = re.sub(r"^import (\w+_pb2)( as [\w.]+)?$", r"from . import \1\2",
                  text, flags=re.MULTILINE)
    path.write_text(text)


def main() -> int:
    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={target}",
            f"--grpc_python_out={target}",
            *[str(PROTO_DIR / name) for name in PROTOS],
        ], check=True)
        for generated in target.glob("*_pb2*.py"):
            relativize_imports(generated)
        (target / "__init__.py").write_text(
            '"""Generated gRPC stubs (scripts/gen_proto.py). Do not edit."""\n')
        print(f"generated stubs in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
