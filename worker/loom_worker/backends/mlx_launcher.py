"""Wrapper entrypoint: apply the MLX memory limit, then run mlx_lm.server.

Split out as a module so the limit is set inside the SAME process that runs
the model (subprocess env cannot carry a Python-API-only setting).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--memory-limit-bytes", type=int, default=0)
    args, passthrough = parser.parse_known_args()

    import mlx.core as mx

    if args.memory_limit_bytes > 0:
        mx.set_memory_limit(args.memory_limit_bytes)

    # Re-enter mlx_lm.server's own CLI with the serving arguments.
    sys.argv = [
        "mlx_lm.server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--host",
        "0.0.0.0",
        *passthrough,
    ]
    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
