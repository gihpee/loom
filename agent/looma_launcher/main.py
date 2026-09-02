"""Container entry point.

Everything the launcher does in one screen: work out which agent to run, make
sure the data directory exists, then hand over and stay out of the way.

Arguments are NOT parsed here. They belong to the agent, and the agent is what
gets replaced by an update — a launcher that understood them would have to be
updated every time they changed, which is exactly what it must not need.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from looma_launcher import payload as payload_mod
from looma_launcher.supervise import Supervisor


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOOMA_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _prepare_root() -> Path:
    root = payload_mod.root()
    for sub in ("agent", "tasks", "envs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def main(argv=None) -> int:
    _setup_logging()
    logger = logging.getLogger("looma_launcher")
    root = _prepare_root()
    chosen = payload_mod.resolve()
    logger.info("data root %s, agent payload %s", root, chosen.describe())
    args = list(sys.argv[1:] if argv is None else argv)
    return Supervisor(chosen, args).run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
