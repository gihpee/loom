"""Run the agent, and keep running it.

The agent is a subprocess. When it exits, the launcher starts it again — the
node owner asked for a machine that stays connected, not for one that quits on
the first unhandled error.

It also installs what the agent downloaded and takes it back out again when it
does not work. "Stood up" means the agent reached the orchestrator and said so
by writing a health marker — not that the process is alive, because a payload
that starts, fails to connect and sits there is alive and useless.

The rollback decision is made HERE, on the node, without asking anyone: the
connection to the orchestrator may be exactly what the new version broke.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

from loom_launcher import payload as payload_mod
from loom_launcher.payload import Payload

logger = logging.getLogger("loom_launcher.supervise")

# Long enough that a crash loop does not spin the CPU, short enough that a
# transient failure does not take the node out of the fleet for minutes.
RESTART_DELAY_S = 3.0
# A restart this soon after start means the agent never really came up.
TOO_SOON_S = 30.0
# How many fast failures of a version that has never once registered before we
# put the previous one back. Two rather than one: a single crash can be the
# machine (a card gone, a full disk), and rolling back would not fix it.
FAILURES_BEFORE_ROLLBACK = 3
# Этим кодом агент говорит, что вышел нарочно — скачал обновление и уступает
# место. Обычный ноль от этого не отличить, а считать плановую остановку
# падением значит подвести исправную версию под откат.
UPDATE_EXIT_CODE = 70


def _why_updates_are_off() -> str:
    """Пусто, если обновления возможны."""
    from loom_launcher.signature import public_key_bytes

    if public_key_bytes() is not None:
        return ""
    return ("в образе этого узла нет ключа релизов, проверить подпись нечем — "
            "обновления по сети выключены. Лечится только пересборкой образа "
            "с ключом и docker pull на узле")


class Supervisor:
    def __init__(self, payload: Payload, agent_args: List[str]) -> None:
        self.payload = payload
        self.agent_args = agent_args
        self.consecutive_failures = 0
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()

    def run_forever(self) -> int:
        self._install_signal_handlers()
        logger.info("launcher starting agent %s", self.payload.describe())
        while not self._stop.is_set():
            self._apply_downloaded()
            started = time.monotonic()
            code = self._run_once()
            if self._stop.is_set():
                return code or 0
            lived = time.monotonic() - started
            if code == UPDATE_EXIT_CODE:
                logger.info("agent %s stepped aside for an update", self.payload.version)
                self.consecutive_failures = 0
                continue
            if lived < TOO_SOON_S:
                self.consecutive_failures += 1
                logger.warning(
                    "agent %s exited after %.1fs with code %s (%d in a row)",
                    self.payload.version, lived, code, self.consecutive_failures,
                )
                self._consider_rollback()
            else:
                self.consecutive_failures = 0
                logger.warning("agent exited with code %s after %.0fs", code, lived)
            time.sleep(RESTART_DELAY_S)
        return 0

    # ---------------------------------------------------------------- updates
    def _apply_downloaded(self) -> None:
        """Install whatever the agent fetched before it stopped.

        Verification happens inside install(). The agent downloads; it does not
        decide what may run — that is the whole reason this code is in the part
        an update cannot replace.
        """
        for manifest in payload_mod.pending():
            installed = payload_mod.install(
                manifest, installed_version=self.payload.version)
            if installed is None:
                continue
            payload_mod.switch_to(installed)
            self.payload = installed
            self.consecutive_failures = 0
            logger.info("now running agent %s", installed.describe())

    def _consider_rollback(self) -> None:
        if self.consecutive_failures < FAILURES_BEFORE_ROLLBACK:
            return
        if self.payload.bundled:
            # Already at the floor. Keep restarting: the problem is not the
            # payload, and there is nothing better to fall back to.
            return
        if payload_mod.health_marker(self.payload.version).exists():
            # It worked before, so the new thing is not what broke. Rolling
            # back would hide a real problem behind a version change.
            logger.error("agent %s keeps failing although it once registered; "
                         "not rolling back", self.payload.version)
            return
        logger.error("agent %s never registered and failed %d times; going back",
                     self.payload.version, self.consecutive_failures)
        restored = payload_mod.roll_back()
        if restored is not None:
            self.payload = restored
            self.consecutive_failures = 0

    def _run_once(self) -> Optional[int]:
        env = dict(os.environ)
        if self.payload.path is not None:
            # The payload's own directory wins over anything installed in the
            # image, so an update actually takes effect.
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{self.payload.path}{os.pathsep}{existing}" if existing else str(self.payload.path)
        env["LOOM_AGENT_VERSION"] = self.payload.version
        # Where the agent says it actually came up. Its absence after repeated
        # fast exits is what triggers a rollback.
        env["LOOM_AGENT_HEALTH_FILE"] = str(payload_mod.health_marker(self.payload.version))
        env["LOOM_AGENT_INCOMING"] = str(payload_mod.incoming_dir())
        # Образ без ключа не примет ни одного релиза. Сказать об этом агенту
        # сейчас — значит увидеть причину в панели; промолчать — значит
        # смотреть, как узел качает, сливается и перезапускается по кругу.
        blocked = _why_updates_are_off()
        if blocked:
            env["LOOM_UPDATES_DISABLED"] = blocked
        argv = [sys.executable, "-m", "loom_agent.main", *self.agent_args]
        self._proc = subprocess.Popen(argv, env=env, start_new_session=True)
        try:
            return self._proc.wait()
        except KeyboardInterrupt:
            return None

    # ------------------------------------------------------------- shutdown
    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, _frame) -> None:
        # `docker stop` sends SIGTERM to the launcher only. Without passing it
        # on, the agent would be killed by the timeout instead of shutting
        # down, and whatever it was doing would be lost rather than finished.
        logger.info("launcher got signal %s; stopping the agent", signum)
        self._stop.set()
        self._terminate_agent()

    def _terminate_agent(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("agent did not stop in time; killing it")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
