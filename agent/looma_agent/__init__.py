"""Looma node agent.

The process that connects a machine to Looma. It holds no models and runs no
computation itself: work arrives as a task, gets its own directory and its own
environment, and runs there. See docs/WORKER_RUNTIME.md.
"""

import os

# The launcher tells us which payload we are. Falls back to the packaged
# version when the agent is started directly, e.g. in tests.
__version__ = os.environ.get("LOOMA_AGENT_VERSION") or "0.1.0"
