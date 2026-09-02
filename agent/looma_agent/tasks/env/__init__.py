"""What has to exist in a task's directory before its command can run.

An environment is expensive to build and cheap to reuse, so it does not live
with the task: it lives in a cache keyed by what was asked for, and outlives
every task that used it. See docs/WORKER_RUNTIME.md.
"""

from looma_agent.tasks.env.base import Environment, NO_ENVIRONMENT
from looma_agent.tasks.env.cache import EnvironmentCache

__all__ = ["Environment", "NO_ENVIRONMENT", "EnvironmentCache"]
