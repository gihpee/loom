"""Running work on somebody else's machine.

Everything the orchestrator asks for — inference, training, a client's own
code — arrives here as a task: a directory, an environment provisioned into it,
and a process. See docs/WORKER_RUNTIME.md.
"""

from looma_agent.tasks.spec import EnvSpec, Resources, TaskRefused, TaskSpec

__all__ = ["EnvSpec", "Resources", "TaskSpec", "TaskRefused"]
