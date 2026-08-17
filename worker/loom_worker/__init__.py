"""Loom worker: executes orchestrator push-commands, contains no planning logic.

Self-contained package: must not import anything from the orchestrator/planning
codebase. The only shared artifact is the protobuf contract (generated stubs
in ``loom_worker.proto``).
"""

__version__ = "0.0.1"
