"""One stage of a model pipeline.

Not part of the agent. It is a payload: installed into a task's environment
when the orchestrator asks for inference, and absent from every node that was
never asked. That is the whole reason the agent image is small enough to pull
in seconds — see docs/WORKER_RUNTIME.md.

It talks to its neighbours by RANK through the agent's channel and has no idea
whether the next stage is on this machine or another continent.
"""
