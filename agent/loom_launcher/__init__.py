"""The launcher: the only part of the node that a network update cannot touch.

It lives in the image, it is small, and it stays that way on purpose. Its job
is to find an agent payload, check it is ours, run it, and put the previous one
back if the new one cannot stand up.

Nothing here imports the agent. The agent is a SUBPROCESS, which is the whole
reason it can be replaced while the container keeps running.
"""
