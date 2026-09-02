"""Moving bytes in and out of a node that nothing can dial into.

Everything travels on the connection the agent opened: no inbound port, no
second protocol, no separate credentials to leak.
"""
