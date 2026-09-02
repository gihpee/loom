"""Looma — multi-model LLM inference marketplace.

Package layout:
- ``looma.planning`` — reusable scheduling library (Phase-1/Phase-2 DP) adapted
  from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
- ``looma.perfmap`` — centralized Perf-map Store (replacement for the DHT data
  source used by the original Parallax Phase-2 routing).
- ``looma.proto`` — control-plane protobuf contract (orchestrator -> worker).
"""

__version__ = "0.0.1"
