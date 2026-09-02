"""Looma orchestrator v0: single model, direct use of the planning library.

Composition:
- ``gateway``   — gRPC ControlGateway server (workers dial in, bidi stream)
- ``controller``— single-model control logic: join -> Phase-1 -> LoadShard/
                  StartServing; telemetry -> Perf-map -> Phase-2
- ``config``    — model + runtime configuration (env / JSON)
"""
