"""Orchestrator configuration: env + model catalog JSON.

Catalog file: {"models": [<ModelSpec dict>, ...]} or a bare list. Each entry =
serialized ModelInfo kwargs plus serving/market fields (see registry.ModelSpec).
A single-model file (Phase-1 style) is accepted via LOOM_MODEL_CONFIG and
wrapped into a one-entry catalog.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from loom.orchestrator.registry import ModelRegistry, ModelSpec


@dataclass
class OrchestratorConfig:
    registry: ModelRegistry
    grpc_port: int = 9000
    http_port: int = 8000
    # Address workers should dial; embedded into every issued join key.
    public_address: str = "127.0.0.1:9000"
    keystore_path: str = ""
    node_token: str = ""  # legacy/dev master token (optional)
    admin_token: str = ""
    perfmap_sync_interval_s: float = 5.0
    heartbeat_timeout_s: float = 30.0
    rebalance_interval_s: float = 60.0
    qps_per_pipeline: float = 10.0
    # Split of a granted VRAM quota between weights and KV cache. Raise
    # param_mem_ratio to fit a bigger model on a card (less KV headroom);
    # lower it for more concurrency per replica.
    param_mem_ratio: float = 0.6
    kvcache_mem_ratio: float = 0.3
    slo_check_interval_s: float = 10.0
    slo_window_s: float = 60.0
    slo_min_samples: int = 10
    slo_boost_factor: float = 2.0

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        if "LOOM_MODEL_CATALOG" in os.environ:
            registry = ModelRegistry.from_catalog_file(os.environ["LOOM_MODEL_CATALOG"])
        elif "LOOM_MODEL_CONFIG" in os.environ:
            raw = json.loads(Path(os.environ["LOOM_MODEL_CONFIG"]).read_text())
            registry = ModelRegistry([ModelSpec.from_dict(raw)])
        else:
            raise RuntimeError("set LOOM_MODEL_CATALOG or LOOM_MODEL_CONFIG")
        grpc_port = int(os.environ.get("LOOM_GRPC_PORT", "9000"))
        return cls(
            registry=registry,
            grpc_port=grpc_port,
            http_port=int(os.environ.get("LOOM_HTTP_PORT", "8000")),
            # Empty means "detect it" (see public_addr.resolve_public_address).
            public_address=os.environ.get("LOOM_PUBLIC_ADDR", ""),
            keystore_path=os.environ.get("LOOM_KEYSTORE_PATH", ""),
            node_token=os.environ.get("LOOM_NODE_TOKEN", ""),
            admin_token=os.environ.get("LOOM_ADMIN_TOKEN", ""),
            perfmap_sync_interval_s=float(os.environ.get("LOOM_PERFMAP_SYNC_S", "5")),
            heartbeat_timeout_s=float(os.environ.get("LOOM_HEARTBEAT_TIMEOUT_S", "30")),
            rebalance_interval_s=float(os.environ.get("LOOM_REBALANCE_S", "60")),
            qps_per_pipeline=float(os.environ.get("LOOM_QPS_PER_PIPELINE", "10")),
            param_mem_ratio=float(os.environ.get("LOOM_PARAM_MEM_RATIO", "0.6")),
            kvcache_mem_ratio=float(os.environ.get("LOOM_KVCACHE_MEM_RATIO", "0.3")),
            slo_check_interval_s=float(os.environ.get("LOOM_SLO_CHECK_S", "10")),
            slo_window_s=float(os.environ.get("LOOM_SLO_WINDOW_S", "60")),
            slo_min_samples=int(os.environ.get("LOOM_SLO_MIN_SAMPLES", "10")),
            slo_boost_factor=float(os.environ.get("LOOM_SLO_BOOST", "2.0")),
        )
