"""Scheduler Pool: one UNMODIFIED planning Scheduler instance per model.

Each instance runs on the sub-pool granted by the Resource Broker, with node
capacities built from the granted VRAM quotas (the explicit capacity input of
Phase-1). Rebalance v0 rebuilds an instance from its current grant — full
re-placement per broker pass; incremental joins land later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loom.logging_config import get_logger
from loom.orchestrator.registry import ModelSpec
from loom.planning import Node, NodeHardwareInfo, Scheduler, ShardCapacity

logger = get_logger(__name__)


@dataclass
class NodeDescriptor:
    """Orchestrator-side record of a registered worker (auto-detected data)."""

    node_id: str
    region: str
    hardware: NodeHardwareInfo
    vram_free_bytes: int  # schedulable now — what the broker may hand out
    vram_total_bytes: int = 0  # device total — backends need it for quota math
    detection_source: str = ""
    agent_version: str = ""


@dataclass
class ModelInstance:
    spec: ModelSpec
    scheduler: Scheduler
    # node_id -> granted quota used to build this instance
    grants: Dict[str, int]


class SchedulerPool:
    def __init__(self, *, param_mem_ratio: float = 0.6, kvcache_mem_ratio: float = 0.3) -> None:
        self._instances: Dict[str, ModelInstance] = {}
        # Must match the broker's split, otherwise Phase-1 would size shards
        # against a different weights budget than the one that was granted.
        self.param_mem_ratio = param_mem_ratio
        self.kvcache_mem_ratio = kvcache_mem_ratio

    def get(self, model_id: str) -> Optional[ModelInstance]:
        return self._instances.get(model_id)

    def model_ids(self) -> List[str]:
        return list(self._instances.keys())

    def drop(self, model_id: str) -> None:
        self._instances.pop(model_id, None)

    def rebuild(
        self,
        spec: ModelSpec,
        grants: Dict[str, int],
        nodes: Dict[str, NodeDescriptor],
    ) -> ModelInstance:
        """(Re)create the per-model scheduler from broker grants and bootstrap it."""
        planning_nodes: List[Node] = []
        for node_id, quota in grants.items():
            desc = nodes.get(node_id)
            if desc is None:
                continue
            capacity = ShardCapacity.from_model_info(
                spec.model_info,
                vram_quota_bytes=quota,
                device="mlx" if desc.hardware.device == "mlx" else "cuda",
                param_mem_ratio=self.param_mem_ratio,
                kvcache_mem_ratio=self.kvcache_mem_ratio,
            )
            planning_nodes.append(
                Node(
                    node_id=node_id,
                    hardware=desc.hardware,
                    model_info=spec.model_info,
                    capacity=capacity,
                )
            )
        scheduler = Scheduler(
            spec.model_info,
            planning_nodes,
            min_nodes_bootstrapping=1,
            routing_strategy="dp",
        )
        ok = scheduler.bootstrap()
        if not ok:
            logger.warning("[Pool] bootstrap failed for model %s on grants %s", spec.model_id, grants)
        instance = ModelInstance(spec=spec, scheduler=scheduler, grants=dict(grants))
        self._instances[spec.model_id] = instance
        return instance

    def shard_plan(self, model_id: str) -> List[Tuple[str, int, int]]:
        """Current (node_id, start_layer, end_layer) for a model, [] if none."""
        instance = self._instances.get(model_id)
        if instance is None:
            return []
        return instance.scheduler.list_node_allocations()
