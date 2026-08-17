"""Resource Broker: slices the shared node pool between models (offline pass).

Implements the greedy first-fit-decreasing algorithm from the specification:

    score(model) = priority(model) * price_willing(model) * demand_qps(model)
    sort models by score desc
    for each model:
        group remaining pool by region
        for each region (by free VRAM desc):
            greedily take VRAM until k pipelines fit
    if a model gets no full pipeline: reduce k (down to 1), else leave
    unscheduled (cross-region bridging with a latency penalty is a later step).

Deliberately greedy — no ILP/LP. Upgrade only after profiling shows it fails
on the target pool size.

Extension over the literal pseudo-code (needed for multi-process isolation on
one GPU): a node is consumed *by bytes*, not whole — leftover VRAM stays in
the pool for lower-score models, which is exactly what "several backend
processes with VRAM quotas on one physical GPU" requires.

The broker only grants (node, vram_quota) sub-pools; the actual pipeline
construction is done by the UNMODIFIED Phase-1 DP per model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loom.logging_config import get_logger
from loom.orchestrator.registry import ModelSpec
from loom.planning import ModelInfo

logger = get_logger(__name__)

GIB = 1024**3


@dataclass
class PoolNode:
    """Broker's view of one physical node."""

    node_id: str
    region: str
    vram_free_bytes: int
    tflops_fp16: float = 1.0


@dataclass
class BrokerPlan:
    """model_id -> {node_id -> vram_quota_bytes}; empty dict = unscheduled."""

    allocations: Dict[str, Dict[str, int]] = field(default_factory=dict)
    unscheduled: List[str] = field(default_factory=list)

    def quota(self, model_id: str, node_id: str) -> int:
        return self.allocations.get(model_id, {}).get(node_id, 0)


def pipeline_vram_bytes(model_info: ModelInfo, *, param_mem_ratio: float = 0.5) -> int:
    """VRAM needed for ONE full pipeline of the model.

    Inverse of the ShardCapacity formula: the param budget must fit all decoder
    layers plus both endpoints (embedding + LM head unless tied).
    """
    per_layer = model_info.decoder_layer_io_bytes(roofline=False)
    endpoints = model_info.embedding_io_bytes * (1 if model_info.tie_embedding else 2)
    param_bytes = model_info.num_layers * per_layer + endpoints
    return math.ceil(param_bytes / param_mem_ratio)


class ResourceBroker:
    def __init__(
        self,
        *,
        qps_per_pipeline: float = 10.0,
        param_mem_ratio: float = 0.5,
        max_pipelines_per_model: int = 8,
    ) -> None:
        self.qps_per_pipeline = qps_per_pipeline
        self.param_mem_ratio = param_mem_ratio
        self.max_pipelines_per_model = max_pipelines_per_model

    def target_pipelines(self, spec: ModelSpec) -> int:
        """k for a model: explicit override, else derived from demand."""
        if spec.target_pipelines > 0:
            return min(spec.target_pipelines, self.max_pipelines_per_model)
        k = math.ceil(spec.demand_qps / self.qps_per_pipeline)
        return max(1, min(k, self.max_pipelines_per_model))

    def min_slice_bytes(self, model_info: ModelInfo) -> int:
        """Smallest useful grant: one decoder layer's worth of param budget."""
        per_layer = model_info.decoder_layer_io_bytes(roofline=False)
        return math.ceil(per_layer / self.param_mem_ratio)

    def layers_fitting(
        self, quota_bytes: int, model_info: ModelInfo, *, endpoint_bytes: int = 0
    ) -> int:
        """How many decoder layers a quota holds — same floor() as Phase-1.

        Counting in LAYERS (not bytes) is what makes multi-stage grants usable:
        each node rounds down independently, so a grant of exactly
        `pipeline_vram_bytes` split over N nodes loses up to N-1 layers to
        rounding and Phase-1 would fail to cover the model.
        """
        per_layer = model_info.decoder_layer_io_bytes(roofline=False)
        budget = math.floor(quota_bytes * self.param_mem_ratio) - endpoint_bytes
        if budget <= 0:
            return 0
        return max(0, math.floor(budget / per_layer))

    def bytes_for_layers(
        self, num_layers: int, model_info: ModelInfo, *, endpoint_bytes: int = 0
    ) -> int:
        """Inverse of `layers_fitting`: quota needed to hold `num_layers`."""
        per_layer = model_info.decoder_layer_io_bytes(roofline=False)
        return math.ceil((num_layers * per_layer + endpoint_bytes) / self.param_mem_ratio)

    def plan(
        self,
        nodes: List[PoolNode],
        models: List[ModelSpec],
        previous: Optional[Dict[str, Dict[str, int]]] = None,
        score_boosts: Optional[Dict[str, float]] = None,
    ) -> BrokerPlan:
        """Slice the pool.

        `previous` (model -> {node -> quota}) enables stickiness: all else
        being equal a model stays on the nodes it already occupies.
        `score_boosts` (model -> multiplier) is the SLO hook: a model
        violating its SLO gets its score boosted for THIS pass (and one extra
        target pipeline), so it wins resources at the next rebalance.
        """
        free: Dict[str, int] = {n.node_id: int(n.vram_free_bytes) for n in nodes}
        node_by_id = {n.node_id: n for n in nodes}
        previous = previous or {}
        score_boosts = score_boosts or {}
        plan = BrokerPlan()

        ordered = sorted(
            models,
            key=lambda s: (-s.score() * score_boosts.get(s.model_id, 1.0), s.model_id),
        )
        for spec in ordered:
            need = pipeline_vram_bytes(spec.model_info, param_mem_ratio=self.param_mem_ratio)
            min_slice = self.min_slice_bytes(spec.model_info)
            k = self.target_pipelines(spec)
            if score_boosts.get(spec.model_id, 1.0) > 1.0:
                k = min(k + 1, self.max_pipelines_per_model)
            sticky = set(previous.get(spec.model_id, {}))

            granted: Dict[str, int] = {}
            pipelines_got = 0
            # Try k pipelines, then degrade k down to 1 (per spec: "снизить k").
            while pipelines_got < k:
                takes = self._allocate_one_pipeline(
                    free,
                    node_by_id,
                    need=need,
                    min_slice=min_slice,
                    sticky=sticky,
                    model_info=spec.model_info,
                )
                if takes is None:
                    break
                for node_id, take in takes.items():
                    free[node_id] -= take
                    granted[node_id] = granted.get(node_id, 0) + take
                pipelines_got += 1

            if pipelines_got == 0:
                logger.warning(
                    "[Broker] model %s (score=%.2f) unscheduled: no region fits %.2f GiB",
                    spec.model_id,
                    spec.score(),
                    need / GIB,
                )
                plan.unscheduled.append(spec.model_id)
                continue
            if pipelines_got < k:
                logger.info(
                    "[Broker] model %s: degraded to %d/%d pipelines",
                    spec.model_id,
                    pipelines_got,
                    k,
                )
            plan.allocations[spec.model_id] = granted
        return plan

    def _allocate_one_pipeline(
        self,
        free: Dict[str, int],
        node_by_id: Dict[str, PoolNode],
        *,
        need: int,
        min_slice: int,
        sticky: Optional[set] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> Optional[Dict[str, int]]:
        """FFD one pipeline inside a single region; None if no region fits."""
        # Stickiness: first try to fit the pipeline ENTIRELY on the nodes the
        # model already occupies (no reshuffle). A partial overlap is worse
        # than none — it splits the pipeline into slivers — so on failure we
        # fall back to plain FFD with no sticky bias at all.
        if sticky:
            result = self._ffd_pass(
                free,
                node_by_id,
                need=need,
                min_slice=min_slice,
                allowed=sticky,
                model_info=model_info,
            )
            if result is not None:
                return result
        return self._ffd_pass(
            free, node_by_id, need=need, min_slice=min_slice, allowed=None, model_info=model_info
        )

    def _ffd_pass(
        self,
        free: Dict[str, int],
        node_by_id: Dict[str, PoolNode],
        *,
        need: int,
        min_slice: int,
        allowed: Optional[set],
        model_info: Optional[ModelInfo] = None,
    ) -> Optional[Dict[str, int]]:
        # Group live free capacity by region.
        regions: Dict[str, List[str]] = {}
        for node_id, f in free.items():
            if f >= min_slice and (allowed is None or node_id in allowed):
                regions.setdefault(node_by_id[node_id].region, []).append(node_id)

        # Regions by total free VRAM desc (spec: "по убыв. свободного VRAM").
        region_order = sorted(
            regions.keys(), key=lambda r: -sum(free[nid] for nid in regions[r])
        )
        for region in region_order:
            node_ids = sorted(regions[region], key=lambda nid: -free[nid])  # FFD
            takes = (
                self._take_by_layers(free, node_ids, model_info)
                if model_info is not None
                else self._take_by_bytes(free, node_ids, need=need, min_slice=min_slice)
            )
            if takes:
                return takes
        return None

    def _take_by_bytes(
        self, free: Dict[str, int], node_ids: List[str], *, need: int, min_slice: int
    ) -> Optional[Dict[str, int]]:
        takes: Dict[str, int] = {}
        remaining = need
        for nid in node_ids:
            if remaining <= 0:
                break
            take = min(free[nid], remaining)
            if take < min_slice and take < remaining:
                continue  # useless sliver on this node
            takes[nid] = take
            remaining -= take
        return takes if remaining <= 0 else None

    def _take_by_layers(
        self, free: Dict[str, int], node_ids: List[str], model_info: ModelInfo
    ) -> Optional[Dict[str, int]]:
        """Grant whole layers per node, mirroring Phase-1's own accounting.

        The first node also pays for the embedding matrix and the last one for
        the LM head, so the grant a node needs depends on its position in the
        chain — which is why this cannot be done in bytes alone.
        """
        total_layers = int(model_info.num_layers)
        embed = model_info.embedding_io_bytes
        head = 0 if model_info.tie_embedding else embed
        takes: Dict[str, int] = {}
        layers_left = total_layers

        for position, nid in enumerate(node_ids):
            if layers_left <= 0:
                break
            endpoint_cost = embed if position == 0 else 0
            # Assume this node might close the chain: reserve the LM head too,
            # so a single-node grant is never short.
            closing_cost = endpoint_cost + head
            layers_if_closing = self.layers_fitting(
                free[nid], model_info, endpoint_bytes=closing_cost
            )
            if layers_if_closing >= layers_left:
                takes[nid] = min(
                    free[nid],
                    self.bytes_for_layers(layers_left, model_info, endpoint_bytes=closing_cost),
                )
                layers_left = 0
                break
            layers_here = self.layers_fitting(
                free[nid], model_info, endpoint_bytes=endpoint_cost
            )
            if layers_here <= 0:
                continue
            takes[nid] = min(
                free[nid],
                self.bytes_for_layers(layers_here, model_info, endpoint_bytes=endpoint_cost),
            )
            layers_left -= layers_here

        return takes if layers_left <= 0 else None
