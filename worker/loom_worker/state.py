"""Worker-local shard state. Pure bookkeeping, no decisions."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ShardStatus(str, Enum):
    LOADING = "loading"
    LOADED = "loaded"
    SERVING = "serving"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class PipelineRole:
    """This node's position in a multi-node pipeline (1 stage = whole model)."""

    pipeline_id: str = ""
    stage_index: int = 0
    num_stages: int = 1
    is_first: bool = True
    is_last: bool = True

    @property
    def is_multi_stage(self) -> bool:
        return self.num_stages > 1


@dataclass
class ShardSpec:
    model_id: str
    start_layer: int
    end_layer: int
    backend_type: str
    weights_uri: str
    vram_quota_bytes: int
    role: PipelineRole = field(default_factory=PipelineRole)


@dataclass
class ShardState:
    spec: ShardSpec
    status: ShardStatus = ShardStatus.LOADING
    backend: Optional[object] = None  # BackendAdapter
    endpoint_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WorkerState:
    node_id: str
    advertise_host: str
    shards: Dict[str, ShardState] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def get(self, model_id: str) -> Optional[ShardState]:
        with self.lock:
            return self.shards.get(model_id)

    def put(self, model_id: str, shard: ShardState) -> None:
        with self.lock:
            self.shards[model_id] = shard

    def pop(self, model_id: str) -> Optional[ShardState]:
        with self.lock:
            return self.shards.pop(model_id, None)

    def snapshot(self) -> Dict[str, ShardState]:
        with self.lock:
            return dict(self.shards)
