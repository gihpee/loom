"""Model Registry: CRUD catalog of models with their ModelInfo and market fields.

Phase 2 keeps it in-memory with a JSON catalog bootstrap; the Postgres `models`
table (perfmap/schema.sql) is the durable home once persistence lands.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from loom.planning import ModelInfo


@dataclass
class ModelSpec:
    """Catalog entry: serving config + market/priority signals for the broker."""

    model_id: str
    weights_uri: str
    backend_type: str
    model_info: ModelInfo
    demand_qps: float = 1.0
    priority: float = 1.0
    price_willing: float = 1.0
    # Explicit override for the number of pipelines the broker should target;
    # 0 = derive from demand_qps (see ResourceBroker.target_pipelines).
    target_pipelines: int = 0
    # SLO: p95 TTFT threshold in ms; None disables SLO monitoring for the model.
    slo_p95_ttft_ms: Optional[float] = None

    def score(self) -> float:
        """Broker ordering: score = priority * price_willing * demand_qps."""
        return self.priority * self.price_willing * self.demand_qps

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelSpec":
        return cls(
            model_id=raw["model_id"],
            weights_uri=raw["weights_uri"],
            backend_type=raw["backend_type"],
            model_info=ModelInfo(**raw["model_info"]),
            demand_qps=float(raw.get("demand_qps", 1.0)),
            priority=float(raw.get("priority", 1.0)),
            price_willing=float(raw.get("price_willing", 1.0)),
            target_pipelines=int(raw.get("target_pipelines", 0)),
            slo_p95_ttft_ms=(
                float(raw["slo_p95_ttft_ms"]) if raw.get("slo_p95_ttft_ms") is not None else None
            ),
        )


class ModelRegistry:
    """Thread-safe model catalog."""

    def __init__(self, specs: Optional[List[ModelSpec]] = None) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, ModelSpec] = {}
        for spec in specs or []:
            self._models[spec.model_id] = spec

    @classmethod
    def from_catalog_file(cls, path: str | Path) -> "ModelRegistry":
        raw = json.loads(Path(path).read_text())
        models = raw["models"] if isinstance(raw, dict) else raw
        return cls([ModelSpec.from_dict(m) for m in models])

    def add(self, spec: ModelSpec) -> None:
        with self._lock:
            self._models[spec.model_id] = spec

    def remove(self, model_id: str) -> Optional[ModelSpec]:
        with self._lock:
            return self._models.pop(model_id, None)

    def get(self, model_id: str) -> Optional[ModelSpec]:
        with self._lock:
            return self._models.get(model_id)

    def list(self) -> List[ModelSpec]:
        with self._lock:
            return list(self._models.values())

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())
