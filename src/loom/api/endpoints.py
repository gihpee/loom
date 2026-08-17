# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/router/main.py — Endpoint, EndpointMetrics, EndpointRegistry
# (регистрация/выбор эндпоинтов + метрики inflight/EMA).
# Изменения (точечная правка по ТЗ):
#   1. Добавлено обязательное поле `model_id` в Endpoint и все операции
#      реестра; select()/candidates() матчат запрос ТОЛЬКО на эндпоинты той же
#      модели (в оригинале реестр — балансировщик одной модели без поля model).
#   2. Ключ реестра — (model_id, base_url) вместо base_url.
#   3. Обрезано под Loom v0: убраны HTTP-конфигурация балансировщика на лету,
#      троттлинг-бакеты и httpx-клиент (метрики пишутся контроллером/прокси
#      напрямую); скоринг/выбор делегируется неизменённым стратегиям из
#      loom/api/lb_strategy.py.
"""Model-aware endpoint registry for the Loom client API."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loom.api.lb_strategy import PerformanceConfig, StrategyName, make_strategy


@dataclass
class EndpointMetrics:
    inflight: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_error_ts: Optional[float] = None

    # Exponential moving averages in milliseconds
    ema_ttft_ms: Optional[float] = None
    ema_tpot_ms: Optional[float] = None

    # Capacity hint from downstream status.
    max_running_request: Optional[int] = None


@dataclass
class Endpoint:
    endpoint_id: str
    model_id: str
    base_url: str
    node_id: str = ""
    enabled: bool = True
    created_ts: float = field(default_factory=time.time)
    metrics: EndpointMetrics = field(default_factory=EndpointMetrics)


class EndpointRegistry:
    """Registry keyed by (model_id, base_url); selection is per-model only."""

    def __init__(
        self,
        *,
        strategy: StrategyName = "round_robin",
        performance_cfg: Optional[PerformanceConfig] = None,
        ema_alpha: float = 0.1,
    ) -> None:
        self._lock = threading.RLock()
        self._endpoints: Dict[Tuple[str, str], Endpoint] = {}
        self._strategy = make_strategy(
            strategy, performance_cfg=performance_cfg or PerformanceConfig()
        )
        self._ema_alpha = ema_alpha

    # Accepted schemes: http/https for directly dialable backends (tests, local
    # dev) and tunnel:// for backends reachable only through the data plane.
    _SCHEMES = ("http://", "https://", "tunnel://")

    def register(self, *, model_id: str, base_url: str, node_id: str = "") -> Endpoint:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(self._SCHEMES):
            raise ValueError(f"base_url must start with one of {self._SCHEMES}")
        with self._lock:
            key = (model_id, base_url)
            existing = self._endpoints.get(key)
            if existing is not None:
                existing.enabled = True
                if node_id:
                    existing.node_id = node_id
                return existing
            ep = Endpoint(
                endpoint_id=str(uuid.uuid4()),
                model_id=model_id,
                base_url=base_url,
                node_id=node_id,
            )
            self._endpoints[key] = ep
            return ep

    def unregister(self, *, model_id: str, base_url: Optional[str] = None, node_id: Optional[str] = None) -> int:
        """Remove endpoints of a model by url or node; both None removes all of the model."""
        removed = 0
        with self._lock:
            for key in list(self._endpoints.keys()):
                mid, url = key
                if mid != model_id:
                    continue
                ep = self._endpoints[key]
                if base_url is not None and url != base_url.strip().rstrip("/"):
                    continue
                if node_id is not None and ep.node_id != node_id:
                    continue
                del self._endpoints[key]
                removed += 1
        return removed

    def candidates(self, model_id: str) -> List[Endpoint]:
        """Endpoints eligible for a request: SAME model only, enabled."""
        with self._lock:
            return [
                ep
                for (mid, _), ep in self._endpoints.items()
                if mid == model_id and ep.enabled
            ]

    def select(self, model_id: str) -> Optional[Endpoint]:
        candidates = self.candidates(model_id)
        if not candidates:
            return None
        return self._strategy.select(candidates)

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted({mid for (mid, _) in self._endpoints.keys()})

    def list_endpoints(self, model_id: Optional[str] = None) -> List[Endpoint]:
        with self._lock:
            return [
                ep
                for (mid, _), ep in self._endpoints.items()
                if model_id is None or mid == model_id
            ]

    # --- metrics hooks used by the API proxy --------------------------------
    def mark_request_start(self, ep: Endpoint) -> None:
        with self._lock:
            ep.metrics.inflight += 1
            ep.metrics.total_requests += 1

    def mark_request_end(self, ep: Endpoint, *, error: bool = False, ttft_ms: Optional[float] = None) -> None:
        with self._lock:
            ep.metrics.inflight = max(0, ep.metrics.inflight - 1)
            if error:
                ep.metrics.total_errors += 1
                ep.metrics.last_error_ts = time.time()
            if ttft_ms is not None:
                prev = ep.metrics.ema_ttft_ms
                ep.metrics.ema_ttft_ms = (
                    ttft_ms if prev is None else (1 - self._ema_alpha) * prev + self._ema_alpha * ttft_ms
                )
