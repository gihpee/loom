"""Perf-map Store: centralized live-performance data keyed by model_id.

Own code (no borrowed Parallax logic). Schema:

- ``(model_id, node_id) -> ShardPerf {start_layer, end_layer, latency_ms,
  current_requests, is_healthy, updated_at}``
- ``(src_node_id, dst_node_id) -> rtt_ms`` (model-agnostic)

Redis key layout (used by :class:`RedisPerfMapStore`):

- ``loom:perf:{model_id}:{node_id}`` — HASH with ShardPerf fields
- ``loom:perf:{model_id}`` — SET of node_ids that have entries
- ``loom:rtt:{src}:{dst}`` — STRING (float ms), with TTL

Entries older than ``ttl_seconds`` are treated as missing: a worker that
stopped reporting must not keep serving stale latencies to Phase-2 routing.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ShardPerf:
    """Live performance snapshot of one model shard on one node."""

    model_id: str
    node_id: str
    start_layer: int
    end_layer: int
    latency_ms: Optional[float] = None
    current_requests: int = 0
    is_healthy: bool = True
    updated_at: float = field(default_factory=time.time)


class PerfMapStore(ABC):
    """Interface for the centralized perf-map storage."""

    @abstractmethod
    def upsert_shard_perf(self, perf: ShardPerf) -> None:
        """Insert or update the live perf entry for (model_id, node_id)."""

    @abstractmethod
    def get_shard_perf(self, model_id: str) -> List[ShardPerf]:
        """Return all non-stale perf entries for a model."""

    @abstractmethod
    def delete_shard_perf(self, model_id: str, node_id: str) -> None:
        """Remove the perf entry for (model_id, node_id)."""

    @abstractmethod
    def upsert_rtt(self, src_node_id: str, dst_node_id: str, rtt_ms: float) -> None:
        """Insert or update a pairwise RTT measurement."""

    @abstractmethod
    def get_rtt_map(self, src_node_id: str) -> Dict[str, float]:
        """Return {dst_node_id: rtt_ms} for all non-stale RTTs from a node."""


class InMemoryPerfMapStore(PerfMapStore):
    """Thread-safe in-memory implementation (tests, single-process demos)."""

    def __init__(self, *, ttl_seconds: float = 60.0) -> None:
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._perf: Dict[Tuple[str, str], ShardPerf] = {}
        self._rtt: Dict[Tuple[str, str], Tuple[float, float]] = {}  # -> (rtt_ms, ts)

    def _fresh(self, ts: float) -> bool:
        return (time.time() - ts) <= self._ttl

    def upsert_shard_perf(self, perf: ShardPerf) -> None:
        with self._lock:
            self._perf[(perf.model_id, perf.node_id)] = perf

    def get_shard_perf(self, model_id: str) -> List[ShardPerf]:
        with self._lock:
            return [
                p
                for (mid, _), p in self._perf.items()
                if mid == model_id and self._fresh(p.updated_at)
            ]

    def delete_shard_perf(self, model_id: str, node_id: str) -> None:
        with self._lock:
            self._perf.pop((model_id, node_id), None)

    def upsert_rtt(self, src_node_id: str, dst_node_id: str, rtt_ms: float) -> None:
        with self._lock:
            self._rtt[(src_node_id, dst_node_id)] = (float(rtt_ms), time.time())

    def get_rtt_map(self, src_node_id: str) -> Dict[str, float]:
        with self._lock:
            return {
                dst: rtt
                for (src, dst), (rtt, ts) in self._rtt.items()
                if src == src_node_id and self._fresh(ts)
            }


class RedisPerfMapStore(PerfMapStore):
    """Redis-backed implementation. Requires the ``redis`` extra."""

    def __init__(self, redis_client, *, ttl_seconds: float = 60.0, prefix: str = "loom") -> None:
        self._r = redis_client
        self._ttl = int(ttl_seconds)
        self._prefix = prefix

    def _perf_key(self, model_id: str, node_id: str) -> str:
        return f"{self._prefix}:perf:{model_id}:{node_id}"

    def _perf_index_key(self, model_id: str) -> str:
        return f"{self._prefix}:perf:{model_id}"

    def _rtt_key(self, src: str, dst: str) -> str:
        return f"{self._prefix}:rtt:{src}:{dst}"

    def upsert_shard_perf(self, perf: ShardPerf) -> None:
        key = self._perf_key(perf.model_id, perf.node_id)
        pipe = self._r.pipeline()
        pipe.hset(
            key,
            mapping={
                "start_layer": perf.start_layer,
                "end_layer": perf.end_layer,
                "latency_ms": "" if perf.latency_ms is None else perf.latency_ms,
                "current_requests": perf.current_requests,
                "is_healthy": int(perf.is_healthy),
                "updated_at": perf.updated_at,
            },
        )
        pipe.expire(key, self._ttl)
        pipe.sadd(self._perf_index_key(perf.model_id), perf.node_id)
        pipe.execute()

    def get_shard_perf(self, model_id: str) -> List[ShardPerf]:
        node_ids = [
            n.decode() if isinstance(n, bytes) else n
            for n in self._r.smembers(self._perf_index_key(model_id))
        ]
        out: List[ShardPerf] = []
        stale: List[str] = []
        for nid in node_ids:
            raw = self._r.hgetall(self._perf_key(model_id, nid))
            if not raw:
                stale.append(nid)
                continue
            d = {
                (k.decode() if isinstance(k, bytes) else k): (
                    v.decode() if isinstance(v, bytes) else v
                )
                for k, v in raw.items()
            }
            out.append(
                ShardPerf(
                    model_id=model_id,
                    node_id=nid,
                    start_layer=int(d["start_layer"]),
                    end_layer=int(d["end_layer"]),
                    latency_ms=float(d["latency_ms"]) if d.get("latency_ms") else None,
                    current_requests=int(d.get("current_requests", 0)),
                    is_healthy=bool(int(d.get("is_healthy", 1))),
                    updated_at=float(d.get("updated_at", 0.0)),
                )
            )
        if stale:
            self._r.srem(self._perf_index_key(model_id), *stale)
        return out

    def delete_shard_perf(self, model_id: str, node_id: str) -> None:
        pipe = self._r.pipeline()
        pipe.delete(self._perf_key(model_id, node_id))
        pipe.srem(self._perf_index_key(model_id), node_id)
        pipe.execute()

    def upsert_rtt(self, src_node_id: str, dst_node_id: str, rtt_ms: float) -> None:
        self._r.set(self._rtt_key(src_node_id, dst_node_id), float(rtt_ms), ex=self._ttl)

    def get_rtt_map(self, src_node_id: str) -> Dict[str, float]:
        pattern = self._rtt_key(src_node_id, "*")
        prefix_len = len(self._rtt_key(src_node_id, ""))
        out: Dict[str, float] = {}
        for key in self._r.scan_iter(match=pattern):
            k = key.decode() if isinstance(key, bytes) else key
            val = self._r.get(k)
            if val is None:
                continue
            out[k[prefix_len:]] = float(val)
        return out
