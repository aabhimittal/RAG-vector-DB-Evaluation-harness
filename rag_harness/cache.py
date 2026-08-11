"""Semantic answer cache — cost optimisation by reuse.

Exact-string caches miss paraphrases: "What is a vector DB?" and "explain vector
databases" are different keys but the same question. A **semantic cache** keys on
the query *embedding* and returns a stored answer when a new query's embedding is
sufficiently similar to a cached one — turning a repeat/near-repeat into a
zero-cost, zero-latency hit.

This complements complexity-based routing: routing makes each call cheaper;
caching removes the call entirely for recurring questions. Both are measured by
the pipeline's cost accounting.

The implementation is a bounded, FIFO-evicted list scanned linearly — fine for
demo scale and easy to reason about. At production scale you would back it with
the same vector store used for retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class SemanticCache(Generic[T]):
    """Bounded semantic cache mapping query embeddings to arbitrary payloads."""

    def __init__(self, *, threshold: float = 0.92, max_size: int = 1024) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.max_size = max_size
        self._entries: List[Tuple[List[float], T]] = []
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query_vec: Sequence[float]) -> Optional[T]:
        """Return the payload of the most similar cached query above threshold."""
        best_payload: Optional[T] = None
        best_sim = self.threshold
        for vec, payload in self._entries:
            sim = _cosine(query_vec, vec)
            if sim >= best_sim:
                best_sim = sim
                best_payload = payload
        if best_payload is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return best_payload

    def put(self, query_vec: Sequence[float], payload: T) -> None:
        """Store a payload, evicting the oldest entry when full (FIFO)."""
        self._entries.append((list(query_vec), payload))
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def clear(self) -> None:
        self._entries.clear()
        self.stats = CacheStats()
