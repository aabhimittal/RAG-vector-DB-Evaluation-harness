"""An in-memory vector database with JSON persistence.

This is a compact but complete vector store: it holds embeddings alongside their
source chunks, performs exact cosine-similarity search (with optional metadata
filtering), and can be saved to / loaded from disk. Vectors are assumed to be
L2-normalised by the embedder, so cosine similarity is a plain dot product; the
store also normalises defensively so it is correct even if a caller forgets.

For production scale you would swap this for FAISS, pgvector, Qdrant, etc. — the
:class:`VectorStore` API (``add`` / ``search`` / ``save`` / ``load``) is
intentionally small so that substitution is straightforward.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from rag_harness.chunking import Chunk


@dataclass
class SearchHit:
    """A single retrieval result."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


class VectorStore:
    """Exact-search, in-memory vector store keyed by chunk id."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._chunks: List[Chunk] = []
        self._index: Dict[str, int] = {}  # chunk id -> position

    # -- Mutation --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Add chunks with their corresponding vectors.

        Adding a chunk id that already exists replaces the prior entry, making
        re-indexing idempotent.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"vector dim {len(vector)} != store dim {self.dim}"
                )
            vec = _l2_normalize(vector)
            if chunk.id in self._index:
                pos = self._index[chunk.id]
                self._vectors[pos] = vec
                self._chunks[pos] = chunk
            else:
                self._index[chunk.id] = len(self._ids)
                self._ids.append(chunk.id)
                self._vectors.append(vec)
                self._chunks.append(chunk)

    # -- Query -----------------------------------------------------------
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        *,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[SearchHit]:
        """Return the ``top_k`` most similar chunks by cosine similarity.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.
            filter_fn: Optional predicate; only chunks for which it returns
                ``True`` are considered (useful for metadata filtering).
        """
        if len(query_vector) != self.dim:
            raise ValueError(
                f"query dim {len(query_vector)} != store dim {self.dim}"
            )
        if not self._vectors or top_k <= 0:
            return []

        q = _l2_normalize(query_vector)
        scored: List[SearchHit] = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if filter_fn is not None and not filter_fn(chunk):
                continue
            score = sum(a * b for a, b in zip(q, vector))
            scored.append(SearchHit(chunk=chunk, score=score))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    # -- Persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "dim": self.dim,
            "items": [
                {"chunk": asdict(chunk), "vector": vector}
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a store previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(dim=int(payload["dim"]))
        chunks: List[Chunk] = []
        vectors: List[List[float]] = []
        for item in payload["items"]:
            chunk_data = item["chunk"]
            chunks.append(
                Chunk(
                    id=chunk_data["id"],
                    text=chunk_data["text"],
                    doc_id=chunk_data["doc_id"],
                    ordinal=chunk_data["ordinal"],
                    metadata=chunk_data.get("metadata", {}),
                )
            )
            vectors.append([float(x) for x in item["vector"]])
        store.add(chunks, vectors)
        return store
