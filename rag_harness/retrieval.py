"""Hybrid retrieval: sparse (BM25) + dense (vector) with rank fusion and MMR.

Dense vector search excels at semantic similarity but can miss exact terms —
acronyms, error codes, product SKUs, rare proper nouns — where a keyword match
is decisive. Sparse lexical search (BM25) is the mirror image. **Hybrid
retrieval** runs both and fuses their rankings with Reciprocal Rank Fusion
(RRF), which is robust because it depends only on rank order, not on the
incompatible score scales of the two retrievers.

An optional **Maximal Marginal Relevance (MMR)** re-ranking step then trades a
little relevance for diversity, so the top-k passages are not near-duplicates of
each other — important when several chunks of one document would otherwise crowd
out coverage.

Everything here is offline, deterministic, and dependency-free.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rag_harness.chunking import Chunk
from rag_harness.embeddings import EmbeddingProvider
from rag_harness.vector_store import SearchHit, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# BM25 sparse index
# --------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over a chunk collection.

    BM25 ranks documents by term frequency (saturating, so repeated terms have
    diminishing return) scaled by inverse document frequency and normalised by
    document length. It is the workhorse lexical retriever and complements dense
    search on exact-term queries.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: List[Chunk] = []
        self._doc_len: List[int] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._df: Dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._index: Dict[str, int] = {}  # chunk id -> position

    def __len__(self) -> int:
        return len(self._chunks)

    def _rebuild_stats(self) -> None:
        total = sum(self._doc_len)
        self._avgdl = (total / len(self._doc_len)) if self._doc_len else 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index (idempotent per chunk id)."""
        for chunk in chunks:
            if chunk.id in self._index:
                # Re-indexing the same id: simplest correct behaviour is to
                # rebuild from scratch, since postings are append-only.
                self._reindex_replacing(chunk)
                continue
            pos = len(self._chunks)
            self._index[chunk.id] = pos
            self._chunks.append(chunk)
            tokens = tokenize(chunk.text)
            self._doc_len.append(len(tokens))
            tf: Dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            for tok, count in tf.items():
                self._postings[tok].append((pos, count))
                self._df[tok] += 1
        self._rebuild_stats()

    def _reindex_replacing(self, chunk: Chunk) -> None:
        # Rare path (duplicate id with new text): rebuild the whole index. Kept
        # simple and correct rather than fast.
        replacement = {c.id: c for c in self._chunks}
        replacement[chunk.id] = chunk
        rebuilt = list(replacement.values())
        self._chunks = []
        self._doc_len = []
        self._postings = defaultdict(list)
        self._df = defaultdict(int)
        self._index = {}
        self.add(rebuilt)

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        # BM25 idf with +1 smoothing to stay non-negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 4) -> List[SearchHit]:
        """Return up to ``top_k`` chunks ranked by BM25 score."""
        if not self._chunks or top_k <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in set(q_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for pos, tf in postings:
                dl = self._doc_len[pos]
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0)
                )
                scores[pos] += idf * (tf * (self.k1 + 1.0)) / denom if denom else 0.0

        # Deterministic ordering: score desc, then chunk id asc for ties.
        ranked = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], self._chunks[kv[0]].id),
        )
        return [
            SearchHit(chunk=self._chunks[pos], score=score)
            for pos, score in ranked[:top_k]
        ]


# --------------------------------------------------------------------------
# Rank fusion + MMR
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked id-lists into one ranking via RRF.

    ``score(id) = sum over lists of 1 / (k + rank)`` with 1-based ranks. Depends
    only on rank position, so it fuses retrievers with incomparable score scales.
    Ties are broken by id for determinism.
    """
    fused: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr_rerank(
    query_vec: Sequence[float],
    candidates: List[SearchHit],
    vectors: Dict[str, Sequence[float]],
    *,
    lambda_param: float = 0.5,
    top_k: int = 4,
) -> List[SearchHit]:
    """Re-rank candidates for relevance *and* diversity (Maximal Marginal
    Relevance).

    Iteratively selects the candidate maximising
    ``lambda * sim(query, d) - (1 - lambda) * max sim(d, already-selected)``.
    ``lambda_param = 1`` is pure relevance; ``0`` is pure diversity.
    """
    if not candidates:
        return []
    remaining = list(candidates)
    selected: List[SearchHit] = []

    while remaining and len(selected) < top_k:
        best_hit = None
        best_score = -math.inf
        for hit in remaining:
            vec = vectors.get(hit.chunk.id)
            if vec is None:
                relevance = hit.score
                diversity_penalty = 0.0
            else:
                relevance = _cosine(query_vec, vec)
                diversity_penalty = max(
                    (_cosine(vec, vectors[s.chunk.id]) for s in selected if s.chunk.id in vectors),
                    default=0.0,
                )
            mmr = lambda_param * relevance - (1.0 - lambda_param) * diversity_penalty
            # Deterministic tie-break by chunk id.
            if mmr > best_score or (
                mmr == best_score and best_hit is not None and hit.chunk.id < best_hit.chunk.id
            ):
                best_score = mmr
                best_hit = hit
        selected.append(best_hit)  # type: ignore[arg-type]
        remaining.remove(best_hit)  # type: ignore[arg-type]
    return selected


# --------------------------------------------------------------------------
# Hybrid retriever
# --------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Retrieved hits plus diagnostics used by the pipeline."""

    hits: List[SearchHit]
    confidence: float  # top dense cosine similarity, in ~[0, 1]
    mode: str
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def contexts(self) -> List[str]:
        return [h.chunk.text for h in self.hits]


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion and optional MMR.

    ``mode`` selects the strategy:

    * ``"dense"``  — vector search only.
    * ``"sparse"`` — BM25 only (falls back to dense if the BM25 index is empty).
    * ``"hybrid"`` — fuse dense and sparse rankings with RRF.

    A dense confidence signal (top cosine similarity) is always computed so the
    pipeline can gate abstention regardless of mode.
    """

    def __init__(
        self,
        store: VectorStore,
        bm25: BM25Index,
        embedder: EmbeddingProvider,
        *,
        mode: str = "hybrid",
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.mode = mode
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.rrf_k = rrf_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(
        self, query: str, query_vec: Sequence[float], top_k: int
    ) -> RetrievalResult:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)

        # Always run dense search — it provides the confidence signal and the
        # vectors MMR needs, even in sparse mode.
        dense_hits = self.store.search(list(query_vec), top_k=candidate_k)
        confidence = max((h.score for h in dense_hits), default=0.0)

        sparse_hits: List[SearchHit] = []
        if self.mode in ("sparse", "hybrid") and len(self.bm25) > 0:
            sparse_hits = self.bm25.search(query, top_k=candidate_k)

        by_id: Dict[str, SearchHit] = {}
        for h in dense_hits:
            by_id.setdefault(h.chunk.id, h)
        for h in sparse_hits:
            by_id.setdefault(h.chunk.id, h)

        if self.mode == "dense" or (self.mode == "sparse" and not sparse_hits):
            ordered = dense_hits
        elif self.mode == "sparse":
            ordered = sparse_hits
        else:  # hybrid
            if sparse_hits:
                fused = reciprocal_rank_fusion(
                    [[h.chunk.id for h in dense_hits], [h.chunk.id for h in sparse_hits]],
                    k=self.rrf_k,
                )
                ordered = [
                    SearchHit(chunk=by_id[cid].chunk, score=fused_score)
                    for cid, fused_score in fused
                    if cid in by_id
                ]
            else:
                ordered = dense_hits

        if self.use_mmr and ordered:
            vectors = {
                cid: v
                for cid, v in ((h.chunk.id, self.store.get_vector(h.chunk.id)) for h in ordered)
                if v is not None
            }
            ordered = mmr_rerank(
                query_vec,
                ordered,
                vectors,
                lambda_param=self.mmr_lambda,
                top_k=top_k,
            )

        hits = ordered[:top_k]
        diagnostics = {
            "dense_candidates": len(dense_hits),
            "sparse_candidates": len(sparse_hits),
            "fused": self.mode == "hybrid" and bool(sparse_hits),
            "mmr": self.use_mmr,
        }
        return RetrievalResult(
            hits=hits, confidence=confidence, mode=self.mode, diagnostics=diagnostics
        )
