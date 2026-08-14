from rag_harness.chunking import Chunk
from rag_harness.embeddings import HashingEmbedder
from rag_harness.retrieval import (
    BM25Index,
    HybridRetriever,
    mmr_rerank,
    reciprocal_rank_fusion,
)
from rag_harness.vector_store import SearchHit, VectorStore


def _chunk(cid, text, doc=None):
    return Chunk(id=cid, text=text, doc_id=doc or cid, ordinal=0)


# --- BM25 ---------------------------------------------------------------
def test_bm25_ranks_exact_term_match_first():
    idx = BM25Index()
    idx.add(
        [
            _chunk("a", "the quarterly revenue report for fiscal year 2024"),
            _chunk("b", "a general discussion of company culture and values"),
            _chunk("c", "notes about lunch options in the cafeteria"),
        ]
    )
    hits = idx.search("revenue report", top_k=2)
    assert hits[0].chunk.id == "a"


def test_bm25_empty_query_and_index():
    idx = BM25Index()
    assert idx.search("anything", top_k=3) == []
    idx.add([_chunk("a", "hello world")])
    assert idx.search("", top_k=3) == []
    assert idx.search("   ", top_k=3) == []


def test_bm25_unknown_terms_score_zero():
    idx = BM25Index()
    idx.add([_chunk("a", "vectors and embeddings")])
    assert idx.search("bananas tropical fruit", top_k=3) == []


def test_bm25_is_deterministic_on_ties():
    idx = BM25Index()
    idx.add([_chunk("b", "alpha"), _chunk("a", "alpha"), _chunk("c", "alpha")])
    hits = idx.search("alpha", top_k=3)
    # Equal scores -> tie-break by chunk id ascending.
    assert [h.chunk.id for h in hits] == ["a", "b", "c"]


def test_bm25_reindex_same_id_replaces():
    idx = BM25Index()
    idx.add([_chunk("a", "first text about cats")])
    idx.add([_chunk("a", "second text about dogs")])
    assert len(idx) == 1
    hits = idx.search("dogs", top_k=1)
    assert hits and hits[0].chunk.id == "a"


# --- RRF ----------------------------------------------------------------
def test_rrf_fuses_and_orders():
    dense = ["x", "y", "z"]
    sparse = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    order = [doc for doc, _ in fused]
    # y is rank1 in sparse and rank2 in dense -> should top x (rank1 dense, rank2 sparse tie)
    assert order[0] in {"x", "y"}
    assert set(order) == {"w", "x", "y", "z"}


def test_rrf_is_deterministic():
    a = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    b = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert a == b


# --- MMR ----------------------------------------------------------------
def test_mmr_prefers_diverse_results():
    # Two near-identical vectors and one different; MMR should not return both
    # near-identical ones when top_k=2.
    q = [1.0, 0.0, 0.0]
    vectors = {
        "dup1": [0.99, 0.14, 0.0],
        "dup2": [0.99, 0.141, 0.0],
        "div": [0.7, 0.0, 0.714],
    }
    cands = [SearchHit(_chunk(cid, cid), 0.0) for cid in ("dup1", "dup2", "div")]
    selected = mmr_rerank(q, cands, vectors, lambda_param=0.5, top_k=2)
    ids = {h.chunk.id for h in selected}
    assert "div" in ids  # diversity picked the different vector
    assert not ("dup1" in ids and "dup2" in ids)


def test_mmr_lambda_one_is_pure_relevance():
    q = [1.0, 0.0]
    vectors = {"a": [0.9, 0.1], "b": [0.5, 0.5], "c": [0.1, 0.9]}
    cands = [SearchHit(_chunk(cid, cid), 0.0) for cid in ("a", "b", "c")]
    selected = mmr_rerank(q, cands, vectors, lambda_param=1.0, top_k=3)
    assert [h.chunk.id for h in selected] == ["a", "b", "c"]


# --- Hybrid retriever ---------------------------------------------------
def _build_hybrid(mode="hybrid", **kw):
    emb = HashingEmbedder(dim=512)
    store = VectorStore(dim=512)
    bm25 = BM25Index()
    chunks = [
        _chunk("c1", "cosine similarity measures the angle between two vectors", "cosine"),
        _chunk("c2", "chunking splits long documents into passages", "chunking"),
        _chunk("c3", "HNSW is an approximate nearest neighbour index for vectors", "vector_db"),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    bm25.add(chunks)
    return emb, HybridRetriever(store, bm25, emb, mode=mode, **kw)


def test_hybrid_retrieval_returns_confidence_and_hits():
    emb, retr = _build_hybrid("hybrid")
    q = "what is cosine similarity"
    res = retr.retrieve(q, emb.embed_one(q), top_k=2)
    assert res.hits
    assert res.mode == "hybrid"
    assert 0.0 <= res.confidence <= 1.0
    assert res.hits[0].chunk.doc_id == "cosine"


def test_hybrid_beats_dense_on_exact_acronym():
    # "HNSW" is an exact term BM25 nails; dense hashing may dilute it.
    emb, retr = _build_hybrid("hybrid")
    q = "HNSW"
    res = retr.retrieve(q, emb.embed_one(q), top_k=1)
    assert res.hits[0].chunk.doc_id == "vector_db"


def test_sparse_mode_uses_bm25():
    emb, retr = _build_hybrid("sparse")
    q = "approximate nearest neighbour"
    res = retr.retrieve(q, emb.embed_one(q), top_k=1)
    assert res.mode == "sparse"
    assert res.hits[0].chunk.doc_id == "vector_db"


def test_dense_mode_ignores_bm25():
    emb, retr = _build_hybrid("dense")
    q = "angle between vectors"
    res = retr.retrieve(q, emb.embed_one(q), top_k=2)
    assert res.mode == "dense"
    assert res.diagnostics["sparse_candidates"] == 0


def test_hybrid_with_mmr_returns_topk():
    emb, retr = _build_hybrid("hybrid", use_mmr=True, mmr_lambda=0.5)
    q = "vectors"
    res = retr.retrieve(q, emb.embed_one(q), top_k=2)
    assert len(res.hits) <= 2
    assert res.diagnostics["mmr"] is True


def test_hybrid_empty_bm25_falls_back_to_dense():
    emb = HashingEmbedder(dim=256)
    store = VectorStore(dim=256)
    c = _chunk("c1", "some indexed content about vectors")
    store.add([c], emb.embed([c.text]))
    retr = HybridRetriever(store, BM25Index(), emb, mode="hybrid")
    q = "vectors"
    res = retr.retrieve(q, emb.embed_one(q), top_k=1)
    assert res.hits and res.hits[0].chunk.id == "c1"
