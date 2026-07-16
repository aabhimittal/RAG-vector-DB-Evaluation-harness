import math

from rag_harness.chunking import Chunk
from rag_harness.embeddings import HashingEmbedder, get_embedder
from rag_harness.vector_store import VectorStore


def test_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed_one("vector databases and similarity search")
    v2 = emb.embed_one("vector databases and similarity search")
    assert v1 == v2
    assert abs(math.sqrt(sum(x * x for x in v1)) - 1.0) < 1e-9


def test_similar_texts_score_higher_than_unrelated():
    emb = HashingEmbedder(dim=512)
    q = emb.embed_one("what is cosine similarity")
    related = emb.embed_one("cosine similarity measures the angle between vectors")
    unrelated = emb.embed_one("bananas are a yellow tropical fruit")
    sim_related = sum(a * b for a, b in zip(q, related))
    sim_unrelated = sum(a * b for a, b in zip(q, unrelated))
    assert sim_related > sim_unrelated


def _chunk(cid, text, doc="d"):
    return Chunk(id=cid, text=text, doc_id=doc, ordinal=0)


def test_vector_store_search_ranks_relevant_first():
    emb = HashingEmbedder(dim=512)
    store = VectorStore(dim=512)
    chunks = [
        _chunk("c1", "cosine similarity measures the angle between two vectors", "cosine"),
        _chunk("c2", "chunking splits documents into smaller passages", "chunking"),
        _chunk("c3", "haiku is the cheapest claude model tier", "claude"),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    hits = store.search(emb.embed_one("what is cosine similarity"), top_k=2)
    assert hits[0].chunk.id == "c1"
    assert len(hits) == 2


def test_vector_store_add_is_idempotent_on_id():
    emb = HashingEmbedder(dim=64)
    store = VectorStore(dim=64)
    c = _chunk("c1", "first version")
    store.add([c], emb.embed([c.text]))
    c2 = _chunk("c1", "second version")
    store.add([c2], emb.embed([c2.text]))
    assert len(store) == 1


def test_vector_store_filter_fn():
    emb = HashingEmbedder(dim=128)
    store = VectorStore(dim=128)
    chunks = [
        _chunk("a", "alpha content", "docA"),
        _chunk("b", "beta content", "docB"),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    hits = store.search(
        emb.embed_one("content"), top_k=5, filter_fn=lambda c: c.doc_id == "docB"
    )
    assert all(h.chunk.doc_id == "docB" for h in hits)


def test_vector_store_save_and_load(tmp_path):
    emb = HashingEmbedder(dim=128)
    store = VectorStore(dim=128)
    chunks = [_chunk("a", "alpha content"), _chunk("b", "beta content")]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    path = str(tmp_path / "index.json")
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    hits = loaded.search(emb.embed_one("alpha"), top_k=1)
    assert hits[0].chunk.id == "a"


def test_get_embedder_factory():
    emb = get_embedder("hashing", dim=32)
    assert emb.dim == 32
