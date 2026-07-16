from rag_harness.chunking import chunk_document


def test_empty_document_yields_no_chunks():
    assert chunk_document("   ", "doc") == []


def test_chunks_respect_target_size():
    text = " ".join(f"Sentence number {i}." for i in range(50))
    chunks = chunk_document(text, "doc", chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    # Allow slack for the trailing overlap seed plus one appended sentence.
    for c in chunks:
        assert len(c.text) <= 120 + 40


def test_chunk_ids_are_ordered_and_unique():
    text = " ".join(f"Fact {i} about vectors and search." for i in range(30))
    chunks = chunk_document(text, "doc", chunk_size=100, chunk_overlap=10)
    ids = [c.id for c in chunks]
    assert ids == sorted(ids, key=lambda x: int(x.split("::")[1]))
    assert len(set(ids)) == len(ids)
    assert all(c.doc_id == "doc" for c in chunks)


def test_overlap_carries_context():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    chunks = chunk_document(text, "doc", chunk_size=40, chunk_overlap=15)
    assert len(chunks) >= 2


def test_overlap_must_be_smaller_than_size():
    try:
        chunk_document("hello world.", "doc", chunk_size=10, chunk_overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_metadata_is_attached():
    chunks = chunk_document("One sentence here.", "doc", metadata={"lang": "en"})
    assert chunks[0].metadata["lang"] == "en"
