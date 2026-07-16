from rag_harness.eval.metrics import (
    exact_match,
    keyword_coverage,
    mrr,
    recall_at_k,
    retrieval_hit,
    token_f1,
)


def test_retrieval_hit():
    assert retrieval_hit(["a", "b"], ["b"]) == 1.0
    assert retrieval_hit(["a", "b"], ["z"]) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0
    assert recall_at_k(["a"], ["a", "b"]) == 0.5
    assert recall_at_k(["a"], []) == 0.0


def test_mrr_rewards_top_rank():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_exact_match_is_normalised():
    assert exact_match("The Cat.", "the cat") == 1.0
    assert exact_match("a dog", "a cat") == 0.0


def test_token_f1():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("cat", "") == 0.0
    partial = token_f1("the cat sat on the mat", "the cat")
    assert 0.0 < partial < 1.0


def test_keyword_coverage():
    assert keyword_coverage("vectors and similarity search", ["vectors", "search"]) == 1.0
    assert keyword_coverage("vectors only", ["vectors", "search"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0
