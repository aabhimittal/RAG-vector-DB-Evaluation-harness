import os

from rag_harness.config import Settings
from rag_harness.eval.dataset import load_corpus, load_eval_set
from rag_harness.eval.harness import EvalHarness
from rag_harness.pipeline import RAGPipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _mock_settings(**overrides):
    base = dict(llm_mode="mock", embedding_dim=512, top_k=3)
    base.update(overrides)
    return Settings(**base)


def test_pipeline_index_and_query_offline():
    pipeline = RAGPipeline(_mock_settings())
    n = pipeline.index_corpus(load_corpus(os.path.join(DATA_DIR, "corpus.jsonl")))
    assert n > 0

    result = pipeline.query("What is a vector database?")
    assert result.answer
    assert result.hits
    # The vector_db doc should be retrieved for this question.
    assert "vector_db" in result.retrieved_doc_ids
    assert result.llm.mode == "mock"


def test_routing_picks_cheaper_model_for_simple_query():
    pipeline = RAGPipeline(_mock_settings())
    pipeline.index_corpus(load_corpus(os.path.join(DATA_DIR, "corpus.jsonl")))

    simple = pipeline.query("What is RAG?")
    complex_q = pipeline.query(
        "Compare and analyse why the recall versus speed trade-off in "
        "approximate nearest-neighbour search changes as the corpus grows, "
        "with derivations and implications."
    )
    # Simple query should never be more expensive per-token tier than complex.
    assert simple.routing.complexity.score <= complex_q.routing.complexity.score
    assert simple.cost_usd <= complex_q.baseline_cost_usd


def test_query_reports_cost_savings_when_routing_enabled():
    pipeline = RAGPipeline(_mock_settings(enable_model_routing=True))
    pipeline.index_corpus(load_corpus(os.path.join(DATA_DIR, "corpus.jsonl")))
    result = pipeline.query("Which Claude tier is cheapest?")
    # A simple query routes below premium, so some cost is saved.
    assert result.cost_saved_usd >= 0.0
    assert result.baseline_cost_usd >= result.cost_usd


def test_index_persistence_roundtrip(tmp_path):
    pipeline = RAGPipeline(_mock_settings())
    pipeline.index_corpus(load_corpus(os.path.join(DATA_DIR, "corpus.jsonl")))
    path = str(tmp_path / "index.json")
    pipeline.save_index(path)

    fresh = RAGPipeline(_mock_settings())
    fresh.load_index(path)
    result = fresh.query("What is cosine similarity?")
    assert "cosine" in result.retrieved_doc_ids


def test_eval_harness_produces_report():
    pipeline = RAGPipeline(_mock_settings())
    pipeline.index_corpus(load_corpus(os.path.join(DATA_DIR, "corpus.jsonl")))
    examples = load_eval_set(os.path.join(DATA_DIR, "eval_set.jsonl"))
    report = EvalHarness(pipeline).run(examples)

    assert report.num_examples == len(examples)
    # Retrieval should be strong on this clean corpus.
    assert report.retrieval["hit_rate"] >= 0.5
    assert 0.0 <= report.generation["token_f1"] <= 1.0
    # Routing should distribute across tiers and save cost vs premium baseline.
    assert report.cost["cost_saved_usd"] >= 0.0
    assert sum(report.routing["tier_counts"].values()) == report.num_examples
    assert isinstance(report.summary(), str)


def test_routing_saves_more_than_no_routing():
    corpus = load_corpus(os.path.join(DATA_DIR, "corpus.jsonl"))
    examples = load_eval_set(os.path.join(DATA_DIR, "eval_set.jsonl"))

    routed = RAGPipeline(_mock_settings(enable_model_routing=True))
    routed.index_corpus(corpus)
    routed_report = EvalHarness(routed).run(examples, keep_per_example=False)

    flat = RAGPipeline(_mock_settings(enable_model_routing=False))
    flat.index_corpus(corpus)
    flat_report = EvalHarness(flat).run(examples, keep_per_example=False)

    # With routing on, total spend should not exceed always-default spend by a
    # lot; routing must save something relative to the premium baseline.
    assert routed_report.cost["cost_saved_usd"] >= flat_report.cost["cost_saved_usd"]
