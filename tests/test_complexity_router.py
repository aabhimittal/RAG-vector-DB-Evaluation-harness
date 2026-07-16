from rag_harness.complexity import ComplexityScorer
from rag_harness.config import Settings
from rag_harness.router import ModelRouter, ModelTier, estimate_cost


def test_simple_query_scores_lower_than_complex():
    scorer = ComplexityScorer()
    simple = scorer.score("What is RAG?")
    complex_q = scorer.score(
        "Compare exact and approximate nearest-neighbour search and analyse "
        "why the recall versus speed trade-off changes as the corpus grows, "
        "with derivations."
    )
    assert simple.score < complex_q.score
    assert 0.0 <= simple.score <= 1.0
    assert 0.0 <= complex_q.score <= 1.0


def test_score_is_bounded_and_signals_present():
    scorer = ComplexityScorer()
    c = scorer.score("Why does chunk size matter and how do you tune it?")
    assert 0.0 <= c.score <= 1.0
    for key in ("length", "reasoning", "structure", "specificity"):
        assert key in c.signals
        assert 0.0 <= c.signals[key] <= 1.0


def test_router_selects_tiers_by_threshold():
    settings = Settings(
        enable_model_routing=True,
        simple_threshold=0.33,
        moderate_threshold=0.66,
    )
    router = ModelRouter(settings)

    simple = router.route("What is a vector database?")
    assert simple.tier == ModelTier.SIMPLE
    assert simple.model == settings.model_simple

    complex_decision = router.route(
        "Explain and analyse why model routing reduces token cost, comparing "
        "the trade-offs and derivations across complexity distributions, step "
        "by step, with examples and implications."
    )
    assert complex_decision.tier == ModelTier.COMPLEX
    assert complex_decision.model == settings.model_complex


def test_router_disabled_uses_default_but_still_scores():
    settings = Settings(enable_model_routing=False, model_default="claude-sonnet-5")
    router = ModelRouter(settings)
    decision = router.route("Why compare and analyse these complex trade-offs?")
    assert decision.model == "claude-sonnet-5"
    assert decision.routing_enabled is False
    # complexity is still attached for measurement
    assert 0.0 <= decision.complexity.score <= 1.0


def test_estimate_cost_orders_tiers():
    haiku = estimate_cost("claude-haiku-4-5", 1000, 500)
    sonnet = estimate_cost("claude-sonnet-5", 1000, 500)
    opus = estimate_cost("claude-opus-4-8", 1000, 500)
    assert haiku < sonnet < opus


def test_estimate_cost_unknown_model_falls_back():
    cost = estimate_cost("some-future-model", 1000, 0)
    assert cost > 0
