"""Evaluation harness for the RAG pipeline.

Provides retrieval metrics (hit-rate, recall@k, MRR), generation metrics
(token-level F1, exact match, keyword coverage), and a runner that scores a
pipeline against a labelled eval set while also reporting the token-optimisation
win (cost spent vs. always-premium baseline).
"""

from rag_harness.eval.dataset import EvalExample, load_eval_set
from rag_harness.eval.harness import EvalReport, EvalHarness
from rag_harness.eval.metrics import (
    exact_match,
    keyword_coverage,
    mrr,
    recall_at_k,
    retrieval_hit,
    token_f1,
)

__all__ = [
    "EvalExample",
    "load_eval_set",
    "EvalReport",
    "EvalHarness",
    "exact_match",
    "keyword_coverage",
    "mrr",
    "recall_at_k",
    "retrieval_hit",
    "token_f1",
]
