"""Retrieval and generation metrics.

Small, dependency-free implementations of the standard RAG evaluation metrics.
All functions are pure and individually unit-tested.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Set

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


# --- Retrieval metrics --------------------------------------------------
def retrieval_hit(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """1.0 if any relevant doc appears in the retrieved set, else 0.0."""
    relevant: Set[str] = set(relevant_doc_ids)
    return 1.0 if any(d in relevant for d in retrieved_doc_ids) else 0.0


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """Fraction of relevant docs that appear in the retrieved set."""
    relevant: Set[str] = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    retrieved: Set[str] = set(retrieved_doc_ids)
    return len(relevant & retrieved) / len(relevant)


def mrr(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant doc (0.0 if none present)."""
    relevant: Set[str] = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# --- Generation metrics -------------------------------------------------
def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalised prediction equals normalised reference."""
    return 1.0 if _tokens(prediction) == _tokens(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD-style)."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Multiset overlap.
    common = 0
    ref_remaining = list(ref_tokens)
    for tok in pred_tokens:
        if tok in ref_remaining:
            ref_remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, keywords: Sequence[str]) -> float:
    """Fraction of required keywords present (as tokens) in the prediction."""
    if not keywords:
        return 1.0
    pred_tokens = set(_tokens(prediction))
    hits = 0
    for kw in keywords:
        kw_tokens = set(_tokens(kw))
        if kw_tokens and kw_tokens <= pred_tokens:
            hits += 1
    return hits / len(keywords)
