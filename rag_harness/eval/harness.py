"""The evaluation runner.

Runs a :class:`~rag_harness.pipeline.RAGPipeline` over a labelled eval set and
aggregates:

* **retrieval quality** — hit-rate, recall@k, MRR
* **generation quality** — token-F1, exact-match, keyword coverage
* **the token-optimisation win** — total cost spent vs. the always-premium
  baseline, plus the distribution of model tiers chosen.

The report is JSON-serialisable so it can be logged, diffed across runs, or
rendered in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from rag_harness.eval.dataset import EvalExample
from rag_harness.eval.metrics import (
    exact_match,
    keyword_coverage,
    mrr,
    recall_at_k,
    retrieval_hit,
    token_f1,
)
from rag_harness.pipeline import RAGPipeline


@dataclass
class EvalReport:
    """Aggregated evaluation results."""

    num_examples: int
    retrieval: Dict[str, float]
    generation: Dict[str, float]
    routing: Dict[str, object]
    cost: Dict[str, float]
    reliability: Dict[str, float] = field(default_factory=dict)
    per_example: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "num_examples": self.num_examples,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "routing": self.routing,
            "cost": self.cost,
            "reliability": self.reliability,
            "per_example": self.per_example,
        }

    def summary(self) -> str:
        """A compact human-readable summary."""
        lines = [
            f"Examples evaluated: {self.num_examples}",
            "",
            "Retrieval:",
            f"  hit_rate   : {self.retrieval['hit_rate']:.3f}",
            f"  recall@k   : {self.retrieval['recall_at_k']:.3f}",
            f"  mrr        : {self.retrieval['mrr']:.3f}",
            "",
            "Generation:",
            f"  token_f1   : {self.generation['token_f1']:.3f}",
            f"  exact_match: {self.generation['exact_match']:.3f}",
            f"  keyword_cov: {self.generation['keyword_coverage']:.3f}",
            "",
            "Model routing (token optimisation):",
            f"  retrieval mode  : {self.routing.get('retrieval_mode', 'dense')}",
            f"  tier counts: {self.routing['tier_counts']}",
            f"  cost spent      : ${self.cost['total_cost_usd']:.6f}",
            f"  premium baseline: ${self.cost['baseline_cost_usd']:.6f}",
            f"  cost saved      : ${self.cost['cost_saved_usd']:.6f} "
            f"({self.cost['savings_pct']:.1f}%)",
        ]
        if self.reliability:
            lines += [
                "",
                "Reliability & security:",
                f"  mean confidence : {self.reliability['mean_confidence']:.3f}",
                f"  abstain rate    : {self.reliability['abstain_rate']:.3f}",
                f"  cache hit rate  : {self.reliability['cache_hit_rate']:.3f}",
                f"  injection flags : {self.reliability['injection_flags']}",
            ]
        return "\n".join(lines)


class EvalHarness:
    """Evaluate a pipeline against a labelled eval set."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, examples: List[EvalExample], *, keep_per_example: bool = True) -> EvalReport:
        n = len(examples)
        if n == 0:
            raise ValueError("eval set is empty")

        sums = {
            "hit_rate": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "keyword_coverage": 0.0,
        }
        gen_counts = {"token_f1": 0, "exact_match": 0, "keyword_coverage": 0}
        tier_counts: Dict[str, int] = {}
        total_cost = 0.0
        baseline_cost = 0.0
        abstained = 0
        cache_hits = 0
        injection_flags = 0
        confidence_sum = 0.0
        per_example: List[Dict] = []

        for ex in examples:
            result = self.pipeline.query(ex.question)
            abstained += 1 if result.abstained else 0
            cache_hits += 1 if result.cache_hit else 0
            injection_flags += result.injection_flags
            confidence_sum += result.confidence

            # Retrieval metrics (only meaningful if labels present).
            retrieved = result.retrieved_doc_ids
            if ex.relevant_doc_ids:
                sums["hit_rate"] += retrieval_hit(retrieved, ex.relevant_doc_ids)
                sums["recall_at_k"] += recall_at_k(retrieved, ex.relevant_doc_ids)
                sums["mrr"] += mrr(retrieved, ex.relevant_doc_ids)

            # Generation metrics (skip when no gold label).
            if ex.answer is not None:
                sums["token_f1"] += token_f1(result.answer, ex.answer)
                sums["exact_match"] += exact_match(result.answer, ex.answer)
                gen_counts["token_f1"] += 1
                gen_counts["exact_match"] += 1
            if ex.keywords:
                sums["keyword_coverage"] += keyword_coverage(result.answer, ex.keywords)
                gen_counts["keyword_coverage"] += 1

            tier = result.routing.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_cost += result.cost_usd
            baseline_cost += result.baseline_cost_usd

            if keep_per_example:
                per_example.append(result.to_dict())

        # Retrieval metrics are averaged over examples that carry labels.
        num_retrieval_labeled = sum(1 for e in examples if e.relevant_doc_ids)
        retr_denom = num_retrieval_labeled or 1
        retrieval = {
            "hit_rate": sums["hit_rate"] / retr_denom,
            "recall_at_k": sums["recall_at_k"] / retr_denom,
            "mrr": sums["mrr"] / retr_denom,
            "num_labeled": num_retrieval_labeled,
        }

        generation = {
            "token_f1": sums["token_f1"] / (gen_counts["token_f1"] or 1),
            "exact_match": sums["exact_match"] / (gen_counts["exact_match"] or 1),
            "keyword_coverage": sums["keyword_coverage"]
            / (gen_counts["keyword_coverage"] or 1),
        }

        cost_saved = max(0.0, baseline_cost - total_cost)
        savings_pct = (cost_saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        cost = {
            "total_cost_usd": total_cost,
            "baseline_cost_usd": baseline_cost,
            "cost_saved_usd": cost_saved,
            "savings_pct": savings_pct,
        }

        routing = {
            "tier_counts": tier_counts,
            "routing_enabled": self.pipeline.settings.enable_model_routing,
            "retrieval_mode": self.pipeline.settings.retrieval_mode,
        }

        reliability = {
            "mean_confidence": confidence_sum / n,
            "abstain_rate": abstained / n,
            "cache_hit_rate": cache_hits / n,
            "injection_flags": injection_flags,
        }

        return EvalReport(
            num_examples=n,
            retrieval=retrieval,
            generation=generation,
            routing=routing,
            cost=cost,
            reliability=reliability,
            per_example=per_example,
        )
