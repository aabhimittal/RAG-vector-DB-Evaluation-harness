"""Complexity-based model router — the token/cost optimisation layer.

The core idea: **not every query needs the biggest model.** A factual lookup
("What is the capital of France?") can be answered by a small, cheap model,
while a multi-step analytical question deserves a premium one. Routing each
query to the cheapest model that can plausibly answer it cuts cost and latency
dramatically at fleet scale without hurting quality on the easy majority.

This module maps a :class:`~rag_harness.complexity.Complexity` score to a
:class:`ModelTier`, and exposes per-model pricing so the harness can report the
**cost actually spent** versus the cost of always using the premium model — the
headline metric for this optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from rag_harness.complexity import Complexity, ComplexityScorer
from rag_harness.config import Settings


class ModelTier(str, Enum):
    """Coarse capability/cost tiers."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Published per-million-token pricing (USD) for the default model line-up.
# Used only for *estimating* and comparing cost; billing is authoritative.
# (input $/1M, output $/1M)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost in USD from token counts.

    Unknown models fall back to Sonnet-tier pricing so estimates never crash.
    """
    price = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    return (
        input_tokens * price["input"] / 1_000_000.0
        + output_tokens * price["output"] / 1_000_000.0
    )


@dataclass
class RoutingDecision:
    """Explains which model was chosen for a query and why."""

    model: str
    tier: ModelTier
    complexity: Complexity
    routing_enabled: bool

    @property
    def reason(self) -> str:
        if not self.routing_enabled:
            return f"routing disabled → default model {self.model}"
        return (
            f"complexity {self.complexity.score:.2f} → {self.tier.value} tier "
            f"→ {self.model}"
        )


class ModelRouter:
    """Select a Claude model per query based on complexity.

    When routing is disabled (``settings.enable_model_routing = False``) the
    router always returns ``settings.model_default`` but still attaches the
    complexity score, so you can measure what routing *would* have done.
    """

    def __init__(
        self,
        settings: Settings,
        scorer: Optional[ComplexityScorer] = None,
    ) -> None:
        self.settings = settings
        self.scorer = scorer or ComplexityScorer()

    def _tier_for(self, complexity: Complexity) -> ModelTier:
        s = complexity.score
        if s < self.settings.simple_threshold:
            return ModelTier.SIMPLE
        if s < self.settings.moderate_threshold:
            return ModelTier.MODERATE
        return ModelTier.COMPLEX

    def _model_for_tier(self, tier: ModelTier) -> str:
        return {
            ModelTier.SIMPLE: self.settings.model_simple,
            ModelTier.MODERATE: self.settings.model_moderate,
            ModelTier.COMPLEX: self.settings.model_complex,
        }[tier]

    def route(self, query: str) -> RoutingDecision:
        """Return a :class:`RoutingDecision` for ``query``."""
        complexity = self.scorer.score(query)
        tier = self._tier_for(complexity)

        if self.settings.enable_model_routing:
            model = self._model_for_tier(tier)
        else:
            model = self.settings.model_default

        return RoutingDecision(
            model=model,
            tier=tier,
            complexity=complexity,
            routing_enabled=self.settings.enable_model_routing,
        )
