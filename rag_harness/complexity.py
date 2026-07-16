"""Query complexity scoring.

The router uses a normalised complexity score in ``[0, 1]`` to decide how much
model horsepower a query deserves. Rather than a black-box classifier, we use a
transparent, explainable set of linguistic signals so that routing decisions can
be audited and tuned. Each signal contributes a weighted, bounded sub-score.

Signals
-------
* **length** — longer queries tend to pack more constraints.
* **reasoning cues** — words like "why", "compare", "derive", "trade-off"
  correlate with multi-step reasoning.
* **question structure** — multiple question marks / clauses / conjunctions.
* **specificity** — numbers, code, and rare/long tokens.

The weights are deliberately simple and live here so they can be tuned against
your own eval set (see the eval harness).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")

# Cue words that signal multi-step or analytical reasoning.
_REASONING_CUES = {
    "why",
    "how",
    "compare",
    "contrast",
    "explain",
    "analyze",
    "analyse",
    "evaluate",
    "derive",
    "prove",
    "justify",
    "trade-off",
    "tradeoff",
    "tradeoffs",
    "implications",
    "difference",
    "differences",
    "relationship",
    "cause",
    "consequence",
    "synthesize",
    "design",
    "optimize",
    "optimise",
    "reason",
    "step-by-step",
}

# Cheap, factual-lookup cues that pull complexity *down*.
_LOOKUP_CUES = {
    "what",
    "who",
    "when",
    "where",
    "which",
    "define",
    "list",
    "name",
}

_MULTI_STEP_CONNECTORS = {
    "and",
    "then",
    "after",
    "before",
    "because",
    "therefore",
    "however",
    "whereas",
    "while",
}


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class Complexity:
    """Result of scoring a query."""

    score: float  # overall complexity in [0, 1]
    signals: Dict[str, float] = field(default_factory=dict)
    tokens: int = 0

    def band(self, simple_threshold: float, moderate_threshold: float) -> str:
        """Return a human-readable band for this score."""
        if self.score < simple_threshold:
            return "simple"
        if self.score < moderate_threshold:
            return "moderate"
        return "complex"


class ComplexityScorer:
    """Turn a natural-language query into a bounded complexity score."""

    def __init__(
        self,
        *,
        length_weight: float = 0.30,
        reasoning_weight: float = 0.40,
        structure_weight: float = 0.15,
        specificity_weight: float = 0.15,
    ) -> None:
        total = (
            length_weight
            + reasoning_weight
            + structure_weight
            + specificity_weight
        )
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        # Normalise so weights always sum to 1 regardless of what the caller
        # passes, keeping the final score bounded in [0, 1].
        self.length_weight = length_weight / total
        self.reasoning_weight = reasoning_weight / total
        self.structure_weight = structure_weight / total
        self.specificity_weight = specificity_weight / total

    def _length_signal(self, tokens: List[str]) -> float:
        # Saturates around ~40 tokens.
        return _clip(len(tokens) / 40.0)

    def _reasoning_signal(self, lower_tokens: List[str]) -> float:
        token_set = set(lower_tokens)
        reasoning_hits = len(token_set & _REASONING_CUES)
        lookup_hits = len(token_set & _LOOKUP_CUES)
        # Each reasoning cue adds meaningfully; factual-lookup cues subtract.
        raw = 0.5 * reasoning_hits - 0.25 * lookup_hits
        return _clip(raw)

    def _structure_signal(self, text: str, lower_tokens: List[str]) -> float:
        question_marks = text.count("?")
        connectors = sum(1 for t in lower_tokens if t in _MULTI_STEP_CONNECTORS)
        commas = text.count(",")
        raw = 0.25 * max(0, question_marks - 1) + 0.15 * connectors + 0.1 * commas
        return _clip(raw)

    def _specificity_signal(self, text: str, tokens: List[str]) -> float:
        has_digits = 1.0 if any(ch.isdigit() for ch in text) else 0.0
        has_code = 1.0 if re.search(r"[`{}()\[\]=<>]|def |class ", text) else 0.0
        long_tokens = sum(1 for t in tokens if len(t) >= 12)
        raw = 0.35 * has_digits + 0.4 * has_code + 0.1 * long_tokens
        return _clip(raw)

    def score(self, query: str) -> Complexity:
        """Score ``query`` and return a :class:`Complexity`."""
        tokens = _WORD_RE.findall(query)
        lower_tokens = [t.lower() for t in tokens]

        signals = {
            "length": self._length_signal(tokens),
            "reasoning": self._reasoning_signal(lower_tokens),
            "structure": self._structure_signal(query, lower_tokens),
            "specificity": self._specificity_signal(query, tokens),
        }

        score = (
            self.length_weight * signals["length"]
            + self.reasoning_weight * signals["reasoning"]
            + self.structure_weight * signals["structure"]
            + self.specificity_weight * signals["specificity"]
        )
        return Complexity(score=_clip(score), signals=signals, tokens=len(tokens))
