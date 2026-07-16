"""LLM client abstraction over the Anthropic Claude API.

Two backends share one interface:

* **api**  — real Claude calls via the official ``anthropic`` SDK.
* **mock** — a deterministic, offline stub that composes an answer from the
  retrieved context. This keeps the whole pipeline runnable (and testable in
  CI) without an API key or network access.

The client returns an :class:`LLMResponse` carrying the answer text plus token
usage, which the pipeline uses for cost accounting and the routing report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from rag_harness.config import Settings

# System prompt is frozen (no per-request interpolation) so that, in a real
# deployment, it caches cleanly as a stable prefix across requests.
SYSTEM_PROMPT = (
    "You are a precise retrieval-augmented assistant. Answer the user's "
    "question using ONLY the numbered context passages provided. Cite the "
    "passages you use with bracketed numbers like [1] or [2]. If the context "
    "does not contain the answer, say you don't have enough information rather "
    "than guessing. Be concise and factual."
)


@dataclass
class LLMResponse:
    """A model response with usage accounting."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    mode: str  # "api" | "mock"


def build_user_prompt(question: str, contexts: List[str]) -> str:
    """Assemble the user turn: numbered context passages then the question."""
    if contexts:
        numbered = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        context_block = f"Context passages:\n{numbered}\n\n"
    else:
        context_block = "Context passages: (none retrieved)\n\n"
    return f"{context_block}Question: {question}"


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for the mock backend."""
    return max(1, len(text) // 4)


class LLMClient:
    """Backend-agnostic wrapper around Claude message generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.resolved_mode()
        self._client = None  # lazily constructed anthropic.Anthropic

    # -- Public API ------------------------------------------------------
    def generate(
        self,
        question: str,
        contexts: List[str],
        model: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Answer ``question`` grounded in ``contexts`` using ``model``."""
        max_tokens = max_tokens or self.settings.max_tokens
        if self.mode == "api":
            return self._generate_api(question, contexts, model, max_tokens)
        return self._generate_mock(question, contexts, model, max_tokens)

    # -- API backend -----------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            import anthropic  # imported lazily so mock mode needs no dependency

            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key
            )
        return self._client

    def _generate_api(
        self, question: str, contexts: List[str], model: str, max_tokens: int
    ) -> LLMResponse:
        client = self._ensure_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(question, contexts)}
            ],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return LLMResponse(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            mode="api",
        )

    # -- Mock backend ----------------------------------------------------
    def _generate_mock(
        self, question: str, contexts: List[str], model: str, max_tokens: int
    ) -> LLMResponse:
        """Deterministic offline answer.

        Picks the context sentence with the greatest word overlap with the
        question and returns it with a citation. This is intentionally simple —
        it exists to exercise the pipeline end-to-end, not to be a good model.
        """
        prompt = build_user_prompt(question, contexts)
        if not contexts:
            answer = "I don't have enough information to answer that."
            return LLMResponse(
                text=answer,
                model=f"mock:{model}",
                input_tokens=_approx_tokens(SYSTEM_PROMPT + prompt),
                output_tokens=_approx_tokens(answer),
                mode="mock",
            )

        q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        best_sentence = ""
        best_idx = 0
        best_overlap = -1
        for idx, ctx in enumerate(contexts):
            for sentence in re.split(r"(?<=[.!?])\s+", ctx):
                s_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(q_words & s_words)
                if overlap > best_overlap and sentence.strip():
                    best_overlap = overlap
                    best_sentence = sentence.strip()
                    best_idx = idx

        if best_overlap <= 0:
            answer = (
                "Based on the provided context, I don't have enough "
                "information to answer that confidently."
            )
        else:
            answer = f"{best_sentence} [{best_idx + 1}]"

        return LLMResponse(
            text=answer,
            model=f"mock:{model}",
            input_tokens=_approx_tokens(SYSTEM_PROMPT + prompt),
            output_tokens=_approx_tokens(answer),
            mode="mock",
        )
