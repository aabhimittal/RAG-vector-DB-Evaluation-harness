"""Central configuration for the RAG harness.

All tunables live here so the rest of the codebase reads configuration from a
single, well-documented place. ``Settings`` is a plain dataclass populated from
environment variables (with sane defaults), which keeps the package free of
heavyweight config dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration for the harness.

    Values are resolved from environment variables when :meth:`from_env` is
    used, otherwise the defaults below apply. Nothing here requires network
    access; the API key is only consulted when a real Claude call is made.
    """

    # --- Embeddings -----------------------------------------------------
    embedding_provider: str = "hashing"  # "hashing" (offline) | "anthropic-proxy"
    embedding_dim: int = 256

    # --- Chunking -------------------------------------------------------
    chunk_size: int = 500  # target characters per chunk
    chunk_overlap: int = 80  # character overlap between adjacent chunks

    # --- Retrieval ------------------------------------------------------
    top_k: int = 4  # chunks retrieved per query

    # --- Model routing (token optimisation) -----------------------------
    # When enabled, query complexity selects the cheapest model that can
    # plausibly answer, trading cost for capability only when needed.
    enable_model_routing: bool = True
    # Complexity thresholds in [0, 1]; a query scoring below `simple_threshold`
    # routes to the cheap tier, below `moderate_threshold` to the mid tier,
    # otherwise to the premium tier.
    simple_threshold: float = 0.33
    moderate_threshold: float = 0.66

    # Model IDs per tier. Defaults follow the current Claude line-up.
    model_simple: str = "claude-haiku-4-5"
    model_moderate: str = "claude-sonnet-5"
    model_complex: str = "claude-opus-4-8"
    # Model used when routing is disabled.
    model_default: str = "claude-sonnet-5"

    max_tokens: int = 1024

    # --- Runtime --------------------------------------------------------
    # "auto" uses the real API when ANTHROPIC_API_KEY is set, otherwise mock.
    llm_mode: str = "auto"  # "auto" | "api" | "mock"
    anthropic_api_key: Optional[str] = field(default=None, repr=False)

    def resolved_mode(self) -> str:
        """Return the effective LLM mode ("api" or "mock")."""
        if self.llm_mode == "mock":
            return "mock"
        if self.llm_mode == "api":
            return "api"
        # auto
        return "api" if self.anthropic_api_key else "mock"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls(
            embedding_provider=os.environ.get("RAG_EMBEDDING_PROVIDER", "hashing"),
            embedding_dim=_env_int("RAG_EMBEDDING_DIM", 256),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 500),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 80),
            top_k=_env_int("RAG_TOP_K", 4),
            enable_model_routing=_env_bool("RAG_ENABLE_ROUTING", True),
            simple_threshold=_env_float("RAG_SIMPLE_THRESHOLD", 0.33),
            moderate_threshold=_env_float("RAG_MODERATE_THRESHOLD", 0.66),
            model_simple=os.environ.get("RAG_MODEL_SIMPLE", "claude-haiku-4-5"),
            model_moderate=os.environ.get("RAG_MODEL_MODERATE", "claude-sonnet-5"),
            model_complex=os.environ.get("RAG_MODEL_COMPLEX", "claude-opus-4-8"),
            model_default=os.environ.get("RAG_MODEL_DEFAULT", "claude-sonnet-5"),
            max_tokens=_env_int("RAG_MAX_TOKENS", 1024),
            llm_mode=os.environ.get("RAG_LLM_MODE", "auto"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
