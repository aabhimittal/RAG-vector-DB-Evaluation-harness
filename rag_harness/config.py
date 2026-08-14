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
    # Retrieval strategy: "dense" (vectors), "sparse" (BM25), or "hybrid"
    # (both, fused with Reciprocal Rank Fusion). Hybrid is the robust default.
    retrieval_mode: str = "hybrid"
    candidate_multiplier: int = 4  # over-retrieve this many x top_k before fusion
    rrf_k: int = 60  # RRF constant; larger flattens the rank weighting
    use_mmr: bool = False  # diversity re-ranking (Maximal Marginal Relevance)
    mmr_lambda: float = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

    # --- Reliability: confidence-gated abstention -----------------------
    # When retrieval confidence (top dense cosine similarity) falls below the
    # threshold, the pipeline abstains WITHOUT calling the LLM — avoiding a
    # confident hallucination and saving the whole generation cost.
    enable_abstention: bool = True
    abstain_threshold: float = 0.10

    # --- Semantic answer cache (cost optimisation) ----------------------
    enable_semantic_cache: bool = False
    cache_similarity_threshold: float = 0.92
    cache_max_size: int = 1024

    # --- Security: prompt-injection defence -----------------------------
    sanitize_context: bool = True

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
            retrieval_mode=os.environ.get("RAG_RETRIEVAL_MODE", "hybrid"),
            candidate_multiplier=_env_int("RAG_CANDIDATE_MULTIPLIER", 4),
            rrf_k=_env_int("RAG_RRF_K", 60),
            use_mmr=_env_bool("RAG_USE_MMR", False),
            mmr_lambda=_env_float("RAG_MMR_LAMBDA", 0.5),
            enable_abstention=_env_bool("RAG_ENABLE_ABSTENTION", True),
            abstain_threshold=_env_float("RAG_ABSTAIN_THRESHOLD", 0.10),
            enable_semantic_cache=_env_bool("RAG_ENABLE_CACHE", False),
            cache_similarity_threshold=_env_float("RAG_CACHE_THRESHOLD", 0.92),
            cache_max_size=_env_int("RAG_CACHE_MAX_SIZE", 1024),
            sanitize_context=_env_bool("RAG_SANITIZE_CONTEXT", True),
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
