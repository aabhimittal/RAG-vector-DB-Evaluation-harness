"""RAG Vector-DB Evaluation Harness.

An end-to-end Retrieval-Augmented Generation system with:

* a lightweight, dependency-free vector database,
* a pluggable embedding layer,
* a full generation pipeline built on the Anthropic Claude API,
* complexity-based **model switching** for token/cost optimisation, and
* an evaluation harness with retrieval and generation metrics.

The package is designed to run fully offline in ``mock`` mode (deterministic
embeddings + a stub LLM) so that tests and demos work without network access
or an API key, while transparently upgrading to real Claude models when
``ANTHROPIC_API_KEY`` is available.
"""

from rag_harness.chunking import Chunk, chunk_document
from rag_harness.complexity import Complexity, ComplexityScorer
from rag_harness.config import Settings
from rag_harness.embeddings import EmbeddingProvider, HashingEmbedder, get_embedder
from rag_harness.llm import LLMClient, LLMResponse
from rag_harness.pipeline import RAGPipeline, RAGResult
from rag_harness.router import ModelRouter, ModelTier, RoutingDecision
from rag_harness.vector_store import SearchHit, VectorStore

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "chunk_document",
    "Complexity",
    "ComplexityScorer",
    "Settings",
    "EmbeddingProvider",
    "HashingEmbedder",
    "get_embedder",
    "LLMClient",
    "LLMResponse",
    "RAGPipeline",
    "RAGResult",
    "ModelRouter",
    "ModelTier",
    "RoutingDecision",
    "SearchHit",
    "VectorStore",
    "__version__",
]
