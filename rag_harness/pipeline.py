"""End-to-end RAG pipeline.

Ties every component together:

``index`` :  documents → chunks → embeddings → vector store
``query`` :  question → **route** (pick model by complexity) → **retrieve**
             (top-k chunks) → **generate** (grounded answer) → cost accounting

The :class:`RAGResult` returned by :meth:`RAGPipeline.query` carries everything
needed to evaluate quality *and* the token-optimisation win: which model was
chosen, why, the retrieved chunks, token usage, and the cost saved versus always
using the premium model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag_harness.chunking import Chunk, chunk_document
from rag_harness.config import Settings
from rag_harness.embeddings import EmbeddingProvider, get_embedder
from rag_harness.llm import LLMClient, LLMResponse
from rag_harness.router import ModelRouter, RoutingDecision, estimate_cost
from rag_harness.vector_store import SearchHit, VectorStore


@dataclass
class RAGResult:
    """The full trace of answering a single query."""

    question: str
    answer: str
    hits: List[SearchHit]
    routing: RoutingDecision
    llm: LLMResponse
    cost_usd: float
    baseline_cost_usd: float  # cost if the premium model were always used

    @property
    def cost_saved_usd(self) -> float:
        return max(0.0, self.baseline_cost_usd - self.cost_usd)

    @property
    def retrieved_doc_ids(self) -> List[str]:
        return [hit.chunk.doc_id for hit in self.hits]

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.llm.model,
            "tier": self.routing.tier.value,
            "complexity": round(self.routing.complexity.score, 4),
            "routing_reason": self.routing.reason,
            "retrieved": [
                {"chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "score": round(h.score, 4)}
                for h in self.hits
            ],
            "input_tokens": self.llm.input_tokens,
            "output_tokens": self.llm.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "baseline_cost_usd": round(self.baseline_cost_usd, 8),
            "cost_saved_usd": round(self.cost_saved_usd, 8),
            "mode": self.llm.mode,
        }


class RAGPipeline:
    """Orchestrates indexing and querying."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder: Optional[EmbeddingProvider] = None,
        store: Optional[VectorStore] = None,
        llm: Optional[LLMClient] = None,
        router: Optional[ModelRouter] = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or get_embedder(
            self.settings.embedding_provider, self.settings.embedding_dim
        )
        self.store = store or VectorStore(dim=self.embedder.dim)
        self.llm = llm or LLMClient(self.settings)
        self.router = router or ModelRouter(self.settings)

    # -- Indexing --------------------------------------------------------
    def index_document(
        self,
        text: str,
        doc_id: str,
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        """Chunk, embed, and add one document. Returns the chunk count."""
        chunks = chunk_document(
            text,
            doc_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            metadata=metadata,
        )
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def index_corpus(self, documents: List[Dict[str, str]]) -> int:
        """Index a list of ``{"id", "text", ...metadata}`` documents.

        Returns the total number of chunks indexed.
        """
        total = 0
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            metadata = {
                k: str(v) for k, v in doc.items() if k not in {"id", "text"}
            }
            total += self.index_document(text, doc_id, metadata=metadata)
        return total

    # -- Retrieval -------------------------------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchHit]:
        top_k = top_k or self.settings.top_k
        query_vec = self.embedder.embed_one(question)
        return self.store.search(query_vec, top_k=top_k)

    # -- Full query ------------------------------------------------------
    def query(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """Answer ``question`` end-to-end with routing, retrieval, generation."""
        routing = self.router.route(question)
        hits = self.retrieve(question, top_k=top_k)
        contexts = [hit.chunk.text for hit in hits]

        response = self.llm.generate(question, contexts, routing.model)

        cost = estimate_cost(
            routing.model, response.input_tokens, response.output_tokens
        )
        # Baseline: what it would cost to always use the premium/complex model.
        baseline_cost = estimate_cost(
            self.settings.model_complex,
            response.input_tokens,
            response.output_tokens,
        )
        return RAGResult(
            question=question,
            answer=response.text,
            hits=hits,
            routing=routing,
            llm=response,
            cost_usd=cost,
            baseline_cost_usd=baseline_cost,
        )

    # -- Persistence -----------------------------------------------------
    def save_index(self, path: str) -> None:
        self.store.save(path)

    def load_index(self, path: str) -> None:
        self.store = VectorStore.load(path)
