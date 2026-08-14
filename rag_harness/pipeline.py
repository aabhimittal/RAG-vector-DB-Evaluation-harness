"""End-to-end RAG pipeline.

Ties every component together:

``index`` :  documents → chunks → embeddings → (dense vector store + BM25 index)
``query`` :  question → **semantic cache** → **route** (model by complexity) →
             **hybrid retrieve** (dense + sparse, fused) → **abstain?** (skip the
             LLM when retrieval is weak) → **sanitise** context (injection
             defence) → **generate** → cost accounting

The :class:`RAGResult` returned by :meth:`RAGPipeline.query` carries everything
needed to evaluate quality *and* the optimisation wins: which model was chosen
and why, the retrieved chunks, retrieval confidence, whether the pipeline
abstained or hit the cache, any neutralised injection attempts, token usage, and
the cost saved versus always using the premium model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from rag_harness.cache import SemanticCache
from rag_harness.chunking import Chunk, chunk_document
from rag_harness.config import Settings
from rag_harness.embeddings import EmbeddingProvider, get_embedder
from rag_harness.llm import LLMClient, LLMResponse
from rag_harness.retrieval import BM25Index, HybridRetriever
from rag_harness.router import ModelRouter, RoutingDecision, estimate_cost
from rag_harness import security
from rag_harness.vector_store import SearchHit, VectorStore

# Fixed message returned when the pipeline abstains — safer than a guess.
ABSTAIN_MESSAGE = (
    "I don't have enough relevant information in the knowledge base to answer "
    "that confidently."
)


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
    # Extended diagnostics for the production-hardening features.
    confidence: float = 1.0
    abstained: bool = False
    cache_hit: bool = False
    retrieval_mode: str = "dense"
    injection_flags: int = 0

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
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "cache_hit": self.cache_hit,
            "retrieval_mode": self.retrieval_mode,
            "injection_flags": self.injection_flags,
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
        self.bm25 = BM25Index()
        # If a pre-populated store was injected, mirror it into the BM25 index so
        # hybrid retrieval works immediately.
        if len(self.store) > 0:
            self.bm25.add(self.store.all_chunks())
        self.llm = llm or LLMClient(self.settings)
        self.router = router or ModelRouter(self.settings)
        self.retriever = self._build_retriever()
        self.cache: Optional[SemanticCache[RAGResult]] = (
            SemanticCache(
                threshold=self.settings.cache_similarity_threshold,
                max_size=self.settings.cache_max_size,
            )
            if self.settings.enable_semantic_cache
            else None
        )

    def _build_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            self.store,
            self.bm25,
            self.embedder,
            mode=self.settings.retrieval_mode,
            candidate_multiplier=self.settings.candidate_multiplier,
            rrf_k=self.settings.rrf_k,
            use_mmr=self.settings.use_mmr,
            mmr_lambda=self.settings.mmr_lambda,
        )

    # -- Indexing --------------------------------------------------------
    def index_document(
        self,
        text: str,
        doc_id: str,
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        """Chunk, embed, and add one document to both indexes."""
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
        self.bm25.add(chunks)
        return len(chunks)

    def index_corpus(self, documents: List[Dict[str, str]]) -> int:
        """Index a list of ``{"id", "text", ...metadata}`` documents."""
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
        """Return the retrieved hits for a question (convenience accessor)."""
        top_k = top_k or self.settings.top_k
        query_vec = self.embedder.embed_one(question)
        return self.retriever.retrieve(question, query_vec, top_k).hits

    # -- Full query ------------------------------------------------------
    def query(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """Answer ``question`` end-to-end with all optimisation layers."""
        top_k = top_k or self.settings.top_k
        query_vec = self.embedder.embed_one(question)
        routing = self.router.route(question)

        # 1) Semantic cache — a near-duplicate question returns instantly at zero
        #    LLM cost.
        if self.cache is not None:
            cached = self.cache.lookup(query_vec)
            if cached is not None:
                return replace(
                    cached,
                    question=question,
                    cache_hit=True,
                    cost_usd=0.0,
                )

        # 2) Hybrid retrieval + confidence.
        retrieval = self.retriever.retrieve(question, query_vec, top_k)
        hits = retrieval.hits

        # 3) Abstain when retrieval is too weak — no LLM call, no cost, no
        #    hallucination.
        if self.settings.enable_abstention and retrieval.confidence < self.settings.abstain_threshold:
            result = RAGResult(
                question=question,
                answer=ABSTAIN_MESSAGE,
                hits=hits,
                routing=routing,
                llm=LLMResponse(
                    text=ABSTAIN_MESSAGE,
                    model=f"abstained:{routing.model}",
                    input_tokens=0,
                    output_tokens=0,
                    mode=self.llm.mode,
                ),
                cost_usd=0.0,
                baseline_cost_usd=0.0,
                confidence=retrieval.confidence,
                abstained=True,
                retrieval_mode=retrieval.mode,
            )
            if self.cache is not None:
                self.cache.put(query_vec, result)
            return result

        # 4) Injection defence — neutralise adversarial instructions in context.
        contexts = retrieval.contexts
        injection_flags = 0
        if self.settings.sanitize_context and contexts:
            contexts, report = security.sanitize_contexts(contexts)
            injection_flags = report.flagged

        # 5) Generate the grounded answer.
        response = self.llm.generate(question, contexts, routing.model)

        cost = estimate_cost(routing.model, response.input_tokens, response.output_tokens)
        baseline_cost = estimate_cost(
            self.settings.model_complex, response.input_tokens, response.output_tokens
        )
        result = RAGResult(
            question=question,
            answer=response.text,
            hits=hits,
            routing=routing,
            llm=response,
            cost_usd=cost,
            baseline_cost_usd=baseline_cost,
            confidence=retrieval.confidence,
            abstained=False,
            retrieval_mode=retrieval.mode,
            injection_flags=injection_flags,
        )
        if self.cache is not None:
            self.cache.put(query_vec, result)
        return result

    def query_batch(
        self, questions: List[str], top_k: Optional[int] = None
    ) -> List[RAGResult]:
        """Answer many questions, sharing the index and cache across them."""
        return [self.query(q, top_k=top_k) for q in questions]

    # -- Persistence -----------------------------------------------------
    def save_index(self, path: str) -> None:
        self.store.save(path)

    def load_index(self, path: str) -> None:
        """Load a persisted vector store and rebuild the BM25 index from it."""
        self.store = VectorStore.load(path)
        self.bm25 = BM25Index()
        self.bm25.add(self.store.all_chunks())
        self.retriever = self._build_retriever()
