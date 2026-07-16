"""Minimal end-to-end example.

Run it with:

    python examples/quickstart.py

Works offline out of the box (mock LLM). Export ANTHROPIC_API_KEY to switch to
real Claude models.
"""

import os
import sys

# Allow running directly from a clone without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_harness import RAGPipeline, Settings  # noqa: E402

DOCUMENTS = [
    {
        "id": "rag",
        "text": (
            "Retrieval-Augmented Generation grounds a language model's answers "
            "in retrieved passages. A RAG pipeline chunks documents, embeds the "
            "chunks, retrieves the most similar chunks for a query, and "
            "generates a grounded answer."
        ),
    },
    {
        "id": "routing",
        "text": (
            "Model routing sends each request to the cheapest model capable of "
            "handling it. Simple lookups go to a small model like Haiku, while "
            "complex multi-step reasoning goes to a premium model like Opus. "
            "Because most traffic is easy, routing cuts total token cost."
        ),
    },
]


def main() -> None:
    # Offline-friendly settings; drop `llm_mode` to auto-detect the API key.
    settings = Settings(llm_mode="auto", embedding_dim=512, top_k=2)
    pipeline = RAGPipeline(settings)

    n_chunks = pipeline.index_corpus(DOCUMENTS)
    print(f"Indexed {len(DOCUMENTS)} documents into {n_chunks} chunks.\n")

    for question in [
        "What is RAG?",  # simple -> cheap model
        "Explain and analyse why model routing reduces token cost as the "
        "distribution of query complexity shifts.",  # complex -> premium model
    ]:
        result = pipeline.query(question)
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        print(f"   {result.routing.reason}")
        print(
            f"   cost ${result.cost_usd:.6f} "
            f"(saved ${result.cost_saved_usd:.6f} vs premium)\n"
        )


if __name__ == "__main__":
    main()
