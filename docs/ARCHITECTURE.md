# Architecture

This document explains how the pieces fit together and why each exists.

## High-level flow

```
                          ┌──────────────────────────────────────────────┐
                          │                 RAGPipeline                   │
                          │                                               │
  documents ──► index ──► │  chunk ─► embed ─► VectorStore.add            │
                          │                                               │
  question  ──► query ──► │  ┌── ModelRouter.route ──────────────┐        │
                          │  │  ComplexityScorer → tier → model  │        │
                          │  └───────────────────────────────────┘        │
                          │           │                                   │
                          │           ▼                                   │
                          │  embed(question) ─► VectorStore.search(top_k) │
                          │           │                                   │
                          │           ▼                                   │
                          │  LLMClient.generate(question, contexts, model)│
                          │           │                                   │
                          │           ▼                                   │
                          │  RAGResult (answer + routing + cost)          │
                          └──────────────────────────────────────────────┘
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Single source of truth for all tunables (`Settings`), env-driven. |
| `chunking.py` | Sentence-aware, overlapping document splitter. |
| `embeddings.py` | `EmbeddingProvider` protocol + offline `HashingEmbedder`. |
| `vector_store.py` | In-memory cosine-similarity store with JSON persistence. |
| `complexity.py` | Transparent, weighted query-complexity scorer. |
| `router.py` | Maps complexity → model tier; per-model cost estimation. |
| `llm.py` | Claude API client with a deterministic offline mock backend. |
| `pipeline.py` | Orchestrates index + query; produces `RAGResult`. |
| `eval/metrics.py` | Retrieval (hit-rate, recall@k, MRR) + generation (F1, EM, coverage). |
| `eval/harness.py` | Runs a pipeline over a labelled set; aggregates a report. |
| `cli.py` | `index` / `query` / `eval` subcommands. |

## Design decisions

**Zero required dependencies for the core.** The hashing embedder and mock LLM
mean the entire pipeline — including the eval harness — runs offline with only
the standard library. This makes the project trivially testable in CI and easy
to explore before committing an API key. The `anthropic` SDK is an *optional*
extra pulled in only when real model calls are made.

**Everything is swappable behind a small interface.** `EmbeddingProvider` is a
`Protocol`; `VectorStore` exposes just `add` / `search` / `save` / `load`. To
move to a production embedder or a FAISS/pgvector backend, implement the
interface and inject it into `RAGPipeline` — no call sites change.

**Routing is explainable, not a black box.** `ComplexityScorer` decomposes into
named, bounded signals (length, reasoning cues, structure, specificity) with
tunable weights, and every `RoutingDecision` carries the score and the reason.
This is what makes the cost/quality trade-off auditable and tunable against your
own eval set.

**The optimisation win is measured, not assumed.** Every query records the cost
actually spent *and* the cost the premium model would have incurred, so the
harness can report exact savings (see the `cost` block in the eval report).

## Extending the system

* **Real embeddings** — implement `EmbeddingProvider.embed`/`embed_one` and
  register it in `get_embedder`.
* **Scale-out vector search** — replace `VectorStore` with a FAISS/Qdrant-backed
  class exposing the same four methods.
* **Better routing** — tune the scorer weights and thresholds in `Settings`, or
  replace `ComplexityScorer` with a learned classifier while keeping the
  `route()` contract.
* **More metrics** — add pure functions to `eval/metrics.py` and aggregate them
  in `EvalHarness.run`.
