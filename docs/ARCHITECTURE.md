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
| `retrieval.py` | BM25 sparse index, RRF fusion, MMR re-ranking, `HybridRetriever`. |
| `cache.py` | Semantic answer cache (`SemanticCache`) keyed on query embeddings. |
| `security.py` | Prompt-injection detection/neutralisation for retrieved context. |
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

## Production-hardening layers (v0.2)

The `query` path is wrapped with four independent, individually-toggleable
layers, ordered so the cheapest exit wins:

```
query ─► semantic cache ─► route ─► hybrid retrieve ─► abstain? ─► sanitise ─► generate
          (free hit)                (dense+sparse)     (no LLM)    (injection)
```

* **Hybrid retrieval** (`retrieval.py`). Dense and sparse (BM25) rankings are
  fused with Reciprocal Rank Fusion — chosen over score-normalisation because it
  depends only on rank order, so the two retrievers' incompatible score scales
  never need reconciling. MMR re-ranking is an optional diversity pass. A dense
  *confidence* signal (top cosine) is always computed, even in sparse mode, so
  the abstention gate has a consistent input.

* **Abstention** (`pipeline.py`). A confidence below `abstain_threshold` short-
  circuits to a fixed "insufficient evidence" answer with **no LLM call** — the
  reliability win (no hallucination) and a cost win (no tokens) in one gate.

* **Semantic cache** (`cache.py`). Keyed on the query embedding so paraphrases
  hit. A hit returns the stored result at zero cost. It sits *before* routing and
  retrieval so a hit skips the entire pipeline.

* **Injection defence** (`security.py`). Retrieved passages are untrusted; the
  sanitiser neutralises recognised override/exfiltration phrasings and defangs
  fake role turns *before* the context reaches the model, complementing the
  system prompt's "context is data" instruction. Neutralisation is conservative
  (defang, don't delete) so the retrieval signal survives.

Each layer defaults to a backward-compatible setting (dense-equivalent hybrid,
abstention on with a low threshold, cache off, sanitise on as a no-op for clean
text), and every effect is surfaced on `RAGResult` and aggregated by the eval
harness.

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
