# RAG Vector-DB Evaluation Harness

An **end-to-end Retrieval-Augmented Generation (RAG)** system built around three
things:

1. a lightweight **vector database** and a full retrieval → generation pipeline,
2. an **evaluation harness** that scores retrieval *and* generation quality, and
3. **complexity-based model routing** — token/cost optimisation by switching
   Claude models per query based on how hard the query is.

It runs **fully offline out of the box** (deterministic embeddings + a mock LLM,
zero third-party dependencies), and transparently upgrades to real Claude models
when `ANTHROPIC_API_KEY` is set. That makes it easy to read, test, and demo
before spending a cent.

```
$ python -m rag_harness.cli eval --corpus data/corpus.jsonl --eval data/eval_set.jsonl

Retrieval:
  hit_rate   : 1.000
  recall@k   : 1.000
  mrr        : 0.938

Generation:
  token_f1   : 0.525
  keyword_cov: 0.792

Model routing (token optimisation):
  tier counts: {'simple': 4, 'moderate': 4}
  cost spent      : $0.010472
  premium baseline: $0.025600
  cost saved      : $0.015128 (59.1%)
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Quick start](#quick-start)
- [Step-by-step: how the pipeline works](#step-by-step-how-the-pipeline-works)
  - [1. Configuration](#1-configuration)
  - [2. Chunking](#2-chunking)
  - [3. Embeddings](#3-embeddings)
  - [4. The vector database](#4-the-vector-database)
  - [5. Complexity scoring](#5-complexity-scoring)
  - [6. Model routing (token optimisation)](#6-model-routing-token-optimisation)
  - [7. Generation](#7-generation)
  - [8. The full pipeline](#8-the-full-pipeline)
  - [9. The evaluation harness](#9-the-evaluation-harness)
- [CLI reference](#cli-reference)
- [Using real Claude models](#using-real-claude-models)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Extending](#extending)

---

## Why this exists

Most RAG demos stop at "retrieve then prompt." This project rounds out the two
things that matter in production:

- **Evaluation.** You cannot improve what you do not measure. The harness
  separates *retrieval* metrics (did we fetch the right documents?) from
  *generation* metrics (was the answer correct?) so you can tell which half of
  the system regressed.

- **Cost.** Sending every query to your most capable model is wasteful — most
  real traffic is easy. This project scores each query's complexity and routes
  it to the cheapest Claude tier that can handle it, then **measures the exact
  savings** against an always-premium baseline (≈59% on the sample set above).

---

## Quick start

No dependencies required for the offline path:

```bash
git clone <this-repo>
cd RAG-vector-DB-Evaluation-harness

# 1) Run the tests (offline, deterministic)
pip install pytest        # only needed to run the test suite
python -m pytest

# 2) Ask a question against the sample corpus
python -m rag_harness.cli query \
  --corpus data/corpus.jsonl \
  --question "Which Claude tier is cheapest?"

# 3) Run the evaluation harness
python -m rag_harness.cli eval \
  --corpus data/corpus.jsonl \
  --eval data/eval_set.jsonl

# 4) Or use the Python API
python examples/quickstart.py
```

Everything above works with **no API key**. To use real Claude models, see
[Using real Claude models](#using-real-claude-models).

---

## Step-by-step: how the pipeline works

The whole system is orchestrated by
[`RAGPipeline`](rag_harness/pipeline.py). Here is what each stage does and why,
in the order data flows through it.

### 1. Configuration

Everything tunable lives in one dataclass,
[`Settings`](rag_harness/config.py) — chunk sizes, `top_k`, routing thresholds,
model IDs per tier, and the LLM mode. It reads from environment variables via
`Settings.from_env()` but has sensible defaults, so nothing else in the codebase
hard-codes a knob.

```python
from rag_harness import Settings
settings = Settings(top_k=4, enable_model_routing=True)
```

### 2. Chunking

[`chunk_document`](rag_harness/chunking.py) splits documents into
**overlapping, sentence-aware** passages. Chunk size is a trade-off: too large
and the relevant signal is diluted (and you waste context tokens); too small and
you lose the context needed to answer. The splitter accumulates whole sentences
up to a target size, then starts a new chunk while carrying a small character
**overlap** so facts spanning a boundary survive.

```python
from rag_harness import chunk_document
chunks = chunk_document(text, doc_id="rag", chunk_size=500, chunk_overlap=80)
```

### 3. Embeddings

An embedding turns text into a fixed-length vector where *semantic similarity
becomes geometric proximity*. Anthropic has no first-party embeddings endpoint,
so the default [`HashingEmbedder`](rag_harness/embeddings.py) uses the
hashing-trick over word n-grams and character shingles to produce **deterministic,
L2-normalised** vectors with **no dependencies**. It is not state-of-the-art, but
it is reproducible and good enough to demonstrate and evaluate the full pipeline.

Because it is hidden behind the `EmbeddingProvider` protocol, swapping in a real
embedding service later is a one-file change.

### 4. The vector database

[`VectorStore`](rag_harness/vector_store.py) holds embeddings alongside their
source chunks and does exact **cosine-similarity** search (a dot product on
normalised vectors), with optional metadata filtering and JSON persistence
(`save` / `load`). Its surface is intentionally tiny — `add`, `search`, `save`,
`load` — so you can replace it with FAISS, Qdrant, or pgvector at scale without
touching any call site.

```python
hits = store.search(query_vector, top_k=4)   # -> List[SearchHit(chunk, score)]
```

### 5. Complexity scoring

This is the heart of the token-optimisation feature.
[`ComplexityScorer`](rag_harness/complexity.py) turns a query into a bounded
score in `[0, 1]` from **transparent, weighted signals**:

| Signal | Intuition |
| --- | --- |
| **length** | longer queries pack more constraints |
| **reasoning cues** | "why / compare / derive / trade-off" imply multi-step reasoning |
| **structure** | multiple clauses, question marks, connectors |
| **specificity** | numbers, code, rare/long tokens |

It is deliberately *not* a black box: every score comes with its per-signal
breakdown, so routing decisions are auditable and the weights are tunable against
your own eval set.

### 6. Model routing (token optimisation)

[`ModelRouter`](rag_harness/router.py) maps the complexity score to a model tier
using two thresholds:

```
score < simple_threshold   → SIMPLE   → claude-haiku-4-5   ($1 / $5  per 1M)
score < moderate_threshold → MODERATE → claude-sonnet-5    ($3 / $15 per 1M)
otherwise                  → COMPLEX  → claude-opus-4-8     ($5 / $25 per 1M)
```

Since most production traffic is easy, this routes the majority of queries to the
cheap tier and reserves the premium model for the hard minority — cutting cost
and latency while preserving quality where it matters. Every `RoutingDecision`
carries the score and a human-readable `reason`, and the router exposes per-model
pricing so the pipeline can compute the **cost saved vs. an always-premium
baseline**.

You can turn routing off (`enable_model_routing=False`) to A/B it — the
complexity is still scored, so you can measure exactly what routing *would* have
saved.

### 7. Generation

[`LLMClient`](rag_harness/llm.py) wraps two interchangeable backends behind one
method:

- **api** — real Claude via the official `anthropic` SDK, using a frozen system
  prompt (so it caches cleanly as a stable prefix) that instructs the model to
  answer **only** from the numbered context passages and to cite them.
- **mock** — a deterministic, offline stub that composes an answer from the
  retrieved context. It exists so the whole pipeline (and CI) runs without a key.

Both return an `LLMResponse` with token usage, which drives cost accounting.

### 8. The full pipeline

`RAGPipeline.query()` ties it together: **route → retrieve → generate → account
cost**, returning a [`RAGResult`](rag_harness/pipeline.py) that carries the
answer, the retrieved chunks, the routing decision, token usage, and the cost
saved versus premium.

```python
from rag_harness import RAGPipeline, Settings

pipe = RAGPipeline(Settings(top_k=3))
pipe.index_corpus([{"id": "doc1", "text": "..."}])
result = pipe.query("What is a vector database?")

print(result.answer)
print(result.routing.reason)          # e.g. "complexity 0.04 → simple tier → claude-haiku-4-5"
print(result.cost_saved_usd)
```

### 9. The evaluation harness

[`EvalHarness`](rag_harness/eval/harness.py) runs the pipeline over a labelled
JSONL eval set and aggregates three families of numbers:

- **Retrieval** — `hit_rate`, `recall@k`, `mrr` (were the right docs fetched?)
- **Generation** — `token_f1`, `exact_match`, `keyword_coverage` (was the answer
  right?)
- **Routing / cost** — tier distribution, total spend, and **savings vs. the
  always-premium baseline**.

```python
from rag_harness.eval import EvalHarness, load_eval_set
report = EvalHarness(pipe).run(load_eval_set("data/eval_set.jsonl"))
print(report.summary())
```

An eval example is one JSON line:

```json
{"question": "What is a vector database?",
 "answer": "A database that indexes and searches high-dimensional vectors.",
 "relevant_doc_ids": ["vector_db"],
 "keywords": ["vector", "search"]}
```

---

## CLI reference

```bash
# Build and persist an index
python -m rag_harness.cli index --corpus data/corpus.jsonl --index out/index.json

# Query a persisted index (or pass --corpus to build on the fly)
python -m rag_harness.cli query --index out/index.json --question "What is RAG?"
python -m rag_harness.cli query --corpus data/corpus.jsonl --question "..." --json

# Evaluate; write a full JSON report
python -m rag_harness.cli eval --corpus data/corpus.jsonl --eval data/eval_set.jsonl \
    --per-example --out out/report.json
```

Global flags: `--no-routing` (disable routing), `--mock` (force offline LLM),
`--top-k N` (retrieval depth).

---

## Using real Claude models

The harness auto-detects the API key. Install the optional SDK and export the
key:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python -m rag_harness.cli query --corpus data/corpus.jsonl --question "What is RAG?"
```

Model IDs per tier default to the current line-up and are configurable
(`RAG_MODEL_SIMPLE`, `RAG_MODEL_MODERATE`, `RAG_MODEL_COMPLEX`). Set
`RAG_LLM_MODE=mock` to force offline mode even with a key present, or
`RAG_LLM_MODE=api` to require the real API.

> The system prompt is frozen (no per-request interpolation) so it forms a
> stable cacheable prefix, and answers are constrained to the retrieved context
> with citations — the two things that make RAG answers grounded and cheap.

---

## Project layout

```
rag_harness/
  config.py         # Settings (all tunables)
  chunking.py       # sentence-aware overlapping splitter
  embeddings.py     # EmbeddingProvider protocol + offline HashingEmbedder
  vector_store.py   # cosine-similarity store + JSON persistence
  complexity.py     # transparent query-complexity scorer
  router.py         # complexity → model tier + cost estimation
  llm.py            # Claude client (api) + deterministic mock
  pipeline.py       # index + query orchestration -> RAGResult
  cli.py            # index / query / eval commands
  eval/
    metrics.py      # hit-rate, recall@k, MRR, F1, EM, keyword coverage
    dataset.py      # JSONL corpus / eval-set loaders
    harness.py      # runner + aggregated report
data/               # sample corpus + eval set
examples/           # quickstart.py
tests/              # 31 unit/integration tests (offline)
docs/ARCHITECTURE.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design rationale and
extension points.

---

## Testing

```bash
pip install pytest
python -m pytest
```

The suite is fully offline and deterministic (31 tests covering chunking,
embeddings, the vector store, complexity, routing, metrics, the pipeline, and the
eval harness).

---

## Extending

- **Real embeddings** — implement `EmbeddingProvider` and register it in
  `get_embedder`.
- **Scale-out search** — reimplement `VectorStore`'s four methods over FAISS /
  Qdrant / pgvector.
- **Smarter routing** — tune the scorer weights/thresholds, or swap in a learned
  classifier behind the same `route()` contract.
- **More metrics** — add pure functions to `eval/metrics.py` and aggregate them
  in `EvalHarness.run`.

## License

MIT — see [LICENSE](LICENSE).
