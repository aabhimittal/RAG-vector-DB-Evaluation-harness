"""Command-line interface for the RAG harness.

Subcommands
-----------
``index`` : build a vector index from a JSONL corpus and persist it.
``query`` : ask a question against a persisted (or freshly built) index.
``eval``  : run the evaluation harness over a corpus + eval set.

Everything runs offline by default (mock LLM + hashing embeddings). Set
``ANTHROPIC_API_KEY`` to use real Claude models.

Examples
--------
    python -m rag_harness.cli index  --corpus data/corpus.jsonl --index out/index.json
    python -m rag_harness.cli query  --index out/index.json --question "What is RAG?"
    python -m rag_harness.cli eval   --corpus data/corpus.jsonl --eval data/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from rag_harness.config import Settings
from rag_harness.eval.dataset import load_corpus, load_eval_set
from rag_harness.eval.harness import EvalHarness
from rag_harness.pipeline import RAGPipeline


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "no_routing", False):
        settings.enable_model_routing = False
    if getattr(args, "mock", False):
        settings.llm_mode = "mock"
    if getattr(args, "top_k", None):
        settings.top_k = args.top_k
    return settings


def cmd_index(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    pipeline = RAGPipeline(settings)
    docs = load_corpus(args.corpus)
    n_chunks = pipeline.index_corpus(docs)
    pipeline.save_index(args.index)
    print(f"Indexed {len(docs)} documents into {n_chunks} chunks -> {args.index}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    pipeline = RAGPipeline(settings)
    if args.index:
        pipeline.load_index(args.index)
    elif args.corpus:
        pipeline.index_corpus(load_corpus(args.corpus))
    else:
        print("error: provide --index or --corpus", file=sys.stderr)
        return 2

    result = pipeline.query(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Q: {result.question}")
        print(f"A: {result.answer}")
        print()
        print(f"routing : {result.routing.reason}")
        print(f"model   : {result.llm.model}  (mode={result.llm.mode})")
        print(
            f"tokens  : in={result.llm.input_tokens} out={result.llm.output_tokens}"
        )
        print(
            f"cost    : ${result.cost_usd:.6f} "
            f"(baseline ${result.baseline_cost_usd:.6f}, "
            f"saved ${result.cost_saved_usd:.6f})"
        )
        print("sources :")
        for hit in result.hits:
            print(f"  - {hit.chunk.id} (score={hit.score:.3f})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    pipeline = RAGPipeline(settings)
    pipeline.index_corpus(load_corpus(args.corpus))
    examples = load_eval_set(args.eval)
    report = EvalHarness(pipeline).run(examples, keep_per_example=args.per_example)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nFull report written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_harness",
        description="RAG vector-DB pipeline with an evaluation harness and "
        "complexity-based model routing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-routing", action="store_true", help="disable model routing")
    common.add_argument("--mock", action="store_true", help="force offline mock LLM")
    common.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")

    p_index = sub.add_parser("index", parents=[common], help="build and save an index")
    p_index.add_argument("--corpus", required=True, help="JSONL corpus path")
    p_index.add_argument("--index", required=True, help="output index path")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", parents=[common], help="answer a question")
    p_query.add_argument("--index", help="persisted index path")
    p_query.add_argument("--corpus", help="corpus to index on the fly")
    p_query.add_argument("--question", required=True, help="the question to ask")
    p_query.add_argument("--json", action="store_true", help="emit JSON")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", parents=[common], help="run the eval harness")
    p_eval.add_argument("--corpus", required=True, help="JSONL corpus path")
    p_eval.add_argument("--eval", required=True, help="JSONL eval-set path")
    p_eval.add_argument("--json", action="store_true", help="emit JSON")
    p_eval.add_argument("--per-example", action="store_true", help="include per-example rows")
    p_eval.add_argument("--out", help="write full JSON report to this path")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
