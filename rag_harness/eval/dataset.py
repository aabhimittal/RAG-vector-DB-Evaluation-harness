"""Eval dataset loading.

An eval set is a JSONL file where each line is an example:

.. code-block:: json

    {
      "question": "What is a vector database?",
      "answer": "A database that indexes and searches high-dimensional vectors.",
      "relevant_doc_ids": ["vector_db"],
      "keywords": ["vector", "search"]
    }

``answer`` and ``keywords`` are optional but power the generation metrics;
``relevant_doc_ids`` powers the retrieval metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalExample:
    """One labelled evaluation example."""

    question: str
    answer: Optional[str] = None
    relevant_doc_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalExample":
        return cls(
            question=data["question"],
            answer=data.get("answer"),
            relevant_doc_ids=list(data.get("relevant_doc_ids", [])),
            keywords=list(data.get("keywords", [])),
        )


def load_eval_set(path: str) -> List[EvalExample]:
    """Load a JSONL eval set into a list of :class:`EvalExample`."""
    examples: List[EvalExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(EvalExample.from_dict(json.loads(line)))
    return examples


def load_corpus(path: str) -> List[dict]:
    """Load a JSONL corpus of ``{"id", "text", ...}`` documents."""
    docs: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs
