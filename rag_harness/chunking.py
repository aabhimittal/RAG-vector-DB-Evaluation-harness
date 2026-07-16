"""Document chunking.

Splits documents into overlapping, sentence-aware chunks. Chunking is the first
step of any RAG pipeline: retrieval quality depends heavily on chunks being
small enough to be specific but large enough to carry context.

The splitter is greedy and boundary-aware — it accumulates sentences until the
target size is reached, then starts a new chunk while carrying a configurable
character overlap so that facts spanning a boundary are not lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Split on sentence terminators followed by whitespace. Kept deliberately simple
# and dependency-free; good enough for prose and markdown corpora.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    id: str
    text: str
    doc_id: str
    ordinal: int  # position of the chunk within its source document
    metadata: Dict[str, str] = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    text: str,
    doc_id: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    metadata: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    """Chunk ``text`` into overlapping, sentence-aware :class:`Chunk` objects.

    Args:
        text: Raw document text.
        doc_id: Stable identifier for the source document.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of trailing context carried into the next
            chunk. Must be smaller than ``chunk_size``.
        metadata: Optional metadata copied onto every chunk.

    Returns:
        A list of chunks in document order. An empty/whitespace document yields
        an empty list.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    metadata = dict(metadata or {})
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_len = 0
    ordinal = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, ordinal
        if not buffer:
            return
        chunk_text = " ".join(buffer).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    id=f"{doc_id}::{ordinal}",
                    text=chunk_text,
                    doc_id=doc_id,
                    ordinal=ordinal,
                    metadata=dict(metadata),
                )
            )
            ordinal += 1
        # Seed the next buffer with a trailing overlap window taken from the
        # end of the chunk we just emitted.
        if chunk_overlap > 0 and chunk_text:
            tail = chunk_text[-chunk_overlap:]
            buffer = [tail]
            buffer_len = len(tail)
        else:
            buffer = []
            buffer_len = 0

    for sentence in sentences:
        # A single sentence longer than the target still becomes its own chunk
        # (after flushing whatever preceded it) rather than being dropped.
        if buffer_len and buffer_len + 1 + len(sentence) > chunk_size:
            flush()
        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    flush()
    return chunks
