"""Embedding providers.

Anthropic does not ship a first-party embeddings endpoint, so this module
provides a deterministic, dependency-free **hashing embedder** as the default.
It produces stable, L2-normalised vectors using the hashing-trick over word and
character n-grams. It is not state-of-the-art, but it is:

* fully offline and reproducible (great for tests and CI),
* fast, and
* good enough to demonstrate and evaluate the end-to-end pipeline.

The :class:`EmbeddingProvider` protocol makes it trivial to swap in a real
embedding service (e.g. Voyage, OpenAI, a local sentence-transformer) later —
implement ``embed`` and register it in :func:`get_embedder`.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that turns text into fixed-length float vectors."""

    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, returning one vector per input."""
        ...

    def embed_one(self, text: str) -> List[float]:
        """Embed a single text."""
        ...


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash(token: str, dim: int) -> int:
    # FNV-1a — a small, fast, deterministic non-cryptographic hash. We avoid the
    # builtin hash() because it is salted per-process (PYTHONHASHSEED), which
    # would make embeddings non-reproducible across runs.
    h = 0x811C9DC5
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % dim


class HashingEmbedder:
    """Deterministic bag-of-n-grams embedder using the hashing trick.

    Each document is represented as a sparse count vector over hashed word
    unigrams, bigrams, and 3-char shingles, then L2-normalised so that cosine
    similarity reduces to a dot product.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _features(self, text: str) -> Iterable[str]:
        tokens = _tokenize(text)
        # Word unigrams and bigrams capture topical signal.
        yield from tokens
        for a, b in zip(tokens, tokens[1:]):
            yield f"{a}_{b}"
        # Character trigrams add robustness to morphology/typos.
        collapsed = "".join(tokens)
        for i in range(len(collapsed) - 2):
            yield "#" + collapsed[i : i + 3]

    def embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for feature in self._features(text):
            vec[_hash(feature, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]


def get_embedder(provider: str = "hashing", dim: int = 256) -> EmbeddingProvider:
    """Factory for embedding providers.

    Args:
        provider: Provider name. Only ``"hashing"`` ships by default; the hook
            is here so real providers can be registered without touching call
            sites.
        dim: Embedding dimensionality.
    """
    if provider in {"hashing", "anthropic-proxy"}:
        # "anthropic-proxy" is reserved for a future real embedding backend;
        # today it falls back to the deterministic hashing embedder so the
        # pipeline always works offline.
        return HashingEmbedder(dim=dim)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
