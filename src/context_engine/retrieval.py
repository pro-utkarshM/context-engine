"""Retriever component for the v1 benchmark.

Implements the ``Retriever`` contract from
``docs/component-interface-spec.md``: input is a query string, a pool
size, and optional corpus-filter metadata; output is an ordered list of
``(chunk_id, score, retriever_name)`` results. The retriever never
enforces a token budget and never mutates corpus artifacts.

The v1 implementation is BM25 lexical retrieval. It is pure-stdlib,
deterministic, fast at v1 scale, and produces clean per-query signal on
the PostgreSQL documentation corpus (the v1 queries share specific
terms with their gold chunks).

A pluggable :class:`Retriever` Protocol lets later phases swap in learned
retrievers without changing the candidate-pool builder or selector.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .artifacts import CorpusChunk


# Standard BM25 parameters; these are the canonical defaults that work
# well on short technical documents and have been stable in IR literature
# for decades. Bumping them is an experiment, not a tuning exercise.
_BM25_K1 = 1.5
_BM25_B = 0.75


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+(?:[./-][a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanum / path-like tokenization. Pure function, no I/O."""
    return _TOKEN_PATTERN.findall(text.lower())


def _metadata_matches(
    chunk_metadata: object,
    filter_metadata: Mapping[str, object] | None,
) -> bool:
    """All filter keys must match the chunk's metadata values exactly.

    A ``None`` or empty filter always matches. Works on Mapping-typed
    metadata and on slot-dataclass ``ChunkMetadata`` (attribute access).
    """
    if not filter_metadata:
        return True
    for key, expected in filter_metadata.items():
        if isinstance(chunk_metadata, Mapping):
            actual = chunk_metadata.get(key)
        else:
            actual = getattr(chunk_metadata, key, None)
        if actual != expected:
            return False
    return True


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    score: float
    retriever_name: str


class Retriever(Protocol):
    """Protocol every v1 retriever must satisfy.

    Concrete implementations must be pure: no corpus mutation, no I/O
    during ``retrieve``. Pass the corpus in via ``index``.
    """

    name: str

    def index(self, chunks: Iterable[CorpusChunk]) -> None: ...

    def retrieve(
        self,
        query: str,
        *,
        pool_size: int,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]: ...


@dataclass(slots=True)
class BM25Retriever:
    """Classic BM25 over chunk text.

    Index is built once per corpus; ``retrieve`` is then a deterministic
    function of the indexed postings and the query. Filter metadata is
    applied as a pre-filter on the candidate set before scoring.
    """

    name: str = "bm25"
    k1: float = _BM25_K1
    b: float = _BM25_B

    # Populated by ``index``. Mutable defaults via field(default_factory=...)
    # so the dataclass actually constructs at import time.
    _chunks_by_id: dict[str, CorpusChunk] = field(default_factory=dict)
    _doc_lens: list[int] = field(default_factory=list)
    _avg_doc_len: float = 0.0
    _df: Counter = field(default_factory=Counter)
    _idf: dict[str, float] = field(default_factory=dict)
    _tf_by_chunk: dict[str, Counter] = field(default_factory=dict)

    def index(self, chunks: Iterable[CorpusChunk]) -> None:
        chunk_list = list(chunks)
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunk_list}
        self._tf_by_chunk = {
            chunk.chunk_id: Counter(tokenize(chunk.text)) for chunk in chunk_list
        }
        self._doc_lens = [sum(tf.values()) for tf in self._tf_by_chunk.values()]
        n_docs = max(len(chunk_list), 1)
        self._avg_doc_len = sum(self._doc_lens) / n_docs if self._doc_lens else 0.0

        df: Counter = Counter()
        for tf in self._tf_by_chunk.values():
            for term in tf:
                df[term] += 1
        self._df = df

        # Standard BM25 IDF, floored at 0 to avoid negative weights on
        # extremely common terms (which would otherwise invert rank).
        self._idf = {
            term: math.log(1.0 + (n_docs - df_term + 0.5) / (df_term + 0.5))
            for term, df_term in df.items()
        }

    def retrieve(
        self,
        query: str,
        *,
        pool_size: int,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]:
        if pool_size <= 0:
            return []
        if not self._chunks_by_id:
            raise ValueError("BM25Retriever.retrieve called before index()")

        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidates = [
            chunk
            for chunk in self._chunks_by_id.values()
            if _metadata_matches(chunk.metadata, filter_metadata)
        ]
        if not candidates:
            return []

        scored: list[tuple[str, float]] = []
        for chunk in candidates:
            tf = self._tf_by_chunk.get(chunk.chunk_id, Counter())
            doc_len = self._doc_lens[
                list(self._chunks_by_id).index(chunk.chunk_id)
            ] if False else sum(tf.values())
            score = self._score(tf, doc_len, query_terms)
            scored.append((chunk.chunk_id, score))

        # Stable sort: score desc, then chunk_id asc for determinism.
        scored.sort(key=lambda item: (-item[1], item[0]))
        top = scored[:pool_size]

        return [
            RetrievalResult(chunk_id=chunk_id, score=score, retriever_name=self.name)
            for chunk_id, score in top
        ]

    def _score(self, tf: Counter, doc_len: int, query_terms: Iterable[str]) -> float:
        if doc_len == 0 or self._avg_doc_len == 0:
            return 0.0
        score = 0.0
        len_norm = 1.0 - self.b + self.b * (doc_len / self._avg_doc_len)
        for term in query_terms:
            if term not in tf:
                continue
            f = tf[term]
            idf = self._idf.get(term, 0.0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * len_norm
            score += idf * (numerator / denominator)
        return score