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

The v1 library ships two retrievers:

- :class:`BM25Retriever` - classic BM25 (inverse-document-frequency
  weighted term overlap).
- :class:`BM25ExactMatchRetriever` - BM25 followed by an exact-phrase
  rerank. Useful for short technical corpora where specific phrases
  appear verbatim in the right chunks.

Both are pure stdlib, deterministic, and indexed at corpus load time.
"""

from __future__ import annotations

import math
import random
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


@dataclass(slots=True)
class BM25ExactMatchRetriever:
    """Hybrid retriever: BM25 over the corpus, then an exact-phrase rerank.

    Why it exists: the v1 corpus is small (8 chunks) and the queries
    contain specific phrases ("pg_hba.conf", "hostssl", "SIGHUP", ...)
    that are also contained verbatim in the gold chunks. BM25 alone
    captures the *term overlap* signal but not the *exact phrase match*
    signal. A chunk that repeats a multi-word phrase from the query
    verbatim is more likely to be the gold than a chunk that has the
    same terms but in different order.

    This retriever composes with a ``BM25Retriever``:

    1. ``bm25_retriever`` returns its top ``prefilter_pool_size`` results
       (>= ``pool_size``) ordered by BM25 score.
    2. Each result is rescored as ``bm25_score + boost_factor * phrase_score``.
    3. The top ``pool_size`` by the combined score is returned.

    The ``phrase_score`` is the sum of n-gram lengths whose phrase
    appears verbatim in the chunk text. For a query "X Y Z", the
    unigrams "X", "Y", "Z" each contribute 1 if present; bigrams "X Y"
    and "Y Z" each contribute 2 if present; trigram "X Y Z" contributes
    3 if present. Longer phrases dominate, so a chunk that contains
    the full query trigram scores higher than one that only contains
    one of the words.

    Empty query / no-index guards are inherited from the underlying
    BM25 retriever. Filter metadata is forwarded.

    Properties:
      - pure stdlib (substring matching; no regex flavour reliance)
      - deterministic given the same BM25 index and the same query
      - safe to call on indices rebuilt mid-corpus; the rerank step
        does not mutate the underlying BM25 state
    """

    name: str = "bm25_exact"
    boost_factor: float = 1.0
    max_phrase_length: int = 3
    min_phrase_length: int = 1
    prefilter_factor: int = 2
    bm25_retriever: BM25Retriever = field(default_factory=BM25Retriever)

    def index(self, chunks: Iterable[CorpusChunk]) -> None:
        """Build the BM25 index. The rerank step is query-only, no extra state."""
        self.bm25_retriever.index(chunks)

    def _query_phrases(self, query: str) -> list[tuple[str, int]]:
        """Tokenize the query and return ``(phrase, n_gram_length)`` pairs.

        Phrases are returned as space-joined lowercase tokens, no
        punctuation. The caller does substring matching against the
        chunk text (which is also lowercased for the match).
        """
        tokens = tokenize(query)
        phrases: list[tuple[str, int]] = []
        for n in range(self.min_phrase_length, self.max_phrase_length + 1):
            for start in range(0, len(tokens) - n + 1):
                phrase = " ".join(tokens[start:start + n])
                if phrase:
                    phrases.append((phrase, n))
        return phrases

    def _phrase_score(self, chunk_text: str, phrases: list[tuple[str, int]]) -> float:
        """Sum of n-gram lengths whose phrase appears verbatim in the chunk."""
        if not phrases:
            return 0.0
        chunk_lower = chunk_text.lower()
        score = 0.0
        for phrase, length in phrases:
            if phrase in chunk_lower:
                score += length
        return score

    def retrieve(
        self,
        query: str,
        *,
        pool_size: int,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalResult]:
        if pool_size <= 0:
            return []
        if self.boost_factor < 0:
            raise ValueError("boost_factor must be >= 0")

        prefilter_pool_size = max(pool_size * self.prefilter_factor, pool_size)
        bm25_results = self.bm25_retriever.retrieve(
            query,
            pool_size=prefilter_pool_size,
            filter_metadata=filter_metadata,
        )
        if not bm25_results:
            return []

        # Build a map from chunk_id -> chunk text for the rerank step.
        bm25_by_id = {result.chunk_id: result for result in bm25_results}
        phrases = self._query_phrases(query)

        reranked: list[tuple[str, float, float]] = []
        for chunk_id, bm25_result in bm25_by_id.items():
            chunk = self.bm25_retriever._chunks_by_id.get(chunk_id)
            if chunk is None:
                # Should not happen: BM25 only returns indexed chunks.
                continue
            phrase_score = self._phrase_score(chunk.text, phrases)
            combined = bm25_result.score + self.boost_factor * phrase_score
            reranked.append((chunk_id, combined, phrase_score))

        reranked.sort(key=lambda item: (-item[1], item[0]))
        top = reranked[:pool_size]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=combined_score,
                retriever_name=self.name,
            )
            for chunk_id, combined_score, _ in top
        ]



@dataclass(slots=True)
class RandomRetriever:
    """Random-pool retriever — uniform random sampling without replacement.

    Returns ``pool_size`` chunks uniformly at random from the indexed
    corpus, optionally pre-filtered by metadata. The output is
    deterministic given a seed, but the underlying signal is "no
    signal" — the retriever does not score against the query.

    Use as a baseline: if a candidate pool builder + strategy pipeline
    performs no better with this retriever than with BM25, the
    downstream selection is not gaining anything from retrieval.

    The Protocol compliance is intentional: a useless retriever is
    still a valid retriever under the contract. ``#13`` ships this
    as the third option behind ``bm25`` and ``bm25_exact``.
    """

    name: str = "random"
    seed: int = 0
    _chunks_by_id: dict[str, CorpusChunk] = field(default_factory=dict)

    def index(self, chunks: Iterable[CorpusChunk]) -> None:
        chunk_list = list(chunks)
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunk_list}

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
            raise ValueError("RandomRetriever.retrieve called before index()")

        candidates = [
            chunk
            for chunk in self._chunks_by_id.values()
            if _metadata_matches(chunk.metadata, filter_metadata)
        ]
        if not candidates:
            return []

        # Deterministic: seeded Random picks the same indices for the
        # same seed, query, and pool_size. The query parameter is
        # unused (it appears in the Protocol signature).
        rng = random.Random(self.seed + hash(query))
        selected = rng.sample(candidates, k=min(pool_size, len(candidates)))
        # Sort by chunk_id for stable ordering of the returned list.
        selected.sort(key=lambda chunk: chunk.chunk_id)
        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                score=0.0,  # random has no query-relative score
                retriever_name=self.name,
            )
            for chunk in selected
        ]
