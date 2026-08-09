"""Tests for the hybrid (BM25 + exact-match rerank) retriever."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from context_engine.artifacts import CorpusChunk
from context_engine.io import load_jsonl
from context_engine.retrieval import (
    BM25ExactMatchRetriever,
    BM25Retriever,
    RetrievalResult,
    tokenize,
)


def _chunk(chunk_id: str, text: str, topic: str = "authentication") -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "x.md",
            "section_path": ["S"],
            "source_type": "doc",
            "text": text,
            "token_count": 100,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": topic},
        }
    )


def test_hybrid_retriever_class_satisfies_protocol():
    """BM25ExactMatchRetriever must satisfy the Retriever Protocol shape."""
    retriever = BM25ExactMatchRetriever()
    assert hasattr(retriever, "name")
    assert hasattr(retriever, "index")
    assert hasattr(retriever, "retrieve")
    assert callable(retriever.index)
    assert callable(retriever.retrieve)


def test_hybrid_retriever_uses_its_own_name_in_results():
    retriever = BM25ExactMatchRetriever()
    retriever.index([_chunk("c1", "pg_hba.conf authentication")])
    results = retriever.retrieve("pg_hba.conf", pool_size=1)
    assert all(r.retriever_name == "bm25_exact" for r in results)


def test_hybrid_retriever_returns_results_in_combined_score_descending_order():
    chunks = [
        _chunk("loose", "totally unrelated content"),
        _chunk("match", "pg_hba.conf controls client authentication rules"),
        _chunk("weak", "pg_hba.conf appears briefly"),
    ]
    retriever = BM25ExactMatchRetriever()
    retriever.index(chunks)
    results = retriever.retrieve("pg_hba.conf controls authentication", pool_size=3)
    assert [r.chunk_id for r in results] == ["match", "weak", "loose"]
    assert results[0].score > results[1].score > results[2].score


def test_hybrid_retriever_rewards_multi_phrase_over_single_term():
    """A chunk with the full bigram should out-rank a chunk with only one of the words."""
    chunks = [
        _chunk("gold", "pg_hba.conf controls authentication"),  # full phrase
        _chunk("distractor", "pg_hba.conf is mentioned once"),  # only one term
    ]
    retriever = BM25ExactMatchRetriever(boost_factor=1.0)
    retriever.index(chunks)
    results = retriever.retrieve("pg_hba.conf controls authentication", pool_size=2)
    assert results[0].chunk_id == "gold"
    assert results[0].score > results[1].score


def test_hybrid_retriever_zero_pool_yields_empty():
    retriever = BM25ExactMatchRetriever()
    retriever.index([_chunk("c1", "anything")])
    assert retriever.retrieve("anything", pool_size=0) == []


def test_hybrid_retriever_empty_query_yields_empty():
    retriever = BM25ExactMatchRetriever()
    retriever.index([_chunk("c1", "anything")])
    assert retriever.retrieve("", pool_size=3) == []


def test_hybrid_retriever_without_index_raises():
    retriever = BM25ExactMatchRetriever()
    with pytest.raises(ValueError):
        retriever.retrieve("anything", pool_size=3)


def test_hybrid_retriever_rejects_negative_boost_factor():
    retriever = BM25ExactMatchRetriever(boost_factor=-1.0)
    retriever.index([_chunk("c1", "anything")])
    with pytest.raises(ValueError, match="boost_factor"):
        retriever.retrieve("anything", pool_size=1)


def test_hybrid_retriever_zero_boost_factor_behaves_like_bm25():
    """With boost_factor=0 the hybrid retriever should return BM25-ranked results."""
    chunks = [
        _chunk("loose", "totally unrelated content"),
        _chunk("match", "pg_hba.conf controls client authentication rules"),
        _chunk("weak", "pg_hba.conf appears briefly"),
    ]
    bm25_only = BM25Retriever()
    bm25_only.index(chunks)
    bm25_results = bm25_only.retrieve("pg_hba.conf authentication", pool_size=3)

    hybrid = BM25ExactMatchRetriever(boost_factor=0.0)
    hybrid.index(chunks)
    hybrid_results = hybrid.retrieve("pg_hba.conf authentication", pool_size=3)

    assert [r.chunk_id for r in hybrid_results] == [r.chunk_id for r in bm25_results]
    assert [r.score for r in hybrid_results] == pytest.approx([r.score for r in bm25_results])


def test_hybrid_retriever_filter_metadata_forwards_to_bm25():
    chunks = [
        _chunk("a1", "pg_hba.conf authentication", topic="authentication"),
        _chunk("r1", "pg_hba.conf replication", topic="replication"),
    ]
    retriever = BM25ExactMatchRetriever()
    retriever.index(chunks)
    results = retriever.retrieve(
        "pg_hba.conf", pool_size=3, filter_metadata={"topic": "authentication"}
    )
    assert [r.chunk_id for r in results] == ["a1"]


def test_hybrid_retriever_prefilter_factor_expands_bm25_pool():
    """When the prefilter top-K is smaller than needed, the rerank still yields ``pool_size``."""
    chunks = [
        _chunk(f"c{i}", f"common term unique-{i}") for i in range(5)
    ]
    retriever = BM25ExactMatchRetriever(prefilter_factor=3)
    retriever.index(chunks)
    # pool_size=2, prefilter pulls 6, rerank returns top 2.
    results = retriever.retrieve("common term", pool_size=2)
    assert len(results) == 2


def test_hybrid_retriever_handles_phrase_min_max_length():
    """min_phrase_length=2 and max_phrase_length=2 should only count bigrams."""
    chunks = [
        _chunk("only_unigram", "pg_hba.conf appears here"),  # only unigram match
        _chunk("bigram_match", "pg_hba.conf authentication"),
    ]
    retriever = BM25ExactMatchRetriever(min_phrase_length=2, max_phrase_length=2)
    retriever.index(chunks)
    results = retriever.retrieve("pg_hba.conf authentication", pool_size=2)
    # The bigram_match chunk should outrank because of the bigram boost.
    assert results[0].chunk_id == "bigram_match"


def test_hybrid_retriever_short_phrase_only_uses_unigrams():
    """When max_phrase_length=1, only unigram counts matter."""
    chunks = [
        _chunk("gold", "pg_hba.conf"),
        _chunk("distractor", "pg_hba.conf appears once"),
    ]
    retriever = BM25ExactMatchRetriever(
        min_phrase_length=1, max_phrase_length=1, boost_factor=1.0
    )
    retriever.index(chunks)
    results = retriever.retrieve("pg_hba.conf", pool_size=2)
    # Both have the unigram "pg_hba.conf"; BM25 should still rank by doc-len-aware signal.
    assert len(results) == 2


def test_hybrid_retriever_returns_expected_result_type():
    retriever = BM25ExactMatchRetriever()
    retriever.index([_chunk("c1", "anything")])
    results = retriever.retrieve("anything", pool_size=1)
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_hybrid_retriever_recovers_gold_in_top_k_on_v1_corpus():
    """Regression: hybrid retriever should also put gold in the top-K on v1.

    The v1 contract expects gold_in_pool=True; the candidate-pool builder
    injects gold if the retriever missed it, but the retriever's own
    ability to surface gold at top-K is the quality signal.
    """
    chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(Path("data/processed/corpus_chunks_v1.jsonl"))
    ]
    queries = load_jsonl(Path("data/processed/queries_v1.jsonl"))

    retriever = BM25ExactMatchRetriever()
    retriever.index(chunks)

    failures: list[str] = []
    for q in queries:
        results = retriever.retrieve(q["query"], pool_size=8)
        gold = q["gold_support_ids"][0]
        top_ids = [r.chunk_id for r in results]
        if gold not in top_ids:
            failures.append(f"  {q['query_id']}: gold={gold} missing: {top_ids}")

    assert not failures, "Hybrid retriever failed on v1:\n" + "\n".join(failures)


def test_hybrid_retriever_at_least_as_good_as_bm25_on_top_k_recovery():
    """The hybrid retriever should not regress gold-in-top-8 vs BM25 on v1.

    A test that says "hybrid is better than BM25" is too strong — the
    rerank is a tie-breaker, not a guarantee. But it should not push
    gold out of the top-8 on any query that BM25 already covers.
    """
    chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(Path("data/processed/corpus_chunks_v1.jsonl"))
    ]
    queries = load_jsonl(Path("data/processed/queries_v1.jsonl"))

    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = BM25ExactMatchRetriever()
    hybrid.index(chunks)

    for q in queries:
        bm25_results = bm25.retrieve(q["query"], pool_size=8)
        hybrid_results = hybrid.retrieve(q["query"], pool_size=8)
        gold = q["gold_support_ids"][0]

        bm25_has_gold = any(r.chunk_id == gold for r in bm25_results)
        hybrid_has_gold = any(r.chunk_id == gold for r in hybrid_results)

        if bm25_has_gold and not hybrid_has_gold:
            pytest.fail(
                f"Hybrid dropped gold from top-8 on {q['query_id']} where BM25 had it"
            )
