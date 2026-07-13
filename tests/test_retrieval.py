"""Tests for the retrieval component."""

from __future__ import annotations

from context_engine.artifacts import CorpusChunk
from context_engine.retrieval import (
    BM25Retriever,
    RetrievalResult,
    _metadata_matches,
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


def test_tokenize_lowercases_and_extracts_alnum() -> None:
    assert tokenize("PostgreSQL 16 pg_hba.conf controls auth") == [
        "postgresql",
        "16",
        "pg_hba.conf",
        "controls",
        "auth",
    ]


def test_tokenize_handles_path_like_and_dash() -> None:
    assert "pg_hba.conf" in tokenize("see pg_hba.conf")
    assert "ssl-on" in tokenize("ssl-on by default")


def test_metadata_matches_empty_filter_always_passes() -> None:
    assert _metadata_matches({"topic": "x"}, None)
    assert _metadata_matches({"topic": "x"}, {})


def test_metadata_matches_exact_match_required() -> None:
    assert _metadata_matches({"topic": "x"}, {"topic": "x"})
    assert not _metadata_matches({"topic": "y"}, {"topic": "x"})


def test_metadata_matches_works_on_slot_dataclass() -> None:
    """ChunkMetadata is a slot-dataclass, not a Mapping. Must still work."""
    from context_engine.artifacts import ChunkMetadata

    md = ChunkMetadata(topic="authentication", subtopic="file-basics")
    assert _metadata_matches(md, {"topic": "authentication"})
    assert not _metadata_matches(md, {"topic": "replication"})


def test_bm25_retrieve_returns_results_in_score_descending_order() -> None:
    retriever = BM25Retriever()
    retriever.index(
        [
            _chunk("loose", "totally unrelated content"),
            _chunk("match", "pg_hba.conf controls client authentication rules"),
            _chunk("weak", "pg_hba.conf appears briefly"),
        ]
    )
    results = retriever.retrieve("pg_hba.conf authentication", pool_size=3)
    assert [r.chunk_id for r in results] == ["match", "weak", "loose"]
    assert results[0].score > results[1].score > results[2].score


def test_bm25_retrieve_respects_pool_size() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk(f"c{i}", f"common term unique-{i}") for i in range(5)])
    results = retriever.retrieve("common term", pool_size=2)
    assert len(results) == 2


def test_bm25_retrieve_zero_pool_yields_empty() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk("c1", "anything")])
    assert retriever.retrieve("anything", pool_size=0) == []


def test_bm25_retrieve_empty_query_yields_empty() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk("c1", "anything")])
    assert retriever.retrieve("", pool_size=3) == []


def test_bm25_retrieve_without_index_raises() -> None:
    retriever = BM25Retriever()
    try:
        retriever.retrieve("anything", pool_size=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError when retrieve called before index")


def test_bm25_retrieve_filter_metadata_is_a_prefilter() -> None:
    retriever = BM25Retriever()
    retriever.index(
        [
            _chunk("auth1", "pg_hba.conf authentication", topic="authentication"),
            _chunk("repl1", "pg_hba.conf replication", topic="replication"),
        ]
    )
    results = retriever.retrieve(
        "pg_hba.conf", pool_size=3, filter_metadata={"topic": "authentication"}
    )
    assert [r.chunk_id for r in results] == ["auth1"]


def test_bm25_retrieve_filter_no_match_yields_empty() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk("c1", "anything", topic="x")])
    assert (
        retriever.retrieve("anything", pool_size=3, filter_metadata={"topic": "y"})
        == []
    )


def test_bm25_results_carry_retriever_name() -> None:
    retriever = BM25Retriever(name="bm25_v1")
    retriever.index([_chunk("c1", "match")])
    results = retriever.retrieve("match", pool_size=1)
    assert results[0].retriever_name == "bm25_v1"


def test_bm25_does_not_mutate_corpus() -> None:
    """Contract: retriever does not mutate corpus artifacts."""
    chunk = _chunk("c1", "pg_hba.conf authentication")
    text_before = chunk.text
    retriever = BM25Retriever()
    retriever.index([chunk])
    retriever.retrieve("anything", pool_size=1)
    assert chunk.text == text_before  # corpus artifact unchanged


def test_bm25_recovers_gold_in_top_k_on_v1_corpus() -> None:
    """Regression test: BM25 puts each gold chunk in the top-K.

    The v1 queries were authored so the gold-support chunk shares
    distinctive terms with its query. BM25 should always recover the
    gold in the candidate pool even if it doesn't always rank it #1.

    Two queries (q_0006 on regex fields, q_0009 on SIGHUP/Windows) have
    adjacent chunks with stronger lexical overlap and rank gold below
    rank 1 on BM25. The candidate-pool builder's contract (gold_in_pool)
    is what guarantees gold reaches the selector, not the retriever.
    """
    from pathlib import Path

    from context_engine.io import load_jsonl

    chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(Path("data/processed/corpus_chunks_v1.jsonl"))
    ]
    queries = load_jsonl(Path("data/processed/queries_v1.jsonl"))

    retriever = BM25Retriever()
    retriever.index(chunks)

    failures: list[str] = []
    for q in queries:
        results = retriever.retrieve(q["query"], pool_size=8)
        gold = q["gold_support_ids"][0]
        top_ids = [r.chunk_id for r in results]
        if gold not in top_ids:
            failures.append(f"  {q['query_id']}: gold={gold} missing from top-8: {top_ids}")

    assert not failures, (
        "BM25 failed to recover gold in top-8:\n" + "\n".join(failures)
    )


def test_bm25_recovers_gold_at_rank_1_on_eight_of_ten_v1_queries() -> None:
    """Sharper regression: on most v1 queries the retriever gets rank 1.

    Two queries fail by design (adjacent chunk has stronger lexical
    overlap). Locking that in protects against regressions in either
    direction: a future tokenizer / k1 / b tweak that drops gold below
    rank 1 on previously-correct queries should fail this test.
    """
    from pathlib import Path

    from context_engine.io import load_jsonl

    chunks = [
        CorpusChunk.from_dict(row)
        for row in load_jsonl(Path("data/processed/corpus_chunks_v1.jsonl"))
    ]
    queries = load_jsonl(Path("data/processed/queries_v1.jsonl"))

    retriever = BM25Retriever()
    retriever.index(chunks)

    rank1_count = 0
    for q in queries:
        results = retriever.retrieve(q["query"], pool_size=8)
        if results and results[0].chunk_id == q["gold_support_ids"][0]:
            rank1_count += 1

    assert rank1_count >= 8, (
        f"Expected BM25 to rank gold #1 on >=8/10 v1 queries; got {rank1_count}/10. "
        "If a tokenizer/parameter change drops rank-1 count below 8, investigate."
    )