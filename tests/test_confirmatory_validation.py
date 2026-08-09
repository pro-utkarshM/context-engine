"""Pre-confirmatory structural validation tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from context_engine.artifacts import (
    CandidatePool,
    CorpusChunk,
    Query,
)


def test_all_queries_have_split_metadata():
    """Every query in the v1 benchmark must have a split field.

    The split is set during authoring: 'development' for q_0001-q_0010,
    'confirmatory' for q_0011-q_0030. Downstream analysis depends on
    this field being present.
    """
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        assert "split" in q["metadata"], f"  {q['query_id']}: missing split"
        assert q["metadata"]["split"] in ("development", "confirmatory"), (
            f"  {q['query_id']}: invalid split {q['metadata']['split']}"
        )


def test_development_set_size_is_ten():
    """The development set is the original 10 queries from prior phases."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    dev = [q for q in queries if q["metadata"]["split"] == "development"]
    assert len(dev) == 10
    for q in dev:
        assert q["query_id"].startswith("q_00") and q["query_id"][3:].isdigit() and int(q["query_id"][3:]) <= 10
        assert q["query_id"] not in [q2["query_id"] for q2 in queries if q2["metadata"]["split"] == "confirmatory"]


def test_confirmatory_set_size_is_twenty():
    """The confirmatory set is the 20 newly authored queries."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    conf = [q for q in queries if q["metadata"]["split"] == "confirmatory"]
    assert len(conf) == 20
    for q in conf:
        assert q["query_id"].startswith("q_00")
        assert q["query_id"] >= "q_0011"


def test_confirmatory_set_diverse_topics():
    """The confirmatory set should cover diverse topics, not just auth."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    conf = [q for q in queries if q["metadata"]["split"] == "confirmatory"]
    topics = Counter(q["metadata"]["topic"] for q in conf)
    # At least 4 distinct topics
    assert len(topics) >= 4, f"only {len(topics)} topics: {list(topics)}"
    # No single topic dominates
    most_common = topics.most_common(1)[0]
    assert most_common[1] <= 7, f"  {most_common[0]}: {most_common[1]} queries (too many)"


def test_confirmatory_set_multihop_share():
    """The confirmatory set should have at least 30% multi-hop queries."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    conf = [q for q in queries if q["metadata"]["split"] == "confirmatory"]
    n_multihop = sum(1 for q in conf if q["metadata"]["requires_multi_hop"])
    assert n_multihop >= 6, f"only {n_multihop} multi-hop queries (need >= 6)"


def test_all_pools_contain_gold_chunks():
    """Every candidate pool must contain all of its query's gold chunks."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    pools_path = Path("data/processed/candidate_pools_v1.jsonl")
    if not queries_path.exists() or not pools_path.exists():
        pytest.skip("data not available")
    with queries_path.open() as f:
        queries = {json.loads(line)["query_id"]: json.loads(line) for line in f if line.strip()}
    with pools_path.open() as f:
        pools = {json.loads(line)["query_id"]: json.loads(line) for line in f if line.strip()}

    for qid, q in queries.items():
        pool = pools[qid]
        for gold_id in q["gold_support_ids"]:
            assert gold_id in pool["candidate_ids"], (
                f"  {qid}: gold {gold_id} not in pool {pool['candidate_ids']}"
            )


def test_no_query_in_development_and_confirmatory():
    """A query must be in exactly one split."""
    queries_path = Path("data/processed/queries_v1.jsonl")
    if not queries_path.exists():
        pytest.skip("queries_v1.jsonl not available")
    with queries_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]
    splits = [q["metadata"]["split"] for q in queries]
    assert all(s in ("development", "confirmatory") for s in splits)
    # No duplicates
    qids = [q["query_id"] for q in queries]
    assert len(qids) == len(set(qids))


def test_corpus_has_at_least_30_chunks():
    """The expanded corpus must have at least 30 chunks to support diverse queries."""
    chunks_path = Path("data/processed/corpus_chunks_v1.jsonl")
    if not chunks_path.exists():
        pytest.skip("corpus_chunks_v1.jsonl not available")
    with chunks_path.open() as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    assert len(chunks) >= 30, f"only {len(chunks)} chunks"


def test_corpus_chunks_topically_diverse():
    """The expanded corpus must cover more than just authentication."""
    chunks_path = Path("data/processed/corpus_chunks_v1.jsonl")
    if not chunks_path.exists():
        pytest.skip("corpus_chunks_v1.jsonl not available")
    with chunks_path.open() as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    topics = Counter(c["metadata"]["topic"] for c in chunks)
    assert len(topics) >= 4, f"only {len(topics)} topics: {list(topics)}"
