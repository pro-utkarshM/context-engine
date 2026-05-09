from context_engine.artifacts import CandidatePool, CorpusChunk, Query
from context_engine.context_sets import generate_context_set


def _chunk(chunk_id: str, token_count: int) -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "auth-pg-hba-conf.html",
            "section_path": ["Client Authentication", "The pg_hba.conf File"],
            "source_type": "doc",
            "text": f"text for {chunk_id}",
            "token_count": token_count,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": "authentication", "subtopic": "test"},
        }
    )


def _query() -> Query:
    return Query.from_dict(
        {
            "query_id": "q_0001",
            "query": "Which configuration file controls client authentication rules in PostgreSQL?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "pg_hba.conf controls client authentication rules.",
            "gold_support_ids": ["gold_a", "gold_b"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "fact_lookup",
            },
        }
    )


def _pool() -> CandidatePool:
    return CandidatePool.from_dict(
        {
            "query_id": "q_0001",
            "candidate_pool_id": "pool_q_0001_v1",
            "candidate_ids": ["gold_a", "gold_b", "d1", "d2"],
            "composition": {
                "gold_count": 2,
                "plausible_count": 2,
                "distractor_count": 0,
                "neutral_count": 0,
            },
            "gold_in_pool": True,
            "candidate_metadata": {
                "gold_a": {"role": "gold"},
                "gold_b": {"role": "gold"},
                "d1": {"role": "distractor", "distractor_type": "stale"},
                "d2": {"role": "distractor", "distractor_type": "topical_wrong"},
            },
        }
    )


def test_gold_only_strategy_keeps_only_gold() -> None:
    context_set = generate_context_set(
        query=_query(),
        candidate_pool=_pool(),
        chunks_by_id={
            "gold_a": _chunk("gold_a", 10),
            "gold_b": _chunk("gold_b", 20),
            "d1": _chunk("d1", 30),
            "d2": _chunk("d2", 40),
        },
        strategy=type("Strategy", (), {"name": "gold_only", "ordering_type": "best_first"})(),
    )
    assert context_set.selected_ids == ["gold_a", "gold_b"]
    assert context_set.token_count == 30
    assert context_set.metadata.contains_all_gold is True


def test_gold_plus_distractors_marks_unknown_distractors() -> None:
    context_set = generate_context_set(
        query=_query(),
        candidate_pool=_pool(),
        chunks_by_id={
            "gold_a": _chunk("gold_a", 10),
            "gold_b": _chunk("gold_b", 20),
            "d1": _chunk("d1", 30),
            "d2": _chunk("d2", 40),
        },
        strategy=type("Strategy", (), {"name": "gold_plus_distractors", "ordering_type": "best_first"})(),
    )
    assert context_set.selected_ids == ["gold_a", "gold_b", "d1", "d2"]
    assert context_set.metadata.distractor_types == ["stale", "topical_wrong"]


def test_minimal_support_can_miss_gold_chunks() -> None:
    context_set = generate_context_set(
        query=_query(),
        candidate_pool=_pool(),
        chunks_by_id={
            "gold_a": _chunk("gold_a", 10),
            "gold_b": _chunk("gold_b", 20),
            "d1": _chunk("d1", 30),
            "d2": _chunk("d2", 40),
        },
        strategy=type("Strategy", (), {"name": "minimal_support", "ordering_type": "best_first"})(),
    )
    assert context_set.selected_ids == ["gold_a"]
    assert context_set.metadata.contains_all_gold is False
    assert context_set.metadata.missing_gold_count == 1
