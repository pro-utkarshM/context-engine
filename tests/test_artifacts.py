from context_engine.artifacts import (
    ArtifactValidationError,
    CandidatePool,
    ContextSet,
    CorpusChunk,
    MarginalImpact,
    Outcome,
    Query,
)


def test_corpus_chunk_round_trip() -> None:
    row = {
        "chunk_id": "pg16_auth_001",
        "doc_version": "16",
        "doc_path": "client-authentication.md",
        "section_path": ["Client Authentication", "The pg_hba.conf File"],
        "source_type": "doc",
        "text": "The pg_hba.conf file controls client authentication...",
        "token_count": 148,
        "chunk_index": 1,
        "prev_chunk_id": None,
        "next_chunk_id": "pg16_auth_002",
        "metadata": {"topic": "authentication", "subtopic": "pg_hba.conf"},
    }
    artifact = CorpusChunk.from_dict(row)
    assert artifact.to_dict() == row


def test_query_allows_nullable_gold_support_ids() -> None:
    row = {
        "query_id": "q_0001",
        "query": "Which file controls client authentication rules in PostgreSQL?",
        "task_type": "doc_qa",
        "difficulty": "easy",
        "gold_answer": "pg_hba.conf controls client authentication rules.",
        "gold_support_ids": None,
        "metadata": {
            "topic": "authentication",
            "requires_multi_hop": False,
            "question_family": "fact_lookup",
        },
    }
    artifact = Query.from_dict(row)
    assert artifact.gold_support_ids is None


def test_candidate_pool_round_trip() -> None:
    row = {
        "query_id": "q_0001",
        "candidate_pool_id": "pool_q_0001_v1",
        "candidate_ids": ["pg16_auth_001", "pg15_auth_003", "pg16_roles_011"],
        "composition": {
            "gold_count": 1,
            "plausible_count": 10,
            "distractor_count": 6,
            "neutral_count": 3,
        },
        "gold_in_pool": True,
    }
    artifact = CandidatePool.from_dict(row)
    assert artifact.to_dict() == row


def test_context_set_round_trip() -> None:
    row = {
        "set_id": "q_0001_set_03",
        "query_id": "q_0001",
        "candidate_pool_id": "pool_q_0001_v1",
        "strategy": "gold_plus_distractors",
        "selected_ids": ["pg16_auth_001", "pg15_auth_003", "pg16_conf_004"],
        "ordering_type": "best_first",
        "token_count": 501,
        "metadata": {
            "contains_all_gold": True,
            "missing_gold_count": 0,
            "distractor_types": ["stale", "topical_wrong"],
        },
    }
    artifact = ContextSet.from_dict(row)
    assert artifact.to_dict() == row


def test_outcome_round_trip() -> None:
    row = {
        "set_id": "q_0001_set_03",
        "query_id": "q_0001",
        "answer": "The file is pg_hba.conf.",
        "scores": {"correctness": 1.0, "support": 1.0, "overall": 0.9},
        "prompt_tokens": 721,
        "completion_tokens": 39,
        "latency_ms": 2410,
        "evaluator_version": "eval_v1",
    }
    artifact = Outcome.from_dict(row)
    assert artifact.to_dict() == row


def test_marginal_impact_supports_positive_removal_delta() -> None:
    row = {
        "query_id": "q_0001",
        "base_set_id": "q_0001_set_03",
        "chunk_id": "pg15_auth_003",
        "operation": "remove",
        "base_score": 0.90,
        "new_score": 0.95,
        "delta": 0.05,
    }
    artifact = MarginalImpact.from_dict(row)
    assert artifact.delta == 0.05


def test_marginal_impact_rejects_wrong_delta() -> None:
    row = {
        "query_id": "q_0001",
        "base_set_id": "q_0001_set_03",
        "chunk_id": "pg15_auth_003",
        "operation": "remove",
        "base_score": 0.90,
        "new_score": 0.95,
        "delta": -0.05,
    }
    try:
        MarginalImpact.from_dict(row)
    except ArtifactValidationError:
        return
    raise AssertionError("Expected signed delta validation to fail")
