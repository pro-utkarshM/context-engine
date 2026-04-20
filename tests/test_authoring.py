from context_engine.authoring import (
    make_context_set,
    make_corpus_chunk,
    make_marginal_impact,
    make_query,
)


def test_make_corpus_chunk_builds_contract_shape() -> None:
    chunk = make_corpus_chunk(
        chunk_id="pg16_auth_001",
        doc_version="16",
        doc_path="client-authentication.md",
        section_path=["Client Authentication", "The pg_hba.conf File"],
        text="The pg_hba.conf file controls client authentication...",
        token_count=148,
        chunk_index=1,
        topic="authentication",
        subtopic="pg_hba.conf",
    )
    assert chunk.metadata.topic == "authentication"
    assert chunk.source_type == "doc"


def test_make_query_defaults_to_doc_qa() -> None:
    query = make_query(
        query_id="q_0001",
        query="Which file controls client authentication rules in PostgreSQL?",
        difficulty="easy",
        gold_answer="pg_hba.conf controls client authentication rules.",
        topic="authentication",
        question_family="fact_lookup",
        gold_support_ids=["pg16_auth_001"],
    )
    assert query.task_type == "doc_qa"
    assert query.gold_support_ids == ["pg16_auth_001"]


def test_make_context_set_embeds_metadata() -> None:
    context_set = make_context_set(
        set_id="q_0001_set_03",
        query_id="q_0001",
        candidate_pool_id="pool_q_0001_v1",
        strategy="gold_plus_distractors",
        selected_ids=["pg16_auth_001", "pg15_auth_003"],
        ordering_type="best_first",
        token_count=501,
        contains_all_gold=True,
        missing_gold_count=0,
        distractor_types=["stale"],
    )
    assert context_set.metadata.contains_all_gold is True
    assert context_set.metadata.distractor_types == ["stale"]


def test_make_marginal_impact_derives_signed_delta() -> None:
    impact = make_marginal_impact(
        query_id="q_0001",
        base_set_id="q_0001_set_03",
        chunk_id="pg15_auth_003",
        operation="remove",
        base_score=0.90,
        new_score=0.95,
    )
    assert impact.delta == 0.05
