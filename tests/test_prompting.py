from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.prompting import assemble_prompt


def test_assemble_prompt_includes_query_and_ordered_chunks() -> None:
    query = Query.from_dict(
        {
            "query_id": "q_0001",
            "query": "Which configuration file controls client authentication rules in PostgreSQL?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "The file is pg_hba.conf.",
            "gold_support_ids": ["c1"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "fact_lookup",
            },
        }
    )
    context_set = ContextSet.from_dict(
        {
            "set_id": "q_0001_gold_only",
            "query_id": "q_0001",
            "candidate_pool_id": "pool_q_0001_v1",
            "strategy": "gold_only",
            "selected_ids": ["c1", "c2"],
            "ordering_type": "best_first",
            "token_count": 123,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": [],
            },
        }
    )
    chunks_by_id = {
        "c1": CorpusChunk.from_dict(
            {
                "chunk_id": "c1",
                "doc_version": "16",
                "doc_path": "auth-pg-hba-conf.html",
                "section_path": ["Client Authentication"],
                "source_type": "doc",
                "text": "Chunk one text.",
                "token_count": 50,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "file"},
            }
        ),
        "c2": CorpusChunk.from_dict(
            {
                "chunk_id": "c2",
                "doc_version": "16",
                "doc_path": "auth-pg-hba-conf.html",
                "section_path": ["Client Authentication"],
                "source_type": "doc",
                "text": "Chunk two text.",
                "token_count": 73,
                "chunk_index": 2,
                "prev_chunk_id": "c1",
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "rules"},
            }
        ),
    }

    payload = assemble_prompt(query=query, context_set=context_set, chunks_by_id=chunks_by_id)

    assert "Question:" in payload.prompt
    assert "Chunk one text." in payload.prompt
    assert "Chunk two text." in payload.prompt
    assert "Return exactly one short answer line." in payload.prompt
    assert "INSUFFICIENT_CONTEXT" in payload.prompt
    assert payload.estimated_prompt_tokens == 123


def test_assemble_prompt_puts_context_before_question() -> None:
    """Lock the prompt structure: Context comes BEFORE Question.

    An empirical ablation (3 queries x 2 variants x 5 runs on the v3
    learned selector) showed Context-first ordering is +0.070 better
    on average than Question-first. The biggest lift is on queries with
    5-chunk contexts including distractors. This test pins the order
    so a future "simplification" doesn't regress the finding.
    """
    query = Query.from_dict({
        "query_id": "q1",
        "query": "Test question?",
        "task_type": "doc_qa",
        "difficulty": "easy",
        "gold_answer": "x",
        "gold_support_ids": ["c1"],
        "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
    })
    context_set = ContextSet.from_dict({
        "set_id": "q1_test",
        "query_id": "q1",
        "candidate_pool_id": "pool_q1",
        "strategy": "test",
        "selected_ids": ["c1"],
        "ordering_type": "best_first",
        "token_count": 50,
        "metadata": {"contains_all_gold": True, "missing_gold_count": 0, "distractor_types": []},
    })
    chunks_by_id = {"c1": CorpusChunk.from_dict({
        "chunk_id": "c1", "doc_version": "16", "doc_path": "x.md", "section_path": ["S"],
        "source_type": "doc", "text": "Chunk text.", "token_count": 50,
        "chunk_index": 1, "prev_chunk_id": None, "next_chunk_id": None,
        "metadata": {"topic": "t", "subtopic": None},
    })}
    payload = assemble_prompt(query=query, context_set=context_set, chunks_by_id=chunks_by_id)

    context_pos = payload.prompt.index("Context:")
    question_pos = payload.prompt.index("Question:")
    assert context_pos < question_pos, (
        f"Context must come before Question in the prompt "
        f"(context at {context_pos}, question at {question_pos})"
    )


"""Tests for the prompt-policy dispatcher."""


from context_engine.artifacts import ContextSet, CorpusChunk, Query
from context_engine.prompting import (
    ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT,
    POLICY_REGISTRY,
    assemble_prompt,
    policy_for_chunk_count,
)


def _query(qid: str = "q1") -> Query:
    return Query.from_dict({
        "query_id": qid,
        "query": "What is pg_hba.conf?",
        "task_type": "doc_qa",
        "difficulty": "easy",
        "gold_answer": "pg_hba.conf",
        "gold_support_ids": ["c1"],
        "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
    })


def _chunk(cid: str = "c1", text: str = "Chunk text.") -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": cid,
            "doc_version": "16",
            "doc_path": "x.md",
            "section_path": ["S"],
            "source_type": "doc",
            "text": text,
            "token_count": 50,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": "t", "subtopic": None},
        }
    )


def _context_set(selected_ids: list[str]) -> ContextSet:
    return ContextSet.from_dict({
        "set_id": "q1_test",
        "query_id": "q1",
        "candidate_pool_id": "pool_q1",
        "strategy": "test",
        "selected_ids": selected_ids,
        "ordering_type": "best_first",
        "token_count": 100,
        "metadata": {
            "contains_all_gold": True,
            "missing_gold_count": 0,
            "distractor_types": [],
        },
    })


def _chunks(n: int) -> dict[str, CorpusChunk]:
    return {f"c{i}": _chunk(f"c{i}", f"Text {i}.") for i in range(1, n + 1)}


def test_policy_registry_contains_all_three_policies() -> None:
    assert set(POLICY_REGISTRY) == {"question_first", "context_first", "adaptive_by_chunk_count"}


def test_assemble_prompt_rejects_unknown_policy() -> None:
    with_exception = None
    try:
        assemble_prompt(
            query=_query(),
            context_set=_context_set(["c1"]),
            chunks_by_id=_chunks(1),
            policy="made_up_policy",
        )
    except ValueError as exc:
        with_exception = exc
    assert with_exception is not None
    assert "made_up_policy" in str(with_exception)


def test_context_first_policy_places_context_before_question() -> None:
    payload = assemble_prompt(
        query=_query(),
        context_set=_context_set(["c1"]),
        chunks_by_id=_chunks(1),
        policy="context_first",
    )
    assert payload.policy == "context_first"
    assert payload.prompt.index("Context:") < payload.prompt.index("Question:")


def test_question_first_policy_places_question_before_context() -> None:
    payload = assemble_prompt(
        query=_query(),
        context_set=_context_set(["c1"]),
        chunks_by_id=_chunks(1),
        policy="question_first",
    )
    assert payload.policy == "question_first"
    assert payload.prompt.index("Question:") < payload.prompt.index("Context:")


def test_adaptive_policy_uses_question_first_for_small_context() -> None:
    """<= threshold chunks -> Question-first."""
    for n in [1, 2]:
        payload = assemble_prompt(
            query=_query(),
            context_set=_context_set([f"c{i}" for i in range(1, n + 1)]),
            chunks_by_id=_chunks(n),
            policy="adaptive_by_chunk_count",
        )
        assert payload.policy == "adaptive_by_chunk_count"
        assert payload.prompt.index("Question:") < payload.prompt.index("Context:"), (
            f"chunk_count={n} should pick Question-first"
        )


def test_adaptive_policy_uses_context_first_for_large_context() -> None:
    """> threshold chunks -> Context-first."""
    for n in [3, 5, 7]:
        payload = assemble_prompt(
            query=_query(),
            context_set=_context_set([f"c{i}" for i in range(1, n + 1)]),
            chunks_by_id=_chunks(n),
            policy="adaptive_by_chunk_count",
        )
        assert payload.policy == "adaptive_by_chunk_count"
        assert payload.prompt.index("Context:") < payload.prompt.index("Question:"), (
            f"chunk_count={n} should pick Context-first"
        )


def test_adaptive_policy_threshold_constant_is_two() -> None:
    """The ``ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT`` is the contract for the
    threshold. Defaults to 2 so contexts with 1 or 2 chunks use
    Question-first, 3+ chunks use Context-first.
    """
    assert ADAPTIVE_QUESTION_FIRST_CHUNK_LIMIT == 2


def test_adaptive_policy_threshold_can_be_overridden() -> None:
    """adaptive_threshold param overrides the default."""
    # With threshold=3, 3-chunk context should pick Question-first (chunk_count <= 3).
    payload = assemble_prompt(
        query=_query(),
        context_set=_context_set(["c1", "c2", "c3"]),
        chunks_by_id=_chunks(3),
        policy="adaptive_by_chunk_count",
        adaptive_threshold=3,
    )
    assert payload.prompt.index("Question:") < payload.prompt.index("Context:")


def test_prompt_policy_does_not_reorder_chunks_internally() -> None:
    """The policy only reorders (Question, Context) at the outer prompt
    level. The chunks within the Context block must appear in the
    upstream ``selected_ids`` order for all policies.
    """
    ids = ["c1", "c2", "c3", "c4", "c5"]
    for policy in ["question_first", "context_first", "adaptive_by_chunk_count"]:
        payload = assemble_prompt(
            query=_query(),
            context_set=_context_set(ids),
            chunks_by_id=_chunks(5),
            policy=policy,
        )
        # Each chunk rank should appear in the same order as selected_ids.
        last_pos = -1
        for cid in ids:
            pos = payload.prompt.index(cid)
            assert pos > last_pos, (
                f"chunk {cid} should appear in selected_ids order; "
                f"policy={policy}, ids={ids}"
            )
            last_pos = pos


def test_policy_for_chunk_count_dispatcher() -> None:
    """``policy_for_chunk_count`` reports the concrete policy the adaptive
    dispatcher would pick. For non-adaptive policies it returns the input.
    """
    # Adaptive dispatches based on the threshold (default 2).
    assert policy_for_chunk_count("adaptive_by_chunk_count", 1) == "question_first"
    assert policy_for_chunk_count("adaptive_by_chunk_count", 2) == "question_first"
    assert policy_for_chunk_count("adaptive_by_chunk_count", 3) == "context_first"
    assert policy_for_chunk_count("adaptive_by_chunk_count", 5) == "context_first"
    # Non-adaptive policies pass through unchanged.
    assert policy_for_chunk_count("question_first", 5) == "question_first"
    assert policy_for_chunk_count("context_first", 1) == "context_first"


def test_default_policy_is_context_first() -> None:
    """The default policy (when none is specified) is context_first."""
    payload = assemble_prompt(
        query=_query(),
        context_set=_context_set(["c1"]),
        chunks_by_id=_chunks(1),
    )
    assert payload.policy == "context_first"
