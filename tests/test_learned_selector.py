"""Tests for the learned selector baseline."""

from __future__ import annotations

import pytest

from context_engine.artifacts import (
    CandidatePool,
    ContextSet,
    ContextSetMetadata,
    CorpusChunk,
    Outcome,
    Query,
)
from context_engine.learned_selector import (
    ChunkUtility,
    build_learned_context_set,
    build_learned_context_sets,
    estimate_chunk_utility,
    select_with_budget,
)


def _chunk(chunk_id: str, tokens: int, topic: str = "t") -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "x.md",
            "section_path": ["S"],
            "source_type": "doc",
            "text": f"text for {chunk_id}",
            "token_count": tokens,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": topic},
        }
    )


def _context_set(set_id: str, query_id: str, selected_ids: list[str]) -> ContextSet:
    return ContextSet(
        set_id=set_id,
        query_id=query_id,
        candidate_pool_id=f"pool_{query_id}",
        strategy="gold_only",
        selected_ids=selected_ids,
        ordering_type="best_first",
        token_count=sum(len(s) for s in selected_ids),
        metadata=ContextSetMetadata(
            contains_all_gold=True, missing_gold_count=0, distractor_types=[]
        ),
    )


def _outcome(set_id: str, query_id: str, overall: float) -> Outcome:
    return Outcome.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "answer": "a",
            "scores": {
                "correctness": overall,
                "support": overall,
                "overall": overall,
            },
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "evaluator_version": "eval_v1",
        }
    )


def _query(query_id: str, gold_ids: list[str]) -> Query:
    return Query.from_dict(
        {
            "query_id": query_id,
            "query": "Q?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "x",
            "gold_support_ids": gold_ids,
            "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
        }
    )


def _candidate_pool(query_id: str, candidate_ids: list[str]) -> CandidatePool:
    return CandidatePool.from_dict(
        {
            "query_id": query_id,
            "candidate_pool_id": f"pool_{query_id}",
            "candidate_ids": candidate_ids,
            "composition": {
                "gold_count": 1,
                "plausible_count": 10,
                "distractor_count": 6,
                "neutral_count": 3,
            },
            "gold_in_pool": True,
        }
    )


def _util(chunk_id: str, utility: float, n: int = 1) -> ChunkUtility:
    return ChunkUtility(chunk_id=chunk_id, utility=utility, sample_size=n)


def test_estimate_chunk_utility_averages_included_scores() -> None:
    context_sets = [
        _context_set("s1", "q1", ["c1", "c2"]),
        _context_set("s2", "q1", ["c2"]),
        _context_set("s3", "q1", ["c3"]),
    ]
    outcomes = [
        _outcome("s1", "q1", 0.8),
        _outcome("s2", "q1", 0.4),
        _outcome("s3", "q1", 0.2),
    ]
    utilities = estimate_chunk_utility(context_sets, outcomes)
    assert utilities["c1"].utility == 0.8
    assert utilities["c1"].sample_size == 1
    assert utilities["c2"].utility == pytest.approx(0.6)  # (0.8 + 0.4) / 2
    assert utilities["c2"].sample_size == 2
    assert utilities["c3"].utility == pytest.approx(0.2)


def test_estimate_chunk_utility_respects_score_axis() -> None:
    context_sets = [_context_set("s1", "q1", ["c1"])]
    outcomes = [
        Outcome.from_dict(
            {
                "set_id": "s1",
                "query_id": "q1",
                "answer": "a",
                "scores": {"correctness": 0.9, "support": 0.1, "overall": 0.6},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "evaluator_version": "eval_v1",
            }
        )
    ]
    utilities = estimate_chunk_utility(context_sets, outcomes, score_axis="correctness")
    assert utilities["c1"].utility == 0.9


def test_select_with_budget_packs_in_descending_utility_order() -> None:
    pool = _candidate_pool("q1", ["c1", "c2", "c3"])
    chunks_by_id = {
        "c1": _chunk("c1", 100),
        "c2": _chunk("c2", 100),
        "c3": _chunk("c3", 100),
    }
    utilities = {
        "c1": _util("c1", 0.9),
        "c2": _util("c2", 0.5),
        "c3": _util("c3", 0.1),
    }
    selected = select_with_budget(
        candidate_pool=pool, utilities=utilities, chunks_by_id=chunks_by_id, token_budget=250
    )
    assert [s.chunk_id for s in selected] == ["c1", "c2"]
    assert [s.rank for s in selected] == [1, 2]


def test_select_with_budget_drops_chunks_that_would_overshoot() -> None:
    pool = _candidate_pool("q1", ["c1", "c2", "c3"])
    chunks_by_id = {
        "c1": _chunk("c1", 100),
        "c2": _chunk("c2", 200),
        "c3": _chunk("c3", 100),
    }
    utilities = {
        "c1": _util("c1", 0.9),
        "c2": _util("c2", 0.8),
        "c3": _util("c3", 0.7),
    }
    # Budget 250: c1 (100) fits, c2 (200) would overshoot (300 > 250), c3 (100) fits.
    selected = select_with_budget(
        candidate_pool=pool, utilities=utilities, chunks_by_id=chunks_by_id, token_budget=250
    )
    chunk_ids = [s.chunk_id for s in selected]
    assert chunk_ids == ["c1", "c3"]


def test_select_with_budget_respects_pool_tiebreak() -> None:
    """Ties break by candidate-pool position so the run is deterministic."""
    pool = _candidate_pool("q1", ["c1", "c2", "c3"])
    chunks_by_id = {
        "c1": _chunk("c1", 100),
        "c2": _chunk("c2", 100),
        "c3": _chunk("c3", 100),
    }
    utilities = {c: _util(c, 0.5) for c in ("c1", "c2", "c3")}
    selected = select_with_budget(
        candidate_pool=pool, utilities=utilities, chunks_by_id=chunks_by_id, token_budget=200
    )
    assert [s.chunk_id for s in selected] == ["c1", "c2"]


def test_select_with_budget_zero_budget_yields_empty() -> None:
    pool = _candidate_pool("q1", ["c1"])
    chunks_by_id = {"c1": _chunk("c1", 100)}
    utilities = {"c1": _util("c1", 0.9)}
    assert select_with_budget(
        candidate_pool=pool, utilities=utilities, chunks_by_id=chunks_by_id, token_budget=0
    ) == []


def test_select_with_budget_seeds_unseen_chunks_with_zero_utility() -> None:
    """Chunks never observed in training still compete on pool tiebreak."""
    pool = _candidate_pool("q1", ["c1", "c2", "c3"])
    chunks_by_id = {
        "c1": _chunk("c1", 100),
        "c2": _chunk("c2", 100),
        "c3": _chunk("c3", 100),
    }
    # Only c1 was ever observed; c2 and c3 fall back to 0 utility but still pack
    # by pool position when budget permits.
    utilities = {"c1": _util("c1", 0.9)}
    selected = select_with_budget(
        candidate_pool=pool, utilities=utilities, chunks_by_id=chunks_by_id, token_budget=300
    )
    assert [s.chunk_id for s in selected] == ["c1", "c2", "c3"]

def test_build_learned_context_set_uses_strategy_learned() -> None:
    query = _query("q1", ["c1"])
    pool = _candidate_pool("q1", ["c1", "c2"])
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}
    utilities = {
        "c1": _util("c1", 0.9),
        "c2": _util("c2", 0.4),
    }
    context_set = build_learned_context_set(
        query=query,
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    assert context_set.strategy == "learned"
    assert context_set.query_id == "q1"
    assert context_set.selected_ids == ["c1", "c2"]
    assert context_set.metadata.contains_all_gold is True


def test_build_learned_context_sets_emits_one_per_query() -> None:
    queries = [_query("q1", ["c1"]), _query("q2", ["c3"])]
    pools = [_candidate_pool("q1", ["c1", "c2"]), _candidate_pool("q2", ["c3", "c4"])]
    context_sets = [
        _context_set("s1", "q1", ["c1", "c2"]),
        _context_set("s2", "q2", ["c3", "c4"]),
    ]
    outcomes = [_outcome("s1", "q1", 0.9), _outcome("s2", "q2", 0.5)]
    chunks_by_id = {c: _chunk(c, 100) for c in ("c1", "c2", "c3", "c4")}
    learned = build_learned_context_sets(
        queries=queries,
        candidate_pools=pools,
        context_sets=context_sets,
        outcomes=outcomes,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    assert [cs.set_id for cs in learned] == ["q1_learned", "q2_learned"]
    assert all(cs.strategy == "learned" for cs in learned)


def test_learned_context_set_round_trips_through_contract_validator(tmp_path) -> None:
    """Learned sets must satisfy the data contract like any other strategy."""
    from context_engine.io import load_jsonl, write_jsonl
    from context_engine.validation import validate_jsonl_file

    query = _query("q1", ["c1"])
    pool = _candidate_pool("q1", ["c1", "c2"])
    chunks_by_id = {"c1": _chunk("c1", 100), "c2": _chunk("c2", 100)}
    utilities = {
        "c1": _util("c1", 0.9),
        "c2": _util("c2", 0.4),
    }
    learned = build_learned_context_set(
        query=query,
        candidate_pool=pool,
        utilities=utilities,
        chunks_by_id=chunks_by_id,
        token_budget=200,
    )
    target = tmp_path / "context_sets_learned_v1.jsonl"
    write_jsonl(target, [learned.to_dict()])

    summary = validate_jsonl_file(target)
    assert summary.artifact_name == "context_sets"
    assert summary.row_count == 1
    assert load_jsonl(target)[0]["strategy"] == "learned"
