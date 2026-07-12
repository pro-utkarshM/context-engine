"""Tests for the marginal-impact computation module."""

from __future__ import annotations

from context_engine.artifacts import (
    ContextSet,
    ContextSetMetadata,
    MarginalImpact,
    Outcome,
    ScoreBundle,
)
from context_engine.marginal_impact import (
    MarginalImpactError,
    compute_marginal_impact,
    evaluate_marginal_impact,
)


def _context_set(set_id: str, selected_ids: list[str], strategy: str = "gold_only") -> ContextSet:
    return ContextSet(
        set_id=set_id,
        query_id="q_0001",
        candidate_pool_id="pool_q_0001_v1",
        strategy=strategy,
        selected_ids=selected_ids,
        ordering_type="best_first",
        token_count=sum(range(1, len(selected_ids) + 1)),
        metadata=ContextSetMetadata(
            contains_all_gold=True,
            missing_gold_count=0,
            distractor_types=[],
        ),
    )


def _outcome(set_id: str, overall: float) -> Outcome:
    return Outcome(
        set_id=set_id,
        query_id="q_0001",
        answer="stub",
        scores=ScoreBundle(correctness=overall, support=overall, overall=overall),
        prompt_tokens=100,
        completion_tokens=1,
        latency_ms=0,
        evaluator_version="eval_v1_model_runner",
    )


def test_compute_marginal_impact_signs_delta() -> None:
    base = _context_set("q1_base", ["c1", "c2"])
    row = compute_marginal_impact(
        base_set=base,
        chunk_id="c1",
        operation="remove",
        base_score=0.8,
        new_score=0.9,
    )
    assert isinstance(row, MarginalImpact)
    assert row.delta == 0.1
    assert row.base_score == 0.8
    assert row.new_score == 0.9
    assert row.operation == "remove"


def test_compute_marginal_impact_negative_delta_when_chunk_helps() -> None:
    base = _context_set("q1_base", ["c1"])
    row = compute_marginal_impact(
        base_set=base,
        chunk_id="c1",
        operation="remove",
        base_score=0.95,
        new_score=0.30,
    )
    assert row.delta < 0
    assert abs(row.delta - (-0.65)) < 1e-9


def test_evaluate_marginal_impact_add_path() -> None:
    base = _context_set("q1_base", ["c1", "c2"])

    def scorer(context_set: ContextSet) -> Outcome:
        return _outcome(context_set.set_id, overall=0.5 if "c3" in context_set.selected_ids else 0.4)

    row = evaluate_marginal_impact(
        base_set=base,
        chunk_id="c3",
        operation="add",
        scorer=scorer,
    )
    assert row.operation == "add"
    assert row.chunk_id == "c3"
    assert row.base_set_id == "q1_base"
    assert row.base_score == 0.4
    assert row.new_score == 0.5
    assert row.delta == 0.1


def test_evaluate_marginal_impact_remove_path() -> None:
    base = _context_set("q1_base", ["c1", "c2", "c3"])

    def scorer(context_set: ContextSet) -> Outcome:
        # Removing c1 raises the score — c1 is a distractor.
        return _outcome(context_set.set_id, overall=0.7 if "c1" not in context_set.selected_ids else 0.4)

    row = evaluate_marginal_impact(
        base_set=base,
        chunk_id="c1",
        operation="remove",
        scorer=scorer,
    )
    assert row.operation == "remove"
    assert row.base_score == 0.4
    assert row.new_score == 0.7
    assert row.delta == 0.3


def test_evaluate_marginal_impact_score_key_isolates_components() -> None:
    base = _context_set("q1_base", ["c1", "c2"])

    def scorer(context_set: ContextSet) -> Outcome:
        return Outcome(
            set_id=context_set.set_id,
            query_id="q_0001",
            answer="x",
            scores=ScoreBundle(correctness=0.8, support=0.6, overall=0.74),
            prompt_tokens=100,
            completion_tokens=1,
            latency_ms=0,
            evaluator_version="eval_v1_model_runner",
        )

    add_corr = evaluate_marginal_impact(
        base_set=base, chunk_id="c3", operation="add", scorer=scorer, score_key="correctness"
    )
    add_support = evaluate_marginal_impact(
        base_set=base, chunk_id="c3", operation="add", scorer=scorer, score_key="support"
    )
    assert add_corr.delta == 0.0
    assert add_support.delta == 0.0


def test_evaluate_marginal_impact_rejects_duplicate_add() -> None:
    base = _context_set("q1_base", ["c1", "c2"])

    def scorer(context_set: ContextSet) -> Outcome:
        return _outcome(context_set.set_id, 0.5)

    try:
        evaluate_marginal_impact(base_set=base, chunk_id="c1", operation="add", scorer=scorer)
    except MarginalImpactError as exc:
        assert "already in base set" in str(exc)
    else:
        raise AssertionError("expected MarginalImpactError on duplicate add")


def test_evaluate_marginal_impact_rejects_missing_remove() -> None:
    base = _context_set("q1_base", ["c1", "c2"])

    def scorer(context_set: ContextSet) -> Outcome:
        return _outcome(context_set.set_id, 0.5)

    try:
        evaluate_marginal_impact(base_set=base, chunk_id="c9", operation="remove", scorer=scorer)
    except MarginalImpactError as exc:
        assert "not in base set" in str(exc)
    else:
        raise AssertionError("expected MarginalImpactError on missing remove")


def test_evaluate_marginal_impact_rejects_last_chunk_remove() -> None:
    base = _context_set("q1_base", ["c1"])

    def scorer(context_set: ContextSet) -> Outcome:
        return _outcome(context_set.set_id, 0.5)

    try:
        evaluate_marginal_impact(base_set=base, chunk_id="c1", operation="remove", scorer=scorer)
    except MarginalImpactError as exc:
        assert "last chunk" in str(exc)
    else:
        raise AssertionError("expected MarginalImpactError on last-chunk remove")


def test_marginal_impact_round_trips_through_artifact_validator(tmp_path) -> None:
    """A row produced by the helper must validate as a marginal_impact artifact."""
    from context_engine.io import load_jsonl, write_jsonl
    from context_engine.validation import validate_jsonl_file

    base = _context_set("q1_base", ["c1", "c2"])
    row = compute_marginal_impact(
        base_set=base,
        chunk_id="c2",
        operation="remove",
        base_score=0.5,
        new_score=0.7,
    )
    target = tmp_path / "marginal_impact_v1.jsonl"
    write_jsonl(target, [row.to_dict()])

    rows = load_jsonl(target)
    assert rows[0]["delta"] == 0.2

    summary = validate_jsonl_file(target)
    assert summary.artifact_name == "marginal_impact"
    assert summary.row_count == 1