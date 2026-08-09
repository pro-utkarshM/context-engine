"""Tests for the replications aggregation module."""

from __future__ import annotations

import json

import pytest

from context_engine.artifacts import ContextSet, Outcome, ScoreBundle
from context_engine.replications import (
    ReplicationPairedDelta,
    ReplicationStrategySummary,
    ReplicationSummary,
    paired_summary,
    summarize_replications,
)


def _context_set(set_id, query_id, strategy):
    return ContextSet.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "candidate_pool_id": f"pool_{query_id}",
            "strategy": strategy,
            "selected_ids": ["c1"],
            "ordering_type": "best_first",
            "token_count": 100,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": [],
            },
        }
    )


def _outcome(set_id, query_id, overall):
    return Outcome.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "answer": "ans",
            "scores": {
                "correctness": overall,
                "support": overall,
                "overall": overall,
            },
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "latency_ms": 0,
            "evaluator_version": "eval_v1",
        }
    )


def _synthetic_run(strategy_to_scores):
    """Build a single run: {strategy: [per_query_overall, ...]}."""
    context_sets: list[ContextSet] = []
    outcomes: list[Outcome] = []
    for strategy, scores in strategy_to_scores.items():
        for query_index, score in enumerate(scores):
            query_id = f"q_{query_index:03d}"
            set_id = f"{query_id}_{strategy}"
            context_sets.append(_context_set(set_id, query_id, strategy))
            outcomes.append(_outcome(set_id, query_id, score))
    return context_sets, outcomes


def test_summarize_replications_aggregates_per_run_means():
    """Three runs of gold_only with deltas [0.10, 0.12, 0.08] -> mean ~0.10."""
    runs = [
        _synthetic_run({"gold_only": [0.50, 0.60, 0.70]}),
        _synthetic_run({"gold_only": [0.52, 0.62, 0.72]}),
        _synthetic_run({"gold_only": [0.48, 0.58, 0.68]}),
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )

    assert summary.n_runs == 3
    assert summary.experiment_name == "exp"
    assert summary.runner == "minimax"
    assert len(summary.strategies) == 1
    strategy = summary.strategies[0]
    assert strategy.strategy == "gold_only"
    assert strategy.n_runs == 3
    assert strategy.n_queries_per_run == 3
    assert strategy.run_means == pytest.approx([0.60, 0.62, 0.58], abs=1e-9)
    assert strategy.run_mean_summary.mean == pytest.approx(0.60, abs=1e-9)
    assert strategy.run_mean_summary.ci_low > 0.55
    assert strategy.run_mean_summary.ci_high < 0.65


def test_summarize_replications_within_run_view_uses_first_run():
    runs = [
        _synthetic_run({"gold_only": [0.10, 0.20, 0.30, 0.40]}),
        _synthetic_run({"gold_only": [0.50, 0.50, 0.50, 0.50]}),
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )
    strategy = summary.strategies[0]
    assert strategy.within_run_query_means == [0.10, 0.20, 0.30, 0.40]
    assert strategy.within_run_query_summary.mean == pytest.approx(0.25, abs=1e-9)


def test_summarize_replications_multiple_strategies_sorted_by_name():
    runs = [
        _synthetic_run({
            "gold_only": [0.5, 0.5],
            "shuffled_order": [0.7, 0.7],
            "gold_plus_distractors": [0.6, 0.6],
        }),
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )
    names = [entry.strategy for entry in summary.strategies]
    assert names == ["gold_only", "gold_plus_distractors", "shuffled_order"]


def test_summarize_replications_identical_runs_collapse_to_point_ci():
    """Same outcome rows across runs -> per-run mean is constant -> CI is a point."""
    runs = [
        _synthetic_run({"gold_only": [0.5, 0.6, 0.7]})
        for _ in range(5)
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )
    strategy = summary.strategies[0]
    assert strategy.run_mean_summary.mean == pytest.approx(0.6, abs=1e-9)
    assert strategy.run_mean_summary.ci_low == pytest.approx(0.6, abs=1e-9)
    assert strategy.run_mean_summary.ci_high == pytest.approx(0.6, abs=1e-9)
    assert strategy.run_mean_summary.std == 0.0


def test_summarize_replications_to_dict_is_serializable():
    runs = [
        _synthetic_run({"gold_only": [0.5, 0.6, 0.7]}),
        _synthetic_run({"gold_only": [0.55, 0.65, 0.75]}),
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )
    payload = summary.to_dict()
    # Round-trip through JSON to lock the wire shape.
    text = json.dumps(payload)
    decoded = json.loads(text)
    assert decoded["experiment_name"] == "exp"
    assert decoded["n_runs"] == 2
    assert decoded["ci_level"] == 0.95
    assert len(decoded["strategies"]) == 1
    entry = decoded["strategies"][0]
    assert entry["strategy"] == "gold_only"
    assert entry["run_means"] == pytest.approx([0.6, 0.65], abs=1e-6)
    assert "ci_low" in entry["run_mean_summary"]
    assert "ci_high" in entry["run_mean_summary"]
    assert "p_value_two_sided" not in entry["run_mean_summary"]


def test_summarize_replications_rejects_no_runs():
    with pytest.raises(ValueError, match="at least one run"):
        summarize_replications(
            [],
            experiment_name="exp",
            runner="minimax",
            model_name="MiniMax-M3",
            artifact_version="v1",
        )


def test_paired_summary_reports_negative_delta_for_left_minus_right():
    """auto - canonical with auto lower than canonical -> negative delta."""
    left = _synthetic_run({"gold_only": [0.7, 0.8, 0.9, 0.6, 0.5]})  # auto
    right = _synthetic_run({"gold_only": [0.5, 0.6, 0.7, 0.4, 0.3]})  # canonical
    pairs = paired_summary(
        [left],
        [right],
        left_pool_source="auto",
        right_pool_source="canonical",
        seed=0,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.strategy == "gold_only"
    assert pair.n_queries == 5
    assert pair.left_pool_source == "auto"
    assert pair.right_pool_source == "canonical"
    assert pair.delta_summary.mean_delta == pytest.approx(0.2, abs=1e-9)
    assert pair.delta_summary.ci_low > 0.15


def test_paired_summary_skips_strategies_missing_on_either_side():
    """Strategies on one side only should not appear in the paired output."""
    left = _synthetic_run({"gold_only": [0.5, 0.6]})
    right = _synthetic_run({"gold_only": [0.7, 0.8], "shuffled_order": [0.4, 0.5]})
    pairs = paired_summary(
        [left],
        [right],
        left_pool_source="auto",
        right_pool_source="canonical",
        seed=0,
    )
    # Only gold_only is common across both sides.
    assert [pair.strategy for pair in pairs] == ["gold_only"]


def test_paired_summary_to_dict_includes_p_value():
    left = _synthetic_run({"gold_only": [0.7, 0.8, 0.9, 0.6, 0.5]})
    right = _synthetic_run({"gold_only": [0.5, 0.6, 0.7, 0.4, 0.3]})
    pairs = paired_summary(
        [left],
        [right],
        left_pool_source="auto",
        right_pool_source="canonical",
        seed=0,
    )
    payload = pairs[0].to_dict()
    assert "p_value_two_sided" in payload["delta_summary"]
    assert payload["delta_summary"]["p_value_two_sided"] < 0.05


def test_paired_summary_requires_runs_on_each_side():
    left = _synthetic_run({"gold_only": [0.5, 0.6]})
    right = _synthetic_run({"gold_only": [0.7, 0.8]})
    with pytest.raises(ValueError, match="at least one run"):
        paired_summary([], [right], left_pool_source="a", right_pool_source="b")
    with pytest.raises(ValueError, match="at least one run"):
        paired_summary([left], [], left_pool_source="a", right_pool_source="b")


def test_replication_dataclass_field_shapes_are_stable():
    """Lock the JSON shape so callers see a stable contract."""
    strategy_fields = set(ReplicationStrategySummary.__dataclass_fields__.keys())
    assert strategy_fields == {
        "strategy",
        "n_runs",
        "n_queries_per_run",
        "run_means",
        "run_mean_summary",
        "within_run_query_means",
        "within_run_query_summary",
    }
    pair_fields = set(ReplicationPairedDelta.__dataclass_fields__.keys())
    assert pair_fields == {
        "strategy",
        "n_queries",
        "left_pool_source",
        "right_pool_source",
        "delta_summary",
    }
    summary_fields = set(ReplicationSummary.__dataclass_fields__.keys())
    assert summary_fields == {
        "experiment_name",
        "runner",
        "model_name",
        "artifact_version",
        "n_runs",
        "n_resamples",
        "ci_level",
        "seed",
        "strategies",
        "paired",
    }
