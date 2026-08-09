import json

from context_engine.analysis import (
    best_strategy_per_query,
    per_set_rows,
    render_csv_per_query,
    render_json_report,
    render_markdown_report,
    render_text_report,
    summarize_by_strategy,
)
from context_engine.artifacts import ContextSet, MarginalImpact, Outcome, Query


def _context_set(set_id: str, query_id: str, strategy: str, selected_ids=None, distractor_types=None, token_count=100) -> ContextSet:
    return ContextSet.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "candidate_pool_id": f"pool_{query_id}",
            "strategy": strategy,
            "selected_ids": selected_ids or ["c1"],
            "ordering_type": "best_first",
            "token_count": token_count,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": distractor_types or [],
            },
        }
    )


def _outcome(set_id: str, query_id: str, correctness: float, support: float, overall: float) -> Outcome:
    return Outcome.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "answer": "answer",
            "scores": {
                "correctness": correctness,
                "support": support,
                "overall": overall,
            },
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "latency_ms": 0,
            "evaluator_version": "eval_v1",
        }
    )


def _query(query_id: str, gold_ids: list[str]) -> Query:
    return Query.from_dict(
        {
            "query_id": query_id,
            "query": "What?",
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "x",
            "gold_support_ids": gold_ids,
            "metadata": {"topic": "t", "requires_multi_hop": False, "question_family": "fact_lookup"},
        }
    )


def test_summarize_by_strategy_aggregates_means() -> None:
    summaries = summarize_by_strategy(
        context_sets=[
            _context_set("s1", "q1", "gold_only"),
            _context_set("s2", "q2", "gold_only"),
            _context_set("s3", "q1", "shuffled_order"),
        ],
        outcomes=[
            _outcome("s1", "q1", 1.0, 1.0, 0.9),
            _outcome("s2", "q2", 0.5, 1.0, 0.6),
            _outcome("s3", "q1", 0.0, 0.5, 0.2),
        ],
    )

    gold_only = next(summary for summary in summaries if summary.strategy == "gold_only")
    assert gold_only.run_count == 2
    assert gold_only.avg_correctness == 0.75


def test_best_strategy_per_query_picks_highest_overall() -> None:
    results = best_strategy_per_query(
        context_sets=[
            _context_set("s1", "q1", "gold_only"),
            _context_set("s2", "q1", "shuffled_order"),
        ],
        outcomes=[
            _outcome("s1", "q1", 1.0, 1.0, 0.9),
            _outcome("s2", "q1", 0.0, 1.0, 0.3),
        ],
    )

    assert results == [results[0]]
    assert results[0].query_id == "q1"
    assert results[0].best_strategy == "gold_only"


def test_render_text_report_contains_expected_sections() -> None:
    report = render_text_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 1.0, 1.0, 0.9)],
    )
    assert "Strategy Summary" in report
    assert "Best Strategy Per Query" in report
    assert "gold_only" in report


def test_render_json_report_is_stable_and_well_typed() -> None:
    context_sets = [
        _context_set("s2", "q1", "shuffled_order"),
        _context_set("s1", "q1", "gold_only"),
    ]
    outcomes = [
        _outcome("s2", "q1", 0.0, 1.0, 0.3),
        _outcome("s1", "q1", 1.0, 1.0, 0.9),
    ]

    raw = render_json_report(context_sets, outcomes)
    payload = json.loads(raw)

    assert [row["strategy"] for row in payload["strategy_summary"]] == sorted(
        row["strategy"] for row in payload["strategy_summary"]
    )
    assert [row["set_id"] for row in payload["per_set"]] == ["s1", "s2"]
    assert payload["best_strategy_per_query"][0]["best_strategy"] == "gold_only"
    assert isinstance(payload["strategy_summary"][0]["avg_correctness"], float)


def test_render_csv_per_query_emits_header_and_rows() -> None:
    csv_text = render_csv_per_query(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 1.0, 1.0, 0.9)],
    )
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("set_id,query_id,strategy")
    assert len(lines) == 2
    assert lines[1].startswith("s1,q1,gold_only")


def test_render_markdown_report_includes_distractor_wins_view() -> None:
    md = render_markdown_report(
        context_sets=[
            _context_set("s1", "q1", "gold_only"),
            _context_set("s2", "q1", "gold_plus_distractors"),
        ],
        outcomes=[
            _outcome("s1", "q1", 1.0, 1.0, 0.7),
            _outcome("s2", "q1", 0.5, 0.8, 0.9),
        ],
    )
    assert "## Distractor-Heavy vs Concise-Context Wins" in md
    assert "distractor_heavy_wins" in md


def test_render_markdown_report_predicted_vs_measured_view() -> None:
    queries_by_id = {
        "q1": _query("q1", ["c1"]),
        "q2": _query("q2", ["c2"]),
    }
    impacts = [
        MarginalImpact.from_dict(
            {"query_id": "q1", "base_set_id": "s1", "chunk_id": "c1",
             "operation": "remove", "base_score": 0.7, "new_score": 0.9, "delta": 0.2}
        ),
        MarginalImpact.from_dict(
            {"query_id": "q1", "base_set_id": "s1", "chunk_id": "c9",
             "operation": "remove", "base_score": 0.7, "new_score": 0.7, "delta": 0.0}
        ),
        MarginalImpact.from_dict(
            {"query_id": "q2", "base_set_id": "s2", "chunk_id": "c2",
             "operation": "remove", "base_score": 0.6, "new_score": 0.8, "delta": 0.2}
        ),
    ]
    md = render_markdown_report(
        context_sets=[_context_set("s1", "q1", "gold_plus_distractors")],
        outcomes=[_outcome("s1", "q1", 1.0, 1.0, 0.7)],
        marginal_impacts=impacts,
        queries_by_id=queries_by_id,
    )
    assert "Predicted vs Measured Marginal Impact" in md
    assert "| gold |" in md
    assert "| distractor |" in md
    assert "| unknown |" not in md


def test_render_json_report_includes_marginal_impact_summary_when_provided() -> None:
    queries_by_id = {"q1": _query("q1", ["c1"])}
    impacts = [
        MarginalImpact.from_dict(
            {"query_id": "q1", "base_set_id": "s1", "chunk_id": "c1",
             "operation": "remove", "base_score": 0.7, "new_score": 0.9, "delta": 0.2}
        )
    ]
    raw = render_json_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 1.0, 1.0, 0.9)],
        marginal_impacts=impacts,
    )
    payload = json.loads(raw)
    assert "marginal_impact_summary" in payload
    assert payload["marginal_impact_summary"]["row_count"] == 1


def test_per_set_rows_are_sorted_and_complete() -> None:
    rows = per_set_rows(
        context_sets=[
            _context_set("s2", "q1", "gold_only", selected_ids=["c1"], distractor_types=["stale"]),
            _context_set("s1", "q1", "gold_only", selected_ids=["c1"]),
        ],
        outcomes=[
            _outcome("s2", "q1", 1.0, 1.0, 0.8),
            _outcome("s1", "q1", 1.0, 1.0, 0.9),
        ],
    )
    assert [row.set_id for row in rows] == ["s1", "s2"]
    assert rows[0].distractor_types == []
    assert rows[1].distractor_types == ["stale"]


def _build_replication_summary():
    """Build a ReplicationSummary with 3 runs of 5 strategies × 4 queries."""
    from context_engine.replications import summarize_replications

    strategies = ("gold_only", "shuffled_order", "gold_plus_distractors", "topk_pool_order", "minimal_support")
    runs = []
    for run_seed in (0.10, 0.12, 0.08):
        context_sets = []
        outcomes = []
        for strategy in strategies:
            for i in range(4):
                score = min(1.0, max(0.0, run_seed + (0.05 * i) + (0.02 if strategy == "gold_plus_distractors" else 0.0)))
                set_id = f"q{i}_{strategy}"
                context_sets.append(_context_set(set_id, f"q{i}", strategy))
                outcomes.append(_outcome(set_id, f"q{i}", score, score, score))
        runs.append((context_sets, outcomes))
    return summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )


def test_render_json_report_includes_replication_summary_when_provided():
    summary = _build_replication_summary()
    raw = render_json_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
        replication_summary=summary,
    )
    payload = json.loads(raw)
    assert "replication_summary" in payload
    assert payload["replication_summary"]["n_runs"] == 3
    assert len(payload["replication_summary"]["strategies"]) == 5
    entry = payload["replication_summary"]["strategies"][0]
    assert "ci_low" in entry["run_mean_summary"]
    assert "ci_high" in entry["run_mean_summary"]
    assert entry["run_mean_summary"]["ci_low"] < entry["run_mean_summary"]["mean"]
    assert entry["run_mean_summary"]["ci_high"] > entry["run_mean_summary"]["mean"]


def test_render_json_report_omits_replication_summary_when_not_provided():
    raw = render_json_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
    )
    payload = json.loads(raw)
    assert "replication_summary" not in payload


def test_render_markdown_report_includes_replication_ci_table():
    summary = _build_replication_summary()
    md = render_markdown_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
        replication_summary=summary,
    )
    assert "## Replication Confidence Intervals" in md
    assert "ci_low" in md
    assert "ci_high" in md
    assert "reliability" in md
    assert "gold_only" in md
    assert "shuffled_order" in md


def test_render_markdown_report_includes_pool_comparison_when_paired_present():
    summary = _build_replication_summary()
    from context_engine.replications import paired_summary

    left = (
        [_context_set(f"q{i}_gold_only", f"q{i}", "gold_only") for i in range(4)],
        [_outcome(f"q{i}_gold_only", f"q{i}", 0.7, 0.7, 0.7) for i in range(4)],
    )
    right = (
        [_context_set(f"q{i}_gold_only", f"q{i}", "gold_only") for i in range(4)],
        [_outcome(f"q{i}_gold_only", f"q{i}", 0.5, 0.5, 0.5) for i in range(4)],
    )
    pairs = paired_summary([left], [right], left_pool_source="auto", right_pool_source="canonical", seed=0)
    summary = summary.with_paired(pairs)

    md = render_markdown_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
        replication_summary=summary,
    )
    assert "## Pool Comparison" in md
    assert "auto - canonical" in md
    assert "delta" in md
    assert "ci_low" in md
    assert "ci_high" in md


def test_render_markdown_report_omits_replication_section_when_not_provided():
    md = render_markdown_report(
        context_sets=[_context_set("s1", "q1", "gold_only")],
        outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
    )
    assert "Replication Confidence Intervals" not in md
    assert "Pool Comparison" not in md


def test_replication_json_shape_is_additive_to_existing_report():
    """Adding the replication section should not drop any existing section."""
    summary = _build_replication_summary()
    base_payload = json.loads(
        render_json_report(
            context_sets=[_context_set("s1", "q1", "gold_only")],
            outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
        )
    )
    full_payload = json.loads(
        render_json_report(
            context_sets=[_context_set("s1", "q1", "gold_only")],
            outcomes=[_outcome("s1", "q1", 0.7, 0.7, 0.7)],
            replication_summary=summary,
        )
    )
    assert set(base_payload.keys()) <= set(full_payload.keys())
    assert "replication_summary" in full_payload
