from context_engine.analysis import best_strategy_per_query, render_text_report, summarize_by_strategy
from context_engine.artifacts import ContextSet, Outcome


def _context_set(set_id: str, query_id: str, strategy: str) -> ContextSet:
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
