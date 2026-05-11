from pathlib import Path

from context_engine.artifacts import ContextSet
from context_engine.io import write_jsonl
from scripts.generate_model_outcomes import _filter_context_sets, _load_existing_outcomes


def test_load_existing_outcomes_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert _load_existing_outcomes(tmp_path / "missing.jsonl") == []


def test_load_existing_outcomes_reads_rows(tmp_path: Path) -> None:
    target = tmp_path / "outcomes.jsonl"
    write_jsonl(target, [{"set_id": "s1", "query_id": "q1"}])
    rows = _load_existing_outcomes(target)
    assert rows == [{"query_id": "q1", "set_id": "s1"}]


def test_filter_context_sets_supports_query_limit_and_start_at() -> None:
    context_sets = [
        ContextSet.from_dict(
            {
                "set_id": set_id,
                "query_id": query_id,
                "candidate_pool_id": f"pool_{query_id}",
                "strategy": "gold_only",
                "selected_ids": ["c1"],
                "ordering_type": "best_first",
                "token_count": 10,
                "metadata": {
                    "contains_all_gold": True,
                    "missing_gold_count": 0,
                    "distractor_types": [],
                },
            }
        )
        for set_id, query_id in [
            ("q1_a", "q1"),
            ("q1_b", "q1"),
            ("q2_a", "q2"),
            ("q2_b", "q2"),
        ]
    ]
    filtered = _filter_context_sets(
        context_sets,
        query_ids={"q2"},
        set_ids=None,
        start_at="q2_a",
        limit=1,
    )
    assert [context_set.set_id for context_set in filtered] == ["q2_a"]
