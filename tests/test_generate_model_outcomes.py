import json
import sys
from pathlib import Path

import pytest

from context_engine.artifacts import ContextSet
from context_engine.io import load_jsonl, write_jsonl
from scripts.generate_model_outcomes import (
    _filter_context_sets,
    _load_existing_outcomes,
    main as generate_model_outcomes_main,
)


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


def _seed_full_pipeline(directory: Path) -> None:
    write_jsonl(
        directory / "corpus_chunks_v1.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_version": "16",
                "doc_path": "x.html",
                "section_path": ["A"],
                "source_type": "doc",
                "text": "alpha",
                "token_count": 10,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "metadata": {"topic": "auth", "subtopic": "x"},
            }
        ],
    )
    write_jsonl(
        directory / "queries_v1.jsonl",
        [
            {
                "query_id": "q1",
                "query": "what?",
                "task_type": "doc_qa",
                "difficulty": "easy",
                "gold_answer": "alpha",
                "gold_support_ids": ["c1"],
                "metadata": {
                    "topic": "auth",
                    "requires_multi_hop": False,
                    "question_family": "fact_lookup",
                },
            }
        ],
    )
    write_jsonl(
        directory / "context_sets_v1.jsonl",
        [
            {
                "set_id": "q1_gold_only",
                "query_id": "q1",
                "candidate_pool_id": "pool_q1",
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
        ],
    )


def test_main_runs_with_config_and_dataset_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_full_pipeline(tmp_path)
    config_path = tmp_path / "exp.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "test",
                "model_name": "stub-model",
                "selector_strategy": "default",
                "evaluator_version": "eval_v1_test",
                "dataset_dir": str(tmp_path),
                "artifact_version": "v1",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_model_outcomes",
            "--config",
            str(config_path),
            "--runner",
            "stub",
        ],
    )
    rc = generate_model_outcomes_main()
    assert rc == 0
    output = tmp_path / "outcomes_model_stub_v1.jsonl"
    assert output.exists()
    rows = load_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["set_id"] == "q1_gold_only"
    assert rows[0]["evaluator_version"] == "eval_v1_test"
