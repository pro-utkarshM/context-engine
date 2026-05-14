from __future__ import annotations

import sys
from pathlib import Path

import pytest

from context_engine.io import load_jsonl, write_jsonl
from scripts.generate_context_sets import main as generate_context_sets_main


def _seed_artifacts(directory: Path) -> None:
    write_jsonl(
        directory / "corpus_chunks_v1.jsonl",
        [
            {
                "chunk_id": "gold_a",
                "doc_version": "16",
                "doc_path": "client-authentication.html",
                "section_path": ["Client Authentication", "pg_hba.conf"],
                "source_type": "doc",
                "text": "Sample text about pg_hba.conf",
                "token_count": 10,
                "chunk_index": 1,
                "prev_chunk_id": None,
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "pg_hba.conf"},
            },
            {
                "chunk_id": "d1",
                "doc_version": "15",
                "doc_path": "client-authentication.html",
                "section_path": ["Client Authentication", "pg_hba.conf"],
                "source_type": "doc",
                "text": "Stale text",
                "token_count": 20,
                "chunk_index": 2,
                "prev_chunk_id": "gold_a",
                "next_chunk_id": None,
                "metadata": {"topic": "authentication", "subtopic": "pg_hba.conf"},
            },
        ],
    )
    write_jsonl(
        directory / "queries_v1.jsonl",
        [
            {
                "query_id": "q_0001",
                "query": "Which file controls authentication?",
                "task_type": "doc_qa",
                "difficulty": "easy",
                "gold_answer": "pg_hba.conf",
                "gold_support_ids": ["gold_a"],
                "metadata": {
                    "topic": "authentication",
                    "requires_multi_hop": False,
                    "question_family": "fact_lookup",
                },
            }
        ],
    )
    write_jsonl(
        directory / "candidate_pools_v1.jsonl",
        [
            {
                "query_id": "q_0001",
                "candidate_pool_id": "pool_q_0001_v1",
                "candidate_ids": ["gold_a", "d1"],
                "composition": {
                    "gold_count": 1,
                    "plausible_count": 0,
                    "distractor_count": 1,
                    "neutral_count": 0,
                },
                "gold_in_pool": True,
                "candidate_metadata": {
                    "gold_a": {"role": "gold"},
                    "d1": {"role": "distractor", "distractor_type": "stale"},
                },
            }
        ],
    )


def test_generate_context_sets_writes_under_dataset_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_artifacts(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["generate_context_sets", "--dataset-dir", str(tmp_path)]
    )
    rc = generate_context_sets_main()
    assert rc == 0

    output = tmp_path / "context_sets_v1.jsonl"
    assert output.exists()
    rows = load_jsonl(output)
    assert len(rows) == 5  # 5 default strategies × 1 query
    strategies = {row["strategy"] for row in rows}
    assert "gold_only" in strategies
    assert "gold_plus_distractors" in strategies


def test_generate_context_sets_respects_artifact_version_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "alt"
    target_dir.mkdir()
    _seed_artifacts(tmp_path)
    for src_name, dst_name in [
        ("corpus_chunks_v1.jsonl", "corpus_chunks_v2.jsonl"),
        ("queries_v1.jsonl", "queries_v2.jsonl"),
        ("candidate_pools_v1.jsonl", "candidate_pools_v2.jsonl"),
    ]:
        (tmp_path / src_name).rename(target_dir / dst_name)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_context_sets",
            "--dataset-dir",
            str(target_dir),
            "--artifact-version",
            "v2",
        ],
    )
    rc = generate_context_sets_main()
    assert rc == 0
    assert (target_dir / "context_sets_v2.jsonl").exists()
