"""Tests for scripts/build_candidate_pools.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from context_engine.artifacts import CandidatePool, Query, CorpusChunk


def _chunk(chunk_id: str, text: str, topic: str = "authentication") -> CorpusChunk:
    return CorpusChunk.from_dict(
        {
            "chunk_id": chunk_id,
            "doc_version": "16",
            "doc_path": "x.md",
            "section_path": ["S"],
            "source_type": "doc",
            "text": text,
            "token_count": 100,
            "chunk_index": 1,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"topic": topic},
        }
    )


def _query(query_id: str, text: str, gold_id: str) -> Query:
    return Query.from_dict(
        {
            "query_id": query_id,
            "query": text,
            "task_type": "doc_qa",
            "difficulty": "easy",
            "gold_answer": "x",
            "gold_support_ids": [gold_id],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": False,
                "question_family": "fact_lookup",
            },
        }
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus_path = tmp_path / "corpus_chunks_v1.jsonl"
    queries_path = tmp_path / "queries_v1.jsonl"
    config_path = tmp_path / "config.json"

    chunks = [
        _chunk("c1", "pg_hba.conf controls client authentication"),
        _chunk("c2", "pg_hba.conf is mentioned here"),
        _chunk("c3", "totally unrelated content"),
        _chunk("c4", "pg_hba.conf authentication rules"),
        _chunk("gold", "pg_hba.conf controls authentication rules"),
    ]
    queries = [
        _query("q1", "pg_hba.conf controls authentication", "gold"),
    ]
    _write_jsonl(corpus_path, [c.to_dict() for c in chunks])
    _write_jsonl(queries_path, [q.to_dict() for q in queries])

    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "test",
                "model_name": "stub",
                "selector_strategy": "default",
                "evaluator_version": "eval_v1",
                "dataset_dir": str(tmp_path),
                "artifact_version": "v1",
            }
        ),
        encoding="utf-8",
    )
    return config_path, corpus_path, queries_path


def _run_build(monkeypatch, tmp_path: Path, *args):
    """Run scripts/build_candidate_pools.py against tmp_path."""
    config_path, _, _ = _setup_workspace(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_candidate_pools", "--config", str(config_path), *args],
    )
    spec = importlib.util.spec_from_file_location(
        "build_candidate_pools", "scripts/build_candidate_pools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def test_build_candidate_pools_bm25_includes_gold(tmp_path: Path, monkeypatch):
    rc = _run_build(monkeypatch, tmp_path, "--retriever", "bm25")
    assert rc == 0
    pools_path = tmp_path / "candidate_pools_v1.jsonl"
    assert pools_path.exists()
    pools = [
        CandidatePool.from_dict(json.loads(line))
        for line in pools_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(pools) == 1
    assert "gold" in pools[0].candidate_ids
    assert pools[0].gold_in_pool


def test_build_candidate_pools_bm25_exact_includes_gold(tmp_path: Path, monkeypatch):
    rc = _run_build(monkeypatch, tmp_path, "--retriever", "bm25_exact")
    assert rc == 0
    pools_path = tmp_path / "candidate_pools_v1.jsonl"
    pools = [
        CandidatePool.from_dict(json.loads(line))
        for line in pools_path.read_text().splitlines()
        if line.strip()
    ]
    assert pools[0].gold_in_pool
    assert "gold" in pools[0].candidate_ids


def test_build_candidate_pools_rejects_unknown_retriever(tmp_path: Path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_build(monkeypatch, tmp_path, "--retriever", "bogus")
