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

def test_build_candidate_pools_multi_hop_includes_all_gold(tmp_path: Path, monkeypatch):
    """Multi-hop queries (multiple gold chunks) must have ALL gold chunks
    in the candidate pool."""
    config_path = tmp_path / "config.json"
    corpus_path = tmp_path / "corpus_chunks_v1.jsonl"
    queries_path = tmp_path / "queries_v1.jsonl"

    # Simulate a multi-hop query with 2 gold chunks
    chunks = [
        _chunk("gold_a", "primary authentication mechanism"),
        _chunk("gold_b", "fallback authentication mechanism"),
        _chunk("c3", "unrelated block"),
        _chunk("c4", "description of auth"),
        _chunk("c5", "history of auth"),
    ]
    queries = [
        _query("q1", "primary authentication and fallback behavior", "gold_a"),
    ]
    # Manually set multiple gold chunks
    queries[0] = Query.from_dict({
        "query_id": "q1",
        "query": "primary authentication and fallback behavior",
        "task_type": "doc_qa",
        "difficulty": "hard",
        "gold_answer": "x",
        "gold_support_ids": ["gold_a", "gold_b"],
        "metadata": {
            "topic": "authentication",
            "requires_multi_hop": True,
            "question_family": "comparison",
        },
    })
    _write_jsonl(corpus_path, [c.to_dict() for c in chunks])
    _write_jsonl(queries_path, [q.to_dict() for q in queries])

    config_path.write_text(
        json.dumps({
            "experiment_name": "test",
            "model_name": "stub",
            "selector_strategy": "default",
            "evaluator_version": "eval_v1",
            "dataset_dir": str(tmp_path),
            "artifact_version": "v1",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_candidate_pools", "--config", str(config_path), "--retriever", "bm25"],
    )
    spec = importlib.util.spec_from_file_location(
        "build_candidate_pools", "scripts/build_candidate_pools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main()
    assert rc == 0

    pools_path = tmp_path / "candidate_pools_v1.jsonl"
    pools = [
        CandidatePool.from_dict(json.loads(line))
        for line in pools_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(pools) == 1
    pool = pools[0]
    # Both gold chunks must be in the pool
    assert "gold_a" in pool.candidate_ids
    assert "gold_b" in pool.candidate_ids
    assert pool.gold_in_pool is True
    assert pool.composition.gold_count == 2


def test_build_candidate_pools_rejects_too_many_gold_for_pool_size(tmp_path: Path, monkeypatch):
    """If the query has more gold chunks than pool_size, the build
    script raises a clear error rather than silently dropping gold."""
    config_path = tmp_path / "config.json"
    corpus_path = tmp_path / "corpus_chunks_v1.jsonl"
    queries_path = tmp_path / "queries_v1.jsonl"

    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(1, 6)]
    queries = [
        Query.from_dict({
            "query_id": "q1",
            "query": "test",
            "task_type": "doc_qa",
            "difficulty": "hard",
            "gold_answer": "x",
            "gold_support_ids": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "metadata": {
                "topic": "authentication",
                "requires_multi_hop": True,
                "question_family": "comparison",
            },
        }),
    ]
    _write_jsonl(corpus_path, [c.to_dict() for c in chunks])
    _write_jsonl(queries_path, [q.to_dict() for q in queries])

    config_path.write_text(
        json.dumps({
            "experiment_name": "test",
            "model_name": "stub",
            "selector_strategy": "default",
            "evaluator_version": "eval_v1",
            "dataset_dir": str(tmp_path),
            "artifact_version": "v1",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_candidate_pools", "--config", str(config_path), "--retriever", "bm25", "--pool-size", "5"],
    )
    spec = importlib.util.spec_from_file_location(
        "build_candidate_pools", "scripts/build_candidate_pools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="6 gold chunks"):
        module.main()
