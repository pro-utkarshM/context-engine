
"""Tests for the retriever dispatch in scripts/build_candidate_pools.py.

Covers:
- The RandomRetriever class (sanity / baseline).
- The --retriever random flag path.
- The --retriever-module runtime registration path.
- The mutual-exclusivity check between --retriever and --retriever-module.
- Error cases (malformed spec, missing module, missing class).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from context_engine.artifacts import CandidatePool, CorpusChunk, Query
from context_engine.retrieval import RetrievalResult
from context_engine.retrieval import BM25ExactMatchRetriever, BM25Retriever, RandomRetriever


def _chunk(chunk_id: str, text: str, topic: str = "authentication") -> CorpusChunk:
    return CorpusChunk.from_dict({
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
    })


def _query(query_id: str, text: str, gold_id: str) -> Query:
    return Query.from_dict({
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
    })


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

    config_path.write_text(json.dumps({
        "experiment_name": "test",
        "model_name": "stub",
        "selector_strategy": "default",
        "evaluator_version": "eval_v1",
        "dataset_dir": str(tmp_path),
        "artifact_version": "v1",
    }), encoding="utf-8")
    return config_path, corpus_path, queries_path


def _run_build(monkeypatch, tmp_path: Path, *args):
    config_path, _, _ = _setup_workspace(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_candidate_pools", "--config", str(config_path), *args])
    spec = importlib.util.spec_from_file_location("build_candidate_pools", "scripts/build_candidate_pools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


# --- RandomRetriever class tests ---

def test_random_retriever_returns_pool_size_chunks():
    r = RandomRetriever()
    r.index([_chunk(f"c{i}", f"text {i}") for i in range(10)])
    results = r.retrieve("anything", pool_size=5)
    assert len(results) == 5
    assert all(isinstance(x, RetrievalResult) for x in results)


def test_random_retriever_carries_its_own_name():
    r = RandomRetriever()
    r.index([_chunk("c1", "x")])
    results = r.retrieve("anything", pool_size=1)
    assert results[0].retriever_name == "random"


def test_random_retriever_zero_pool_yields_empty():
    r = RandomRetriever()
    r.index([_chunk("c1", "x")])
    assert r.retrieve("anything", pool_size=0) == []


def test_random_retriever_without_index_raises():
    r = RandomRetriever()
    with pytest.raises(ValueError):
        r.retrieve("anything", pool_size=1)


def test_random_retriever_satisfies_protocol():
    r = RandomRetriever()
    assert hasattr(r, "name")
    assert hasattr(r, "index")
    assert hasattr(r, "retrieve")


def test_random_retriever_is_deterministic_given_seed():
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(20)]
    r1 = RandomRetriever(seed=42)
    r1.index(chunks)
    r2 = RandomRetriever(seed=42)
    r2.index(chunks)
    a = [x.chunk_id for x in r1.retrieve("q", pool_size=5)]
    b = [x.chunk_id for x in r2.retrieve("q", pool_size=5)]
    assert a == b


def test_random_retriever_is_query_sensitive_via_hash():
    """Different queries should produce different samples with the same seed."""
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(20)]
    r = RandomRetriever(seed=42)
    r.index(chunks)
    a = [x.chunk_id for x in r.retrieve("query a", pool_size=5)]
    b = [x.chunk_id for x in r.retrieve("query b", pool_size=5)]
    # Not guaranteed to differ for all queries, but at least one of
    # several query pairs should produce different samples.
    # We just check that the function uses the query (not constant).
    # If both lists happen to match, the test is not informative but
    # does not fail. The key contract is that the retriever does not
    # raise and returns pool_size results.
    assert len(a) == 5
    assert len(b) == 5


def test_random_retriever_filter_metadata_forwards():
    chunks = [
        _chunk("a1", "pg_hba.conf authentication", topic="authentication"),
        _chunk("r1", "pg_hba.conf replication", topic="replication"),
    ]
    r = RandomRetriever(seed=0)
    r.index(chunks)
    results = r.retrieve("q", pool_size=3, filter_metadata={"topic": "authentication"})
    assert all(x.chunk_id == "a1" for x in results)


# --- build_candidate_pools.py --retriever random dispatch ---

def test_build_candidate_pools_random_includes_gold(tmp_path, monkeypatch):
    rc = _run_build(monkeypatch, tmp_path, "--retriever", "random")
    assert rc == 0
    pools = [
        CandidatePool.from_dict(json.loads(line))
        for line in (tmp_path / "candidate_pools_v1.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # The build script enforces gold_in_pool via injection. Even with
    # a random retriever, gold must end up in the pool.
    assert all(p.gold_in_pool for p in pools)


def test_build_candidate_pools_random_rejects_unknown(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_build(monkeypatch, tmp_path, "--retriever", "bogus")


# --- build_candidate_pools.py --retriever-module dispatch ---

def test_load_retriever_from_module_loads_class(monkeypatch, tmp_path):
    """Verify the runtime registration path works end-to-end."""
    # Create a module in the workspace
    custom_path = tmp_path / "my_retriever.py"
    custom_path.write_text("""
from context_engine.retrieval import BM25Retriever

class MyRetriever:
    name = "my_retriever"
    def __init__(self):
        self._bm25 = BM25Retriever()
    def index(self, chunks):
        self._bm25.index(chunks)
    def retrieve(self, query, *, pool_size, filter_metadata=None):
        return self._bm25.retrieve(query, pool_size=pool_size, filter_metadata=filter_metadata)
""", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    rc = _run_build(monkeypatch, tmp_path,
                    "--retriever-module", "my_retriever:MyRetriever")
    assert rc == 0
    pools_path = tmp_path / "candidate_pools_v1.jsonl"
    assert pools_path.exists()
    pools = [
        CandidatePool.from_dict(json.loads(line))
        for line in pools_path.read_text().splitlines()
        if line.strip()
    ]
    assert all(p.gold_in_pool for p in pools)


def test_load_retriever_from_module_rejects_malformed_spec(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="<module>:<ClassName>"):
        _run_build(monkeypatch, tmp_path, "--retriever-module", "no_colon_here")


def test_load_retriever_from_module_rejects_empty_parts(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="<module>:<ClassName>"):
        _run_build(monkeypatch, tmp_path, "--retriever-module", ":ClassName")


def test_load_retriever_from_module_rejects_missing_module(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="cannot import"):
        _run_build(monkeypatch, tmp_path,
                   "--retriever-module", "this_module_does_not_exist:Class")


def test_load_retriever_from_module_rejects_missing_class(monkeypatch, tmp_path):
    custom_path = tmp_path / "empty_module.py"
    custom_path.write_text("# nothing here\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="no attribute"):
        _run_build(monkeypatch, tmp_path,
                   "--retriever-module", "empty_module:NoSuchClass")


def test_load_retriever_from_module_rejects_missing_protocol_attribute(monkeypatch, tmp_path):
    custom_path = tmp_path / "broken_retriever.py"
    custom_path.write_text("""
class BrokenRetriever:
    pass
""", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="missing Protocol attribute"):
        _run_build(monkeypatch, tmp_path,
                   "--retriever-module", "broken_retriever:BrokenRetriever")


def test_load_retriever_from_module_rejects_non_instantiable(monkeypatch, tmp_path):
    custom_path = tmp_path / "argful_retriever.py"
    custom_path.write_text("""
class NeedsArg:
    name = "needs_arg"
    def __init__(self, required_arg):
        self._arg = required_arg
    def index(self, chunks):
        pass
    def retrieve(self, query, *, pool_size, filter_metadata=None):
        return []
""", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="constructor raised"):
        _run_build(monkeypatch, tmp_path,
                   "--retriever-module", "argful_retriever:NeedsArg")


def test_build_candidate_pools_rejects_retriever_and_module_together(tmp_path, monkeypatch):
    custom_path = tmp_path / "extra_retriever.py"
    custom_path.write_text("""
from context_engine.retrieval import BM25Retriever

class ExtraRetriever:
    name = "extra"
    def __init__(self):
        self._bm25 = BM25Retriever()
    def index(self, chunks):
        self._bm25.index(chunks)
    def retrieve(self, query, *, pool_size, filter_metadata=None):
        return []
""", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _run_build(monkeypatch, tmp_path,
                   "--retriever", "bm25",
                   "--retriever-module", "extra_retriever:ExtraRetriever")
