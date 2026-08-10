"""Integrity tests for the Phase O confirmatory audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from context_engine.paired_query import paired_query_summary
from scripts.confirmatory_analysis_unified import load_canonical_outcomes, load_learned_outcomes, run_paired


ROOT = Path(__file__).resolve().parents[1]


def _qids(start: int, stop: int) -> list[str]:
    return [f"q_{number:04d}" for number in range(start, stop + 1)]


def _run_artifact_view(learned_dir: str, canonical_dir: str, qids: list[str]) -> dict:
    learned = load_learned_outcomes(ROOT / learned_dir)
    canonical = load_canonical_outcomes(ROOT / canonical_dir)
    topk = {
        query_id: strategies["topk_pool_order"]
        for query_id, strategies in canonical.items()
        if "topk_pool_order" in strategies
    }
    return run_paired(learned, topk, "topk_pool_order", qids)


def test_pr21_development_regression_delta():
    result = _run_artifact_view(
        "data/processed/learned_v3_context_first",
        "data/processed/canon_r4_context_first",
        _qids(1, 10),
    )
    assert result["mean_delta"] == pytest.approx(0.0588)


def test_pr22_development_regression_delta():
    result = _run_artifact_view(
        "data/processed/learned_v3_confirmatory",
        "data/processed/canon_r4_confirmatory",
        _qids(1, 10),
    )
    assert result["mean_delta"] == pytest.approx(-0.0588)


def test_pr22_confirmatory_regression_delta():
    result = _run_artifact_view(
        "data/processed/learned_v3_confirmatory",
        "data/processed/canon_r4_confirmatory",
        _qids(11, 30),
    )
    assert result["mean_delta"] == pytest.approx(0.0504)


def test_swapping_left_and_right_flips_delta():
    forward = paired_query_summary(
        {"q1": [0.7], "q2": [0.8]},
        {"q1": [0.5], "q2": [0.6]},
        left_label="learned",
        right_label="comparison",
        n_resamples=200,
        seed=0,
    )
    backward = paired_query_summary(
        {"q1": [0.5], "q2": [0.6]},
        {"q1": [0.7], "q2": [0.8]},
        left_label="comparison",
        right_label="learned",
        n_resamples=200,
        seed=0,
    )
    assert forward.delta_summary.mean_delta == pytest.approx(-backward.delta_summary.mean_delta)
    assert forward.delta_summary.ci_low == pytest.approx(-backward.delta_summary.ci_high)
    assert forward.delta_summary.ci_high == pytest.approx(-backward.delta_summary.ci_low)


def test_positive_delta_is_a_win():
    result = paired_query_summary(
        {"q1": [0.7]}, {"q1": [0.5]}, left_label="learned", right_label="comparison"
    )
    assert result.per_query_deltas["q1"] > 0


def test_negative_delta_is_a_loss():
    result = paired_query_summary(
        {"q1": [0.5]}, {"q1": [0.7]}, left_label="learned", right_label="comparison"
    )
    assert result.per_query_deltas["q1"] < 0


def test_zero_delta_is_a_tie():
    result = paired_query_summary(
        {"q1": [0.5]}, {"q1": [0.5]}, left_label="learned", right_label="comparison"
    )
    assert result.per_query_deltas["q1"] == 0


def _load_queries() -> list[dict]:
    with (ROOT / "data/processed/queries_v1.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_split_membership_has_no_overlap():
    queries = _load_queries()
    development = {query["query_id"] for query in queries if query["metadata"]["split"] == "development"}
    confirmatory = {query["query_id"] for query in queries if query["metadata"]["split"] == "confirmatory"}
    assert development.isdisjoint(confirmatory)
    assert development | confirmatory == {query["query_id"] for query in queries}


def test_development_split_has_exact_qid_range():
    queries = _load_queries()
    development = {
        query["query_id"] for query in queries if query["metadata"]["split"] == "development"
    }
    assert development == set(_qids(1, 10))


def test_confirmatory_split_has_exact_qid_range():
    queries = _load_queries()
    confirmatory = {
        query["query_id"] for query in queries if query["metadata"]["split"] == "confirmatory"
    }
    assert confirmatory == set(_qids(11, 30))


def test_full_multihop_percentage_is_not_27_percent():
    queries = _load_queries()
    count = sum(query["metadata"]["requires_multi_hop"] for query in queries)
    percentage = count / len(queries) * 100
    assert count == 8
    assert percentage == pytest.approx(26.6666666667)
    assert percentage != 27


def test_frozen_input_hashes_match():
    expected = {
        "queries_v1.jsonl": "588c3fb1858092e7434496cae4a884e65ef3d482c1d2092d90982f66d75b8f35",
        "corpus_chunks_v1.jsonl": "1e1c37bca3eba0b0d2902e9d661b752af9ccdd96e99bb728bd5040fbcb9186b2",
        "candidate_pools_v1.jsonl": "9b9d15a5b925e36df6d77d85d6f11930f9f7b6b744926881e76d4d81c86ee028",
        "context_sets_v1.jsonl": "1e311e6203c8ee927b2867de055aaee426dfe4009a0a719261883cbf7cdc96de",
        "marginal_impact_minimax_v1.jsonl": "a489fe4cdce9185c1e40414feaeba453eb1e06b3d29660c8abd52c800b679420",
    }
    for name, digest in expected.items():
        path = ROOT / "data/processed" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_unified_analysis_uses_paired_query_summary():
    script = ROOT / "scripts/confirmatory_analysis_unified.py"
    assert script.is_file()
    assert "paired_query_summary" in script.read_text()
